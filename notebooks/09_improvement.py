
import argparse
import ast
import csv
import getpass
import hashlib
import importlib.util
import json
import math
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import psycopg
from psycopg import sql


DATABASE = "adaptive_healthcare_phase9_benchmark"
ALLOWED_HOSTS = {"localhost", "127.0.0.1"}
ALLOWED_PORT = 5433

CANDIDATES = [
    ("patient_procedure", ["patient_id", "procedure_id"]),
    ("claim", ["payer_id", "patient_id"]),
    ("lab_result", ["loinc_code", "patient_id"]),
    ("prescription", ["patient_id", "prescription_id"]),
    ("lab_result", ["patient_id", "result_time DESC"]),
]

class IntentionalRollback(Exception):
    pass


def sha(value):
    return hashlib.sha256(
        str(value).encode("utf-8")
    ).hexdigest()[:20]


def as_float(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def as_bool(value):
    return str(value).strip().lower() in {
        "true", "1", "yes"
    }


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def save_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fields is None:
        fields = list(dict.fromkeys(
            key for row in rows for key in row
        ))

    if not fields:
        fields = ["status"]

    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
            extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            data,
            indent=2,
            default=str,
            allow_nan=False
        ),
        encoding="utf-8"
    )


def safe_select(query):
    query = str(query or "").strip()

    if not re.match(r"(?is)^SELECT\b", query):
        return False

    if ";" in query.rstrip(";"):
        return False

    forbidden = (
        r"\b(INSERT|UPDATE|DELETE|MERGE|DROP|ALTER|CREATE|"
        r"TRUNCATE|GRANT|REVOKE|COPY|CALL|DO|EXECUTE|"
        r"VACUUM|ANALYZE|LOCK|SET|RESET|INTO|"
        r"PG_SLEEP|DBLINK|NEXTVAL|SETVAL|"
        r"LO_IMPORT|LO_EXPORT|PG_ADVISORY)\b"
    )

    if re.search(forbidden, query, re.IGNORECASE):
        return False

    if re.search(
        r"\bFOR\s+(UPDATE|SHARE|NO\s+KEY\s+UPDATE)\b",
        query,
        re.IGNORECASE
    ):
        return False

    return True


def index_columns(columns):
    output = []

    for item in columns:
        match = re.fullmatch(
            r"([A-Za-z_][A-Za-z_0-9]*)"
            r"(?:\s+(ASC|DESC))?",
            item.strip(),
            re.IGNORECASE
        )

        if not match:
            raise ValueError(f"Invalid index column: {item}")

        name, direction = match.groups()

        part = sql.Identifier(name)

        if direction:
            part += sql.SQL(" " + direction.upper())

        output.append(part)

    return sql.SQL(", ").join(output)


def create_index_sql(table, columns, name):
    return sql.SQL(
        "CREATE INDEX {} ON public.{} USING btree ({})"
    ).format(
        sql.Identifier(name),
        sql.Identifier(table),
        index_columns(columns)
    )


def verify_database(conn):
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                current_database(),
                current_setting('port')::integer,
                current_user
            """
        )

        dbname, internal_port, username = cursor.fetchone()

        if dbname != DATABASE or internal_port != 5432:
            raise RuntimeError(
                f"Unsafe database connection: {dbname}:{internal_port}"
            )

        cursor.execute("SHOW transaction_read_only")

        if cursor.fetchone()[0] != "off":
            raise RuntimeError(
                "Benchmark connection must permit temporary index DDL"
            )

    print(
        f"Verified Docker database: {dbname} | User: {username}",
        flush=True
    )


def get_columns(conn, table):
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = %s
            """,
            (table,)
        )

        return {row[0] for row in cursor.fetchall()}


def existing_index(conn, table, columns):
    desired = [
        item.split()[0].lower() for item in columns
    ]

    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT indexdef
            FROM pg_indexes
            WHERE schemaname = 'public'
              AND tablename = %s
            """,
            (table,)
        )

        definitions = [row[0] for row in cursor.fetchall()]

    for definition in definitions:
        match = re.search(r"\(([^)]+)\)", definition)

        if not match:
            continue

        existing = [
            item.strip().strip('"').split()[0].lower()
            for item in match.group(1).split(",")
        ]

        if existing[:len(desired)] == desired:
            return True

    return False


def load_positive_queries(workload_path, detail_path):
    workload = read_csv(workload_path)
    details = read_csv(detail_path)

    lookup = defaultdict(dict)
    frequencies = Counter()

    for row in workload:
        query = str(row.get("query_text") or "").strip()
        query = query.rstrip(";").strip()

        if not as_bool(row.get("safe", "True")):
            continue

        if not safe_select(query):
            continue

        family = str(row.get("query_family") or "").strip()
        name = str(row.get("query_name") or "").strip()

        key = (family, name)
        identity = sha(query)

        frequencies[(key, identity)] += 1

        lookup[key][identity] = {
            "query": query,
            "query_hash": identity,
            "query_family": family,
            "query_name": name
        }

    positive = defaultdict(dict)

    for row in details:
        if not as_bool(row.get("hypothetical_index_used")):
            continue

        gain = as_float(row.get("gain_pct"))

        if gain <= 0:
            continue

        table = str(row.get("table") or "").strip()
        columns = str(row.get("columns") or "").strip()

        key = (
            str(row.get("query_family") or "").strip(),
            str(row.get("query_name") or "").strip()
        )

        for identity, item in lookup.get(key, {}).items():
            record = dict(item)
            record["hypopg_family_gain_pct"] = gain
            record["workload_frequency"] = frequencies[
                (key, identity)
            ]

            candidate_key = (table, columns)

            previous = positive[candidate_key].get(identity)

            if previous is None:
                positive[candidate_key][identity] = record

            elif gain > previous["hypopg_family_gain_pct"]:
                positive[candidate_key][identity] = record

    return positive, workload


def select_queries(positive, table, columns, limit):
    key = (table, " | ".join(columns))

    queries = list(positive.get(key, {}).values())

    queries.sort(
        key=lambda item: (
            -item["workload_frequency"],
            -item["hypopg_family_gain_pct"],
            item["query_hash"]
        )
    )

    return queries[:limit]


def explain_analyze(conn, query):
    if not safe_select(query):
        raise ValueError("Unsafe SQL rejected")

    with conn.cursor() as cursor:
        cursor.execute(
            "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + query
        )

        result = cursor.fetchone()[0]

    if isinstance(result, str):
        result = json.loads(result)

    if isinstance(result, list):
        result = result[0]

    root = result["Plan"]

    indexes = []

    def visit(node):
        if node.get("Index Name"):
            indexes.append(str(node["Index Name"]))

        for child in node.get("Plans", []):
            visit(child)

    visit(root)

    def total_buffers(node, field):
        total = as_float(node.get(field))
        for child in node.get("Plans", []):
            total += total_buffers(child, field)
        return total

    return {
        "execution_ms": as_float(result.get("Execution Time")),
        "planning_ms": as_float(result.get("Planning Time")),
        "planner_cost": as_float(root.get("Total Cost")),
        "shared_hit_blocks": as_float(
            root.get("Shared Hit Blocks")
        ),
        "shared_read_blocks": as_float(
            root.get("Shared Read Blocks")
        ),
        "indexes_used": " | ".join(indexes),
        "root_plan": root.get("Node Type", "")
    }


def measure_batch(
    conn,
    queries,
    stage,
    warmups,
    repeats,
    seed
):
    rng = random.Random(seed)

    records = []

    for item in queries:
        for _ in range(warmups):
            try:
                explain_analyze(conn, item["query"])
            except Exception:
                break

    sequence = [
        (item, repeat)
        for repeat in range(repeats)
        for item in queries
    ]

    rng.shuffle(sequence)

    for position, (item, repeat) in enumerate(sequence, 1):
        record = {
            "stage": stage,
            "query_hash": item["query_hash"],
            "query_family": item["query_family"],
            "query_name": item["query_name"],
            "hypopg_family_gain_pct": item[
                "hypopg_family_gain_pct"
            ],
            "workload_frequency": item["workload_frequency"],
            "repeat": repeat + 1,
            "sequence": position,
            "success": False,
            "error": ""
        }

        try:
            measured = explain_analyze(conn, item["query"])

            record.update(measured)
            record["success"] = True

        except Exception as exc:
            record["error"] = str(exc)[:400]

            # A statement timeout can abort the current transaction.
            # Stop this batch so the outer transaction can roll back.
            records.append(record)
            break

        records.append(record)

    return records


def run_transaction_batch(
    conn,
    table,
    columns,
    index_name,
    queries,
    stage,
    args,
    round_number
):
    batch = []
    index_size = None

    try:
        with conn.transaction():
            with conn.cursor() as cursor:
                cursor.execute(
                    "SET LOCAL lock_timeout = '3000ms'"
                )

                # FIX: PostgreSQL SET does not accept $1 here.
                cursor.execute(
                    "SELECT set_config('statement_timeout', %s, true)",
                    (f"{args.timeout_ms}ms",)
                )

                if stage == "indexed":
                    cursor.execute(
                        create_index_sql(
                            table,
                            columns,
                            index_name
                        )
                    )

                    # No ANALYZE here: use the same existing
                    # table statistics in both conditions.
                    cursor.execute(
                        "SELECT pg_relation_size(%s::regclass)",
                        (index_name,)
                    )

                    index_size = cursor.fetchone()[0]

            batch = measure_batch(
                conn=conn,
                queries=queries,
                stage=stage,
                warmups=args.warmups,
                repeats=args.repeats,
                seed=args.seed + round_number * 101
            )

            # Never commit experimental indexes.
            raise IntentionalRollback()

    except IntentionalRollback:
        pass

    return batch, index_size


def percentile(values, percentage):
    if not values:
        return None

    ordered = sorted(values)

    position = (
        (len(ordered) - 1) * percentage / 100.0
    )

    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)

    fraction = position - lower

    return (
        ordered[lower] * (1 - fraction)
        + ordered[upper] * fraction
    )


def summarize_queries(trials, table, columns):
    grouped = defaultdict(lambda: defaultdict(list))
    info = {}

    for record in trials:
        identity = record["query_hash"]
        info[identity] = record

        if record.get("success"):
            grouped[identity][record["stage"]].append(
                as_float(record.get("execution_ms"))
            )

    summaries = []

    for identity, stages in grouped.items():
        baseline = stages.get("baseline", [])
        indexed = stages.get("indexed", [])

        if not baseline or not indexed:
            continue

        before = statistics.median(baseline)
        after = statistics.median(indexed)

        gain = (
            100 * (before - after) / before
            if before > 0 else 0
        )

        summaries.append({
            "table": table,
            "columns": " | ".join(columns),
            "query_hash": identity,
            "query_family": info[identity]["query_family"],
            "query_name": info[identity]["query_name"],
            "workload_frequency": info[identity]["workload_frequency"],
            "baseline_median_ms": round(before, 6),
            "indexed_median_ms": round(after, 6),
            "baseline_p95_ms": round(
                percentile(baseline, 95), 6
            ),
            "indexed_p95_ms": round(
                percentile(indexed, 95), 6
            ),
            "measured_gain_pct": round(gain, 6),
            "speedup": (
                round(before / after, 6)
                if after > 0 else None
            ),
            "baseline_samples": len(baseline),
            "indexed_samples": len(indexed),
            "status": (
                "improved" if gain > 1
                else "regressed" if gain < -1
                else "neutral"
            )
        })

    return summaries


def benchmark_candidate(
    conn,
    table,
    columns,
    queries,
    args
):
    label = f"{table}({', '.join(columns)})"

    print(f"\n[9C] {label}", flush=True)

    required = {
        item.split()[0] for item in columns
    }

    missing = required - get_columns(conn, table)

    if missing:
        return {
            "candidate": label,
            "status": "missing_columns",
            "error": ", ".join(sorted(missing))
        }, [], []

    if existing_index(conn, table, columns):
        return {
            "candidate": label,
            "status": "existing_equivalent_index"
        }, [], []

    if not queries:
        return {
            "candidate": label,
            "status": "no_matched_queries"
        }, [], []

    index_name = "p9cd_" + sha(label)[:12]

    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT EXISTS(
                SELECT 1
                FROM pg_class
                WHERE relname = %s
            )
            """,
            (index_name,)
        )

        if cursor.fetchone()[0]:
            raise RuntimeError(
                f"Index name already exists: {index_name}"
            )

    trials = []
    index_size = None
    errors = []

    for round_number in range(args.rounds):
        stages = (
            ["baseline", "indexed"]
            if round_number % 2 == 0
            else ["indexed", "baseline"]
        )

        for stage in stages:
            try:
                batch, size = run_transaction_batch(
                    conn,
                    table,
                    columns,
                    index_name,
                    queries,
                    stage,
                    args,
                    round_number
                )

                if size is not None:
                    index_size = size

                for record in batch:
                    record["round"] = round_number + 1
                    record["table"] = table
                    record["columns"] = " | ".join(columns)

                trials.extend(batch)

                failures = sum(
                    not item.get("success", False)
                    for item in batch
                )

                print(
                    f"  Round {round_number + 1} "
                    f"{stage}: {len(batch)} trials, "
                    f"{failures} failures",
                    flush=True
                )

            except Exception as exc:
                errors.append(
                    f"round={round_number + 1}, "
                    f"stage={stage}: {exc}"
                )

                print(
                    "  Stage error:",
                    str(exc)[:250],
                    flush=True
                )

    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT EXISTS(
                SELECT 1
                FROM pg_class
                WHERE relname = %s
            )
            """,
            (index_name,)
        )

        if cursor.fetchone()[0]:
            raise RuntimeError(
                "Experimental index was not removed: "
                + index_name
            )

    query_results = summarize_queries(
        trials,
        table,
        columns
    )

    weighted_before = sum(
        item["baseline_median_ms"]
        * item["workload_frequency"]
        for item in query_results
    )

    weighted_after = sum(
        item["indexed_median_ms"]
        * item["workload_frequency"]
        for item in query_results
    )

    gain = (
        100 * (weighted_before - weighted_after)
        / weighted_before
        if weighted_before > 0 else None
    )

    index_used = any(
        index_name in record.get("indexes_used", "")
        for record in trials
        if record.get("stage") == "indexed"
        and record.get("success")
    )

    successful_trials = sum(
        record.get("success", False)
        for record in trials
    )

    failed_trials = sum(
        not record.get("success", False)
        for record in trials
    )

    expected_trials = (
        len(queries) * args.repeats * args.rounds * 2
    )

    completed = (
        len(query_results) == len(queries)
        and successful_trials == expected_trials
        and failed_trials == 0
        and not errors
    )

    result = {
        "candidate": label,
        "table": table,
        "columns": " | ".join(columns),
        "status": (
            "completed" if completed
            else "partial" if query_results
            else "failed"
        ),
        "queries_requested": len(queries),
        "queries_measured": len(query_results),
        "successful_trials": successful_trials,
        "failed_trials": failed_trials,
        "expected_trials": expected_trials,
        "index_used": index_used,
        "index_size_mb": (
            round(index_size / 1048576, 6)
            if index_size is not None else None
        ),
        "measured_gain_pct": (
            round(gain, 6)
            if gain is not None else None
        ),
        "improved_queries": sum(
            item["status"] == "improved"
            for item in query_results
        ),
        "regressed_queries": sum(
            item["status"] == "regressed"
            for item in query_results
        ),
        "error": " | ".join(errors)[:1000]
    }

    print(
        "  Status:", result["status"],
        "| Measured gain:", result["measured_gain_pct"],
        "| Queries:", result["queries_measured"],
        "| Index used:", result["index_used"],
        flush=True
    )

    return result, trials, query_results


def load_optimizer(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        "phase9_original_optimizer",
        path
    )

    module = importlib.util.module_from_spec(spec)

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def integrate_optimizer(module):
    original_generate = module.generate_candidates
    original_index_definition = module.index_definition
    original_relevant = module.relevant_queries

    def enhanced_generate(frame, metadata, existing):
        candidates = original_generate(
            frame,
            metadata,
            existing
        )

        catalog = module.schema_catalog(metadata)

        existing_keys = {
            (
                candidate.table_schema,
                candidate.table_name,
                tuple(candidate.columns)
            )
            for candidate in candidates
        }

        for table, columns in CANDIDATES:
            table_key = ("public", table)

            if table_key not in catalog:
                continue

            required = {
                item.split()[0].lower()
                for item in columns
            }

            if not required.issubset(catalog[table_key]):
                continue

            normalized_key = (
                "public",
                table,
                tuple(columns)
            )

            if normalized_key in existing_keys:
                continue

            matching = frame[
                frame["safe"]
            ]["query_text"].astype(str).apply(
                lambda query: bool(
                    re.search(
                        r"(?i)\b(?:FROM|JOIN)\s+"
                        r"(?:public\.)?"
                        + re.escape(table)
                        + r"\b",
                        query
                    )
                )
            )

            frequency = int(matching.sum())

            if frequency == 0:
                continue

            candidate = module.Candidate(
                candidate_id=module.candidate_key(
                    "public",
                    table,
                    columns,
                    "btree"
                ),
                table_schema="public",
                table_name=table,
                columns=columns,
                index_type="btree",
                source="phase9b_workload_aware",
                frequency=frequency
            )

            candidates.append(candidate)
            existing_keys.add(normalized_key)

        priority = {
            (table, tuple(columns)): position
            for position, (table, columns)
            in enumerate(CANDIDATES)
        }

        candidates.sort(
            key=lambda candidate: priority.get(
                (
                    candidate.table_name,
                    tuple(candidate.columns)
                ),
                len(priority) + 1
            )
        )

        return candidates

    def enhanced_index_definition(
        candidate,
        name,
        hypothetical=False
    ):
        if any(
            " " in column.strip()
            for column in candidate.columns
        ):
            return sql.SQL(
                "CREATE INDEX {} ON {}.{} USING btree ({})"
            ).format(
                sql.Identifier(name),
                sql.Identifier(candidate.table_schema),
                sql.Identifier(candidate.table_name),
                index_columns(candidate.columns)
            )

        return original_index_definition(
            candidate,
            name,
            hypothetical
        )

    def enhanced_relevant_queries(candidate, frame):
        queries = original_relevant(candidate, frame)

        if candidate.source != "phase9b_workload_aware":
            return queries

        table = candidate.table_name

        additional = []

        for _, row in frame[frame["safe"]].iterrows():
            query = str(row["query_text"])

            if re.search(
                r"(?i)\b(?:FROM|JOIN)\s+(?:public\.)?"
                + re.escape(table)
                + r"\b",
                query
            ):
                additional.append(query)

        return list(dict.fromkeys(queries + additional))

    module.generate_candidates = enhanced_generate
    module.index_definition = enhanced_index_definition
    module.relevant_queries = enhanced_relevant_queries


def parse_candidate_columns(value):
    if isinstance(value, list):
        return value

    try:
        parsed = ast.literal_eval(str(value))

        if isinstance(parsed, list):
            return parsed

    except (ValueError, SyntaxError):
        pass

    return []


def run_phase9d(module, args, output, feedback):
    namespace = argparse.Namespace(
        mode="evaluate",
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        database=args.database,
        workload=args.optimizer_workload,
        reference_workload=None,
        model_dir=args.model_dir,
        output=str(output / "phase9d_optimizer"),
        max_queries=args.max_queries,
        max_candidates=args.max_candidates,
        max_experiments=0,
        benchmark_queries=args.queries_per_candidate,
        scale_query_limit=0,
        repeats=args.repeats,
        bootstrap=1000,
        seed=args.seed,
        timeout_ms=args.timeout_ms,
        minimum_gain_pct=args.minimum_gain_pct,
        maximum_index_mb=512.0,
        maximum_risk=0.5,
        allow_index_changes=False,
        confirm_database=None,
        scales=None
    )

    module.run(namespace)

    root = output / "phase9d_optimizer"

    directories = sorted(
        [p for p in root.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    if not directories:
        raise RuntimeError(
            "Phase 9D optimizer produced no output"
        )

    latest = directories[0]

    ranked_file = latest / "ranked_candidates.csv"

    ranked = (
        read_csv(ranked_file)
        if ranked_file.exists() else []
    )

    feedback_lookup = {
        (
            row.get("table"),
            row.get("columns")
        ): row
        for row in feedback
    }

    integrated = []

    for row in ranked:
        columns = parse_candidate_columns(
            row.get("columns", "")
        )

        key = (
            row.get("table_name"),
            " | ".join(columns)
        )

        measurement = feedback_lookup.get(key, {})

        integrated.append({
            **row,
            "phase9c_status": measurement.get(
                "status", "not_benchmarked"
            ),
            "phase9c_measured_gain_pct": measurement.get(
                "measured_gain_pct"
            ),
            "phase9c_queries_measured": measurement.get(
                "queries_measured"
            ),
            "phase9c_index_used": measurement.get(
                "index_used"
            ),
            "automatic_deployment_authorized": False
        })

    save_csv(
        output / "phase9d_integrated_candidates.csv",
        integrated
    )

    return latest, integrated


def main():
    parser = argparse.ArgumentParser(
        description="Phase 9C + 9D Integrated Research Pipeline"
    )

    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5433)
    parser.add_argument("--database", default=DATABASE)
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password")

    parser.add_argument("--workload", required=True)
    parser.add_argument("--phase9b-details", required=True)
    parser.add_argument("--optimizer-workload", required=True)

    parser.add_argument(
        "--optimizer-script",
        default="notebooks/09_optimizer.py"
    )

    parser.add_argument("--model-dir", required=True)

    parser.add_argument(
        "--output",
        default="results/phase9_cd"
    )

    parser.add_argument(
        "--queries-per-candidate",
        type=int,
        default=5
    )

    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--timeout-ms", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-queries", type=int, default=950)
    parser.add_argument("--max-candidates", type=int, default=40)

    parser.add_argument(
        "--minimum-gain-pct",
        type=float,
        default=5.0
    )

    parser.add_argument(
        "--confirm-physical-indexes",
        action="store_true"
    )

    args = parser.parse_args()

    if (
        args.host not in ALLOWED_HOSTS
        or args.port != ALLOWED_PORT
        or args.database != DATABASE
    ):
        raise RuntimeError(
            "Only the isolated Docker benchmark database "
            "at localhost:5433 is permitted."
        )

    if (
        args.repeats < 1
        or args.rounds < 1
        or args.queries_per_candidate < 1
        or args.warmups < 0
    ):
        raise ValueError("Invalid benchmark parameters")

    if not args.password:
        args.password = getpass.getpass(
            "Docker PostgreSQL password: "
        )

    output = (
        Path(args.output)
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    output.mkdir(parents=True, exist_ok=True)

    positive, workload = load_positive_queries(
        args.workload,
        args.phase9b_details
    )

    feedback = []
    trials = []
    query_results = []

    if args.confirm_physical_indexes:
        with psycopg.connect(
            host=args.host,
            port=args.port,
            dbname=args.database,
            user=args.user,
            password=args.password,
            autocommit=True,
            connect_timeout=10
        ) as conn:

            verify_database(conn)

            for table, columns in CANDIDATES[:3]:
                queries = select_queries(
                    positive,
                    table,
                    columns,
                    args.queries_per_candidate
                )

                result, measured, summarized = (
                    benchmark_candidate(
                        conn,
                        table,
                        columns,
                        queries,
                        args
                    )
                )

                feedback.append(result)
                trials.extend(measured)
                query_results.extend(summarized)

    else:
        print(
            "Physical benchmarking skipped. "
            "Use --confirm-physical-indexes to authorize "
            "rollback-only index experiments."
        )

    save_csv(
        output / "phase9c_candidate_results.csv",
        feedback
    )

    save_csv(
        output / "phase9c_trials.csv",
        trials
    )

    save_csv(
        output / "phase9c_query_results.csv",
        query_results
    )

    module = load_optimizer(
        args.optimizer_script
    )

    integrate_optimizer(module)

    latest, integrated = run_phase9d(
        module,
        args,
        output,
        feedback
    )

    report = {
        "project": "Adaptive Healthcare Knowledge Database",
        "phase": "9C + 9D",
        "database": DATABASE,
        "workload_rows": len(workload),
        "physical_benchmark_authorized": (
            args.confirm_physical_indexes
        ),
        "benchmark_candidates": len(feedback),
        "completed_benchmarks": sum(
            row.get("status") == "completed"
            for row in feedback
        ),
        "positive_measured_candidates": sum(
            row.get("status") == "completed"
            and as_float(row.get("measured_gain_pct")) > 0
            for row in feedback
        ),
        "integrated_candidates": len(integrated),
        "optimizer_output": str(latest),
        "v6_feature_compatibility_verified": False,
        "automatic_deployment_authorized": False,
        "limitations": [
            "Phase 9B query matching uses family and name, not exact SQL identity.",
            "Only a bounded sample of the workload is physically benchmarked.",
            "The measured gain is frequency-weighted within that sample, not across all 950 workload rows.",
            "Measurements can be affected by caching and background system activity.",
            "A small positive gain is not proof of statistical significance.",
            "Index write overhead is not measured.",
            "The original optimizer is patched in memory only.",
            "Model feature and regression target compatibility remain unverified.",
            "No permanent index deployment is authorized."
        ]
    }

    save_json(
        output / "phase9cd_final_report.json",
        report
    )

    print("\n" + "=" * 75)
    print("PHASE 9C + 9D COMPLETE")
    print("=" * 75)

    print(
        "Benchmark candidates:",
        report["benchmark_candidates"]
    )

    print(
        "Completed benchmarks:",
        report["completed_benchmarks"]
    )

    print(
        "Positive measured candidates:",
        report["positive_measured_candidates"]
    )

    print(
        "Integrated optimizer candidates:",
        report["integrated_candidates"]
    )

    print("Optimizer results:", latest)
    print("Combined results:", output)
    print("Automatic deployment: DISABLED")
    print("=" * 75)


if __name__ == "__main__":
    main()
