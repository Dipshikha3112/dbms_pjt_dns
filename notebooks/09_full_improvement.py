
import argparse
import ast
import csv
import getpass
import hashlib
import json
import math
import random
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import psycopg
from psycopg import sql


BENCHMARK_DATABASE = "adaptive_healthcare_phase9_benchmark"

DEFAULT_CANDIDATES = [
    ("patient_procedure", ["patient_id", "procedure_id"]),
    ("claim", ["payer_id", "patient_id"]),
    ("lab_result", ["loinc_code", "patient_id"]),
    ("prescription", ["patient_id", "prescription_id"]),
    ("lab_result", ["patient_id", "result_time DESC"]),
    ("prescription_item", ["prescription_id", "rxcui"]),
    ("patient_allergy", ["patient_id"]),
    ("lab_result", ["loinc_code"]),
    ("claim", ["payer_id"]),
    ("patient_procedure", ["patient_id"]),
]


def hash_text(value):
    return hashlib.sha256(
        str(value).encode("utf-8")
    ).hexdigest()[:16]


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, records):
    records = list(records)

    fields = list(dict.fromkeys(
        key
        for record in records
        for key in record
    )) or ["status"]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(records)


def write_json(path, data):
    Path(path).write_text(
        json.dumps(
            data,
            indent=2,
            default=str,
            allow_nan=False
        ),
        encoding="utf-8"
    )


def parse_columns(value):
    if isinstance(value, list):
        return [str(v).strip() for v in value]

    if not value:
        return []

    try:
        parsed = ast.literal_eval(str(value))

        if isinstance(parsed, (list, tuple)):
            return [str(v).strip() for v in parsed]
    except (ValueError, SyntaxError):
        pass

    return [
        value.strip()
        for value in re.split(r"\s*[|,]\s*", str(value))
        if value.strip()
    ]


def canonical_columns(columns):
    return "|".join(
        re.sub(r"\s+", " ", c.strip().lower())
        for c in columns
    )


def column_parts(column):
    match = re.fullmatch(
        r"([A-Za-z_][A-Za-z_0-9]*)"
        r"(?:\s+(ASC|DESC))?",
        column.strip(),
        re.IGNORECASE
    )

    if not match:
        raise ValueError(
            f"Unsupported index column: {column}"
        )

    return (
        match.group(1),
        (match.group(2) or "ASC").upper()
    )


def index_statement(table, columns):
    parts = []

    for column in columns:
        name, direction = column_parts(column)

        parts.append(
            sql.SQL("{} {}").format(
                sql.Identifier(name),
                sql.SQL(direction)
            )
        )

    return sql.SQL(
        "CREATE INDEX ON public.{} USING btree ({})"
    ).format(
        sql.Identifier(table),
        sql.SQL(", ").join(parts)
    )


def safe_query(query):
    query = str(query or "").strip().rstrip(";").strip()

    if not re.match(r"^SELECT\b", query, re.IGNORECASE):
        return False

    if ";" in query or "--" in query or "/*" in query:
        return False

    forbidden = (
        r"\b(INSERT|UPDATE|DELETE|MERGE|CREATE|DROP|"
        r"ALTER|TRUNCATE|COPY|CALL|DO|EXECUTE|GRANT|"
        r"REVOKE|SET|RESET|LOCK|INTO|PG_SLEEP|"
        r"NEXTVAL|SETVAL|DBLINK|LO_IMPORT|LO_EXPORT)\b"
        r"|\bFOR\s+(UPDATE|SHARE)\b"
    )

    return not bool(
        re.search(forbidden, query, re.IGNORECASE)
    )


def load_workload(path):
    original = read_csv(path)
    unique = {}

    for record in original:
        query = str(
            record.get("query_text") or ""
        ).strip().rstrip(";").strip()

        if not safe_query(query):
            continue

        if (
            "safe" in record
            and str(record["safe"]).lower()
            not in ("true", "yes", "1")
        ):
            continue

        query_hash = hash_text(query)

        if query_hash not in unique:
            unique[query_hash] = {
                "query_hash": query_hash,
                "query_text": query,
                "query_family": str(
                    record.get("query_family") or ""
                ),
                "query_name": str(
                    record.get("query_name") or ""
                ),
                "frequency": 0
            }

        unique[query_hash]["frequency"] += 1

    return original, list(unique.values())


def load_candidates(path):
    candidates = list(DEFAULT_CANDIDATES)

    if path:
        for record in read_csv(path):
            table = str(
                record.get("table_name")
                or record.get("table")
                or ""
            ).strip()

            columns = parse_columns(
                record.get("columns")
            )

            if table and columns:
                candidates.append((table, columns))

    unique = {}

    for table, columns in candidates:
        unique[
            (table, canonical_columns(columns))
        ] = (table, columns)

    return list(unique.values())


def query_references_table(query, table):
    pattern = (
        r"\b(?:FROM|JOIN)\s+"
        r"(?:\"?public\"?\.)?"
        r"\"?" + re.escape(table) + r"\"?\b"
    )

    return bool(
        re.search(pattern, query, re.IGNORECASE)
    )


def verify_database(connection):
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                current_database(),
                current_setting('port')::integer,
                pg_backend_pid()
            """
        )

        database, port, backend = cursor.fetchone()

        if database != BENCHMARK_DATABASE or port != 5432:
            raise RuntimeError(
                "Unsafe database connection refused"
            )

        cursor.execute(
            """
            SELECT EXISTS(
                SELECT 1
                FROM pg_extension
                WHERE extname = 'hypopg'
            )
            """
        )

        if not cursor.fetchone()[0]:
            raise RuntimeError(
                "HypoPG extension is unavailable"
            )

    print(
        "Verified database:",
        database,
        "| Backend:",
        backend,
        flush=True
    )


def load_metadata(connection):
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT table_name, column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
            """
        )

        available = defaultdict(set)

        for table, column in cursor.fetchall():
            available[table].add(column)

        cursor.execute(
            """
            SELECT
                t.relname,
                i.relname,
                am.amname,
                ix.indisvalid,
                pg_get_indexdef(ix.indexrelid),
                (
                    SELECT array_agg(
                        pg_get_indexdef(ix.indexrelid, k, true)
                        ORDER BY k
                    )
                    FROM generate_series(1, ix.indnkeyatts) k
                )
            FROM pg_index ix
            JOIN pg_class t ON t.oid = ix.indrelid
            JOIN pg_class i ON i.oid = ix.indexrelid
            JOIN pg_namespace ns ON ns.oid = t.relnamespace
            JOIN pg_am am ON am.oid = i.relam
            WHERE ns.nspname = 'public'
            """
        )

        indexes = [
            {
                "table": row[0],
                "name": row[1],
                "method": row[2],
                "valid": row[3],
                "definition": row[4],
                "key_columns": row[5]
            }
            for row in cursor.fetchall()
        ]

    return available, indexes


def find_overlapping_indexes(table, columns, indexes):
    required = [column_parts(c) for c in columns]
    matches = []

    for index in indexes:
        if (
            index["table"] != table
            or index["method"] != "btree"
            or not index["valid"]
        ):
            continue

        actual = []

        for expression in index["key_columns"] or []:
            try:
                actual.append(
                    column_parts(
                        expression.replace('"', "")
                    )
                )
            except ValueError:
                actual = []
                break

        if (
            len(actual) >= len(required)
            and actual[:len(required)] == required
        ):
            matches.append(index["name"])

    return matches


def explain(connection, query):
    with connection.cursor() as cursor:
        cursor.execute(
            "EXPLAIN (FORMAT JSON) " + query
        )

        result = cursor.fetchone()[0]

    if isinstance(result, str):
        result = json.loads(result)

    plan = result[0]["Plan"]
    index_names = []

    def walk(node):
        if node.get("Index Name"):
            index_names.append(node["Index Name"])

        for child in node.get("Plans", []):
            walk(child)

    walk(plan)

    return {
        "cost": float(plan["Total Cost"]),
        "indexes": index_names
    }


def reset_hypopg(connection):
    with connection.cursor() as cursor:
        cursor.execute("SELECT hypopg_reset()")


def create_hypothetical_index(connection, table, columns):
    statement = index_statement(
        table, columns
    ).as_string(connection)

    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT * FROM hypopg_create_index(%s)",
            (statement,)
        )

        oid, name = cursor.fetchone()

        cursor.execute(
            "SELECT indexrelid FROM hypopg_list_indexes"
        )

        registered = {
            int(row[0])
            for row in cursor.fetchall()
        }

    if int(oid) not in registered:
        raise RuntimeError(
            "Hypothetical index was not registered"
        )

    return str(name)


def evaluate_candidate(
    connection,
    table,
    columns,
    queries,
    protocol,
    split,
    seed
):
    reset_hypopg(connection)
    results = []

    try:
        baseline = {}

        for item in queries:
            try:
                baseline[item["query_hash"]] = explain(
                    connection,
                    item["query_text"]
                )
            except Exception as exc:
                results.append({
                    "candidate": table,
                    "query_hash": item["query_hash"],
                    "protocol": protocol,
                    "split": split,
                    "seed": seed,
                    "status": "baseline_error",
                    "error": str(exc)[:300]
                })

        hypothetical_name = create_hypothetical_index(
            connection, table, columns
        )

        for item in queries:
            query_hash = item["query_hash"]

            if query_hash not in baseline:
                continue

            before = baseline[query_hash]

            record = {
                "table": table,
                "columns": canonical_columns(columns),
                "query_hash": query_hash,
                "query_family": item["query_family"],
                "query_name": item["query_name"],
                "frequency": item["frequency"],
                "protocol": protocol,
                "split": split,
                "seed": seed,
                "before_cost": before["cost"],
                "status": "ok",
                "error": ""
            }

            try:
                after = explain(
                    connection,
                    item["query_text"]
                )

                gain = (
                    100.0
                    * (before["cost"] - after["cost"])
                    / before["cost"]
                    if before["cost"] > 0
                    else 0.0
                )

                record.update({
                    "after_cost": after["cost"],
                    "gain_pct": gain,
                    "index_used": (
                        hypothetical_name
                        in after["indexes"]
                    )
                })

            except Exception as exc:
                record["status"] = "indexed_error"
                record["error"] = str(exc)[:300]

            results.append(record)

    finally:
        reset_hypopg(connection)

    return results


def aggregate(records):
    valid = [
        record
        for record in records
        if record.get("status") == "ok"
    ]

    before = sum(
        record["before_cost"] * record["frequency"]
        for record in valid
    )

    after = sum(
        record["after_cost"] * record["frequency"]
        for record in valid
    )

    return {
        "queries": len(valid),
        "before_cost": before,
        "after_cost": after,
        "gain_pct": (
            100.0 * (before - after) / before
            if before > 0 else None
        ),
        "index_used": sum(
            bool(record.get("index_used"))
            for record in valid
        ),
        "errors": sum(
            record.get("status") != "ok"
            for record in records
        )
    }


def split_within_template(workload, seed, fraction):
    groups = defaultdict(list)

    for item in workload:
        key = (
            item["query_family"],
            item["query_name"]
        )
        groups[key].append(item)

    random_generator = random.Random(seed)

    discovery = []
    holdout = []

    for group in sorted(groups):
        queries = sorted(
            groups[group],
            key=lambda item: item["query_hash"]
        )

        random_generator.shuffle(queries)

        if len(queries) < 2:
            discovery.extend(queries)
            continue

        count = max(
            1,
            min(
                len(queries) - 1,
                round(len(queries) * fraction)
            )
        )

        holdout.extend(queries[:count])
        discovery.extend(queries[count:])

    return discovery, holdout


def split_unseen_template(workload, seed, fraction):
    groups = defaultdict(list)

    for item in workload:
        key = (
            item["query_family"],
            item["query_name"]
        )
        groups[key].append(item)

    keys = sorted(groups)

    random.Random(seed).shuffle(keys)

    if len(keys) < 2:
        return [], []

    count = max(
        1,
        min(len(keys) - 1, round(len(keys) * fraction))
    )

    held_out_groups = set(keys[:count])

    discovery = []
    holdout = []

    for item in workload:
        key = (
            item["query_family"],
            item["query_name"]
        )

        if key in held_out_groups:
            holdout.append(item)
        else:
            discovery.append(item)

    return discovery, holdout


def bootstrap_interval(records, seed, iterations):
    valid = [
        record
        for record in records
        if record.get("status") == "ok"
    ]

    if len(valid) < 2:
        return None, None

    rng = random.Random(seed)
    gains = []

    for _ in range(iterations):
        sample = rng.choices(
            valid,
            k=len(valid)
        )

        summary = aggregate(sample)

        if summary["gain_pct"] is not None:
            gains.append(summary["gain_pct"])

    if not gains:
        return None, None

    gains.sort()

    return (
        gains[int(0.025 * (len(gains) - 1))],
        gains[int(0.975 * (len(gains) - 1))]
    )


def evaluate_portfolio(connection, candidates, queries):
    reset_hypopg(connection)

    try:
        baseline = []

        for item in queries:
            try:
                plan = explain(
                    connection,
                    item["query_text"]
                )

                baseline.append(
                    (item, plan["cost"])
                )
            except Exception:
                continue

        hypothetical_names = []

        for table, columns in candidates:
            name = create_hypothetical_index(
                connection,
                table,
                columns
            )

            hypothetical_names.append(name)

        total_before = 0.0
        total_after = 0.0
        successful = 0
        used_indexes = set()

        for item, before in baseline:
            try:
                after = explain(
                    connection,
                    item["query_text"]
                )
            except Exception:
                continue

            weight = item["frequency"]

            total_before += before * weight
            total_after += after["cost"] * weight
            successful += 1

            used_indexes.update(
                set(after["indexes"])
                & set(hypothetical_names)
            )

        return {
            "queries": successful,
            "indexes": len(candidates),
            "used_indexes": len(used_indexes),
            "before_cost": total_before,
            "after_cost": total_after,
            "gain_pct": (
                100.0
                * (total_before - total_after)
                / total_before
                if total_before > 0
                else None
            )
        }

    finally:
        reset_hypopg(connection)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 9 V2 research evaluation"
    )

    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5433)
    parser.add_argument(
        "--database",
        default=BENCHMARK_DATABASE
    )
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password")

    parser.add_argument("--workload", required=True)
    parser.add_argument("--original-ranked")

    parser.add_argument(
        "--output",
        default="results/phase9_publication_v2"
    )

    parser.add_argument(
        "--seeds",
        default="42,43,44,45,46"
    )

    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.25
    )

    parser.add_argument(
        "--max-queries-per-candidate",
        type=int,
        default=60
    )

    parser.add_argument(
        "--portfolio-query-limit",
        type=int,
        default=100
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000
    )

    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=15000
    )

    args = parser.parse_args()

    if (
        args.host not in ("localhost", "127.0.0.1")
        or args.port != 5433
        or args.database != BENCHMARK_DATABASE
    ):
        raise RuntimeError(
            "Only the isolated Docker benchmark "
            "database on localhost:5433 is permitted."
        )

    if not 0 < args.holdout_fraction < 1:
        raise ValueError("Invalid holdout fraction")

    if min(
        args.max_queries_per_candidate,
        args.portfolio_query_limit,
        args.top_k,
        args.bootstrap,
        args.timeout_ms
    ) < 1:
        raise ValueError("All limits must be positive")

    seeds = [
        int(value.strip())
        for value in args.seeds.split(",")
        if value.strip()
    ]

    if not seeds:
        raise ValueError("At least one seed is required")

    password = (
        args.password
        if args.password is not None
        else getpass.getpass(
            "Docker PostgreSQL password: "
        )
    )

    output = (
        Path(args.output)
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    output.mkdir(parents=True, exist_ok=True)

    original, workload = load_workload(
        args.workload
    )

    candidates = load_candidates(
        args.original_ranked
    )

    query_results = []
    candidate_results = []
    portfolio_results = []
    diagnostics = []
    split_manifest = []

    with psycopg.connect(
        host=args.host,
        port=args.port,
        dbname=args.database,
        user=args.user,
        password=password,
        autocommit=True,
        connect_timeout=10
    ) as connection:

        verify_database(connection)

        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT set_config('statement_timeout', %s, false)",
                (f"{args.timeout_ms}ms",)
            )

            cursor.execute(
                "SELECT set_config('default_transaction_read_only', 'on', false)"
            )

        available, indexes = load_metadata(
            connection
        )

        write_csv(
            output / "existing_indexes.csv",
            indexes
        )

        valid_candidates = []

        for table, columns in candidates:
            if (
                table not in available
                or any(
                    column_parts(column)[0]
                    not in available[table]
                    for column in columns
                )
            ):
                diagnostics.append({
                    "candidate": table,
                    "status": "missing_schema"
                })
                continue

            valid_candidates.append(
                (table, columns)
            )

        protocols = {
            "within_template": split_within_template,
            "unseen_template": split_unseen_template
        }

        for protocol, split_function in protocols.items():
            for seed in seeds:
                discovery, holdout = split_function(
                    workload,
                    seed,
                    args.holdout_fraction
                )

                print(
                    f"\nPROTOCOL: {protocol} | SEED: {seed}",
                    flush=True
                )

                print(
                    "Discovery:",
                    len(discovery),
                    "| Holdout:",
                    len(holdout),
                    flush=True
                )

                for split_name, items in (
                    ("discovery", discovery),
                    ("holdout", holdout)
                ):
                    for item in items:
                        split_manifest.append({
                            "protocol": protocol,
                            "seed": seed,
                            "split": split_name,
                            "query_hash": item["query_hash"],
                            "query_family": item["query_family"],
                            "query_name": item["query_name"]
                        })

                run_summaries = []

                for table, columns in valid_candidates:
                    label = (
                        f"{table}("
                        f"{canonical_columns(columns)})"
                    )

                    overlaps = find_overlapping_indexes(
                        table,
                        columns,
                        indexes
                    )

                    train_queries = [
                        item
                        for item in discovery
                        if query_references_table(
                            item["query_text"],
                            table
                        )
                    ][:args.max_queries_per_candidate]

                    test_queries = [
                        item
                        for item in holdout
                        if query_references_table(
                            item["query_text"],
                            table
                        )
                    ][:args.max_queries_per_candidate]

                    summary = {
                        "protocol": protocol,
                        "seed": seed,
                        "table": table,
                        "columns": canonical_columns(columns),
                        "candidate": label,
                        "overlapping_indexes": "|".join(overlaps),
                        "discovery_available": len(train_queries),
                        "holdout_available": len(test_queries)
                    }

                    if not train_queries or not test_queries:
                        summary.update({
                            "status": "insufficient_coverage",
                            "discovery_gain_pct": None,
                            "holdout_gain_pct": None
                        })

                        candidate_results.append(summary)

                        diagnostics.append({
                            "candidate": label,
                            "protocol": protocol,
                            "seed": seed,
                            "status": "insufficient_coverage",
                            "discovery": len(train_queries),
                            "holdout": len(test_queries)
                        })

                        continue

                    try:
                        train_results = evaluate_candidate(
                            connection,
                            table,
                            columns,
                            train_queries,
                            protocol,
                            "discovery",
                            seed
                        )

                        test_results = evaluate_candidate(
                            connection,
                            table,
                            columns,
                            test_queries,
                            protocol,
                            "holdout",
                            seed
                        )

                        query_results.extend(
                            train_results + test_results
                        )

                        train = aggregate(train_results)
                        test = aggregate(test_results)

                        ci_low, ci_high = bootstrap_interval(
                            test_results,
                            seed,
                            args.bootstrap
                        )

                        summary.update({
                            "status": (
                                "ok"
                                if not (
                                    train["errors"]
                                    or test["errors"]
                                )
                                else "partial_error"
                            ),
                            "discovery_gain_pct": train["gain_pct"],
                            "holdout_gain_pct": test["gain_pct"],
                            "discovery_success": train["queries"],
                            "holdout_success": test["queries"],
                            "discovery_index_used": train["index_used"],
                            "holdout_index_used": test["index_used"],
                            "holdout_ci_low": ci_low,
                            "holdout_ci_high": ci_high,
                            "query_errors": (
                                train["errors"]
                                + test["errors"]
                            )
                        })

                    except Exception as exc:
                        summary.update({
                            "status": "failed",
                            "error": str(exc)[:400]
                        })

                        diagnostics.append({
                            "candidate": label,
                            "protocol": protocol,
                            "seed": seed,
                            "status": "failed",
                            "error": str(exc)[:400]
                        })

                        try:
                            reset_hypopg(connection)
                        except Exception:
                            pass

                    candidate_results.append(summary)
                    run_summaries.append(summary)

                eligible = [
                    item
                    for item in run_summaries
                    if item["status"] == "ok"
                    and not item["overlapping_indexes"]
                    and item.get("discovery_gain_pct") is not None
                ]

                rng = random.Random(seed)

                methods = {
                    "planner_gain": sorted(
                        eligible,
                        key=lambda item: item["discovery_gain_pct"],
                        reverse=True
                    ),
                    "index_usage": sorted(
                        eligible,
                        key=lambda item: item["discovery_index_used"],
                        reverse=True
                    ),
                    "composite_planner": sorted(
                        [
                            item for item in eligible
                            if "|" in item["columns"]
                        ],
                        key=lambda item: item["discovery_gain_pct"],
                        reverse=True
                    ),
                    "random": rng.sample(
                        eligible,
                        len(eligible)
                    ) if eligible else []
                }

                for method, ranked in methods.items():
                    selected = ranked[:args.top_k]

                    selected_indexes = [
                        (
                            item["table"],
                            item["columns"].split("|")
                        )
                        for item in selected
                    ]

                    if not selected_indexes:
                        continue

                    try:
                        result = evaluate_portfolio(
                            connection,
                            selected_indexes,
                            holdout[
                                :args.portfolio_query_limit
                            ]
                        )

                        portfolio_results.append({
                            "protocol": protocol,
                            "seed": seed,
                            "method": method,
                            "selection": ";".join(
                                item["candidate"]
                                for item in selected
                            ),
                            **result
                        })

                    except Exception as exc:
                        diagnostics.append({
                            "protocol": protocol,
                            "seed": seed,
                            "method": method,
                            "status": "portfolio_error",
                            "error": str(exc)[:400]
                        })

    write_csv(
        output / "query_level.csv",
        query_results
    )

    write_csv(
        output / "candidate_splits.csv",
        candidate_results
    )

    write_csv(
        output / "portfolio_comparison.csv",
        portfolio_results
    )

    write_csv(
        output / "coverage_and_errors.csv",
        diagnostics
    )

    write_csv(
        output / "split_manifest.csv",
        split_manifest
    )

    grouped = defaultdict(list)

    for record in portfolio_results:
        if record["gain_pct"] is not None:
            grouped[
                (record["protocol"], record["method"])
            ].append(record["gain_pct"])

    cross_seed_results = []

    for (protocol, method), values in grouped.items():
        cross_seed_results.append({
            "protocol": protocol,
            "method": method,
            "runs": len(values),
            "mean_gain_pct": statistics.mean(values),
            "median_gain_pct": statistics.median(values),
            "min_gain_pct": min(values),
            "max_gain_pct": max(values)
        })

    write_csv(
        output / "cross_seed_summary.csv",
        cross_seed_results
    )

    report = {
        "project": "Adaptive Healthcare Knowledge Database",
        "phase": "Phase 9 V2",
        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "database": BENCHMARK_DATABASE,
        "workload_records": len(original),
        "safe_unique_sql": len(workload),
        "candidates": len(valid_candidates),
        "seeds": seeds,
        "protocols": list(protocols),
        "candidate_evaluations": len(candidate_results),
        "successful_candidate_evaluations": sum(
            record["status"] == "ok"
            for record in candidate_results
        ),
        "coverage_gaps": sum(
            record["status"] == "insufficient_coverage"
            for record in candidate_results
        ),
        "portfolio_evaluations": len(portfolio_results),
        "v6_model_ablation": "NOT EXECUTED",
        "physical_benchmark": "NOT EXECUTED",
        "write_overhead": "NOT MEASURED",
        "automatic_deployment": False,
        "limitations": [
            "Previously discovered candidates are reused.",
            "Results are retrospective.",
            "HypoPG costs are not execution times.",
            "SQL validation is not a security sandbox.",
            "Only trusted workload files should be used.",
            "Index overlap detection is conservative.",
            "Confidence intervals use SQL variants as sampling units.",
            "No validated V6 model comparison is included.",
            "No physical indexes are created."
        ]
    }

    write_json(
        output / "final_report.json",
        report
    )

    print("\n" + "=" * 75)
    print("PHASE 9 V2 EVALUATION COMPLETE")
    print("=" * 75)

    for key, value in report.items():
        if key != "limitations":
            print(f"{key}: {value}")

    print("\nResults:", output)
    print("Automatic deployment: DISABLED")
    print("=" * 75)


if __name__ == "__main__":
    main()
