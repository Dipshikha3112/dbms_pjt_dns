
import argparse
import ast
import csv
import getpass
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import psycopg
from psycopg import sql


BENCHMARK_DB = "adaptive_healthcare_phase9_benchmark"

DEFAULT_CANDIDATES = [
    ("patient_procedure", ["patient_id", "procedure_id"]),
    ("claim", ["payer_id", "patient_id"]),
    ("lab_result", ["loinc_code", "patient_id"]),
    ("prescription", ["patient_id", "prescription_id"]),
    ("lab_result", ["patient_id", "result_time DESC"]),
]

def digest(value):
    return hashlib.sha256(
        str(value).encode("utf-8")
    ).hexdigest()[:16]


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, columns=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if columns is None:
        columns = list(dict.fromkeys(
            k for row in rows for k in row.keys()
        ))

    if not columns:
        columns = ["status"]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=columns,
            extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, obj):
    Path(path).write_text(
        json.dumps(obj, indent=2, default=str, allow_nan=False),
        encoding="utf-8"
    )


def numeric(value, default=0.0):
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except (ValueError, TypeError):
        return default


def truthy(value):
    return str(value).strip().lower() in (
        "true", "yes", "1"
    )


def parse_columns(value):
    if isinstance(value, list):
        return [str(x).strip() for x in value]

    value = str(value or "").strip()

    if not value:
        return []

    if value.startswith("["):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed]
        except (SyntaxError, ValueError):
            pass

    return [
        part.strip()
        for part in value.split("|")
        if part.strip()
    ]


def columns_key(columns):
    return " | ".join(
        re.sub(r"\s+", " ", c.strip()).lower()
        for c in columns
    )


def safe_select(query):
    query = str(query or "").strip()

    if not re.match(r"(?is)^SELECT\b", query):
        return False

    if ";" in query.rstrip(";"):
        return False

    if re.search(
        r"\b(INSERT|UPDATE|DELETE|MERGE|DROP|ALTER|CREATE|"
        r"TRUNCATE|GRANT|REVOKE|COPY|CALL|DO|EXECUTE|"
        r"VACUUM|ANALYZE|LOCK|SET|RESET|INTO|"
        r"PG_SLEEP|DBLINK|NEXTVAL|SETVAL|"
        r"LO_IMPORT|LO_EXPORT|PG_ADVISORY)\b",
        query,
        re.IGNORECASE
    ):
        return False

    if re.search(
        r"\bFOR\s+(UPDATE|SHARE|NO\s+KEY\s+UPDATE)\b",
        query,
        re.IGNORECASE
    ):
        return False

    return True


def table_referenced(query, table):
    return bool(re.search(
        r"(?i)\b(?:FROM|JOIN)\s+(?:\"?public\"?\.)?"
        r"\"?" + re.escape(table) + r"\"?\b",
        query
    ))


def column_expression(column):
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

    name, direction = match.groups()

    result = sql.Identifier(name)

    if direction:
        result += sql.SQL(" " + direction.upper())

    return result


def index_statement(table, columns):
    return sql.SQL(
        "CREATE INDEX ON public.{} USING btree ({})"
    ).format(
        sql.Identifier(table),
        sql.SQL(", ").join(
            column_expression(c) for c in columns
        )
    ).as_string(None)


def inspect_plan(plan):
    indexes = []
    nodes = []

    def walk(node):
        nodes.append(str(node.get("Node Type", "")))

        if node.get("Index Name"):
            indexes.append(str(node["Index Name"]))

        for child in node.get("Plans", []):
            walk(child)

        for child in node.get("InitPlan", []):
            walk(child)

    walk(plan)

    return {
        "cost": numeric(plan.get("Total Cost")),
        "indexes": indexes,
        "nodes": nodes,
        "root_node": plan.get("Node Type", "")
    }


def explain(conn, query):
    with conn.cursor() as cur:
        cur.execute(
            "EXPLAIN (FORMAT JSON) " + query
        )

        result = cur.fetchone()[0]

    if isinstance(result, str):
        result = json.loads(result)

    if isinstance(result, list):
        result = result[0]

    return inspect_plan(result["Plan"])


def reset_hypopg(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT hypopg_reset()")


def create_hypothetical(conn, table, columns):
    statement = index_statement(table, columns)

    with conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM hypopg_create_index(%s)",
            (statement,)
        )

        row = cur.fetchone()

    if not row:
        raise RuntimeError(
            "HypoPG did not return an index"
        )

    return {
        "oid": int(row[0]),
        "name": str(row[1]),
        "statement": statement
    }


def verify_connection(conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                current_database(),
                current_user,
                pg_backend_pid(),
                current_setting('port')::integer
            """
        )

        database, user, pid, internal_port = cur.fetchone()

        if database != BENCHMARK_DB:
            raise RuntimeError(
                f"Refusing unexpected database: {database}"
            )

        if internal_port != 5432:
            raise RuntimeError(
                f"Unexpected internal port: {internal_port}"
            )

        cur.execute(
            "SELECT EXISTS("
            "SELECT 1 FROM pg_extension WHERE extname='hypopg'"
            ")"
        )

        if not cur.fetchone()[0]:
            raise RuntimeError(
                "HypoPG extension is not installed"
            )

    print(
        f"Verified: {database} | User: {user} | "
        f"Backend PID: {pid}"
    )


def verify_columns(conn, table, columns):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = %s
            """,
            (table,)
        )

        available = {r[0].lower() for r in cur.fetchall()}

    required = {
        c.split()[0].lower() for c in columns
    }

    missing = required - available

    if missing:
        raise RuntimeError(
            "Missing columns: " + ", ".join(sorted(missing))
        )


def load_workload(path):
    rows = read_csv(path)

    distinct = {}
    frequency = defaultdict(int)

    for row in rows:
        query = str(row.get("query_text") or "").strip()
        query = query.rstrip(";").strip()

        if not safe_select(query):
            continue

        if "safe" in row and not truthy(row["safe"]):
            continue

        family = str(row.get("query_family") or "").strip()
        name = str(row.get("query_name") or "").strip()

        query_hash = digest(query)

        key = (family, name, query_hash)

        frequency[key] += 1

        distinct[key] = {
            "query_text": query,
            "query_hash": query_hash,
            "query_family": family,
            "query_name": name
        }

    result = []

    for key, item in distinct.items():
        item = dict(item)
        item["frequency"] = frequency[key]
        result.append(item)

    return result, rows


def load_phase9b(path):
    records = read_csv(path)

    selected = defaultdict(list)

    for row in records:
        if not truthy(
            row.get("hypothetical_index_used")
        ):
            continue

        gain = numeric(row.get("gain_pct"))

        if gain <= 0:
            continue

        table = str(row.get("table") or "").strip()
        columns = parse_columns(row.get("columns"))

        if not table or not columns:
            continue

        selected[
            (table, columns_key(columns))
        ].append({
            "family": str(
                row.get("query_family") or ""
            ).strip(),
            "name": str(
                row.get("query_name") or ""
            ).strip(),
            "phase9b_gain_pct": gain
        })

    return selected


def choose_queries(
    workload,
    phase9b,
    table,
    columns,
    max_queries
):
    matches = phase9b.get(
        (table, columns_key(columns)),
        []
    )

    matching_families = {
        (r["family"], r["name"])
        for r in matches
    }

    selected = []

    for item in workload:
        if not table_referenced(
            item["query_text"], table
        ):
            continue

        exact_family = (
            item["query_family"],
            item["query_name"]
        )

        if matching_families:
            if exact_family not in matching_families:
                continue

        selected.append(dict(item))

    selected.sort(
        key=lambda item: (
            -item["frequency"],
            item["query_family"],
            item["query_hash"]
        )
    )

    return selected[:max_queries]


def run_whatif(
    conn,
    table,
    columns,
    queries,
    timeout_ms
):
    results = []

    reset_hypopg(conn)

    try:
        baseline = {}

        for item in queries:
            try:
                before = explain(
                    conn,
                    item["query_text"]
                )

                baseline[item["query_hash"]] = before

            except Exception as exc:
                results.append({
                    "table": table,
                    "columns": columns_key(columns),
                    "query_hash": item["query_hash"],
                    "query_family": item["query_family"],
                    "query_name": item["query_name"],
                    "frequency": item["frequency"],
                    "status": "baseline_error",
                    "error": str(exc)[:400]
                })

        hypothetical = create_hypothetical(
            conn, table, columns
        )

        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM hypopg_list_indexes"
            )

            registered = cur.fetchall()

        if not any(
            int(row[0]) == hypothetical["oid"]
            for row in registered
        ):
            raise RuntimeError(
                "Hypothetical index not registered"
            )

        for item in queries:
            identity = item["query_hash"]

            if identity not in baseline:
                continue

            before = baseline[identity]

            record = {
                "table": table,
                "columns": columns_key(columns),
                "query_hash": identity,
                "query_family": item["query_family"],
                "query_name": item["query_name"],
                "frequency": item["frequency"],
                "before_cost": before["cost"],
                "hypothetical_index_name": hypothetical["name"],
                "status": "ok",
                "error": ""
            }

            try:
                after = explain(
                    conn,
                    item["query_text"]
                )

                used = hypothetical["name"] in after["indexes"]

                gain = (
                    100.0
                    * (before["cost"] - after["cost"])
                    / before["cost"]
                    if before["cost"] > 0 else 0.0
                )

                record.update({
                    "after_cost": after["cost"],
                    "gain_pct": round(gain, 6),
                    "hypothetical_index_used": used,
                    "before_root_node": before["root_node"],
                    "after_root_node": after["root_node"],
                    "after_indexes": " | ".join(
                        after["indexes"]
                    )
                })

            except Exception as exc:
                record["status"] = "indexed_error"
                record["error"] = str(exc)[:400]

            results.append(record)

    finally:
        reset_hypopg(conn)

    return results


def aggregate_candidate(table, columns, rows):
    valid = [
        r for r in rows
        if r.get("status") == "ok"
    ]

    before = sum(
        numeric(r["before_cost"]) * int(r["frequency"])
        for r in valid
    )

    after = sum(
        numeric(r["after_cost"]) * int(r["frequency"])
        for r in valid
    )

    gain = (
        100.0 * (before - after) / before
        if before > 0 else None
    )

    used = [
        r for r in valid
        if r.get("hypothetical_index_used")
    ]

    return {
        "table": table,
        "columns": columns_key(columns),
        "status": (
            "completed" if valid
            else "no_successful_queries"
        ),
        "queries_evaluated": len(valid),
        "queries_using_hypothetical_index": len(used),
        "positive_queries": sum(
            numeric(r.get("gain_pct")) > 0
            for r in valid
        ),
        "negative_queries": sum(
            numeric(r.get("gain_pct")) < 0
            for r in valid
        ),
        "before_weighted_cost": round(before, 6),
        "after_weighted_cost": round(after, 6),
        "estimated_gain_pct": (
            round(gain, 6) if gain is not None else None
        ),
        "evidence_source": "same_session_hypopg_explain"
    }


def load_phase9c(path):
    if not path:
        return {}

    rows = read_csv(path)

    return {
        (
            str(row.get("table") or "").strip(),
            columns_key(
                parse_columns(row.get("columns"))
            )
        ): row
        for row in rows
    }


def load_original_rankings(path):
    if not path:
        return []

    path = Path(path)

    if not path.exists():
        return []

    return read_csv(path)


def integrate_evidence(
    estimated,
    physical,
    original_ranked
):
    original_map = {}

    for row in original_ranked:
        key = (
            str(row.get("table_name") or "").strip(),
            columns_key(
                parse_columns(row.get("columns"))
            )
        )

        original_map[key] = row

    integrated = []

    for estimate in estimated:
        key = (
            estimate["table"],
            estimate["columns"]
        )

        physical_row = physical.get(key, {})
        original = original_map.get(key, {})

        physical_valid = (
            physical_row.get("status") == "completed"
            and truthy(physical_row.get("index_used"))
        )

        physical_gain = (
            numeric(
                physical_row.get("measured_gain_pct")
            )
            if physical_valid else None
        )

        planner_gain = estimate.get(
            "estimated_gain_pct"
        )

        planner_positive = (
            planner_gain is not None
            and planner_gain > 0
            and estimate[
                "queries_using_hypothetical_index"
            ] > 0
        )

        physical_positive = (
            physical_gain is not None
            and physical_gain > 0
        )

        if planner_positive and physical_positive:
            evidence = "planner_and_physical_positive"
        elif planner_positive:
            evidence = "planner_positive_only"
        elif physical_positive:
            evidence = "physical_positive_only"
        else:
            evidence = "insufficient_positive_evidence"

        integrated.append({
            "table": estimate["table"],
            "columns": estimate["columns"],
            "candidate_id": original.get(
                "candidate_id", digest(key)
            ),
            "original_source": original.get(
                "source", "phase9d"
            ),
            "original_approval": original.get(
                "approved", "unknown"
            ),
            "original_governance_reason": original.get(
                "reason", "not_in_original_rankings"
            ),
            "phase9d_estimated_gain_pct": planner_gain,
            "phase9d_queries_evaluated": estimate[
                "queries_evaluated"
            ],
            "phase9d_index_used_count": estimate[
                "queries_using_hypothetical_index"
            ],
            "phase9c_measured_gain_pct": physical_gain,
            "phase9c_status": physical_row.get(
                "status", "not_benchmarked"
            ),
            "evidence_class": evidence,
            "manual_review_recommended": (
                planner_positive and physical_positive
            ),
            "automatic_approval": False,
            "automatic_deployment": False
        })

    integrated.sort(
        key=lambda row: (
            row["evidence_class"]
            != "planner_and_physical_positive",
            -numeric(
                row["phase9d_estimated_gain_pct"]
            )
        )
    )

    return integrated


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Phase 9D: workload-aware same-session "
            "HypoPG validation and evidence integration"
        )
    )

    parser.add_argument(
        "--host", default="localhost"
    )

    parser.add_argument(
        "--port", type=int, default=5433
    )

    parser.add_argument(
        "--database", default=BENCHMARK_DB
    )

    parser.add_argument(
        "--user", default="postgres"
    )

    parser.add_argument("--password")

    parser.add_argument(
        "--workload", required=True
    )

    parser.add_argument(
        "--phase9b-details", required=True
    )

    parser.add_argument(
        "--phase9c-results", required=True
    )

    parser.add_argument(
        "--original-ranked", required=True
    )

    parser.add_argument(
        "--max-queries-per-candidate",
        type=int,
        default=30
    )

    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=15000
    )

    parser.add_argument(
        "--output",
        default="results/phase9d_fixed"
    )

    args = parser.parse_args()

    if (
        args.host not in ("localhost", "127.0.0.1")
        or args.port != 5433
        or args.database != BENCHMARK_DB
    ):
        raise RuntimeError(
            "Only the isolated Docker benchmark "
            "database on localhost:5433 is allowed."
        )

    if args.max_queries_per_candidate < 1:
        raise ValueError(
            "max-queries-per-candidate must be positive"
        )

    if args.timeout_ms < 1:
        raise ValueError(
            "timeout-ms must be positive"
        )

    if args.password is None:
        args.password = getpass.getpass(
            "Docker PostgreSQL password: "
        )

    workload, raw_workload = load_workload(
        args.workload
    )

    phase9b = load_phase9b(
        args.phase9b_details
    )

    phase9c = load_phase9c(
        args.phase9c_results
    )

    original_ranked = load_original_rankings(
        args.original_ranked
    )

    output = (
        Path(args.output)
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    output.mkdir(parents=True, exist_ok=True)

    candidates = []

    for table, columns in DEFAULT_CANDIDATES:
        candidates.append((table, columns))

    for row in original_ranked:
        table = str(
            row.get("table_name") or ""
        ).strip()

        columns = parse_columns(
            row.get("columns")
        )

        if table and columns:
            candidates.append((table, columns))

    unique = {}

    for table, columns in candidates:
        key = (
            table,
            columns_key(columns)
        )

        unique[key] = (table, columns)

    all_query_results = []
    candidate_results = []

    with psycopg.connect(
        host=args.host,
        port=args.port,
        dbname=args.database,
        user=args.user,
        password=args.password,
        autocommit=True,
        connect_timeout=10
    ) as conn:

        verify_connection(conn)

        with conn.cursor() as cur:
            cur.execute(
                "SELECT set_config('statement_timeout', %s, false)",
                (f"{args.timeout_ms}ms",)
            )

            cur.execute(
                "SELECT set_config('default_transaction_read_only', 'on', false)"
            )

        for number, (table, columns) in enumerate(
            unique.values(), 1
        ):
            label = (
                f"{table}({', '.join(columns)})"
            )

            print(
                f"\n[9D {number}/{len(unique)}] {label}",
                flush=True
            )

            try:
                verify_columns(
                    conn, table, columns
                )

                selected = choose_queries(
                    workload,
                    phase9b,
                    table,
                    columns,
                    args.max_queries_per_candidate
                )

                if not selected:
                    candidate_results.append({
                        "table": table,
                        "columns": columns_key(columns),
                        "status": "no_matching_queries",
                        "queries_evaluated": 0,
                        "queries_using_hypothetical_index": 0,
                        "positive_queries": 0,
                        "estimated_gain_pct": None
                    })

                    print(
                        "  No matching queries",
                        flush=True
                    )
                    continue

                results = run_whatif(
                    conn,
                    table,
                    columns,
                    selected,
                    args.timeout_ms
                )

                for row in results:
                    row["candidate_id"] = digest(
                        (table, columns_key(columns))
                    )

                all_query_results.extend(results)

                summary = aggregate_candidate(
                    table, columns, results
                )

                candidate_results.append(summary)

                print(
                    "  Queries:",
                    summary["queries_evaluated"],
                    "| Index used:",
                    summary[
                        "queries_using_hypothetical_index"
                    ],
                    "| Estimated gain:",
                    summary["estimated_gain_pct"],
                    "%",
                    flush=True
                )

            except Exception as exc:
                candidate_results.append({
                    "table": table,
                    "columns": columns_key(columns),
                    "status": "failed",
                    "error": str(exc)[:500],
                    "queries_evaluated": 0,
                    "queries_using_hypothetical_index": 0,
                    "positive_queries": 0,
                    "estimated_gain_pct": None
                })

                print(
                    "  Error:",
                    str(exc)[:300],
                    flush=True
                )

                try:
                    reset_hypopg(conn)
                except Exception:
                    pass

    integrated = integrate_evidence(
        candidate_results,
        phase9c,
        original_ranked
    )

    write_csv(
        output / "phase9d_query_level.csv",
        all_query_results
    )

    write_csv(
        output / "phase9d_candidate_summary.csv",
        candidate_results
    )

    write_csv(
        output / "phase9d_integrated_evidence.csv",
        integrated
    )

    positive = [
        row for row in candidate_results
        if row.get("estimated_gain_pct") is not None
        and row["estimated_gain_pct"] > 0
    ]

    review = [
        row for row in integrated
        if row["manual_review_recommended"]
    ]

    report = {
        "phase": "9D",
        "database": BENCHMARK_DB,
        "workload_rows": len(raw_workload),
        "unique_safe_sql_variants": len(workload),
        "candidates_evaluated": len(candidate_results),
        "queries_explained": sum(
            row.get("status") == "ok"
            for row in all_query_results
        ),
        "positive_planner_candidates": len(positive),
        "positive_planner_and_physical": len(review),
        "model": (
            "Original V6 model outputs preserved externally; "
            "no model predictions recalculated"
        ),
        "original_governance": (
            "Preserved as reported in original-ranked CSV"
        ),
        "automatic_deployment": False,
        "notes": [
            "HypoPG planner costs are not execution times.",
            "Phase 9C results are reused without rerunning benchmarks.",
            "Phase 9B family/name matching does not prove exact SQL identity.",
            "Query sampling can bias estimated workload improvements.",
            "No clinical outcome or decision-support accuracy is measured.",
            "No permanent indexes are created.",
            "Governance eligibility is not automatically overridden."
        ]
    }

    write_json(
        output / "phase9d_report.json",
        report
    )

    print("\n" + "=" * 72)
    print("PHASE 9D COMPLETE")
    print("=" * 72)

    for key, value in report.items():
        if key != "notes":
            print(f"{key}: {value}")

    print("\nResults:", output)
    print("Automatic deployment: DISABLED")
    print("=" * 72)


if __name__ == "__main__":
    main()
