
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import psycopg
from psycopg import sql
from psycopg.rows import dict_row


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "results" / "phase7_monitoring"

DEFAULT_WORKLOAD = (
    ROOT / "results" / "phase6_workloads"
    / "combined_workload.csv"
)

SQL_COLUMNS = [
    "sql_query", "query_sql", "sql",
    "query_text", "query", "statement"
]

PARAM_COLUMNS = [
    "parameters", "params", "query_params",
    "parameter_values"
]

DB_CONFIG = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", "5432")),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", ""),
}

FEATURE_COLUMNS = [
    "total_cost",
    "estimated_rows",
    "actual_rows",
    "planning_time_ms",
    "execution_time_ms",
    "total_node_time_ms",
    "plan_nodes",
    "max_plan_depth",
    "sequential_scans",
    "index_scans",
    "index_only_scans",
    "bitmap_index_scans",
    "bitmap_heap_scans",
    "nested_loop_joins",
    "hash_joins",
    "merge_joins",
    "sort_nodes",
    "aggregate_nodes",
    "filter_nodes",
    "join_filter_nodes",
    "rows_removed_by_filter",
    "shared_hit_blocks",
    "shared_read_blocks",
    "shared_dirtied_blocks",
    "shared_written_blocks",
    "local_hit_blocks",
    "local_read_blocks",
    "temp_read_blocks",
    "temp_written_blocks",
    "io_read_time_ms",
    "io_write_time_ms",
    "parallel_workers_planned",
    "parallel_workers_launched",
]


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def safe_filename(value):
    return re.sub(
        r"[^A-Za-z0-9_.-]+", "_", str(value)
    )[:100]


def fingerprint(query):
    query = re.sub(
        r"'(?:''|[^'])*'",
        "?",
        query
    )
    query = re.sub(r"\b\d+(?:\.\d+)?\b", "?", query)
    query = re.sub(r"\s+", " ", query)
    return query.strip().lower()


def query_hash(query):
    return hashlib.sha256(
        fingerprint(query).encode("utf-8")
    ).hexdigest()


def is_safe_select(query):
    """
    Conservative SQL validation.

    Only plain SELECT statements are accepted.
    Statements containing potentially unsafe SQL
    constructs are rejected before execution.
    """

    text = query.strip()

    if not text:
        return False

    # Remove simple SQL comments before checking.
    text = re.sub(r"--[^\n]*", " ", text)
    text = re.sub(
        r"/\*.*?\*/", " ", text, flags=re.DOTALL
    )

    # Reject multiple statements.
    if ";" in text.rstrip(";"):
        return False

    text = text.rstrip(";").strip()

    if not re.match(
        r"^SELECT\b", text, flags=re.IGNORECASE
    ):
        return False

    forbidden = [
        "INSERT", "UPDATE", "DELETE",
        "DROP", "ALTER", "TRUNCATE",
        "CREATE", "COPY", "CALL",
        "DO", "MERGE", "GRANT",
        "REVOKE", "VACUUM",
        "FOR UPDATE", "FOR SHARE",
        "INTO",
    ]

    for word in forbidden:
        if re.search(
            r"\b" + word + r"\b",
            text,
            flags=re.IGNORECASE
        ):
            return False

    return True


def load_workload(path, limit=None):
    if not path.exists():
        raise FileNotFoundError(
            f"Workload file not found: {path}"
        )

    with path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []

        sql_column = next(
            (c for c in SQL_COLUMNS if c in columns),
            None
        )

        if sql_column is None:
            raise ValueError(
                "No SQL column found in workload.\n"
                f"Available columns: {columns}\n"
                f"Expected one of: {SQL_COLUMNS}"
            )

        param_column = next(
            (c for c in PARAM_COLUMNS if c in columns),
            None
        )

        records = []

        for index, row in enumerate(reader):
            if limit is not None and index >= limit:
                break

            query = (row.get(sql_column) or "").strip()

            params = None
            if param_column:
                raw = (row.get(param_column) or "").strip()
                if raw:
                    try:
                        params = json.loads(raw)
                    except json.JSONDecodeError:
                        raise ValueError(
                            f"Invalid JSON parameters at "
                            f"workload row {index + 1}"
                        )

            records.append({
                "sequence": index + 1,
                "query": query,
                "parameters": params,
                "source_hash": row.get(
                    "query_hash", ""
                ),
                "fingerprint": fingerprint(query),
                "fingerprint_hash": query_hash(query),
                "query_family": row.get(
                    "query_family", "unknown"
                ),
                "query_name": row.get(
                    "query_name", "unknown"
                ),
                "workload_type": row.get(
                    "workload_type", "unknown"
                ),
                "complexity": row.get(
                    "complexity", "unknown"
                ),
            })

    return records


def connect(database):
    return psycopg.connect(
        dbname=database,
        connect_timeout=15,
        **DB_CONFIG
    )


def create_monitoring_schema(conn):
    """
    Monitoring records are stored in a separate
    schema to avoid conflicts with existing tables.
    """

    with conn.cursor() as cur:
        cur.execute("""
            CREATE SCHEMA IF NOT EXISTS phase7_monitoring
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS
            phase7_monitoring.query_log (
                id BIGSERIAL PRIMARY KEY,
                run_id TEXT NOT NULL,
                scale TEXT NOT NULL,
                sequence_number INTEGER NOT NULL,
                query_hash TEXT,
                fingerprint_hash TEXT,
                query_family TEXT,
                query_name TEXT,
                workload_type TEXT,
                complexity TEXT,
                status TEXT NOT NULL,
                planning_time_ms DOUBLE PRECISION,
                execution_time_ms DOUBLE PRECISION,
                wall_time_ms DOUBLE PRECISION,
                estimated_cost DOUBLE PRECISION,
                estimated_rows DOUBLE PRECISION,
                actual_rows DOUBLE PRECISION,
                plan_features JSONB,
                error_message TEXT,
                recorded_at TIMESTAMPTZ
                    DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS
            idx_phase7_run
            ON phase7_monitoring.query_log(run_id)
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS
            idx_phase7_fingerprint
            ON phase7_monitoring.query_log(
                fingerprint_hash
            )
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS
            idx_phase7_family
            ON phase7_monitoring.query_log(
                query_family
            )
        """)

    conn.commit()


def blank_features():
    return {key: 0 for key in FEATURE_COLUMNS}


def extract_plan_features(document):
    """
    Recursively walk PostgreSQL's JSON plan tree.

    Buffer statistics are summed only at the root
    node, avoiding double counting child buffers.
    """

    if isinstance(document, str):
        document = json.loads(document)

    if isinstance(document, list):
        document = document[0]

    if not isinstance(document, dict):
        raise ValueError("Unexpected EXPLAIN JSON structure")

    root = document.get("Plan")

    if not root:
        raise ValueError("EXPLAIN output has no Plan")

    features = blank_features()

    features["planning_time_ms"] = float(
        document.get("Planning Time") or 0
    )
    features["execution_time_ms"] = float(
        document.get("Execution Time") or 0
    )
    features["total_cost"] = float(
        root.get("Total Cost") or 0
    )
    features["estimated_rows"] = float(
        root.get("Plan Rows") or 0
    )
    features["actual_rows"] = float(
        root.get("Actual Rows") or 0
    )

    root_buffers = {
        "shared_hit_blocks": "Shared Hit Blocks",
        "shared_read_blocks": "Shared Read Blocks",
        "shared_dirtied_blocks": "Shared Dirtied Blocks",
        "shared_written_blocks": "Shared Written Blocks",
        "local_hit_blocks": "Local Hit Blocks",
        "local_read_blocks": "Local Read Blocks",
        "temp_read_blocks": "Temp Read Blocks",
        "temp_written_blocks": "Temp Written Blocks",
        "io_read_time_ms": "I/O Read Time",
        "io_write_time_ms": "I/O Write Time",
    }

    for target, source in root_buffers.items():
        features[target] = float(
            root.get(source) or 0
        )

    def visit(node, depth=1):
        kind = node.get("Node Type", "")

        features["plan_nodes"] += 1
        features["max_plan_depth"] = max(
            features["max_plan_depth"], depth
        )

        node_time = float(
            node.get("Actual Total Time") or 0
        )
        loops = float(node.get("Actual Loops") or 0)

        # Diagnostic aggregate, not additive query latency.
        features["total_node_time_ms"] += (
            node_time * loops
        )

        mapping = {
            "Seq Scan": "sequential_scans",
            "Index Scan": "index_scans",
            "Index Only Scan": "index_only_scans",
            "Bitmap Index Scan": "bitmap_index_scans",
            "Bitmap Heap Scan": "bitmap_heap_scans",
            "Nested Loop": "nested_loop_joins",
            "Hash Join": "hash_joins",
            "Merge Join": "merge_joins",
            "Sort": "sort_nodes",
        }

        if kind in mapping:
            features[mapping[kind]] += 1

        if "Aggregate" in kind:
            features["aggregate_nodes"] += 1

        if node.get("Filter") is not None:
            features["filter_nodes"] += 1

        if node.get("Join Filter") is not None:
            features["join_filter_nodes"] += 1

        features["rows_removed_by_filter"] += float(
            node.get("Rows Removed by Filter") or 0
        )

        features["parallel_workers_planned"] += int(
            node.get("Workers Planned") or 0
        )
        features["parallel_workers_launched"] += int(
            node.get("Workers Launched") or 0
        )

        for child in node.get("Plans", []):
            visit(child, depth + 1)

    visit(root)
    return features


def execute_monitored_query(
    conn, record, timeout_ms
):
    start = time.perf_counter()

    result = {
        **record,
        "status": "error",
        "error": "",
        "wall_time_ms": 0,
        "features": blank_features(),
        "plan": None,
    }

    query = record["query"]

    if not is_safe_select(query):
        result["status"] = "rejected"
        result["error"] = (
            "Not an allowed plain SELECT statement"
        )
        return result

    try:
        # A transaction-scoped read-only setting protects
        # the clinical data from modifications.
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute(
                    "SET TRANSACTION READ ONLY"
                )

                cur.execute(
                    "SELECT set_config("
                    "'statement_timeout', %s, true)",
                    (str(timeout_ms),)
                )

                cur.execute(
                    "SELECT set_config("
                    "'lock_timeout', '3000', true)"
                )

                cur.execute(
                    "EXPLAIN (ANALYZE, BUFFERS, "
                    "FORMAT JSON) " + query,
                    record["parameters"]
                )

                document = cur.fetchone()[0]

                result["plan"] = document
                result["features"] = (
                    extract_plan_features(document)
                )
                result["status"] = "success"

    except psycopg.errors.QueryCanceled as exc:
        result["status"] = "timeout"
        result["error"] = str(exc).splitlines()[0]

    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc).splitlines()[0]

    result["wall_time_ms"] = (
        time.perf_counter() - start
    ) * 1000

    return result


def insert_monitoring_record(
    conn, run_id, scale, result
):
    f = result["features"]

    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO phase7_monitoring.query_log (
                run_id,
                scale,
                sequence_number,
                query_hash,
                fingerprint_hash,
                query_family,
                query_name,
                workload_type,
                complexity,
                status,
                planning_time_ms,
                execution_time_ms,
                wall_time_ms,
                estimated_cost,
                estimated_rows,
                actual_rows,
                plan_features,
                error_message
            )
            VALUES (
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s::jsonb, %s
            )
        """, (
            run_id,
            scale,
            result["sequence"],
            result["source_hash"],
            result["fingerprint_hash"],
            result["query_family"],
            result["query_name"],
            result["workload_type"],
            result["complexity"],
            result["status"],
            f["planning_time_ms"],
            f["execution_time_ms"],
            result["wall_time_ms"],
            f["total_cost"],
            f["estimated_rows"],
            f["actual_rows"],
            json.dumps(f),
            result["error"],
        ))

    conn.commit()


def write_csv(path, rows):
    if not rows:
        return

    columns = list(rows[0].keys())

    with path.open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=columns
        )
        writer.writeheader()
        writer.writerows(rows)


def summarize_by(rows, key):
    groups = defaultdict(list)

    for row in rows:
        if row["status"] == "success":
            groups[row[key]].append(
                row["execution_time_ms"]
            )

    output = []

    for name, values in sorted(groups.items()):
        values = sorted(values)

        p95_index = min(
            len(values) - 1,
            max(0, int(0.95 * len(values) + 0.999) - 1)
        )

        output.append({
            key: name,
            "successful_queries": len(values),
            "mean_ms": round(
                statistics.mean(values), 3
            ),
            "median_ms": round(
                statistics.median(values), 3
            ),
            "p95_ms": round(
                values[p95_index], 3
            ),
            "max_ms": round(max(values), 3),
        })

    return output


def build_exports(results, output_dir):
    flat = []
    fingerprints = Counter()

    for result in results:
        f = result["features"]

        row = {
            "sequence": result["sequence"],
            "query_hash": result["source_hash"],
            "fingerprint_hash": result["fingerprint_hash"],
            "query_family": result["query_family"],
            "query_name": result["query_name"],
            "workload_type": result["workload_type"],
            "complexity": result["complexity"],
            "status": result["status"],
            "wall_time_ms": result["wall_time_ms"],
            "error": result["error"],
        }

        row.update(f)
        flat.append(row)

        fingerprints[result["fingerprint_hash"]] += 1

    for row in flat:
        row["frequency"] = fingerprints[
            row["fingerprint_hash"]
        ]

    write_csv(
        output_dir / "monitoring_results.csv",
        flat
    )

    write_csv(
        output_dir / "ai_training_features.csv",
        flat
    )

    write_csv(
        output_dir / "query_family_summary.csv",
        summarize_by(flat, "query_family")
    )

    write_csv(
        output_dir / "workload_summary.csv",
        summarize_by(flat, "workload_type")
    )

    write_csv(
        output_dir / "complexity_summary.csv",
        summarize_by(flat, "complexity")
    )

    frequency_rows = [
        {
            "fingerprint_hash": key,
            "frequency": count,
        }
        for key, count in fingerprints.most_common()
    ]

    write_csv(
        output_dir / "query_frequency.csv",
        frequency_rows
    )

    failures = [
        row for row in flat
        if row["status"] != "success"
    ]

    write_csv(
        output_dir / "failed_queries.csv",
        failures
    )

    successful = [
        row for row in flat
        if row["status"] == "success"
    ]

    successful.sort(
        key=lambda r: r["execution_time_ms"],
        reverse=True
    )

    write_csv(
        output_dir / "slowest_queries.csv",
        successful[:50]
    )

    return flat


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--database",
        default="adaptive_healthcare_scale_10k"
    )

    parser.add_argument(
        "--scale",
        default="10K"
    )

    parser.add_argument(
        "--workload",
        default=str(DEFAULT_WORKLOAD)
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None
    )

    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=20000
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true"
    )

    args = parser.parse_args()

    if args.timeout_ms < 1000:
        raise ValueError(
            "Timeout must be at least 1000 ms"
        )

    run_id = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S_%f")

    output_dir = (
        OUTPUT
        / safe_filename(args.scale)
        / run_id
    )

    output_dir.mkdir(
        parents=True, exist_ok=True
    )

    print("=" * 75)
    print("PHASE 7 - POSTGRESQL QUERY MONITORING")
    print("=" * 75)
    print("Database:", args.database)
    print("Scale:", args.scale)
    print("Run ID:", run_id)
    print("Output:", output_dir)

    records = load_workload(
        Path(args.workload),
        args.limit
    )

    print("Queries loaded:", len(records))

    if not records:
        raise ValueError("Workload is empty")

    if not DB_CONFIG["password"]:
        from getpass import getpass
        DB_CONFIG["password"] = getpass(
            "PostgreSQL password: "
        )

    print("\nConnecting to PostgreSQL...")

    # Monitoring connection has write access only
    # for storing its own monitoring records.
    monitor_conn = connect(args.database)

    create_monitoring_schema(monitor_conn)

    # Separate connection for read-only query execution.
    query_conn = connect(args.database)

    query_conn.autocommit = True

    results = []
    status_counts = Counter()

    plans_dir = output_dir / "execution_plans"
    plans_dir.mkdir(exist_ok=True)

    print("\nStarting monitoring...\n")

    try:
        for i, record in enumerate(records, start=1):
            result = execute_monitored_query(
                query_conn,
                record,
                args.timeout_ms
            )

            results.append(result)
            status_counts[result["status"]] += 1

            insert_monitoring_record(
                monitor_conn,
                run_id,
                args.scale,
                result
            )

            if result["plan"] is not None:
                plan_path = plans_dir / (
                    f"plan_{i:04d}.json"
                )

                plan_path.write_text(
                    json.dumps(
                        result["plan"],
                        indent=2,
                        default=str
                    ),
                    encoding="utf-8"
                )

            if i % 25 == 0 or i == len(records):
                print(
                    f"Processed {i}/{len(records)} | "
                    f"Success: {status_counts['success']} | "
                    f"Timeout: {status_counts['timeout']} | "
                    f"Errors: {status_counts['error']} | "
                    f"Rejected: {status_counts['rejected']}",
                    flush=True
                )

    except KeyboardInterrupt:
        print(
            "\nInterrupted; exporting completed records."
        )

    finally:
        query_conn.close()
        monitor_conn.close()

    flat = build_exports(
        results,
        output_dir
    )

    success = [
        r for r in flat
        if r["status"] == "success"
    ]

    latency = [
        r["execution_time_ms"]
        for r in success
    ]

    report = {
        "phase": 7,
        "run_id": run_id,
        "database": args.database,
        "scale": args.scale,
        "workload_file": str(args.workload),
        "timestamp": utcnow(),
        "planned_queries": len(records),
        "processed_queries": len(results),
        "status_counts": dict(status_counts),
        "success_rate_percent": round(
            100 * len(success) / len(results), 3
        ) if results else 0,
        "mean_execution_time_ms": (
            statistics.mean(latency)
            if latency else None
        ),
        "median_execution_time_ms": (
            statistics.median(latency)
            if latency else None
        ),
        "distinct_fingerprints": len({
            r["fingerprint_hash"]
            for r in results
        }),
        "features": FEATURE_COLUMNS,
        "notes": [
            "EXPLAIN ANALYZE executes SELECT queries.",
            "Execution is transaction-scoped read-only.",
            "Timeouts may terminate expensive queries.",
            "Planner costs are estimates, not milliseconds.",
            "Buffer statistics are from root plan nodes.",
            "Failed queries are retained for analysis.",
            "Fingerprint hashes are generated from SQL text.",
        ],
    }

    (output_dir / "phase7_report.json").write_text(
        json.dumps(
            report,
            indent=2,
            default=str
        ),
        encoding="utf-8"
    )

    print("\n" + "=" * 75)
    print("PHASE 7 MONITORING COMPLETE")
    print("=" * 75)
    print("Processed:", len(results))
    print("Successful:", status_counts["success"])
    print("Timeouts:", status_counts["timeout"])
    print("Errors:", status_counts["error"])
    print("Rejected:", status_counts["rejected"])
    print(
        "Distinct fingerprints:",
        report["distinct_fingerprints"]
    )
    print("Saved:", output_dir)

    print("\nOutputs:")
    for path in sorted(output_dir.iterdir()):
        print(" -", path.name)


if __name__ == "__main__":
    main()
