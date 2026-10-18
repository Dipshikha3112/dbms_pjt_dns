
import argparse
import hashlib
import html
import json
import os
import re
import time
from collections import Counter
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import pandas as pd
import psycopg
from psycopg.rows import dict_row

try:
    import psutil
except ImportError:
    psutil = None


ROOT = Path(__file__).resolve().parents[1]
DATABASE = "adaptive_healthcare_phase9_benchmark"
SCHEMA = "phase7_monitoring"

WORKLOAD = (
    ROOT / "results"
    / "phase9_research_optimizer"
    / "20261017_185042_5b6ae098"
    / "workload_validation.csv"
)


def connect(args):
    if args.database != DATABASE:
        raise RuntimeError("Production databases are prohibited")

    if args.host not in ("127.0.0.1", "localhost"):
        raise RuntimeError("Only localhost is permitted")

    if args.port != 5433:
        raise RuntimeError("Only benchmark port 5433 is permitted")

    return psycopg.connect(
        host=args.host,
        port=args.port,
        dbname=args.database,
        user=args.user,
        password=args.password,
        connect_timeout=15,
        row_factory=dict_row
    )


def query(args, statement, params=None):
    with connect(args) as conn:
        with conn.cursor() as cur:
            cur.execute(statement, params)
            if cur.description:
                return cur.fetchall()
    return []


def digest(value):
    return hashlib.sha256(
        str(value).encode("utf-8")
    ).hexdigest()[:20]


def fingerprint(sql):
    sql = sql.lower()
    sql = re.sub(r"'(?:''|[^'])*'", " ? ", sql)
    sql = re.sub(r"\b\d+(?:\.\d+)?\b", " ? ", sql)
    sql = re.sub(r"\s+", " ", sql)
    return digest(sql)


def safe_sql(sql):
    sql = str(sql).strip()

    if not re.match(r"^SELECT\b", sql, re.I):
        return False

    if ";" in sql:
        return False

    forbidden = (
        r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|"
        r"TRUNCATE|COPY|CALL|DO|EXECUTE|VACUUM|"
        r"ANALYZE|LOCK|SET|RESET|INTO)\b"
        r"|\bFOR\s+(UPDATE|SHARE)\b"
    )

    return re.search(forbidden, sql, re.I) is None


def load_workload(path):
    df = pd.read_csv(path, low_memory=False)

    required = {
        "query_text",
        "query_family",
        "query_name"
    }

    if not required.issubset(df.columns):
        raise RuntimeError(
            f"Missing columns: {required - set(df.columns)}"
        )

    df = df.dropna(subset=["query_text"]).copy()
    df["query_text"] = df["query_text"].astype(str).str.strip()
    df = df[df["query_text"].map(safe_sql)].copy()

    df["query_id"] = df["query_text"].map(digest)
    df["fingerprint"] = df["query_text"].map(fingerprint)

    return df.drop_duplicates("query_id").reset_index(drop=True)


# ============================================================
# COMPONENT 1 — PERSISTENT QUERY_LOG TABLE
# ============================================================

def initialize_database(args):
    query(args, f"""
        CREATE SCHEMA IF NOT EXISTS {SCHEMA}
    """)

    query(args, f"""
        CREATE TABLE IF NOT EXISTS {SCHEMA}.query_log (
            id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            collected_at TIMESTAMPTZ DEFAULT NOW(),
            query_id TEXT NOT NULL,
            fingerprint TEXT NOT NULL,
            query_family TEXT,
            query_name TEXT,
            query_text TEXT NOT NULL,
            status TEXT NOT NULL,
            error_message TEXT,
            planning_ms DOUBLE PRECISION,
            execution_ms DOUBLE PRECISION,
            wall_ms DOUBLE PRECISION,
            planner_cost DOUBLE PRECISION,
            actual_rows DOUBLE PRECISION,
            total_node_rows DOUBLE PRECISION,
            actual_loops DOUBLE PRECISION,
            shared_hit BIGINT,
            shared_read BIGINT,
            shared_dirtied BIGINT,
            shared_written BIGINT,
            local_hit BIGINT,
            local_read BIGINT,
            temp_read BIGINT,
            temp_written BIGINT,
            io_read_ms DOUBLE PRECISION,
            io_write_ms DOUBLE PRECISION,
            seq_scans INTEGER,
            index_scans INTEGER,
            index_only_scans INTEGER,
            bitmap_scans INTEGER,
            join_nodes INTEGER,
            sort_nodes INTEGER,
            plan_json JSONB
        )
    """)

    query(args, f"""
        CREATE INDEX IF NOT EXISTS idx_phase7_fingerprint_time
        ON {SCHEMA}.query_log(fingerprint, collected_at DESC)
    """)

    query(args, f"""
        CREATE INDEX IF NOT EXISTS idx_phase7_log_time
        ON {SCHEMA}.query_log(collected_at DESC)
    """)

    query(args, f"""
        CREATE TABLE IF NOT EXISTS {SCHEMA}.resource_log (
            id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            collected_at TIMESTAMPTZ DEFAULT NOW(),
            cpu_percent DOUBLE PRECISION,
            ram_percent DOUBLE PRECISION,
            ram_used_mb DOUBLE PRECISION,
            db_connections INTEGER,
            active_connections INTEGER,
            database_size_mb DOUBLE PRECISION
        )
    """)

    query(args, f"""
        CREATE TABLE IF NOT EXISTS {SCHEMA}.monitor_runs (
            id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            started_at TIMESTAMPTZ DEFAULT NOW(),
            finished_at TIMESTAMPTZ,
            requested INTEGER,
            successful INTEGER,
            failed INTEGER
        )
    """)

    print("Phase 7 monitoring tables initialized.")


# ============================================================
# COMPONENT 2 — ACTUAL EXECUTION-TIME MONITORING
# COMPONENT 3 — PLANNING-TIME MONITORING
# COMPONENT 4 — ACTUAL ROW COUNTS AND LOOPS
# COMPONENT 5 — BUFFER AND I/O MONITORING
# COMPONENT 8 — SEQUENTIAL / INDEX SCAN STATISTICS
# ============================================================

def plan_nodes(plan):
    yield plan

    for child in plan.get("Plans", []):
        yield from plan_nodes(child)


def extract_metrics(document):
    if isinstance(document, str):
        document = json.loads(document)

    if isinstance(document, list):
        document = document[0]

    plan = document["Plan"]
    nodes = list(plan_nodes(plan))

    types = Counter(
        str(n.get("Node Type", "")).lower()
        for n in nodes
    )

    def root_value(name):
        return int(plan.get(name, 0) or 0)

    return {
        "planning_ms": document.get("Planning Time", 0),
        "execution_ms": document.get("Execution Time", 0),
        "planner_cost": plan.get("Total Cost", 0),
        "actual_rows": plan.get("Actual Rows", 0),
        "actual_loops": plan.get("Actual Loops", 0),
        "total_node_rows": sum(
            float(n.get("Actual Rows", 0) or 0)
            * float(n.get("Actual Loops", 0) or 0)
            for n in nodes
        ),
        "shared_hit": root_value("Shared Hit Blocks"),
        "shared_read": root_value("Shared Read Blocks"),
        "shared_dirtied": root_value("Shared Dirtied Blocks"),
        "shared_written": root_value("Shared Written Blocks"),
        "local_hit": root_value("Local Hit Blocks"),
        "local_read": root_value("Local Read Blocks"),
        "temp_read": root_value("Temp Read Blocks"),
        "temp_written": root_value("Temp Written Blocks"),
        "io_read_ms": plan.get("I/O Read Time", 0),
        "io_write_ms": plan.get("I/O Write Time", 0),
        "seq_scans": types["seq scan"],
        "index_scans": types["index scan"],
        "index_only_scans": types["index only scan"],
        "bitmap_scans": (
            types["bitmap index scan"]
            + types["bitmap heap scan"]
        ),
        "join_nodes": sum(
            count
            for name, count in types.items()
            if "join" in name or name == "nested loop"
        ),
        "sort_nodes": (
            types["sort"]
            + types["incremental sort"]
        ),
        "plan_json": json.dumps(document)
    }


def measure_query(args, row):
    started = time.perf_counter()

    result = {
        "query_id": row["query_id"],
        "fingerprint": row["fingerprint"],
        "query_family": str(row["query_family"]),
        "query_name": str(row["query_name"]),
        "query_text": row["query_text"],
        "status": "error",
        "error_message": None
    }

    try:
        with connect(args) as conn:
            with conn.cursor() as cur:
                cur.execute("SET TRANSACTION READ ONLY")

                cur.execute(
                    "SELECT set_config('statement_timeout', %s, true)",
                    (f"{args.timeout_ms}ms",)
                )
                cur.execute(
                    "SELECT set_config('lock_timeout', %s, true)",
                    ("2000ms",)
                )

                cur.execute(
                    "EXPLAIN (ANALYZE, BUFFERS, "
                    "FORMAT JSON, TIMING ON) "
                    + row["query_text"]
                )

                document = cur.fetchone()["QUERY PLAN"]

                result.update(extract_metrics(document))
                result["status"] = "ok"

            conn.rollback()

    except Exception as exc:
        result["error_message"] = (
            f"{type(exc).__name__}: {str(exc)[:500]}"
        )

    result["wall_ms"] = (
        time.perf_counter() - started
    ) * 1000

    return result


def save_measurement(args, record):
    columns = list(record)
    placeholders = ", ".join(["%s"] * len(columns))

    statement = (
        f"INSERT INTO {SCHEMA}.query_log "
        f"({', '.join(columns)}) "
        f"VALUES ({placeholders})"
    )

    query(
        args,
        statement,
        [record[c] for c in columns]
    )


# ============================================================
# COMPONENT 6 — QUERY FREQUENCY TRACKING
# ============================================================

def frequency_report(args, hours=24):
    return query(args, f"""
        SELECT
            fingerprint,
            MIN(query_family) AS query_family,
            COUNT(*) AS observed_executions,
            COUNT(*) FILTER (
                WHERE status = 'ok'
            ) AS successful_executions,
            ROUND(AVG(execution_ms)::numeric, 3)
                AS avg_execution_ms,
            ROUND(MAX(execution_ms)::numeric, 3)
                AS max_execution_ms
        FROM {SCHEMA}.query_log
        WHERE collected_at >=
            NOW() - (%s * INTERVAL '1 hour')
        GROUP BY fingerprint
        ORDER BY observed_executions DESC
        LIMIT 50
    """, (hours,))


# ============================================================
# COMPONENT 11 — CPU/RAM AND DATABASE RESOURCE MONITORING
# ============================================================

def collect_resources(args):
    cpu = None
    ram = None
    used = None

    if psutil:
        cpu = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        ram = memory.percent
        used = memory.used / (1024 ** 2)

    stats = query(args, """
        SELECT
            COUNT(*) AS connections,
            COUNT(*) FILTER (
                WHERE state = 'active'
            ) AS active
        FROM pg_stat_activity
        WHERE datname = current_database()
    """)[0]

    size = query(args, """
        SELECT pg_database_size(
            current_database()
        ) / 1048576.0 AS size_mb
    """)[0]["size_mb"]

    query(args, f"""
        INSERT INTO {SCHEMA}.resource_log (
            cpu_percent,
            ram_percent,
            ram_used_mb,
            db_connections,
            active_connections,
            database_size_mb
        )
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (
        cpu,
        ram,
        used,
        stats["connections"],
        stats["active"],
        size
    ))


# ============================================================
# COMPONENT 7 — CONTINUOUS MONITORING SERVICE
# ============================================================

def run_cycle(args, workload, cycle_number=0):
    if args.sample_size == 0:
        selected = workload
    else:
        selected = workload.sample(
            n=min(args.sample_size, len(workload)),
            random_state=args.seed + cycle_number
        )

    run_id = query(args, f"""
        INSERT INTO {SCHEMA}.monitor_runs (
            requested, successful, failed
        )
        VALUES (%s, 0, 0)
        RETURNING id
    """, (len(selected),))[0]["id"]

    success = 0
    failed = 0

    for number, (_, row) in enumerate(
        selected.iterrows(), 1
    ):
        measurement = measure_query(args, row)
        save_measurement(args, measurement)

        if measurement["status"] == "ok":
            success += 1
        else:
            failed += 1

        if number % 25 == 0 or number == len(selected):
            print(
                f"Measured {number}/{len(selected)} | "
                f"Success={success} | Failed={failed}",
                flush=True
            )

    try:
        collect_resources(args)
    except Exception as exc:
        print("Resource collection warning:", exc)

    query(args, f"""
        UPDATE {SCHEMA}.monitor_runs
        SET
            finished_at = NOW(),
            successful = %s,
            failed = %s
        WHERE id = %s
    """, (success, failed, run_id))

    return {
        "run_id": run_id,
        "requested": len(selected),
        "successful": success,
        "failed": failed
    }


def continuous_monitoring(args, workload):
    cycle = 0

    print("Continuous monitoring started.")
    print("Press Ctrl+C to stop.")

    try:
        while (
            args.max_cycles == 0
            or cycle < args.max_cycles
        ):
            started = time.monotonic()

            result = run_cycle(
                args, workload, cycle
            )

            print(result)

            cycle += 1

            if (
                args.max_cycles
                and cycle >= args.max_cycles
            ):
                break

            elapsed = time.monotonic() - started

            time.sleep(
                max(0, args.interval_seconds - elapsed)
            )

    except KeyboardInterrupt:
        print("Monitoring stopped.")


# ============================================================
# COMPONENT 9 — MONITORING API
# ============================================================

def summary(args):
    return query(args, f"""
        SELECT
            COUNT(*) AS total_executions,
            COUNT(*) FILTER (
                WHERE status = 'ok'
            ) AS successful,
            COUNT(*) FILTER (
                WHERE status = 'error'
            ) AS failed,
            COUNT(DISTINCT fingerprint)
                AS unique_fingerprints,
            ROUND(AVG(execution_ms)::numeric, 3)
                AS avg_execution_ms,
            ROUND(MAX(execution_ms)::numeric, 3)
                AS max_execution_ms,
            ROUND(AVG(planning_ms)::numeric, 3)
                AS avg_planning_ms,
            SUM(shared_hit) AS shared_hits,
            SUM(shared_read) AS shared_reads,
            SUM(seq_scans) AS seq_scans,
            SUM(index_scans) AS index_scans
        FROM {SCHEMA}.query_log
    """)[0]


def recent_queries(args, limit=30):
    return query(args, f"""
        SELECT
            collected_at,
            query_family,
            query_name,
            fingerprint,
            status,
            execution_ms,
            planning_ms,
            actual_rows,
            shared_read,
            seq_scans,
            index_scans
        FROM {SCHEMA}.query_log
        ORDER BY id DESC
        LIMIT %s
    """, (limit,))


def resource_history(args, limit=30):
    return query(args, f"""
        SELECT *
        FROM {SCHEMA}.resource_log
        ORDER BY id DESC
        LIMIT %s
    """, (limit,))


def serialize(value):
    if isinstance(value, datetime):
        return value.isoformat()

    if hasattr(value, "item"):
        return value.item()

    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


# ============================================================
# COMPONENT 10 — MONITORING DASHBOARD
# ============================================================

def html_table(rows):
    if not rows:
        return "<p>No measurements available yet.</p>"

    columns = list(rows[0])

    header = "".join(
        f"<th>{html.escape(str(c))}</th>"
        for c in columns
    )

    body = ""

    for row in rows:
        cells = "".join(
            f"<td>{html.escape(str(row.get(c, '')))}</td>"
            for c in columns
        )

        body += f"<tr>{cells}</tr>"

    return (
        f"<table><thead><tr>{header}</tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def dashboard(args):
    metrics = summary(args)

    cards = "".join(
        "<div class='card'>"
        f"<small>{html.escape(str(k))}</small>"
        f"<strong>{html.escape(str(v))}</strong>"
        "</div>"
        for k, v in metrics.items()
    )

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport"
              content="width=device-width, initial-scale=1">
        <title>Healthcare Query Monitoring</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background: #f5f8fc;
                color: #18314c;
                max-width: 1250px;
                margin: 30px auto;
                padding: 15px;
            }}
            .cards {{
                display: grid;
                grid-template-columns:
                    repeat(auto-fit, minmax(170px, 1fr));
                gap: 12px;
            }}
            .card {{
                background: white;
                padding: 18px;
                border: 1px solid #dce6f0;
                border-radius: 12px;
            }}
            small {{
                display: block;
                color: #61778d;
            }}
            strong {{
                display: block;
                font-size: 24px;
                margin-top: 10px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                background: white;
                font-size: 12px;
            }}
            td, th {{
                border-bottom: 1px solid #e3ebf3;
                padding: 10px;
                text-align: left;
            }}
            th {{ background: #e8f0f8; }}
            .scroll {{ overflow-x: auto; }}
            h2 {{ margin-top: 30px; }}
        </style>
    </head>
    <body>
        <h1>Adaptive Healthcare Knowledge Database</h1>
        <p>Phase 7 — Query Monitoring Dashboard</p>
        <p>
            Benchmark database |
            <a href="/">Refresh</a> |
            <a href="/api/summary">JSON API</a>
        </p>

        <div class="cards">{cards}</div>

        <h2>Most Frequent Query Fingerprints</h2>
        <div class="scroll">
            {html_table(frequency_report(args))}
        </div>

        <h2>Recent Query Executions</h2>
        <div class="scroll">
            {html_table(recent_queries(args))}
        </div>

        <h2>CPU, RAM and Database Resources</h2>
        <div class="scroll">
            {html_table(resource_history(args, 10))}
        </div>
    </body>
    </html>
    """


def start_api(args):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlparse(self.path)
            params = parse_qs(path.query)

            try:
                limit = min(
                    200,
                    max(1, int(params.get("limit", [30])[0]))
                )

                if path.path == "/":
                    content = dashboard(args).encode("utf-8")
                    mime = "text/html; charset=utf-8"

                else:
                    if path.path == "/api/health":
                        data = {
                            "status": "ok",
                            "database": args.database
                        }

                    elif path.path == "/api/summary":
                        data = summary(args)

                    elif path.path == "/api/recent":
                        data = recent_queries(args, limit)

                    elif path.path == "/api/frequency":
                        hours = min(
                            720,
                            max(
                                1,
                                int(params.get("hours", [24])[0])
                            )
                        )
                        data = frequency_report(args, hours)

                    elif path.path == "/api/resources":
                        data = resource_history(args, limit)

                    else:
                        self.send_error(404)
                        return

                    content = json.dumps(
                        data,
                        default=serialize
                    ).encode("utf-8")

                    mime = "application/json; charset=utf-8"

                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Cache-Control", "no-store")
                self.send_header(
                    "Content-Length", str(len(content))
                )
                self.end_headers()
                self.wfile.write(content)

            except Exception as exc:
                self.send_error(500, str(exc)[:160])

    server = ThreadingHTTPServer(
        ("127.0.0.1", args.api_port),
        Handler
    )

    print(
        f"Dashboard: http://127.0.0.1:{args.api_port}"
    )

    print("Press Ctrl+C to stop API.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


# ============================================================
# COMPONENT 12 — MONITORING VALIDATION
# ============================================================

def validate_monitoring(args):
    metrics = summary(args)

    invalid = query(args, f"""
        SELECT COUNT(*) AS invalid_rows
        FROM {SCHEMA}.query_log
        WHERE status = 'ok'
        AND (
            execution_ms IS NULL
            OR execution_ms < 0
            OR planning_ms IS NULL
            OR planning_ms < 0
            OR actual_rows < 0
            OR shared_hit < 0
            OR shared_read < 0
        )
    """)[0]["invalid_rows"]

    incomplete = query(args, f"""
        SELECT COUNT(*) AS incomplete_runs
        FROM {SCHEMA}.monitor_runs
        WHERE finished_at IS NOT NULL
        AND requested <> successful + failed
    """)[0]["incomplete_runs"]

    errors = query(args, f"""
        SELECT
            error_message,
            COUNT(*) AS occurrences
        FROM {SCHEMA}.query_log
        WHERE status = 'error'
        GROUP BY error_message
        ORDER BY occurrences DESC
        LIMIT 10
    """)

    overhead = query(args, f"""
        SELECT
            ROUND(
                AVG(
                    wall_ms - execution_ms - planning_ms
                )::numeric, 3
            ) AS avg_nonexecution_wall_ms
        FROM {SCHEMA}.query_log
        WHERE status = 'ok'
    """)[0]

    result = {
        "total_executions": metrics["total_executions"],
        "successful": metrics["successful"],
        "failed": metrics["failed"],
        "invalid_metric_rows": invalid,
        "incomplete_run_records": incomplete,
        "observed_overhead": overhead,
        "errors": errors,
        "basic_validation_passed": (
            invalid == 0
            and incomplete == 0
            and metrics["total_executions"] > 0
        )
    }

    print(json.dumps(
        result,
        indent=2,
        default=serialize
    ))


# ============================================================
# MAIN — RUN ALL PHASE 7 COMPONENTS
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=[
            "init",
            "once",
            "continuous",
            "serve",
            "summary",
            "resources",
            "validate"
        ]
    )

    parser.add_argument(
        "--host", default="127.0.0.1"
    )

    parser.add_argument(
        "--port", type=int, default=5433
    )

    parser.add_argument(
        "--database", default=DATABASE
    )

    parser.add_argument(
        "--user", default="postgres"
    )

    parser.add_argument(
        "--password",
        default=os.getenv("PGPASSWORD")
    )

    parser.add_argument(
        "--workload",
        type=Path,
        default=WORKLOAD
    )

    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=10000
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=20
    )

    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=900
    )

    parser.add_argument(
        "--max-cycles",
        type=int,
        default=0
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--api-port",
        type=int,
        default=8765
    )

    args = parser.parse_args()

    if (
        args.timeout_ms < 100
        or args.sample_size < 0
        or args.interval_seconds < 30
        or args.max_cycles < 0
    ):
        parser.error("Invalid monitoring limits")

    if args.command == "init":
        initialize_database(args)
        return

    if args.command == "serve":
        start_api(args)
        return

    if args.command == "summary":
        print(json.dumps(
            summary(args),
            indent=2,
            default=serialize
        ))
        return

    if args.command == "resources":
        collect_resources(args)
        print("Resource sample recorded.")
        return

    if args.command == "validate":
        validate_monitoring(args)
        return

    workload = load_workload(args.workload)

    print("=" * 75)
    print("PHASE 7 — HEALTHCARE QUERY MONITORING")
    print("=" * 75)
    print("Database:", args.database)
    print("Validated queries:", len(workload))
    print("Sample size:", args.sample_size)
    print("Physical index modifications: NONE")
    print("=" * 75)

    if args.command == "once":
        print(run_cycle(args, workload))
    else:
        continuous_monitoring(args, workload)


if __name__ == "__main__":
    main()
