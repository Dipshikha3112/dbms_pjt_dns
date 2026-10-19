
#!/usr/bin/env python3

import argparse
import csv
import json
import os
import sqlite3
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request

from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

DEFAULT_PHASE9 = (
    ROOT
    / "results"
    / "phase8_phase9_v7_audit"
    / "20261018_130202_315654"
)

DEFAULT_PHASE12 = ROOT / "results" / "phase12_store_v3"

CHECKS = []

API_ENDPOINTS = [
    ("optimizer_summary", "/api/optimizer/summary"),
    ("optimizer_budget_sweep", "/api/optimizer/budget-sweep"),
    ("optimizer_reset_counters", "/api/optimizer/reset-counters"),
    ("graph_stats", "/api/graph/stats"),
    ("graph_patients", "/api/graph/patients"),
    ("monitoring_summary", "/api/monitoring/summary"),
    ("monitoring_recent", "/api/monitoring/recent"),
    ("monitoring_top_fingerprints", "/api/monitoring/top-fingerprints"),
    ("retrieval_doctor", "/api/retrieval/doctor"),
    ("system_capabilities", "/api/system/capabilities"),
]


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def record(name, status, message, duration, details=None):
    item = {
        "name": name,
        "status": status,
        "message": message,
        "duration_ms": round(duration * 1000, 2),
        "details": details or {},
    }

    CHECKS.append(item)

    symbol = {
        "PASS": "[PASS]",
        "FAIL": "[FAIL]",
        "WARN": "[WARN]",
        "SKIP": "[SKIP]",
    }.get(status, "[INFO]")

    print(f"{symbol:6} {name}: {message}")

    return item


def run_check(name, callback):
    start = time.perf_counter()

    try:
        status, message, details = callback()

    except Exception as exc:
        status = "FAIL"
        message = f"{type(exc).__name__}: {exc}"
        details = {}

    return record(
        name,
        status,
        message,
        time.perf_counter() - start,
        details,
    )


def http_request(url, method="GET", payload=None, timeout=15):
    headers = {
        "Accept": "application/json",
        "User-Agent": "DBMS-System-Health-Check/1.0",
    }

    body = None

    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method=method,
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout,
        ) as response:
            raw = response.read(2_000_000)
            status_code = response.status
            content_type = response.headers.get(
                "Content-Type", ""
            )

    except urllib.error.HTTPError as exc:
        raw = exc.read(2000)
        status_code = exc.code
        content_type = exc.headers.get(
            "Content-Type", ""
        )

    decoded = raw.decode("utf-8", errors="replace")

    try:
        data = json.loads(decoded)
    except json.JSONDecodeError:
        data = None

    return {
        "status_code": status_code,
        "content_type": content_type,
        "json": data,
        "body_preview": decoded[:500],
    }


def postgres_check(args):
    try:
        import psycopg
    except ImportError:
        return (
            "FAIL",
            "psycopg not installed",
            {"install": "pip install psycopg[binary]"},
        )

    connection_kwargs = {
        "host": args.pg_host,
        "port": args.pg_port,
        "dbname": args.pg_database,
        "user": args.pg_user,
        "connect_timeout": 8,
    }

    password = os.getenv("PGPASSWORD")

    if password:
        connection_kwargs["password"] = password

    with psycopg.connect(**connection_kwargs) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT current_database(), "
                "current_user, version()"
            )
            database, user, version = cursor.fetchone()

            cursor.execute(
                "SELECT COUNT(*) "
                "FROM information_schema.tables "
                "WHERE table_schema = 'public'"
            )
            table_count = cursor.fetchone()[0]

    details = {
        "database": database,
        "user": user,
        "postgres_version": version.split(",")[0],
        "public_table_count": table_count,
    }

    if table_count == 0:
        return (
            "WARN",
            "PostgreSQL connected but no public tables found",
            details,
        )

    return (
        "PASS",
        f"Connected to {database}; {table_count} public tables",
        details,
    )


def neo4j_check(args):
    try:
        from neo4j import GraphDatabase
    except ImportError:
        return (
            "FAIL",
            "neo4j driver not installed",
            {"install": "pip install neo4j"},
        )

    username = args.neo4j_user
    password = os.getenv("NEO4J_PASSWORD", "")

    if not password:
        return (
            "FAIL",
            "NEO4J_PASSWORD environment variable is missing",
            {},
        )

    driver = GraphDatabase.driver(
        args.neo4j_uri,
        auth=(username, password),
        connection_timeout=8,
    )

    try:
        driver.verify_connectivity()

        with driver.session(
            database=args.neo4j_database
        ) as session:
            result = session.run(
                "MATCH (n) RETURN count(n) AS count"
            )
            node_count = result.single()["count"]

            result = session.run(
                "MATCH ()-[r]->() RETURN count(r) AS count"
            )
            relationship_count = result.single()["count"]

    finally:
        driver.close()

    details = {
        "uri": args.neo4j_uri,
        "database": args.neo4j_database,
        "nodes": node_count,
        "relationships": relationship_count,
    }

    if node_count == 0:
        return (
            "WARN",
            "Neo4j connected but graph has no nodes",
            details,
        )

    return (
        "PASS",
        f"{node_count:,} nodes, "
        f"{relationship_count:,} relationships",
        details,
    )


def phase9_check(args):
    folder = Path(args.phase9_dir)

    if not folder.exists():
        return (
            "FAIL",
            f"Phase 9 audit directory not found: {folder}",
            {},
        )

    csv_files = sorted(folder.rglob("*.csv"))
    json_files = sorted(folder.rglob("*.json"))

    readable = []
    invalid = []

    for path in csv_files:
        try:
            with path.open(
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as handle:
                reader = csv.reader(handle)
                header = next(reader, None)

            if not header:
                invalid.append(str(path.relative_to(folder)))
            else:
                readable.append(str(path.relative_to(folder)))

        except Exception as exc:
            invalid.append(
                f"{path.relative_to(folder)}: {exc}"
            )

    details = {
        "directory": str(folder),
        "csv_count": len(csv_files),
        "json_count": len(json_files),
        "readable_csvs": readable[:100],
        "invalid_csvs": invalid[:50],
    }

    if not csv_files:
        return (
            "FAIL",
            "No Phase 9 CSV results found",
            details,
        )

    if invalid:
        return (
            "WARN",
            f"{len(invalid)} CSV files failed validation",
            details,
        )

    return (
        "PASS",
        f"{len(readable)} readable Phase 9 CSV files",
        details,
    )



def phase12_check(args):
    folder = Path(args.phase12_dir)

    if not folder.is_dir():
        return (
            "FAIL",
            f"Phase 12 directory not found: {folder}",
            {},
        )

    sqlite_path = folder / "literature.sqlite"
    joblib_path = folder / "retrieval.joblib"

    details = {
        "directory": str(folder),
        "sqlite_exists": sqlite_path.is_file(),
        "joblib_exists": joblib_path.is_file(),
    }

    if not sqlite_path.is_file():
        return (
            "FAIL",
            "literature.sqlite is missing",
            details,
        )

    try:
        uri = (
            "file:"
            + urllib.parse.quote(str(sqlite_path.resolve()))
            + "?mode=ro"
        )

        connection = sqlite3.connect(
            uri,
            uri=True,
            timeout=10,
        )

        try:
            integrity = connection.execute(
                "PRAGMA integrity_check"
            ).fetchone()[0]

            if integrity != "ok":
                return (
                    "FAIL",
                    f"SQLite integrity failed: {integrity}",
                    details,
                )

            required_tables = {
                "documents",
                "chunks",
                "chunk_fts",
            }

            actual_tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table'"
                )
            }

            missing = required_tables - actual_tables

            if missing:
                details["missing_tables"] = sorted(missing)
                return (
                    "FAIL",
                    "Required Phase 12 tables are missing",
                    details,
                )

            document_count = connection.execute(
                "SELECT COUNT(*) FROM documents"
            ).fetchone()[0]

            chunk_count = connection.execute(
                "SELECT COUNT(*) FROM chunks"
            ).fetchone()[0]

            fts_count = connection.execute(
                "SELECT COUNT(*) FROM chunk_fts"
            ).fetchone()[0]

        finally:
            connection.close()

    except Exception as exc:
        return (
            "FAIL",
            f"Phase 12 SQLite validation failed: {exc}",
            details,
        )

    details.update({
        "sqlite_integrity": "ok",
        "document_count": document_count,
        "chunk_count": chunk_count,
        "fts_count": fts_count,
        "sqlite_size_bytes": sqlite_path.stat().st_size,
        "joblib_size_bytes": (
            joblib_path.stat().st_size
            if joblib_path.is_file()
            else 0
        ),
    })

    if document_count == 0 or chunk_count == 0:
        return (
            "FAIL",
            "Phase 12 literature index is empty",
            details,
        )

    if fts_count != chunk_count:
        return (
            "WARN",
            "FTS count does not match chunk count",
            details,
        )

    if not joblib_path.is_file():
        return (
            "WARN",
            "SQLite index works, but retrieval.joblib is missing",
            details,
        )

    return (
        "PASS",
        f"Phase 12 index ready: {document_count} documents, "
        f"{chunk_count} chunks, {fts_count} FTS entries; "
        "retrieval artifact present",
        details,
    )


def api_check(args, name, endpoint):
    url = args.api_url.rstrip("/") + endpoint

    try:
        response = http_request(
            url,
            timeout=args.api_timeout,
        )
    except Exception as exc:
        return (
            "FAIL",
            f"HTTP request failed: {exc}",
            {"url": url},
        )

    status_code = response["status_code"]
    data = response["json"]

    details = {
        "url": url,
        "http_status": status_code,
        "content_type": response["content_type"],
    }

    if isinstance(data, dict):
        details["response_keys"] = list(data.keys())[:30]

    if status_code != 200:
        details["body_preview"] = response["body_preview"]
        return (
            "FAIL",
            f"HTTP {status_code}",
            details,
        )

    if data is None:
        return (
            "WARN",
            "HTTP 200 but response is not JSON",
            details,
        )

    if isinstance(data, dict):
        if data.get("ok") is False:
            return (
                "FAIL",
                "Endpoint reports ok=false",
                details,
            )

        if data.get("error"):
            details["reported_error"] = str(
                data["error"]
            )[:300]

            return (
                "FAIL",
                "Endpoint returned an error",
                details,
            )

    return (
        "PASS",
        "HTTP 200 and valid JSON",
        details,
    )


def extract_patient_id(payload):
    if isinstance(payload, list):
        patients = payload

    elif isinstance(payload, dict):
        patients = None

        for key in (
            "patients",
            "items",
            "results",
            "data",
        ):
            value = payload.get(key)

            if isinstance(value, list):
                patients = value
                break

        if patients is None:
            return None

    else:
        return None

    for patient in patients:
        if isinstance(patient, dict):
            for key in (
                "patient_id",
                "id",
                "patientId",
            ):
                value = patient.get(key)

                if value is not None:
                    return str(value)

        elif isinstance(patient, (str, int)):
            return str(patient)

    return None


def patient_endpoint_check(args):
    url = (
        args.api_url.rstrip("/")
        + "/api/graph/patients"
    )

    response = http_request(
        url,
        timeout=args.api_timeout,
    )

    if response["status_code"] != 200:
        return (
            "SKIP",
            "Cannot discover patient ID: patient list API failed",
            {},
        )

    patient_id = extract_patient_id(
        response["json"]
    )

    if not patient_id:
        return (
            "WARN",
            "Patient list returned no discoverable patient ID",
            {},
        )

    encoded_id = urllib.parse.quote(
        patient_id,
        safe="",
    )

    return api_check(
        args,
        "graph_patient_detail",
        f"/api/graph/patient/{encoded_id}",
    )


def retrieval_ask_check(args):
    if args.skip_ask:
        return (
            "SKIP",
            "Retrieval query skipped by --skip-ask",
            {},
        )

    url = (
        args.api_url.rstrip("/")
        + "/api/retrieval/ask"
    )

    query = urllib.parse.urlencode({
        "q": (
            "What evidence is available about "
            "apixaban in atrial fibrillation?"
        )
    })

    response = http_request(
        url + "?" + query,
        timeout=args.ask_timeout,
    )

    details = {
        "url": url,
        "http_status": response["status_code"],
    }

    if response["status_code"] != 200:
        details["body_preview"] = response["body_preview"]
        return (
            "FAIL",
            f"HTTP {response['status_code']}",
            details,
        )

    data = response["json"]

    if not isinstance(data, dict):
        return (
            "WARN",
            "Retrieval endpoint returned unexpected JSON",
            details,
        )

    details["response_keys"] = list(data.keys())[:30]

    if data.get("error") or data.get("ok") is False:
        return (
            "FAIL",
            "Retrieval endpoint reports an error",
            details,
        )

    return (
        "PASS",
        "Retrieval endpoint responded successfully",
        details,
    )


def dashboard_check(args):
    url = args.api_url.rstrip("/") + "/"

    response = http_request(
        url,
        timeout=args.api_timeout,
    )

    status = response["status_code"]

    if status == 200:
        return (
            "PASS",
            "Dashboard root is reachable",
            {"http_status": status},
        )

    return (
        "WARN",
        f"Dashboard root returned HTTP {status}",
        {"http_status": status},
    )


def openapi_check(args):
    url = args.api_url.rstrip("/") + "/openapi.json"

    response = http_request(
        url,
        timeout=args.api_timeout,
    )

    if response["status_code"] != 200:
        return (
            "FAIL",
            "OpenAPI specification unavailable",
            {"http_status": response["status_code"]},
        )

    data = response["json"]

    if not isinstance(data, dict):
        return (
            "FAIL",
            "OpenAPI response is not a JSON object",
            {},
        )

    paths = data.get("paths", {})

    return (
        "PASS",
        f"OpenAPI available with {len(paths)} paths",
        {"path_count": len(paths)},
    )


def summarize(checks):
    counts = {
        status: sum(
            item["status"] == status
            for item in checks
        )
        for status in ("PASS", "FAIL", "WARN", "SKIP")
    }

    if counts["FAIL"]:
        overall = "FAIL"
    elif counts["WARN"]:
        overall = "WARN"
    elif counts["SKIP"]:
        overall = "INCOMPLETE"
    else:
        overall = "PASS"

    return {
        "overall_status": overall,
        "total_checks": len(checks),
        "counts": counts,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Adaptive Healthcare Database "
            "System Health Check"
        )
    )

    parser.add_argument(
        "--output",
        default=str(
            ROOT / "results"
            / "system_health"
            / "system_health.json"
        ),
    )

    parser.add_argument(
        "--pg-host",
        default=os.getenv("PGHOST", "127.0.0.1"),
    )

    parser.add_argument(
        "--pg-port",
        type=int,
        default=int(os.getenv("PGPORT", "5433")),
    )

    parser.add_argument(
        "--pg-database",
        default=os.getenv(
            "PGDATABASE",
            "adaptive_healthcare_phase9_benchmark",
        ),
    )

    parser.add_argument(
        "--pg-user",
        default=os.getenv("PGUSER", "postgres"),
    )

    parser.add_argument(
        "--neo4j-uri",
        default=os.getenv(
            "NEO4J_URI",
            "bolt://127.0.0.1:7687",
        ),
    )

    parser.add_argument(
        "--neo4j-user",
        default=os.getenv(
            "NEO4J_USER",
            "neo4j",
        ),
    )

    parser.add_argument(
        "--neo4j-database",
        default=os.getenv(
            "NEO4J_DATABASE",
            "neo4j",
        ),
    )

    parser.add_argument(
        "--phase9-dir",
        default=str(DEFAULT_PHASE9),
    )

    parser.add_argument(
        "--phase12-dir",
        default=str(DEFAULT_PHASE12),
    )

    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000",
    )

    parser.add_argument(
        "--api-timeout",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--ask-timeout",
        type=int,
        default=90,
    )

    parser.add_argument(
        "--skip-ask",
        action="store_true",
    )

    args = parser.parse_args()

    started = now_iso()
    started_clock = time.perf_counter()

    print("=" * 75)
    print("ADAPTIVE HEALTHCARE DATABASE — SYSTEM HEALTH CHECK")
    print("=" * 75)
    print(f"Project: {ROOT}")
    print(f"Started: {started}")
    print("=" * 75)

    run_check(
        "postgresql",
        lambda: postgres_check(args),
    )

    run_check(
        "neo4j",
        lambda: neo4j_check(args),
    )

    run_check(
        "phase9_optimizer_results",
        lambda: phase9_check(args),
    )

    run_check(
        "phase12_retrieval_index",
        lambda: phase12_check(args),
    )

    run_check(
        "fastapi_dashboard",
        lambda: dashboard_check(args),
    )

    run_check(
        "fastapi_openapi",
        lambda: openapi_check(args),
    )

    for name, endpoint in API_ENDPOINTS:
        run_check(
            f"api_{name}",
            lambda name=name, endpoint=endpoint: api_check(
                args,
                name,
                endpoint,
            ),
        )

    run_check(
        "api_graph_patient_detail",
        lambda: patient_endpoint_check(args),
    )

    run_check(
        "api_retrieval_ask",
        lambda: retrieval_ask_check(args),
    )

    summary = summarize(CHECKS)

    report = {
        "project": "Adaptive Healthcare Knowledge Database",
        "started_at": started,
        "finished_at": now_iso(),
        "duration_seconds": round(
            time.perf_counter() - started_clock,
            3,
        ),
        "configuration": {
            "postgresql": {
                "host": args.pg_host,
                "port": args.pg_port,
                "database": args.pg_database,
            },
            "neo4j": {
                "uri": args.neo4j_uri,
                "database": args.neo4j_database,
            },
            "phase9_dir": args.phase9_dir,
            "phase12_dir": args.phase12_dir,
            "api_url": args.api_url,
        },
        "summary": summary,
        "checks": CHECKS,
    }

    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 75)
    print("HEALTH CHECK SUMMARY")
    print("=" * 75)

    for status, count in summary["counts"].items():
        print(f"{status:6}: {count}")

    print(f"OVERALL: {summary['overall_status']}")
    print(f"REPORT:  {output.resolve()}")
    print("=" * 75)

    return 1 if summary["overall_status"] == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
