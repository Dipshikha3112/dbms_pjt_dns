import os
import time
from pathlib import Path

import pandas as pd
import psycopg2

ROOT = Path(__file__).resolve().parent.parent

INPUT = ROOT / "results" / "phase6_workloads" / "combined_workload.csv"
OUTPUT_DIR = ROOT / "results" / "phase5_scaling" / "scale_50k"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "adaptive_healthcare_scale_50k",
    "user": "postgres",
    "password": os.environ.get("PGPASSWORD", ""),
    "connect_timeout": 15,
}

TIMEOUT_MS = 30000

workload = pd.read_csv(INPUT)
results = []

conn = psycopg2.connect(**DB_CONFIG)
conn.autocommit = False

print(f"Database: {DB_CONFIG['dbname']}")
print(f"Workload queries: {len(workload)}")
print("Starting 10K-scale benchmark...\n", flush=True)

for i, row in workload.iterrows():
    sql = str(row["sql"]).strip().rstrip(";")
    start = time.perf_counter()

    status = "success"
    error = ""
    returned_rows = 0

    try:
        with conn.cursor() as cur:
            cur.execute("BEGIN READ ONLY")
            cur.execute(f"SET LOCAL statement_timeout = {TIMEOUT_MS}")
            cur.execute(sql)

            if cur.description is not None:
                while True:
                    batch = cur.fetchmany(1000)
                    if not batch:
                        break
                    returned_rows += len(batch)

        conn.rollback()

    except Exception as exc:
        conn.rollback()
        status = "timeout" if "statement timeout" in str(exc).lower() else "error"
        error = str(exc).splitlines()[0][:300]

    elapsed_ms = (time.perf_counter() - start) * 1000

    results.append({
        "query_hash": row["query_hash"],
        "workload_type": row["workload_type"],
        "query_family": row["query_family"],
        "query_name": row["query_name"],
        "complexity": row["complexity"],
        "execution_time_ms": round(elapsed_ms, 3),
        "returned_rows": returned_rows,
        "status": status,
        "error": error,
    })

    if (i + 1) % 25 == 0 or i + 1 == len(workload):
        print(
            f"Completed {i + 1}/{len(workload)} queries",
            flush=True
        )
        pd.DataFrame(results).to_csv(
            OUTPUT_DIR / "benchmark_results.csv",
            index=False
        )

conn.close()

df = pd.DataFrame(results)
successful = df[df["status"] == "success"]

print("\nBENCHMARK COMPLETE")
print("Successful:", len(successful))
print("Errors/timeouts:", len(df) - len(successful))

if not successful.empty:
    print("Mean latency (ms):", round(successful["execution_time_ms"].mean(), 3))
    print("Median latency (ms):", round(successful["execution_time_ms"].median(), 3))
    print("P95 latency (ms):", round(successful["execution_time_ms"].quantile(0.95), 3))

print("Results:", OUTPUT_DIR / "benchmark_results.csv")
