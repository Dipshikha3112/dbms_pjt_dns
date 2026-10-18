import json
import getpass
from datetime import datetime
from pathlib import Path
import psycopg

DB_NAME = "adaptive_healthcare_scale_50k"
OUTPUT = Path("phase5_scaling/scale_50000/results/phase3_etl_report.json")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

password = getpass.getpass("PostgreSQL password: ")

with psycopg.connect(
    host="localhost",
    port=5432,
    dbname=DB_NAME,
    user="postgres",
    password=password
) as conn:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT pg_size_pretty(pg_database_size(current_database())),
                   pg_database_size(current_database())
        """)
        size, size_bytes = cur.fetchone()

        cur.execute("""
            SELECT relname, n_live_tup
            FROM pg_stat_user_tables
            WHERE schemaname = 'public'
            ORDER BY relname
        """)
        table_counts = {
            name: int(count)
            for name, count in cur.fetchall()
        }

        cur.execute("""
            SELECT viewname
            FROM pg_views
            WHERE schemaname = 'public'
            ORDER BY viewname
        """)
        view_names = [row[0] for row in cur.fetchall()]

        cur.execute("""
            SELECT COUNT(*)
            FROM pg_constraint
            WHERE contype = 'f'
              AND connamespace = 'public'::regnamespace
        """)
        foreign_key_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM patient")
        exact_patient_count = cur.fetchone()[0]

report = {
    "database": {
        "name": DB_NAME,
        "host": "localhost",
        "port": 5432,
        "size": size,
        "size_bytes": size_bytes
    },
    "datasets": {
        "synthea": True,
        "loinc": True,
        "rxnorm": True
    },
    "table_counts": table_counts,
    "table_counts_method": "PostgreSQL statistics estimates, not exact counts",
    "exact_patient_count": exact_patient_count,
    "foreign_key_count": foreign_key_count,
    "view_names": view_names,
    "view_counts": {name: None for name in view_names},
    "view_counts_status": "Not evaluated during recovery",
    "status": "RECOVERED_REPORT_VALIDATION_PENDING",
    "recovered_at": datetime.now().isoformat()
}

OUTPUT.write_text(json.dumps(report, indent=4), encoding="utf-8")

print("\nREPORT RECOVERY COMPLETE")
print("Database:", DB_NAME)
print("Patients:", exact_patient_count)
print("Database size:", size)
print("Tables:", len(table_counts))
print("Foreign keys:", foreign_key_count)
print("Views:", len(view_names))
print("Report saved:", OUTPUT)
