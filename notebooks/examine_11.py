
import getpass
import json
from datetime import datetime
from pathlib import Path

import psycopg
from neo4j import GraphDatabase, READ_ACCESS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "phase10_8_patient_bridge"
OUT.mkdir(parents=True, exist_ok=True)

pg_password = getpass.getpass("PostgreSQL password: ")
neo_password = getpass.getpass("Neo4j password: ")

PG_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "user": "postgres",
    "password": pg_password,
    "options": "-c default_transaction_read_only=on",
}

def get_pg_patients(database):
    with psycopg.connect(dbname=database, **PG_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT patient_id::text, synthea_id::text "
                "FROM public.patient"
            )
            return cur.fetchall()

original = get_pg_patients("adaptive_healthcare_db")
scaled = get_pg_patients("adaptive_healthcare_scale_10k")

with GraphDatabase.driver(
    "bolt://127.0.0.1:7687",
    auth=("neo4j", neo_password)
) as driver:
    with driver.session(
        database="neo4j",
        default_access_mode=READ_ACCESS
    ) as session:
        graph_ids = {
            str(record["patient_id"])
            for record in session.run(
                "MATCH (p:Patient) "
                "RETURN p.patient_id AS patient_id"
            )
            if record["patient_id"] is not None
        }

original_ids = {str(row[0]) for row in original}
original_uuids = {
    str(row[1]).strip().lower()
    for row in original
    if row[1] is not None
}
scaled_uuids = {
    str(row[1]).strip().lower()
    for row in scaled
    if row[1] is not None
}

graph_original_overlap = graph_ids & original_ids
original_scaled_overlap = original_uuids & scaled_uuids

report = {
    "original_pg_patients": len(original),
    "scaled_pg_patients": len(scaled),
    "neo4j_unique_patient_ids": len(graph_ids),
    "original_pg_unique_patient_ids": len(original_ids),
    "graph_original_exact_matches": len(graph_original_overlap),
    "graph_original_match_percentage": round(
        100 * len(graph_original_overlap) / len(graph_ids), 2
    ) if graph_ids else 0,
    "graph_ids_missing_from_original": len(
        graph_ids - original_ids
    ),
    "original_ids_missing_from_graph": len(
        original_ids - graph_ids
    ),
    "original_scaled_synthea_uuid_matches": len(
        original_scaled_overlap
    ),
    "original_scaled_uuid_overlap_percentage": round(
        100 * len(original_scaled_overlap) / len(original_uuids), 2
    ) if original_uuids else 0,
    "original_graph_exact_set_equal": graph_ids == original_ids,
    "original_scaled_uuid_set_equal": (
        original_uuids == scaled_uuids
    ),
}

output = OUT / (
    "patient_bridge_"
    + datetime.now().strftime("%Y%m%d_%H%M%S")
    + ".json"
)

output.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)

print("\n" + "=" * 65)
print("PHASE 10.8 — PATIENT IDENTITY BRIDGE VALIDATION")
print("=" * 65)

for key, value in report.items():
    print(f"{key}: {value}")

print("\nReport saved:", output)
print("All database operations were read-only.")
