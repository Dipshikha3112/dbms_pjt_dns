
import argparse
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import psycopg
from neo4j import GraphDatabase


ROOT = Path(__file__).resolve().parent.parent


def normalize_id(value):
    if value is None:
        return ""
    return str(value).strip().lower()


def normalize_code(value):
    if value is None:
        return ""
    return str(value).strip().upper()


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, default=str),
        encoding="utf-8"
    )


def get_sql_patients(conn, selected_id=None):
    with conn.cursor() as cur:
        if selected_id:
            cur.execute(
                """
                SELECT patient_id::text, synthea_id::text
                FROM public.patient
                WHERE patient_id::text = %s
                   OR synthea_id::text = %s
                """,
                (selected_id, selected_id)
            )
        else:
            cur.execute(
                """
                SELECT patient_id::text, synthea_id::text
                FROM public.patient
                ORDER BY patient_id
                LIMIT 1
                """
            )

        return [
            {"patient_id": row[0], "synthea_id": row[1]}
            for row in cur.fetchall()
        ]


def get_graph_patients(session):
    records = session.run(
        """
        MATCH (p:Patient)
        RETURN elementId(p) AS element_id,
               p.patient_id AS patient_id,
               p.synthea_id AS synthea_id
        """
    )

    return [dict(record) for record in records]


def resolve_identity(sql_patient, graph_patients):
    sql_id = normalize_id(sql_patient["patient_id"])
    synthea_id = normalize_id(sql_patient["synthea_id"])

    direct_matches = [
        p for p in graph_patients
        if normalize_id(p["patient_id"]) == sql_id
    ]

    if len(direct_matches) == 1:
        return {
            "status": "VERIFIED",
            "method": "EXACT_SOURCE_PATIENT_ID",
            "graph_patient": direct_matches[0]
        }

    if len(direct_matches) > 1:
        return {
            "status": "AMBIGUOUS",
            "method": "DUPLICATE_GRAPH_PATIENT_ID",
            "matches": len(direct_matches)
        }

    if not synthea_id:
        return {
            "status": "UNMATCHED",
            "reason": "SQL patient has no Synthea source ID"
        }

    source_matches = [
        p for p in graph_patients
        if synthea_id in {
            normalize_id(p["synthea_id"]),
            normalize_id(p["patient_id"])
        }
    ]

    if len(source_matches) == 1:
        return {
            "status": "CANDIDATE_REQUIRES_PROVENANCE",
            "method": "SYNTHEA_ID_MATCH",
            "graph_patient": source_matches[0],
            "reason": (
                "Identifier matches, but original graph import "
                "provenance must confirm the namespace"
            )
        }

    if len(source_matches) > 1:
        return {
            "status": "AMBIGUOUS",
            "method": "DUPLICATE_SOURCE_ID",
            "matches": len(source_matches)
        }

    return {
        "status": "UNMATCHED",
        "reason": "No matching graph patient identifier"
    }


def get_sql_medications(conn, patient_id):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT pi.rxcui::text
            FROM public.prescription p
            JOIN public.prescription_item pi
              ON p.prescription_id = pi.prescription_id
            WHERE p.patient_id::text = %s
              AND pi.rxcui IS NOT NULL
            """,
            (patient_id,)
        )

        return {
            normalize_code(row[0])
            for row in cur.fetchall()
            if normalize_code(row[0])
        }


def get_graph_medications(session, element_id):
    records = session.run(
        """
        MATCH (p:Patient)-[:HAS_MEDICATION]->(d:Drug)
        WHERE elementId(p) = $element_id
        RETURN d.rxcui AS rxcui
        """,
        element_id=element_id
    )

    return {
        normalize_code(record["rxcui"])
        for record in records
        if normalize_code(record["rxcui"])
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pg-database",
        default="adaptive_healthcare_db"
    )
    parser.add_argument("--pg-host", default="127.0.0.1")
    parser.add_argument("--pg-port", type=int, default=5432)
    parser.add_argument("--pg-user", default="postgres")
    parser.add_argument("--patient-id", default=None)
    parser.add_argument(
        "--neo4j-uri",
        default="bolt://127.0.0.1:7687"
    )
    parser.add_argument(
        "--output",
        default=str(
            ROOT / "results" / "phase10_bridge"
            / "identity_bridge_report.json"
        )
    )

    args = parser.parse_args()

    pg_password = os.getenv("PGPASSWORD")
    neo_password = os.getenv("NEO4J_PASSWORD")

    if not pg_password or not neo_password:
        raise RuntimeError(
            "Set PGPASSWORD and NEO4J_PASSWORD first"
        )

    report = {
        "phase": "10",
        "task": "PostgreSQL-Neo4j Identity Bridge",
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "postgres_database": args.pg_database,
        "read_only": True,
        "patients": []
    }

    with psycopg.connect(
        host=args.pg_host,
        port=args.pg_port,
        dbname=args.pg_database,
        user=args.pg_user,
        password=pg_password,
        options="-c default_transaction_read_only=on"
    ) as conn:

        with GraphDatabase.driver(
            args.neo4j_uri,
            auth=("neo4j", neo_password)
        ) as driver:

            driver.verify_connectivity()

            with driver.session(
                database="neo4j"
            ) as session:

                sql_patients = get_sql_patients(
                    conn, args.patient_id
                )

                if not sql_patients:
                    raise RuntimeError(
                        "No matching PostgreSQL patient found"
                    )

                graph_patients = get_graph_patients(session)

                for patient in sql_patients:
                    identity = resolve_identity(
                        patient, graph_patients
                    )

                    result = {
                        "sql_patient": patient,
                        "identity_status": identity["status"],
                        "identity_method": identity.get("method"),
                        "identity_details": identity,
                        "medication_comparison": None
                    }

                    if identity["status"] == "VERIFIED":
                        graph_patient = identity["graph_patient"]

                        sql_codes = get_sql_medications(
                            conn, patient["patient_id"]
                        )

                        graph_codes = get_graph_medications(
                            session,
                            graph_patient["element_id"]
                        )

                        matched = sql_codes & graph_codes
                        sql_only = sql_codes - graph_codes
                        graph_only = graph_codes - sql_codes

                        result["medication_comparison"] = {
                            "sql_codes": sorted(sql_codes),
                            "graph_codes": sorted(graph_codes),
                            "verified_codes": sorted(matched),
                            "sql_only_codes": sorted(sql_only),
                            "graph_only_codes": sorted(graph_only),
                            "consistent": sql_codes == graph_codes
                        }

                    report["patients"].append(result)

    verified = [
        p for p in report["patients"]
        if p["identity_status"] == "VERIFIED"
    ]

    consistent = [
        p for p in verified
        if p["medication_comparison"]["consistent"]
    ]

    report["summary"] = {
        "patients_checked": len(report["patients"]),
        "verified_identities": len(verified),
        "consistent_medication_sets": len(consistent),
        "unresolved_identities": (
            len(report["patients"]) - len(verified)
        ),
        "acceptance_pass": (
            len(report["patients"]) == 1
            and len(consistent) == 1
        ),
        "scope": (
            "Identity and distinct RxCUI medication pairs only; "
            "conditions, labs, event counts and timestamps "
            "are not validated by this script"
        )
    }

    save_json(Path(args.output), report)

    print(json.dumps(report["summary"], indent=2))
    print(f"\nReport saved: {args.output}")


if __name__ == "__main__":
    main()
