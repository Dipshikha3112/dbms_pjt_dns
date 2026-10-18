
from __future__ import annotations

import argparse
import csv
import getpass
import hashlib
import json
import os
import re
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import psycopg
from psycopg import sql
from neo4j import GraphDatabase, READ_ACCESS


ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "results" / "phase10_8_patient_root_cause"

ID_FIELD = re.compile(
    r"(^id$|_id$|uuid|synthea|source|external|"
    r"person_id|patient_id|identifier|record_id)",
    re.IGNORECASE,
)

UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{12}$",
    re.IGNORECASE,
)

HEX32_PATTERN = re.compile(r"^[0-9a-f]{32}$", re.I)


def log(message):
    print(message, flush=True)


def value_string(value):
    if value is None:
        return None

    if isinstance(value, (list, dict)):
        return None

    value = str(value).strip()
    return value if value else None


def normalized_uuid(value):
    value = value_string(value)

    if not value:
        return None

    if UUID_PATTERN.fullmatch(value):
        return value.replace("-", "").lower()

    if HEX32_PATTERN.fullmatch(value):
        return value.lower()

    return None


def fingerprint(value):
    value = value_string(value)

    if not value:
        return None

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()[:16]


def classify(value):
    value = value_string(value)

    if value is None:
        return "NULL"

    if UUID_PATTERN.fullmatch(value):
        return "UUID_HYPHENATED"

    if HEX32_PATTERN.fullmatch(value):
        return "UUID_OR_HEX32"

    if value.isdigit():
        return "NUMERIC"

    if re.fullmatch(r"[A-Za-z]+[-_]?\d+", value):
        return "PREFIXED_NUMERIC"

    if re.fullmatch(r"[0-9a-f-]{20,}", value, re.I):
        return "HEX_LIKE"

    return "OTHER_STRING"


def save_json(path, obj):
    path.write_text(
        json.dumps(obj, indent=2, default=str),
        encoding="utf-8"
    )


def save_csv(path, rows, columns):
    with path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=columns,
            extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def graph_read(driver, database, query, parameters=None):
    with driver.session(
        database=database,
        default_access_mode=READ_ACCESS
    ) as session:
        result = session.run(query, parameters or {})
        records = [dict(record) for record in result]
        result.consume()
        return records


def get_pg_columns(conn):
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'patient'
            ORDER BY ordinal_position
        """)

        return [
            {"name": row[0], "type": row[1]}
            for row in cursor.fetchall()
        ]


def get_pg_patients(conn, columns):
    if not columns:
        return []

    query = sql.SQL(
        "SELECT {} FROM public.patient"
    ).format(
        sql.SQL(", ").join(
            sql.Identifier(column)
            for column in columns
        )
    )

    with conn.cursor() as cursor:
        cursor.execute(query)
        return [
            dict(zip(columns, row))
            for row in cursor.fetchall()
        ]


def extract_graph_ids(records, field):
    values = []

    for record in records:
        raw = record["properties"].get(field)

        if isinstance(raw, list):
            values.extend(
                item for item in raw
                if value_string(item) is not None
            )
        elif value_string(raw) is not None:
            values.append(raw)

    return values


def profile_values(values):
    strings = [
        value_string(value)
        for value in values
    ]

    strings = [
        value for value in strings
        if value is not None
    ]

    counts = Counter(classify(value) for value in strings)

    lengths = [len(value) for value in strings]

    return {
        "non_null_values": len(strings),
        "distinct_values": len(set(strings)),
        "format_counts": dict(counts),
        "min_length": min(lengths) if lengths else None,
        "median_length": (
            statistics.median(lengths)
            if lengths else None
        ),
        "max_length": max(lengths) if lengths else None,
        "sample_value_hashes": [
            fingerprint(value)
            for value in strings[:5]
        ]
    }


def collect_identifier_fields(graph_records, pg_columns):
    graph_properties = set()

    for record in graph_records:
        graph_properties.update(
            record["properties"].keys()
        )

    graph_ids = sorted(
        field for field in graph_properties
        if ID_FIELD.search(field)
    )

    pg_ids = sorted(
        item["name"]
        for item in pg_columns
        if ID_FIELD.search(item["name"])
    )

    return graph_properties, graph_ids, pg_ids


def identifier_comparisons(
    graph_records, pg_records,
    graph_fields, pg_fields
):
    graph_values = {
        field: extract_graph_ids(graph_records, field)
        for field in graph_fields
    }

    pg_values = {
        field: [
            record[field]
            for record in pg_records
            if value_string(record[field]) is not None
        ]
        for field in pg_fields
    }

    rows = []

    for graph_field in graph_fields:
        graph_set = {
            value_string(value)
            for value in graph_values[graph_field]
            if value_string(value) is not None
        }

        graph_uuid = {
            normalized_uuid(value)
            for value in graph_set
            if normalized_uuid(value) is not None
        }

        for pg_field in pg_fields:
            pg_set = {
                value_string(value)
                for value in pg_values[pg_field]
                if value_string(value) is not None
            }

            pg_uuid = {
                normalized_uuid(value)
                for value in pg_set
                if normalized_uuid(value) is not None
            }

            exact = graph_set & pg_set
            uuid_matches = graph_uuid & pg_uuid

            rows.append({
                "graph_field": graph_field,
                "postgres_field": pg_field,
                "graph_distinct": len(graph_set),
                "postgres_distinct": len(pg_set),
                "exact_overlap": len(exact),
                "uuid_normalized_overlap": len(uuid_matches),
                "status": (
                    "CANDIDATE_EXACT_BRIDGE"
                    if exact else
                    "CANDIDATE_UUID_BRIDGE"
                    if uuid_matches else
                    "NO_OVERLAP"
                )
            })

    return rows, graph_values, pg_values


def inspect_relationships(driver, database):
    return graph_read(
        driver,
        database,
        """
        MATCH (p:Patient)-[r]-(n)
        RETURN
            type(r) AS relationship_type,
            labels(n) AS neighbor_labels,
            count(*) AS count
        ORDER BY count DESC
        """
    )


def inspect_sql_foreign_keys(conn):
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT
                tc.table_name,
                kcu.column_name,
                ccu.table_name AS referenced_table,
                ccu.column_name AS referenced_column
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage ccu
              ON tc.constraint_name = ccu.constraint_name
             AND tc.table_schema = ccu.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
              AND tc.table_schema = 'public'
              AND ccu.table_name = 'patient'
            ORDER BY tc.table_name, kcu.column_name
        """)

        return [
            {
                "table": row[0],
                "column": row[1],
                "referenced_table": row[2],
                "referenced_column": row[3]
            }
            for row in cursor.fetchall()
        ]


def inspect_local_import_files():
    roots = [
        ROOT / "notebooks",
        ROOT / "scripts",
    ]

    keywords = [
        "patient_id",
        "synthea_id",
        "MERGE (p:Patient",
        "CREATE (p:Patient",
        "Patient {",
        "RESET_GRAPH",
    ]

    results = []

    for folder in roots:
        if not folder.exists():
            continue

        for path in sorted(folder.glob("*.py")):
            if path.name == Path(__file__).name:
                continue

            try:
                if path.stat().st_size > 2_000_000:
                    continue

                text = path.read_text(
                    encoding="utf-8",
                    errors="replace"
                )

                found = [
                    keyword
                    for keyword in keywords
                    if keyword.lower() in text.lower()
                ]

                if found:
                    results.append({
                        "file": str(path.relative_to(ROOT)),
                        "matched_keywords": found
                    })

            except OSError:
                continue

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--neo4j-uri",
        default="bolt://127.0.0.1:7687"
    )
    parser.add_argument(
        "--neo4j-user",
        default="neo4j"
    )
    parser.add_argument(
        "--neo4j-database",
        default="neo4j"
    )
    parser.add_argument(
        "--neo4j-password",
        default=os.getenv("NEO4J_PASSWORD")
    )
    parser.add_argument(
        "--ask-neo4j-password",
        action="store_true"
    )

    parser.add_argument(
        "--pg-host",
        default="localhost"
    )
    parser.add_argument(
        "--pg-port",
        type=int,
        default=5432
    )
    parser.add_argument(
        "--pg-user",
        default="postgres"
    )
    parser.add_argument(
        "--pg-database",
        default="adaptive_healthcare_scale_10k"
    )
    parser.add_argument(
        "--pg-password",
        default=os.getenv("PGPASSWORD")
    )
    parser.add_argument(
        "--ask-pg-password",
        action="store_true"
    )

    args = parser.parse_args()

    if args.ask_neo4j_password:
        args.neo4j_password = getpass.getpass(
            "Neo4j password: "
        )

    if args.ask_pg_password:
        args.pg_password = getpass.getpass(
            "PostgreSQL password: "
        )

    if not args.neo4j_password or not args.pg_password:
        parser.error("Both passwords are required.")

    output = (
        OUT_ROOT /
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )

    output.mkdir(parents=True, exist_ok=False)

    started = time.perf_counter()

    log("=" * 72)
    log("PHASE 10.8 - PATIENT IDENTITY ROOT-CAUSE INVESTIGATION")
    log("=" * 72)

    driver = GraphDatabase.driver(
        args.neo4j_uri,
        auth=(args.neo4j_user, args.neo4j_password)
    )

    try:
        driver.verify_connectivity()

        log("\n[1/6] Reading Neo4j patient properties")

        graph_records = graph_read(
            driver,
            args.neo4j_database,
            """
            MATCH (p:Patient)
            RETURN properties(p) AS properties
            """
        )

        log(f"Neo4j patients: {len(graph_records):,}")

        log("\n[2/6] Inspecting PostgreSQL patient schema")

        with psycopg.connect(
            host=args.pg_host,
            port=args.pg_port,
            user=args.pg_user,
            password=args.pg_password,
            dbname=args.pg_database,
            autocommit=True,
            options=(
                "-c default_transaction_read_only=on "
                "-c statement_timeout=120000"
            )
        ) as conn:

            pg_columns = get_pg_columns(conn)

            graph_properties, graph_fields, pg_fields = (
                collect_identifier_fields(
                    graph_records, pg_columns
                )
            )

            pg_records = get_pg_patients(
                conn, pg_fields
            )

            foreign_keys = inspect_sql_foreign_keys(conn)

        log(f"PostgreSQL patients: {len(pg_records):,}")

        log("\n[3/6] Profiling identifier formats")

        graph_profiles = {
            field: profile_values(
                extract_graph_ids(graph_records, field)
            )
            for field in graph_fields
        }

        pg_profiles = {
            field: profile_values([
                record[field]
                for record in pg_records
            ])
            for field in pg_fields
        }

        save_json(
            output / "identifier_profiles.json",
            {
                "neo4j": graph_profiles,
                "postgres": pg_profiles
            }
        )

        log("\n[4/6] Testing identifier bridges")

        comparisons, _, _ = identifier_comparisons(
            graph_records,
            pg_records,
            graph_fields,
            pg_fields
        )

        save_csv(
            output / "identifier_bridge_tests.csv",
            comparisons,
            [
                "graph_field",
                "postgres_field",
                "graph_distinct",
                "postgres_distinct",
                "exact_overlap",
                "uuid_normalized_overlap",
                "status"
            ]
        )

        exact_pairs = [
            row for row in comparisons
            if row["exact_overlap"] > 0
        ]

        uuid_pairs = [
            row for row in comparisons
            if row["uuid_normalized_overlap"] > 0
        ]

        log(f"Exact identifier field pairs: {len(exact_pairs)}")
        log(f"UUID-normalized field pairs: {len(uuid_pairs)}")

        log("\n[5/6] Inspecting patient graph connectivity")

        relationships = inspect_relationships(
            driver, args.neo4j_database
        )

        save_json(
            output / "patient_relationship_patterns.json",
            relationships
        )

        log("\n[6/6] Inspecting import provenance clues")

        import_files = inspect_local_import_files()

        save_json(
            output / "import_provenance_candidates.json",
            import_files
        )

        summary = {
            "phase": "10.8_PATIENT_ROOT_CAUSE",
            "graph_patients": len(graph_records),
            "postgres_patients": len(pg_records),
            "graph_properties": sorted(graph_properties),
            "postgres_columns": pg_columns,
            "graph_identifier_fields": graph_fields,
            "postgres_identifier_fields": pg_fields,
            "exact_bridge_field_pairs": exact_pairs,
            "uuid_normalized_bridge_field_pairs": uuid_pairs,
            "relationship_patterns": relationships,
            "postgres_patient_foreign_keys": foreign_keys,
            "local_import_script_candidates": import_files,
            "patient_identity_verified": False,
            "root_cause_status": (
                "IDENTIFIER_CANDIDATES_FOUND_REVIEW_PROVENANCE"
                if exact_pairs or uuid_pairs
                else "NO_IDENTIFIER_BRIDGE_FOUND"
            ),
            "interpretation": (
                "Absence of identifier overlap does not "
                "prove distinct patient populations. "
                "Provenance and original import mapping "
                "must be investigated."
            ),
            "read_only": True,
            "elapsed_seconds": round(
                time.perf_counter() - started, 3
            )
        }

        save_json(
            output / "patient_root_cause_report.json",
            summary
        )

        log("\n" + "=" * 72)
        log("PATIENT ROOT-CAUSE DIAGNOSTIC COMPLETE")
        log("=" * 72)

        print(json.dumps({
            "graph_patients": len(graph_records),
            "postgres_patients": len(pg_records),
            "graph_identifier_fields": graph_fields,
            "postgres_identifier_fields": pg_fields,
            "exact_bridge_field_pairs": len(exact_pairs),
            "uuid_normalized_bridge_field_pairs": len(uuid_pairs),
            "graph_relationship_patterns": len(relationships),
            "import_script_candidates": len(import_files),
            "root_cause_status": summary["root_cause_status"],
            "elapsed_seconds": summary["elapsed_seconds"]
        }, indent=2))

        log("\nResults saved to:")
        log(str(output))

    finally:
        driver.close()


if __name__ == "__main__":
    main()
