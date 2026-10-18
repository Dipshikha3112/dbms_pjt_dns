
from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import psycopg
from psycopg import sql
from neo4j import GraphDatabase, READ_ACCESS


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = ROOT / "results" / "phase10_8_resolution"


def log(message):
    print(message, flush=True)


def normalize(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def identifier(value):
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def cyname(value):
    return "`" + str(value).replace("`", "``") + "`"


def save_csv(path, rows, columns):
    pd.DataFrame(rows, columns=columns).to_csv(
        path, index=False
    )


def save_json(path, data):
    path.write_text(
        json.dumps(data, indent=2, default=str),
        encoding="utf-8"
    )


def neo_read(driver, database, query, params=None):
    with driver.session(
        database=database,
        default_access_mode=READ_ACCESS
    ) as session:
        result = session.run(query, params or {})
        rows = [record.data() for record in result]
        result.consume()
        return rows


def pg_schema(conn):
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT c.table_name, c.column_name
            FROM information_schema.columns c
            JOIN information_schema.tables t
              ON c.table_schema = t.table_schema
             AND c.table_name = t.table_name
            WHERE c.table_schema = 'public'
              AND t.table_type = 'BASE TABLE'
        """)

        result = defaultdict(set)

        for table, column in cursor.fetchall():
            result[table].add(column)

        return dict(result)


def graph_properties(driver, database, label):
    query = (
        f"MATCH (n:{cyname(label)}) "
        "UNWIND keys(n) AS property "
        "RETURN DISTINCT property"
    )

    return {
        row["property"]
        for row in neo_read(driver, database, query)
    }


def choose_field(available, candidates):
    lookup = {
        field.lower(): field
        for field in available
    }

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def pg_fetch(conn, table, columns):
    statement = sql.SQL(
        "SELECT {} FROM {}"
    ).format(
        sql.SQL(", ").join(
            sql.Identifier(column)
            for column in columns
        ),
        sql.Identifier("public", table)
    )

    with conn.cursor() as cursor:
        cursor.execute(statement)
        return [
            dict(zip(columns, row))
            for row in cursor.fetchall()
        ]


def graph_fetch(driver, database, label, columns):
    expressions = ", ".join(
        f"n.{cyname(column)} AS {cyname(column)}"
        for column in columns
    )

    query = (
        f"MATCH (n:{cyname(label)}) "
        f"RETURN {expressions}"
    )

    return neo_read(driver, database, query)


def compare_values(graph_value, postgres_value):
    if graph_value is None or postgres_value is None:
        return "NOT_COMPARABLE"

    if normalize(graph_value) == normalize(postgres_value):
        return "MATCH"

    return "MISMATCH"


def resolution_status(code_status, description_status):
    if code_status == "MATCH" and description_status == "MATCH":
        return "BOTH_MATCH"

    if code_status == "MATCH" and description_status == "MISMATCH":
        return "CODE_MATCH_DESCRIPTION_DIFFERENT"

    if code_status == "MISMATCH" and description_status == "MATCH":
        return "DESCRIPTION_MATCH_CODE_DIFFERENT"

    if code_status == "MISMATCH" and description_status == "MISMATCH":
        return "GENUINE_ATTRIBUTE_CONFLICT"

    if "MATCH" in (code_status, description_status):
        return "ATTRIBUTE_AGREEMENT"

    if "MISMATCH" in (code_status, description_status):
        return "ATTRIBUTE_CONFLICT"

    return "IDENTIFIER_ONLY"


def validate_concepts(
    driver,
    database,
    conn,
    schema,
    label,
    table,
    key,
    description_candidates,
    filename,
    output,
    compare_source_code=False
):
    log(f"\nValidating {label} -> {table}")

    columns = [
        "graph_id",
        "postgres_match_count",
        "graph_code",
        "postgres_code",
        "graph_description",
        "postgres_description",
        "code_status",
        "description_status",
        "resolution_status"
    ]

    if table not in schema:
        save_csv(output / filename, [], columns)
        return {"status": "MISSING_TABLE"}

    graph_fields = graph_properties(
        driver, database, label
    )

    pg_fields = schema[table]

    if key not in graph_fields or key not in pg_fields:
        save_csv(output / filename, [], columns)
        return {"status": "MISSING_IDENTIFIER"}

    graph_code = (
        "source_code"
        if compare_source_code and "source_code" in graph_fields
        else None
    )

    postgres_code = (
        "source_code"
        if compare_source_code and "source_code" in pg_fields
        else None
    )

    graph_description = choose_field(
        graph_fields, description_candidates
    )

    postgres_description = choose_field(
        pg_fields, description_candidates
    )

    selected_graph = list(dict.fromkeys(
        field for field in [
            key, graph_code, graph_description
        ] if field
    ))

    selected_pg = list(dict.fromkeys(
        field for field in [
            key, postgres_code, postgres_description
        ] if field
    ))

    graph_rows = graph_fetch(
        driver, database, label, selected_graph
    )

    postgres_rows = pg_fetch(
        conn, table, selected_pg
    )

    postgres_lookup = defaultdict(list)

    for row in postgres_rows:
        value = identifier(row[key])
        if value is not None:
            postgres_lookup[value].append(row)

    results = []

    priority = {
        "BOTH_MATCH": 0,
        "CODE_MATCH_DESCRIPTION_DIFFERENT": 1,
        "DESCRIPTION_MATCH_CODE_DIFFERENT": 2,
        "ATTRIBUTE_AGREEMENT": 3,
        "IDENTIFIER_ONLY": 4,
        "ATTRIBUTE_CONFLICT": 5,
        "GENUINE_ATTRIBUTE_CONFLICT": 6
    }

    for graph_row in graph_rows:
        graph_id = identifier(graph_row[key])
        matches = postgres_lookup.get(graph_id, [])

        graph_code_value = (
            graph_row.get(graph_code)
            if graph_code else None
        )

        graph_description_value = (
            graph_row.get(graph_description)
            if graph_description else None
        )

        if not matches:
            results.append({
                "graph_id": graph_id,
                "postgres_match_count": 0,
                "graph_code": graph_code_value,
                "postgres_code": None,
                "graph_description": graph_description_value,
                "postgres_description": None,
                "code_status": "NOT_COMPARABLE",
                "description_status": "NOT_COMPARABLE",
                "resolution_status": "UNMATCHED"
            })
            continue

        candidates = []

        for postgres_row in matches:
            pg_code_value = (
                postgres_row.get(postgres_code)
                if postgres_code else None
            )

            pg_description_value = (
                postgres_row.get(postgres_description)
                if postgres_description else None
            )

            code_status = compare_values(
                graph_code_value,
                pg_code_value
            )

            description_status = compare_values(
                graph_description_value,
                pg_description_value
            )

            status = resolution_status(
                code_status, description_status
            )

            candidates.append({
                "graph_id": graph_id,
                "postgres_match_count": len(matches),
                "graph_code": graph_code_value,
                "postgres_code": pg_code_value,
                "graph_description": graph_description_value,
                "postgres_description": pg_description_value,
                "code_status": code_status,
                "description_status": description_status,
                "resolution_status": status
            })

        best = min(
            candidates,
            key=lambda row: priority.get(
                row["resolution_status"], 99
            )
        )

        results.append(best)

    save_csv(output / filename, results, columns)

    counts = Counter(
        row["resolution_status"]
        for row in results
    )

    report = {
        "status": "COMPLETED",
        "graph_rows": len(graph_rows),
        "postgres_rows": len(postgres_rows),
        "matched_identifiers": sum(
            row["postgres_match_count"] > 0
            for row in results
        ),
        "graph_code_field": graph_code,
        "postgres_code_field": postgres_code,
        "graph_description_field": graph_description,
        "postgres_description_field": postgres_description,
        "available_graph_fields": sorted(graph_fields),
        "available_postgres_fields": sorted(pg_fields),
        "resolution_counts": dict(counts)
    }

    log(json.dumps(
        report["resolution_counts"], indent=2
    ))

    return report


def validate_patients(
    driver, database, conn, schema, output
):
    log("\nValidating patient identity")

    graph_fields = graph_properties(
        driver, database, "Patient"
    )

    postgres_fields = schema.get("patient", set())

    graph_candidates = sorted(
        field for field in graph_fields
        if re.search(r"id|uuid|synthea|source", field, re.I)
    )

    postgres_candidates = sorted(
        field for field in postgres_fields
        if field in {
            "patient_id", "synthea_id",
            "source_id", "uuid", "id"
        }
    )

    graph_rows = neo_read(
        driver,
        database,
        "MATCH (n:Patient) RETURN properties(n) AS properties"
    )

    postgres_rows = (
        pg_fetch(conn, "patient", postgres_candidates)
        if postgres_candidates else []
    )

    pg_sets = {
        field: {
            identifier(row[field])
            for row in postgres_rows
            if identifier(row[field]) is not None
        }
        for field in postgres_candidates
    }

    comparisons = []

    for graph_field in graph_candidates:
        graph_values = {
            identifier(row["properties"].get(graph_field))
            for row in graph_rows
        }

        graph_values.discard(None)

        for postgres_field in postgres_candidates:
            overlap = graph_values & pg_sets[postgres_field]

            comparisons.append({
                "graph_field": graph_field,
                "postgres_field": postgres_field,
                "graph_distinct_values": len(graph_values),
                "postgres_distinct_values": len(
                    pg_sets[postgres_field]
                ),
                "exact_overlap": len(overlap),
                "matching_examples": json.dumps(
                    sorted(overlap)[:10]
                )
            })

    save_csv(
        output / "patient_identifier_pairs.csv",
        comparisons,
        [
            "graph_field", "postgres_field",
            "graph_distinct_values",
            "postgres_distinct_values",
            "exact_overlap", "matching_examples"
        ]
    )

    report = {
        "graph_patients": len(graph_rows),
        "postgres_patients": len(postgres_rows),
        "graph_properties": sorted(graph_fields),
        "postgres_columns": sorted(postgres_fields),
        "candidate_identifier_comparisons": comparisons,
        "matching_field_pairs": sum(
            row["exact_overlap"] > 0
            for row in comparisons
        ),
        "patient_identity_verified": False,
        "conclusion": (
            "Exact identifier overlap is only candidate "
            "evidence. A patient identity bridge requires "
            "verified source provenance."
        )
    }

    save_json(
        output / "patient_identity_diagnostics.json",
        report
    )

    log(
        f"Patient identifier pairs with overlap: "
        f"{report['matching_field_pairs']}"
    )

    return report


def verify_relationships(driver, database, output):
    log("\nAuditing clinical relationship verification")

    patterns = neo_read(
        driver,
        database,
        """
        MATCH (a)-[r]->(b)
        RETURN
            labels(a) AS source_labels,
            type(r) AS relationship_type,
            labels(b) AS target_labels,
            count(*) AS relationship_count
        ORDER BY relationship_count DESC
        """
    )

    rows = []

    for pattern in patterns:
        rows.append({
            "source_labels": json.dumps(
                pattern["source_labels"]
            ),
            "relationship_type": pattern["relationship_type"],
            "target_labels": json.dumps(
                pattern["target_labels"]
            ),
            "graph_relationship_count": pattern[
                "relationship_count"
            ],
            "sql_relationship_verified": False,
            "status": "NOT_VERIFIED",
            "reason": (
                "Requires a confirmed patient/entity bridge "
                "and independently matching clinical events."
            )
        })

    save_csv(
        output / "relationship_verification.csv",
        rows,
        [
            "source_labels", "relationship_type",
            "target_labels", "graph_relationship_count",
            "sql_relationship_verified", "status", "reason"
        ]
    )

    return {
        "patterns_inspected": len(rows),
        "independently_verified_relationships": 0,
        "status": "BLOCKED_PENDING_EVENT_MAPPING"
    }


def validate_identifier_sets(
    driver, database, conn, schema, output
):
    log("\nComparing complete identifier result sets")

    specifications = [
        ("disease", "Disease", "condition", "condition_id"),
        ("drug", "Drug", "drug_concept", "rxcui"),
        ("lab", "LabTest", "loinc_concept", "loinc_code")
    ]

    rows = []

    for domain, label, table, key in specifications:
        if table not in schema or key not in schema[table]:
            continue

        if key not in graph_properties(driver, database, label):
            continue

        graph_rows = neo_read(
            driver,
            database,
            (
                f"MATCH (n:{cyname(label)}) "
                f"WHERE n.{cyname(key)} IS NOT NULL "
                f"RETURN DISTINCT toString(n.{cyname(key)}) "
                "AS identifier"
            )
        )

        postgres_rows = pg_fetch(
            conn, table, [key]
        )

        graph_ids = {
            identifier(row["identifier"])
            for row in graph_rows
            if identifier(row["identifier"]) is not None
        }

        postgres_ids = {
            identifier(row[key])
            for row in postgres_rows
            if identifier(row[key]) is not None
        }

        common = graph_ids & postgres_ids

        rows.append({
            "domain": domain,
            "graph_label": label,
            "postgres_table": table,
            "identifier": key,
            "graph_distinct": len(graph_ids),
            "postgres_distinct": len(postgres_ids),
            "shared_identifiers": len(common),
            "graph_only": len(graph_ids - postgres_ids),
            "postgres_only": len(postgres_ids - graph_ids),
            "identifier_sets_equal": graph_ids == postgres_ids,
            "clinical_equivalence": "NOT_ESTABLISHED"
        })

        log(
            f"{domain}: shared={len(common):,}, "
            f"graph_only={len(graph_ids - postgres_ids):,}, "
            f"postgres_only={len(postgres_ids - graph_ids):,}"
        )

    save_csv(
        output / "query_equivalence.csv",
        rows,
        [
            "domain", "graph_label", "postgres_table",
            "identifier", "graph_distinct",
            "postgres_distinct", "shared_identifiers",
            "graph_only", "postgres_only",
            "identifier_sets_equal", "clinical_equivalence"
        ]
    )

    return rows


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
        parser.error("Both database passwords are required.")

    output = (
        OUTPUT_ROOT
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )

    output.mkdir(parents=True, exist_ok=False)

    started = time.perf_counter()

    log("=" * 72)
    log("PHASE 10.8 - CLINICAL MAPPING RESOLUTION")
    log("=" * 72)

    driver = GraphDatabase.driver(
        args.neo4j_uri,
        auth=(args.neo4j_user, args.neo4j_password)
    )

    try:
        driver.verify_connectivity()

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

            schema = pg_schema(conn)

            log("\n10.8A - Disease resolution")

            disease = validate_concepts(
                driver,
                args.neo4j_database,
                conn,
                schema,
                "Disease",
                "condition",
                "condition_id",
                [
                    "name", "description", "display",
                    "condition_name", "label", "term"
                ],
                "disease_resolution.csv",
                output,
                compare_source_code=True
            )

            log("\n10.8B - Lab resolution")

            lab = validate_concepts(
                driver,
                args.neo4j_database,
                conn,
                schema,
                "LabTest",
                "loinc_concept",
                "loinc_code",
                [
                    "name", "description",
                    "long_common_name", "long_name",
                    "short_name", "component",
                    "display", "test_name", "label"
                ],
                "lab_resolution.csv",
                output
            )

            log("\n10.8C - Patient diagnostics")

            patient = validate_patients(
                driver,
                args.neo4j_database,
                conn,
                schema,
                output
            )

            log("\n10.8D - Relationship validation")

            relationships = verify_relationships(
                driver,
                args.neo4j_database,
                output
            )

            log("\n10.8E - Identifier equivalence")

            query_results = validate_identifier_sets(
                driver,
                args.neo4j_database,
                conn,
                schema,
                output
            )

            log("\n10.8F - Mapping registry")

            registry = []

            for domain, result in [
                ("disease", disease),
                ("lab", lab)
            ]:
                counts = result.get(
                    "resolution_counts", {}
                )

                registry.append({
                    "domain": domain,
                    "identifier_matches": result.get(
                        "matched_identifiers", 0
                    ),
                    "attribute_agreements": sum(
                        counts.get(status, 0)
                        for status in [
                            "BOTH_MATCH",
                            "ATTRIBUTE_AGREEMENT",
                            "CODE_MATCH_DESCRIPTION_DIFFERENT",
                            "DESCRIPTION_MATCH_CODE_DIFFERENT"
                        ]
                    ),
                    "attribute_conflicts": sum(
                        counts.get(status, 0)
                        for status in [
                            "GENUINE_ATTRIBUTE_CONFLICT",
                            "ATTRIBUTE_CONFLICT"
                        ]
                    ),
                    "status": (
                        "REVIEW_REQUIRED"
                        if counts.get("GENUINE_ATTRIBUTE_CONFLICT", 0)
                        or counts.get("ATTRIBUTE_CONFLICT", 0)
                        else "IDENTIFIER_VALIDATED_WITH_LIMITATIONS"
                    )
                })

            drug_result = next(
                (
                    row for row in query_results
                    if row["domain"] == "drug"
                ),
                None
            )

            registry.append({
                "domain": "drug",
                "identifier_matches": (
                    drug_result["shared_identifiers"]
                    if drug_result else 0
                ),
                "attribute_agreements": None,
                "attribute_conflicts": None,
                "status": (
                    "IDENTIFIER_VALIDATED; "
                    "NAME_AGREEMENT_REPORTED_IN_PHASE_10_7"
                )
            })

            registry.append({
                "domain": "patient",
                "identifier_matches": 0,
                "attribute_agreements": None,
                "attribute_conflicts": None,
                "status": "PATIENT_IDENTITY_NOT_VERIFIED"
            })

            save_csv(
                output / "mapping_registry.csv",
                registry,
                [
                    "domain", "identifier_matches",
                    "attribute_agreements",
                    "attribute_conflicts", "status"
                ]
            )

            report = {
                "phase": "10.8",
                "completed_at": datetime.now().isoformat(),
                "disease": disease,
                "lab": lab,
                "patient": {
                    "graph_patients": patient["graph_patients"],
                    "postgres_patients": patient["postgres_patients"],
                    "matching_identifier_field_pairs": patient[
                        "matching_field_pairs"
                    ],
                    "identity_verified": False
                },
                "relationships": relationships,
                "identifier_equivalence": query_results,
                "clinically_equivalent_queries_verified": 0,
                "read_only": True,
                "duration_seconds": round(
                    time.perf_counter() - started, 3
                ),
                "phase11_readiness": (
                    "Concept-level benchmarks can proceed "
                    "with validated identifiers. Patient-level "
                    "hybrid queries require a verified identity "
                    "bridge and relationship evidence."
                )
            }

            save_json(
                output / "phase10_8_report.json",
                report
            )

            log("\n" + "=" * 72)
            log("PHASE 10.8 COMPLETE")
            log("=" * 72)

            log(json.dumps({
                "disease": disease.get(
                    "resolution_counts", {}
                ),
                "lab": lab.get(
                    "resolution_counts", {}
                ),
                "patient_identifier_pairs_with_overlap": patient[
                    "matching_field_pairs"
                ],
                "identifier_equivalence": [
                    {
                        "domain": row["domain"],
                        "shared": row["shared_identifiers"],
                        "sets_equal": row["identifier_sets_equal"]
                    }
                    for row in query_results
                ],
                "duration_seconds": report[
                    "duration_seconds"
                ]
            }, indent=2))

            log("\nResults saved to:")
            log(str(output))

    finally:
        driver.close()


if __name__ == "__main__":
    main()
