
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
OUTPUT_ROOT = ROOT / "results" / "phase10_7_semantic_validation"

MAPPING_SPECS = [
    {
        "domain": "disease",
        "label": "Disease",
        "table": "condition",
        "graph_id": "condition_id",
        "pg_id": "condition_id",
        "code_candidates": ["source_code"],
        "description_candidates": [
            "name", "display", "description",
            "condition_name", "label", "term"
        ],
        "filename": "disease_mapping_validation.csv"
    },
    {
        "domain": "drug",
        "label": "Drug",
        "table": "drug_concept",
        "graph_id": "rxcui",
        "pg_id": "rxcui",
        "code_candidates": [],
        "description_candidates": [
            "name", "display", "description",
            "drug_name", "label", "term"
        ],
        "filename": "drug_mapping_validation.csv"
    },
    {
        "domain": "lab",
        "label": "LabTest",
        "table": "loinc_concept",
        "graph_id": "loinc_code",
        "pg_id": "loinc_code",
        "code_candidates": [],
        "description_candidates": [
            "name", "display", "description",
            "test_name", "label", "term"
        ],
        "filename": "lab_mapping_validation.csv"
    }
]

MAPPING_COLUMNS = [
    "domain", "graph_label", "postgres_table",
    "graph_identifier", "postgres_identifier",
    "graph_id", "postgres_matches",
    "identifier_status", "graph_code",
    "postgres_code", "code_status",
    "graph_description", "postgres_description",
    "description_status", "semantic_status"
]


def log(message=""):
    print(message, flush=True)


def heading(message):
    log("\n" + "=" * 76)
    log(message)
    log("=" * 76)


def cypher_identifier(value):
    return "`" + str(value).replace("`", "``") + "`"


def canonical_id(value):
    if value is None:
        return None

    value = str(value).strip()

    return value if value else None


def normalize_text(value):
    if value is None:
        return ""

    return re.sub(
        r"\s+", " ", str(value)
    ).strip().casefold()


def save_csv(path, records, columns):
    pd.DataFrame(
        records,
        columns=columns
    ).to_csv(path, index=False)


def save_json(path, data):
    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            default=str
        ),
        encoding="utf-8"
    )


def neo_read(driver, database, query, params=None):
    with driver.session(
        database=database,
        default_access_mode=READ_ACCESS
    ) as session:
        result = session.run(query, params or {})

        rows = [
            record.data()
            for record in result
        ]

        result.consume()

        return rows


def postgres_connect(args):
    return psycopg.connect(
        host=args.pg_host,
        port=args.pg_port,
        user=args.pg_user,
        password=args.pg_password,
        dbname=args.pg_database,
        connect_timeout=15,
        autocommit=True,
        options=(
            "-c default_transaction_read_only=on "
            "-c statement_timeout=60000"
        )
    )


def postgres_schema(conn):
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT
                c.table_name,
                c.column_name,
                c.data_type
            FROM information_schema.columns c
            JOIN information_schema.tables t
              ON c.table_schema = t.table_schema
             AND c.table_name = t.table_name
            WHERE c.table_schema = 'public'
              AND t.table_type = 'BASE TABLE'
            ORDER BY
                c.table_name,
                c.ordinal_position
        """)

        result = defaultdict(dict)

        for table, column, data_type in cursor.fetchall():
            result[table][column] = data_type

        return dict(result)


def neo_properties(
    driver,
    database,
    label,
    sample_size
):
    query = (
        f"MATCH (n:{cypher_identifier(label)}) "
        "WITH n LIMIT $sample "
        "UNWIND keys(n) AS property "
        "RETURN DISTINCT property"
    )

    rows = neo_read(
        driver,
        database,
        query,
        {"sample": sample_size}
    )

    return {
        row["property"]
        for row in rows
    }


def choose_field(available, candidates):
    lookup = {
        str(field).lower(): field
        for field in available
    }

    for candidate in candidates:
        match = lookup.get(candidate.lower())

        if match:
            return match

    return None


def neo_entities(
    driver,
    database,
    label,
    fields,
    limit
):
    selections = ", ".join(
        f"n.{cypher_identifier(field)} "
        f"AS {cypher_identifier(field)}"
        for field in fields
    )

    query = (
        f"MATCH (n:{cypher_identifier(label)}) "
        f"RETURN {selections} "
        "LIMIT $limit"
    )

    return neo_read(
        driver,
        database,
        query,
        {"limit": limit}
    )


def postgres_entities(
    conn,
    table,
    fields,
    limit
):
    statement = sql.SQL(
        "SELECT {} FROM {} LIMIT %s"
    ).format(
        sql.SQL(", ").join(
            sql.Identifier(field)
            for field in fields
        ),
        sql.Identifier("public", table)
    )

    with conn.cursor() as cursor:
        cursor.execute(statement, (limit,))
        rows = cursor.fetchall()

    return [
        dict(zip(fields, row))
        for row in rows
    ]


def compare_attribute(graph_value, pg_values):
    if graph_value is None:
        return "NOT_COMPARABLE"

    valid = [
        normalize_text(value)
        for value in pg_values
        if value is not None
    ]

    if not valid:
        return "NOT_COMPARABLE"

    graph_normalized = normalize_text(graph_value)

    if not graph_normalized:
        return "NOT_COMPARABLE"

    if graph_normalized in valid:
        return "MATCH"

    return "MISMATCH"


def validate_concept_mapping(
    driver,
    database,
    conn,
    schema,
    args,
    spec,
    output
):
    domain = spec["domain"]
    label = spec["label"]
    table = spec["table"]

    heading(
        f"10.7 - {domain.upper()} SEMANTIC VALIDATION"
    )

    if table not in schema:
        log(f"Missing PostgreSQL table: {table}")

        save_csv(
            output / spec["filename"],
            [],
            MAPPING_COLUMNS
        )

        return [], {
            "status": "MISSING_POSTGRES_TABLE"
        }

    pg_columns = set(schema[table])

    graph_columns = neo_properties(
        driver,
        database,
        label,
        args.property_sample
    )

    graph_id = spec["graph_id"]
    pg_id = spec["pg_id"]

    if graph_id not in graph_columns:
        log(f"Missing Neo4j property: {graph_id}")

        save_csv(
            output / spec["filename"],
            [],
            MAPPING_COLUMNS
        )

        return [], {
            "status": "MISSING_GRAPH_IDENTIFIER"
        }

    if pg_id not in pg_columns:
        log(f"Missing PostgreSQL column: {pg_id}")

        save_csv(
            output / spec["filename"],
            [],
            MAPPING_COLUMNS
        )

        return [], {
            "status": "MISSING_POSTGRES_IDENTIFIER"
        }

    graph_code = choose_field(
        graph_columns,
        spec["code_candidates"]
    )

    pg_code = choose_field(
        pg_columns,
        spec["code_candidates"]
    )

    graph_description = choose_field(
        graph_columns,
        spec["description_candidates"]
    )

    pg_description = choose_field(
        pg_columns,
        spec["description_candidates"]
    )

    graph_fields = list(dict.fromkeys(
        field
        for field in [
            graph_id,
            graph_code,
            graph_description
        ]
        if field
    ))

    pg_fields = list(dict.fromkeys(
        field
        for field in [
            pg_id,
            pg_code,
            pg_description
        ]
        if field
    ))

    graph_rows = neo_entities(
        driver,
        database,
        label,
        graph_fields,
        args.max_graph_nodes
    )

    pg_rows = postgres_entities(
        conn,
        table,
        pg_fields,
        args.max_pg_rows
    )

    pg_lookup = defaultdict(list)

    for row in pg_rows:
        identifier = canonical_id(row.get(pg_id))

        if identifier is not None:
            pg_lookup[identifier].append(row)

    results = []

    for graph_row in graph_rows:
        identifier = canonical_id(
            graph_row.get(graph_id)
        )

        matches = (
            pg_lookup.get(identifier, [])
            if identifier is not None
            else []
        )

        code_status = "NOT_COMPARABLE"
        description_status = "NOT_COMPARABLE"

        graph_code_value = (
            graph_row.get(graph_code)
            if graph_code else None
        )

        graph_description_value = (
            graph_row.get(graph_description)
            if graph_description else None
        )

        pg_code_value = None
        pg_description_value = None

        if matches and graph_code and pg_code:
            pg_code_values = [
                row.get(pg_code)
                for row in matches
            ]

            pg_code_value = pg_code_values[0]

            code_status = compare_attribute(
                graph_code_value,
                pg_code_values
            )

        if (
            matches
            and graph_description
            and pg_description
        ):
            pg_description_values = [
                row.get(pg_description)
                for row in matches
            ]

            pg_description_value = (
                pg_description_values[0]
            )

            description_status = compare_attribute(
                graph_description_value,
                pg_description_values
            )

        if identifier is None:
            identifier_status = "NULL_IDENTIFIER"

        elif matches:
            identifier_status = "MATCH"

        else:
            identifier_status = "NO_MATCH"

        if not matches:
            semantic_status = "UNMATCHED"

        elif (
            code_status == "MISMATCH"
            or description_status == "MISMATCH"
        ):
            semantic_status = "CONFLICT_REVIEW"

        elif (
            code_status == "MATCH"
            or description_status == "MATCH"
        ):
            semantic_status = "ATTRIBUTE_AGREEMENT"

        else:
            semantic_status = "IDENTIFIER_ONLY"

        results.append({
            "domain": domain,
            "graph_label": label,
            "postgres_table": table,
            "graph_identifier": graph_id,
            "postgres_identifier": pg_id,
            "graph_id": identifier,
            "postgres_matches": len(matches),
            "identifier_status": identifier_status,
            "graph_code": graph_code_value,
            "postgres_code": pg_code_value,
            "code_status": code_status,
            "graph_description": graph_description_value,
            "postgres_description": (
                pg_description_value
            ),
            "description_status": description_status,
            "semantic_status": semantic_status
        })

    save_csv(
        output / spec["filename"],
        results,
        MAPPING_COLUMNS
    )

    counts = Counter(
        row["semantic_status"]
        for row in results
    )

    stats = {
        "status": "COMPLETED",
        "graph_rows_examined": len(graph_rows),
        "postgres_rows_loaded": len(pg_rows),
        "identifier_matches": sum(
            row["identifier_status"] == "MATCH"
            for row in results
        ),
        "attribute_agreements": counts[
            "ATTRIBUTE_AGREEMENT"
        ],
        "identifier_only": counts[
            "IDENTIFIER_ONLY"
        ],
        "conflicts": counts[
            "CONFLICT_REVIEW"
        ],
        "unmatched": counts["UNMATCHED"],
        "graph_code_field": graph_code,
        "postgres_code_field": pg_code,
        "graph_description_field": graph_description,
        "postgres_description_field": pg_description,
        "graph_limit_reached": (
            len(graph_rows) >= args.max_graph_nodes
        ),
        "postgres_limit_reached": (
            len(pg_rows) >= args.max_pg_rows
        )
    }

    log(
        f"Graph rows: {len(graph_rows):,} | "
        f"PostgreSQL rows: {len(pg_rows):,}"
    )

    log(
        f"Identifier matches: "
        f"{stats['identifier_matches']:,}"
    )

    log(
        f"Attribute agreements: "
        f"{stats['attribute_agreements']:,}"
    )

    log(
        f"Conflicts requiring review: "
        f"{stats['conflicts']:,}"
    )

    return results, stats


PATIENT_COLUMNS = [
    "graph_patient_id",
    "postgres_patient_id_match",
    "postgres_synthea_id_match",
    "matched_by",
    "status"
]


def validate_patient_identity(
    driver,
    database,
    conn,
    schema,
    args,
    output
):
    heading("10.7D - PATIENT IDENTITY VALIDATION")

    if "patient" not in schema:
        save_csv(
            output / "patient_identity_validation.csv",
            [],
            PATIENT_COLUMNS
        )

        return {
            "status": "MISSING_POSTGRES_PATIENT_TABLE"
        }

    graph_columns = neo_properties(
        driver,
        database,
        "Patient",
        args.property_sample
    )

    graph_id = choose_field(
        graph_columns,
        ["patient_id", "synthea_id", "id"]
    )

    pg_columns = schema["patient"]

    pg_fields = [
        field
        for field in ["patient_id", "synthea_id"]
        if field in pg_columns
    ]

    if not graph_id or not pg_fields:
        save_csv(
            output / "patient_identity_validation.csv",
            [],
            PATIENT_COLUMNS
        )

        return {
            "status": "MISSING_PATIENT_IDENTIFIER"
        }

    graph_rows = neo_entities(
        driver,
        database,
        "Patient",
        [graph_id],
        args.max_graph_nodes
    )

    pg_rows = postgres_entities(
        conn,
        "patient",
        pg_fields,
        args.max_pg_rows
    )

    pg_identifier_sets = {}

    for field in pg_fields:
        pg_identifier_sets[field] = {
            canonical_id(row[field])
            for row in pg_rows
            if canonical_id(row[field]) is not None
        }

    results = []

    for graph_row in graph_rows:
        identifier = canonical_id(
            graph_row.get(graph_id)
        )

        patient_id_match = (
            identifier is not None
            and identifier in pg_identifier_sets.get(
                "patient_id", set()
            )
        )

        synthea_id_match = (
            identifier is not None
            and identifier in pg_identifier_sets.get(
                "synthea_id", set()
            )
        )

        matched_by = []

        if patient_id_match:
            matched_by.append("patient_id")

        if synthea_id_match:
            matched_by.append("synthea_id")

        results.append({
            "graph_patient_id": identifier,
            "postgres_patient_id_match": patient_id_match,
            "postgres_synthea_id_match": synthea_id_match,
            "matched_by": ",".join(matched_by),
            "status": (
                "IDENTIFIER_MATCH"
                if matched_by
                else "NO_MATCH"
            )
        })

    save_csv(
        output / "patient_identity_validation.csv",
        results,
        PATIENT_COLUMNS
    )

    matching_count = sum(
        row["status"] == "IDENTIFIER_MATCH"
        for row in results
    )

    stats = {
        "status": "COMPLETED",
        "graph_identifier": graph_id,
        "graph_patients_examined": len(graph_rows),
        "postgres_patients_loaded": len(pg_rows),
        "matching_graph_patients": matching_count,
        "graph_limit_reached": (
            len(graph_rows) >= args.max_graph_nodes
        ),
        "postgres_limit_reached": (
            len(pg_rows) >= args.max_pg_rows
        ),
        "patient_identity_semantically_verified": False
    }

    log(
        f"Matching graph patient identifiers: "
        f"{matching_count:,}/{len(graph_rows):,}"
    )

    return stats


RELATIONSHIP_COLUMNS = [
    "source_labels",
    "relationship_type",
    "target_labels",
    "graph_relationship_count",
    "validation_status",
    "notes"
]


def validate_relationship_inventory(
    driver,
    database,
    output
):
    heading("10.7E - CLINICAL RELATIONSHIP AUDIT")

    rows = neo_read(
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

    results = []

    for row in rows:
        results.append({
            "source_labels": json.dumps(
                row["source_labels"]
            ),
            "relationship_type": row[
                "relationship_type"
            ],
            "target_labels": json.dumps(
                row["target_labels"]
            ),
            "graph_relationship_count": row[
                "relationship_count"
            ],
            "validation_status": "NOT_VERIFIED",
            "notes": (
                "Independent SQL relationship validation "
                "requires an explicit entity/event mapping."
            )
        })

    save_csv(
        output / "relationship_validation.csv",
        results,
        RELATIONSHIP_COLUMNS
    )

    log(
        f"Relationship patterns inspected: "
        f"{len(results)}"
    )

    return results


REGISTRY_COLUMNS = [
    "mapping_id",
    "domain",
    "graph_label",
    "graph_identifier",
    "postgres_table",
    "postgres_identifier",
    "identifier_matches",
    "attribute_agreements",
    "conflicts",
    "status"
]

QUERY_PAIR_COLUMNS = [
    "pair_id",
    "domain",
    "cypher",
    "sql",
    "parameters_json",
    "verification_status",
    "notes"
]


def create_mapping_registry(
    domain_results,
    domain_stats,
    output
):
    heading("10.7F - MAPPING REGISTRY")

    registry = []
    query_pairs = []

    for spec in MAPPING_SPECS:
        domain = spec["domain"]
        results = domain_results.get(domain, [])
        stats = domain_stats.get(domain, {})

        matches = stats.get(
            "identifier_matches", 0
        )

        agreements = stats.get(
            "attribute_agreements", 0
        )

        conflicts = stats.get(
            "conflicts", 0
        )

        if stats.get("status") != "COMPLETED":
            status = "NOT_TESTED"

        elif agreements > 0:
            status = "ATTRIBUTE_AGREEMENT_OBSERVED"

        elif matches > 0:
            status = "IDENTIFIER_OVERLAP_ONLY"

        else:
            status = "NO_IDENTIFIER_MATCH"

        registry.append({
            "mapping_id": (
                f"MAP{len(registry) + 1:03d}"
            ),
            "domain": domain,
            "graph_label": spec["label"],
            "graph_identifier": spec["graph_id"],
            "postgres_table": spec["table"],
            "postgres_identifier": spec["pg_id"],
            "identifier_matches": matches,
            "attribute_agreements": agreements,
            "conflicts": conflicts,
            "status": status
        })

        matched_record = next(
            (
                row
                for row in results
                if row["identifier_status"] == "MATCH"
            ),
            None
        )

        if matched_record is None:
            continue

        graph_label = cypher_identifier(
            spec["label"]
        )

        graph_property = cypher_identifier(
            spec["graph_id"]
        )

        cypher_query = (
            f"MATCH (n:{graph_label}) "
            f"WHERE toString(n.{graph_property}) "
            "= $identifier "
            f"RETURN toString(n.{graph_property}) "
            "AS identifier"
        )

        sql_query = sql.SQL("""
            SELECT {column}::text AS identifier
            FROM {table}
            WHERE {column}::text = %(identifier)s
        """).format(
            column=sql.Identifier(
                spec["pg_id"]
            ),
            table=sql.Identifier(
                "public",
                spec["table"]
            )
        ).as_string()

        query_pairs.append({
            "pair_id": (
                f"PAIR{len(query_pairs) + 1:03d}"
            ),
            "domain": domain,
            "cypher": cypher_query,
            "sql": sql_query,
            "parameters_json": json.dumps({
                "identifier": matched_record["graph_id"]
            }),
            "verification_status": (
                "IDENTIFIER_EXISTENCE_TESTED"
            ),
            "notes": (
                "Identifier lookup candidate only. "
                "Clinical-result equivalence not established."
            )
        })

    save_csv(
        output / "mapping_registry.csv",
        registry,
        REGISTRY_COLUMNS
    )

    save_csv(
        output / "verified_query_pairs.csv",
        query_pairs,
        QUERY_PAIR_COLUMNS
    )

    log(f"Registry entries: {len(registry)}")
    log(f"Identifier lookup pairs: {len(query_pairs)}")

    return registry, query_pairs


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Phase 10.7 - Read-only semantic "
            "mapping verification"
        )
    )

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

    parser.add_argument(
        "--property-sample",
        type=int,
        default=1000
    )

    parser.add_argument(
        "--max-graph-nodes",
        type=int,
        default=600000
    )

    parser.add_argument(
        "--max-pg-rows",
        type=int,
        default=200000
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

    if not args.neo4j_password:
        parser.error("Neo4j password required.")

    if not args.pg_password:
        parser.error("PostgreSQL password required.")

    if min(
        args.property_sample,
        args.max_graph_nodes,
        args.max_pg_rows
    ) < 1:
        parser.error(
            "Sample and row limits must be positive."
        )

    heading(
        "PHASE 10.7 - SEMANTIC MAPPING VERIFICATION"
    )

    output = (
        OUTPUT_ROOT
        / datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )
    )

    output.mkdir(
        parents=True,
        exist_ok=False
    )

    started = time.perf_counter()

    driver = GraphDatabase.driver(
        args.neo4j_uri,
        auth=(
            args.neo4j_user,
            args.neo4j_password
        )
    )

    try:
        driver.verify_connectivity()

        with postgres_connect(args) as conn:
            schema = postgres_schema(conn)

            domain_results = {}
            domain_stats = {}

            for spec in MAPPING_SPECS:
                records, stats = validate_concept_mapping(
                    driver,
                    args.neo4j_database,
                    conn,
                    schema,
                    args,
                    spec,
                    output
                )

                domain_results[
                    spec["domain"]
                ] = records

                domain_stats[
                    spec["domain"]
                ] = stats

            patient_stats = validate_patient_identity(
                driver,
                args.neo4j_database,
                conn,
                schema,
                args,
                output
            )

            relationship_results = (
                validate_relationship_inventory(
                    driver,
                    args.neo4j_database,
                    output
                )
            )

            registry, query_pairs = (
                create_mapping_registry(
                    domain_results,
                    domain_stats,
                    output
                )
            )

            report = {
                "phase": "10.7",
                "completed_at": (
                    datetime.now().isoformat()
                ),
                "neo4j_database": args.neo4j_database,
                "postgres_database": args.pg_database,
                "domain_validation": domain_stats,
                "patient_validation": patient_stats,
                "relationship_patterns": len(
                    relationship_results
                ),
                "semantically_verified_relationships": 0,
                "mapping_registry_entries": len(
                    registry
                ),
                "identifier_query_pairs": len(
                    query_pairs
                ),
                "clinically_equivalent_query_pairs": 0,
                "phase11_readiness": (
                    "PARTIAL - identifier lookup "
                    "comparisons can be prepared; "
                    "patient-level and clinical "
                    "relationship equivalence "
                    "remain unverified."
                ),
                "read_only": True,
                "duration_seconds": round(
                    time.perf_counter() - started,
                    3
                ),
                "limitations": [
                    "Exact identifier matches do not prove "
                    "clinical semantic equivalence.",
                    "Description comparison uses normalized "
                    "exact text, not ontology synonym matching.",
                    "Patient identity requires source "
                    "provenance confirmation.",
                    "Clinical graph relationships have not "
                    "been independently verified against SQL.",
                    "Queries may be truncated by configured "
                    "graph and PostgreSQL row limits.",
                    "No databases are modified."
                ]
            }

            save_json(
                output / "phase10_7_report.json",
                report
            )

            heading("PHASE 10.7 COMPLETE")

            log(
                json.dumps(
                    report,
                    indent=2
                )
            )

            log("\nResults saved to:")
            log(str(output))

    finally:
        driver.close()


if __name__ == "__main__":
    main()
