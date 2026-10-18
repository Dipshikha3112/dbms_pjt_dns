
import argparse
import csv
import getpass
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import psycopg
from neo4j import GraphDatabase, READ_ACCESS


ROOT = Path(__file__).resolve().parent.parent


def normalize_id(value):
    return str(value).strip() if value is not None else ""


def normalize_name(value):
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def save_csv(path, rows, columns):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=columns, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def get_pg_drugs(conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT rxcui::text, name::text
            FROM public.drug_concept
        """)
        return cur.fetchall()


def get_graph_drugs(session):
    result = session.run("""
        MATCH (d:Drug)
        RETURN
            d.rxcui AS rxcui,
            d.name AS name,
            properties(d) AS properties,
            elementId(d) AS element_id
    """)
    return [dict(row) for row in result]


def get_graph_relationship_evidence(session, identifiers):
    evidence = defaultdict(
        lambda: {
            "relationships": Counter(),
            "neighbor_labels": Counter(),
            "patient_ids": set(),
            "relationship_count": 0,
        }
    )

    query = """
    UNWIND $identifiers AS identifier
    MATCH (d:Drug {rxcui: identifier})
    OPTIONAL MATCH (d)-[r]-(neighbor)
    RETURN
        identifier,
        type(r) AS relationship_type,
        CASE
            WHEN r IS NULL THEN null
            WHEN startNode(r) = d THEN 'OUT'
            ELSE 'IN'
        END AS direction,
        labels(neighbor) AS neighbor_labels,
        CASE
            WHEN 'Patient' IN labels(neighbor)
            THEN neighbor.patient_id
            ELSE null
        END AS patient_id
    """

    for start in range(0, len(identifiers), 200):
        batch = identifiers[start:start + 200]

        for record in session.run(query, identifiers=batch):
            code = normalize_id(record["identifier"])
            relationship = record["relationship_type"]

            if relationship is None:
                continue

            item = evidence[code]
            item["relationship_count"] += 1

            key = f"{record['direction']}:{relationship}"
            item["relationships"][key] += 1

            for label in record["neighbor_labels"] or []:
                item["neighbor_labels"][label] += 1

            if record["patient_id"] is not None:
                item["patient_ids"].add(
                    normalize_id(record["patient_id"])
                )

    return evidence


def get_prescription_codes(conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT pi.rxcui::text
            FROM public.prescription p
            JOIN public.prescription_item pi
                ON p.prescription_id = pi.prescription_id
            WHERE pi.rxcui IS NOT NULL
        """)
        return {
            normalize_id(row[0])
            for row in cur.fetchall()
        }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pg-original",
        default="adaptive_healthcare_db"
    )
    parser.add_argument("--pg-host", default="127.0.0.1")
    parser.add_argument("--pg-port", type=int, default=5432)
    parser.add_argument("--pg-user", default="postgres")

    parser.add_argument(
        "--neo4j-uri",
        default="bolt://127.0.0.1:7687"
    )
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-database", default="neo4j")

    parser.add_argument(
        "--ask-pg-password",
        action="store_true"
    )
    parser.add_argument(
        "--ask-neo4j-password",
        action="store_true"
    )

    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    output = (
        ROOT
        / "results"
        / "phase10_9_drug_provenance"
        / stamp
    )
    output.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                output / "audit.log",
                encoding="utf-8"
            ),
        ],
    )

    pg_password = (
        getpass.getpass("PostgreSQL password: ")
        if args.ask_pg_password
        else None
    )

    neo_password = (
        getpass.getpass("Neo4j password: ")
        if args.ask_neo4j_password
        else None
    )

    summary = {
        "phase": "10.9",
        "status": "STARTED",
        "postgres_database": args.pg_original,
        "neo4j_database": args.neo4j_database,
        "database_modifications": "NONE",
    }

    try:
        logging.info("Connecting to PostgreSQL")

        with psycopg.connect(
            host=args.pg_host,
            port=args.pg_port,
            dbname=args.pg_original,
            user=args.pg_user,
            password=pg_password,
            options=(
                "-c default_transaction_read_only=on "
                "-c statement_timeout=120000"
            ),
        ) as conn:

            logging.info("Connecting to Neo4j")

            with GraphDatabase.driver(
                args.neo4j_uri,
                auth=(args.neo4j_user, neo_password),
            ) as driver:

                driver.verify_connectivity()

                with driver.session(
                    database=args.neo4j_database,
                    default_access_mode=READ_ACCESS,
                ) as session:

                    logging.info(
                        "Loading PostgreSQL drug concepts"
                    )
                    pg_rows = get_pg_drugs(conn)

                    pg_map = defaultdict(set)
                    pg_name_to_codes = defaultdict(set)

                    for code, name in pg_rows:
                        code = normalize_id(code)
                        name = normalize_name(name)

                        pg_map[code].add(name)

                        if name:
                            pg_name_to_codes[name].add(code)

                    logging.info(
                        "Loading Neo4j Drug nodes"
                    )
                    graph_rows = get_graph_drugs(session)

                    graph_map = defaultdict(list)

                    for row in graph_rows:
                        code = normalize_id(row["rxcui"])
                        graph_map[code].append(row)

                    graph_ids = set(graph_map)
                    postgres_ids = set(pg_map)

                    graph_only_ids = sorted(
                        graph_ids - postgres_ids
                    )
                    postgres_only_ids = sorted(
                        postgres_ids - graph_ids
                    )
                    shared_ids = sorted(
                        graph_ids & postgres_ids
                    )

                    logging.info(
                        "Graph-only drug identifiers: %d",
                        len(graph_only_ids),
                    )
                    logging.info(
                        "Shared drug identifiers: %d",
                        len(shared_ids),
                    )
                    logging.info(
                        "PostgreSQL-only identifiers: %d",
                        len(postgres_only_ids),
                    )

                    logging.info(
                        "Inspecting graph-only relationships"
                    )
                    evidence = get_graph_relationship_evidence(
                        session,
                        graph_only_ids,
                    )

                    logging.info(
                        "Checking prescription usage"
                    )
                    prescribed_codes = get_prescription_codes(
                        conn
                    )

                    graph_only_rows = []
                    classifications = Counter()
                    property_counts = Counter()
                    relationship_counts = Counter()

                    for code in graph_only_ids:
                        for node in graph_map[code]:
                            name = str(node["name"] or "")
                            normalized_name = normalize_name(name)

                            props = node["properties"] or {}

                            ev = evidence[code]

                            matching_name_codes = sorted(
                                pg_name_to_codes.get(
                                    normalized_name, set()
                                )
                            )

                            has_patient = bool(ev["patient_ids"])
                            used_in_prescriptions = (
                                code in prescribed_codes
                            )

                            if has_patient or used_in_prescriptions:
                                category = (
                                    "CLINICAL_USAGE_REVIEW"
                                )
                            elif matching_name_codes:
                                category = (
                                    "NAME_MATCH_DIFFERENT_ID_REVIEW"
                                )
                            elif ev["relationship_count"] > 0:
                                category = (
                                    "CONNECTED_SUPPLEMENTARY_REVIEW"
                                )
                            else:
                                category = (
                                    "ISOLATED_CONCEPT_REVIEW"
                                )

                            classifications[category] += 1

                            for key in props:
                                property_counts[key] += 1

                            relationship_counts.update(
                                ev["relationships"]
                            )

                            graph_only_rows.append({
                                "rxcui": code,
                                "name": name,
                                "element_id": node["element_id"],
                                "classification": category,
                                "used_in_postgres_prescriptions":
                                    used_in_prescriptions,
                                "patient_count":
                                    len(ev["patient_ids"]),
                                "relationship_count":
                                    ev["relationship_count"],
                                "relationship_types": json.dumps(
                                    dict(ev["relationships"])
                                ),
                                "neighbor_labels": json.dumps(
                                    dict(ev["neighbor_labels"])
                                ),
                                "matching_postgres_name_rxcuis":
                                    " | ".join(matching_name_codes),
                                "properties_json": json.dumps(
                                    props,
                                    default=str,
                                    ensure_ascii=False,
                                ),
                            })

                    logging.info(
                        "Checking shared identifier-name agreement"
                    )

                    name_mismatches = []

                    for code in shared_ids:
                        for node in graph_map[code]:
                            graph_name = normalize_name(
                                node["name"]
                            )

                            if graph_name not in pg_map[code]:
                                name_mismatches.append({
                                    "rxcui": code,
                                    "graph_name": node["name"],
                                    "postgres_names": " | ".join(
                                        sorted(pg_map[code])
                                    ),
                                })

                    graph_only_columns = [
                        "rxcui",
                        "name",
                        "element_id",
                        "classification",
                        "used_in_postgres_prescriptions",
                        "patient_count",
                        "relationship_count",
                        "relationship_types",
                        "neighbor_labels",
                        "matching_postgres_name_rxcuis",
                        "properties_json",
                    ]

                    save_csv(
                        output / "graph_only_drugs.csv",
                        graph_only_rows,
                        graph_only_columns,
                    )

                    save_csv(
                        output / "shared_name_mismatches.csv",
                        name_mismatches,
                        [
                            "rxcui",
                            "graph_name",
                            "postgres_names",
                        ],
                    )

                    save_csv(
                        output / "postgres_only_drugs.csv",
                        [
                            {
                                "rxcui": code,
                                "postgres_names": " | ".join(
                                    sorted(pg_map[code])
                                ),
                            }
                            for code in postgres_only_ids
                        ],
                        ["rxcui", "postgres_names"],
                    )

                    save_csv(
                        output / "graph_only_property_keys.csv",
                        [
                            {
                                "property": key,
                                "node_count": count,
                            }
                            for key, count
                            in property_counts.most_common()
                        ],
                        ["property", "node_count"],
                    )

                    save_csv(
                        output / "graph_only_relationship_types.csv",
                        [
                            {
                                "relationship": key,
                                "count": count,
                            }
                            for key, count
                            in relationship_counts.most_common()
                        ],
                        ["relationship", "count"],
                    )

                    priority_order = {
                        "CLINICAL_USAGE_REVIEW": 0,
                        "NAME_MATCH_DIFFERENT_ID_REVIEW": 1,
                        "CONNECTED_SUPPLEMENTARY_REVIEW": 2,
                        "ISOLATED_CONCEPT_REVIEW": 3,
                    }

                    priority_rows = sorted(
                        graph_only_rows,
                        key=lambda row: (
                            priority_order[row["classification"]],
                            -row["relationship_count"],
                            row["rxcui"],
                        ),
                    )

                    save_csv(
                        output / "review_priority.csv",
                        priority_rows,
                        graph_only_columns,
                    )

                    summary.update({
                        "status": (
                            "COMPLETED_WITH_REVIEW_ITEMS"
                            if graph_only_ids or name_mismatches
                            else "PASS"
                        ),
                        "graph_drug_nodes": len(graph_rows),
                        "graph_unique_identifiers": len(graph_ids),
                        "postgres_drug_rows": len(pg_rows),
                        "postgres_unique_identifiers":
                            len(postgres_ids),
                        "shared_identifiers": len(shared_ids),
                        "graph_only_identifiers":
                            len(graph_only_ids),
                        "postgres_only_identifiers":
                            len(postgres_only_ids),
                        "shared_name_mismatch_rows":
                            len(name_mismatches),
                        "graph_duplicate_identifier_nodes":
                            len(graph_rows) - len(graph_ids),
                        "graph_only_codes_used_in_prescriptions":
                            len(
                                set(graph_only_ids)
                                & prescribed_codes
                            ),
                        "graph_only_codes_with_patient_neighbors":
                            sum(
                                bool(evidence[code]["patient_ids"])
                                for code in graph_only_ids
                            ),
                        "graph_only_codes_with_relationships":
                            sum(
                                evidence[code]["relationship_count"] > 0
                                for code in graph_only_ids
                            ),
                        "classification_counts":
                            dict(classifications),
                        "property_key_counts":
                            dict(property_counts),
                        "relationship_type_counts":
                            dict(relationship_counts),
                        "provenance_verified": False,
                        "note": (
                            "Graph topology and properties provide "
                            "provenance clues, not authoritative proof. "
                            "Graph-only identifiers require source review."
                        ),
                    })

    except Exception as exc:
        summary["status"] = "ERRORS_PRESENT"
        summary["error"] = (
            f"{type(exc).__name__}: {exc}"
        )
        logging.exception("Phase 10.9 failed")

    finally:
        summary["output_directory"] = str(output)
        summary["completed_at"] = (
            datetime.now().isoformat()
        )

        with open(
            output / "summary.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                summary,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print("\n" + "=" * 70)
        print("PHASE 10.9 - DRUG PROVENANCE AUDIT")
        print("=" * 70)

        for key, value in summary.items():
            print(f"{key}: {value}")

        print("\nDatabase modifications: NONE")

    return (
        1
        if summary["status"] == "ERRORS_PRESENT"
        else 0
    )


if __name__ == "__main__":
    sys.exit(main())
