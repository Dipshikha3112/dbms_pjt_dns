
from __future__ import annotations

import argparse
import csv
import getpass
import json
import os
import re
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import psycopg
from neo4j import GraphDatabase, READ_ACCESS


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = ROOT / "results" / "phase10_8_disease_reconciliation"

SUFFIX_PATTERN = re.compile(
    r"\s*\((?:disorder|finding|situation|procedure|"
    r"observable entity|clinical finding|event|"
    r"morphologic abnormality|body structure)\)\s*$",
    re.IGNORECASE,
)

STOP_WORDS = {
    "a", "an", "and", "the", "of", "in", "on",
    "with", "without", "due", "to", "unspecified",
}

# Only formatting is removed from clinical codes.
# Do not strip punctuation or leading zeros from codes.
def normalize_code(value):
    if value is None:
        return ""
    return unicodedata.normalize(
        "NFKC", str(value)
    ).strip().upper()


def normalize_text(value):
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.casefold().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_clinical_term(value):
    text = normalize_text(value)
    text = SUFFIX_PATTERN.sub("", text)
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_similarity(a, b):
    a_tokens = {
        token for token in normalize_clinical_term(a).split()
        if token not in STOP_WORDS
    }
    b_tokens = {
        token for token in normalize_clinical_term(b).split()
        if token not in STOP_WORDS
    }

    if not a_tokens or not b_tokens:
        return 0.0

    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def description_similarity(a, b):
    left = normalize_clinical_term(a)
    right = normalize_clinical_term(b)

    if not left or not right:
        return 0.0

    sequence = SequenceMatcher(
        None, left, right, autojunk=False
    ).ratio()

    token = token_similarity(a, b)

    return round(
        0.65 * sequence + 0.35 * token, 4
    )


def csv_write(path, rows, columns):
    with path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=columns, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def json_write(path, value):
    path.write_text(
        json.dumps(value, indent=2, default=str),
        encoding="utf-8"
    )


def graph_read(driver, database):
    with driver.session(
        database=database,
        default_access_mode=READ_ACCESS
    ) as session:
        result = session.run("""
            MATCH (d:Disease)
            RETURN
                d.condition_id AS condition_id,
                d.source_code AS source_code,
                d.name AS description,
                properties(d) AS properties
        """)
        records = [dict(row) for row in result]
        result.consume()
        return records


def pg_read(conn):
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT condition_id, source_code, description
            FROM public.condition
        """)
        return [
            {
                "condition_id": row[0],
                "source_code": row[1],
                "description": row[2]
            }
            for row in cursor.fetchall()
        ]


def pg_columns(conn, table):
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = %s
            ORDER BY ordinal_position
        """, (table,))
        return [row[0] for row in cursor.fetchall()]


def candidate_record(graph, pg, method, score, status):
    return {
        "graph_condition_id": graph["condition_id"],
        "postgres_condition_id": pg["condition_id"],
        "graph_source_code": graph["source_code"],
        "postgres_source_code": pg["source_code"],
        "graph_description": graph["description"],
        "postgres_description": pg["description"],
        "matching_method": method,
        "similarity_score": round(score, 4),
        "status": status
    }


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

    parser.add_argument(
        "--candidate-threshold",
        type=float,
        default=0.65
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5
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

    if not 0 <= args.candidate_threshold <= 1:
        parser.error("Candidate threshold must be between 0 and 1.")

    if args.top_k < 1:
        parser.error("--top-k must be at least 1.")

    output = (
        OUTPUT_ROOT /
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)

    started = time.perf_counter()

    print("=" * 72)
    print("PHASE 10.8 - ADVANCED DISEASE RECONCILIATION")
    print("=" * 72)

    driver = GraphDatabase.driver(
        args.neo4j_uri,
        auth=(args.neo4j_user, args.neo4j_password)
    )

    try:
        driver.verify_connectivity()

        graph = graph_read(
            driver, args.neo4j_database
        )

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
            postgres = pg_read(conn)
            condition_columns = pg_columns(conn, "condition")

        print(f"Neo4j diseases: {len(graph)}")
        print(f"PostgreSQL conditions: {len(postgres)}")

        pg_by_code = defaultdict(list)
        graph_by_code = defaultdict(list)

        for row in postgres:
            code = normalize_code(row["source_code"])
            if code:
                pg_by_code[code].append(row)

        for row in graph:
            code = normalize_code(row["source_code"])
            if code:
                graph_by_code[code].append(row)

        verified = []
        terminology_variants = []
        description_conflicts = []
        ambiguous = []
        unmatched = []
        candidates = []

        for disease in graph:
            code = normalize_code(disease["source_code"])
            matches = pg_by_code.get(code, []) if code else []

            if matches:
                if (
                    len(matches) != 1
                    or len(graph_by_code[code]) != 1
                ):
                    ambiguous.append({
                        "graph_condition_id": disease["condition_id"],
                        "graph_source_code": disease["source_code"],
                        "reason": "NON_UNIQUE_SOURCE_CODE",
                        "postgres_candidate_ids": "|".join(
                            str(row["condition_id"])
                            for row in matches
                        )
                    })
                    continue

                pg = matches[0]

                raw_graph = normalize_text(
                    disease["description"]
                )
                raw_pg = normalize_text(
                    pg["description"]
                )

                clinical_graph = normalize_clinical_term(
                    disease["description"]
                )
                clinical_pg = normalize_clinical_term(
                    pg["description"]
                )

                score = description_similarity(
                    disease["description"],
                    pg["description"]
                )

                if raw_graph and raw_graph == raw_pg:
                    verified.append(
                        candidate_record(
                            disease, pg,
                            "EXACT_SOURCE_CODE_AND_DESCRIPTION",
                            1.0,
                            "VERIFIED_EXACT"
                        )
                    )

                elif (
                    clinical_graph
                    and clinical_graph == clinical_pg
                ):
                    terminology_variants.append(
                        candidate_record(
                            disease, pg,
                            "SOURCE_CODE_AND_NORMALIZED_DESCRIPTION",
                            1.0,
                            "NORMALIZED_TERMINOLOGY_AGREEMENT"
                        )
                    )

                else:
                    description_conflicts.append(
                        candidate_record(
                            disease, pg,
                            "SOURCE_CODE_WITH_DIFFERENT_DESCRIPTION",
                            score,
                            "DESCRIPTION_REVIEW_REQUIRED"
                        )
                    )

                continue

            unmatched.append({
                "graph_condition_id": disease["condition_id"],
                "graph_source_code": disease["source_code"],
                "graph_description": disease["description"],
                "reason": (
                    "MISSING_SOURCE_CODE"
                    if not code
                    else "SOURCE_CODE_NOT_FOUND_IN_POSTGRES"
                )
            })

            scored = []

            for pg in postgres:
                score = description_similarity(
                    disease["description"],
                    pg["description"]
                )

                if score >= args.candidate_threshold:
                    scored.append((score, pg))

            scored.sort(
                key=lambda pair: pair[0],
                reverse=True
            )

            for score, pg in scored[:args.top_k]:
                candidates.append(
                    candidate_record(
                        disease, pg,
                        "DESCRIPTION_SIMILARITY_ONLY",
                        score,
                        "CANDIDATE_NOT_VERIFIED"
                    )
                )

        mapping_columns = [
            "graph_condition_id",
            "postgres_condition_id",
            "graph_source_code",
            "postgres_source_code",
            "graph_description",
            "postgres_description",
            "matching_method",
            "similarity_score",
            "status"
        ]

        csv_write(
            output / "verified_exact_mappings.csv",
            verified,
            mapping_columns
        )

        csv_write(
            output / "normalized_terminology_matches.csv",
            terminology_variants,
            mapping_columns
        )

        csv_write(
            output / "description_conflicts.csv",
            description_conflicts,
            mapping_columns
        )

        csv_write(
            output / "unmatched_candidates.csv",
            candidates,
            mapping_columns
        )

        csv_write(
            output / "unmatched_graph_diseases.csv",
            unmatched,
            [
                "graph_condition_id",
                "graph_source_code",
                "graph_description",
                "reason"
            ]
        )

        csv_write(
            output / "ambiguous_source_codes.csv",
            ambiguous,
            [
                "graph_condition_id",
                "graph_source_code",
                "reason",
                "postgres_candidate_ids"
            ]
        )

        all_mappings = (
            verified
            + terminology_variants
            + description_conflicts
        )

        csv_write(
            output / "disease_mapping_registry.csv",
            all_mappings,
            mapping_columns
        )

        matched_pg_ids = {
            str(row["postgres_condition_id"])
            for row in all_mappings
        }

        postgres_unmapped = [
            row for row in postgres
            if str(row["condition_id"]) not in matched_pg_ids
        ]

        csv_write(
            output / "postgres_unmapped_conditions.csv",
            postgres_unmapped,
            [
                "condition_id",
                "source_code",
                "description"
            ]
        )

        summary = {
            "phase": "10.8_ADVANCED_RECONCILIATION",
            "graph_diseases": len(graph),
            "postgres_conditions": len(postgres),
            "exact_verified": len(verified),
            "normalized_terminology_agreements": len(
                terminology_variants
            ),
            "source_code_matches_requiring_description_review": len(
                description_conflicts
            ),
            "unmatched_graph_diseases": len(unmatched),
            "ambiguous_source_codes": len(ambiguous),
            "unmatched_candidate_rows": len(candidates),
            "total_unique_source_code_matches": len(
                all_mappings
            ),
            "postgres_conditions_not_mapped": len(
                postgres_unmapped
            ),
            "condition_table_columns": condition_columns,
            "candidate_similarity_threshold": (
                args.candidate_threshold
            ),
            "mapping_ready": (
                len(verified) + len(terminology_variants)
                == len(graph)
                and not description_conflicts
                and not unmatched
                and not ambiguous
            ),
            "clinical_code_system_verified": False,
            "read_only": True,
            "duration_seconds": round(
                time.perf_counter() - started, 3
            )
        }

        json_write(
            output / "disease_reconciliation_summary.json",
            summary
        )

        print("\n" + "=" * 72)
        print("ADVANCED DISEASE RECONCILIATION RESULTS")
        print("=" * 72)
        print(json.dumps(summary, indent=2))
        print("\nResults saved to:")
        print(output)

    finally:
        driver.close()


if __name__ == "__main__":
    main()
