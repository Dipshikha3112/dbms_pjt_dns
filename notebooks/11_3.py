
from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import psycopg
from psycopg import sql as psql
from neo4j import GraphDatabase


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = ROOT / "results" / "phase10_6_hybrid_audit"

ID_HINTS = (
    "id", "code", "rxcui", "snomed", "loinc",
    "icd", "mondo", "mesh", "chebi", "cui"
)

CLINICAL_HINTS = (
    "patient", "disease", "drug", "lab",
    "diagnosis", "condition", "medication",
    "prescription", "observation"
)


def log(message=""):
    print(message, flush=True)


def heading(message):
    log("\n" + "=" * 76)
    log(message)
    log("=" * 76)


def save_csv(path, rows, columns=None):
    pd.DataFrame(
        rows,
        columns=columns if not rows else None
    ).to_csv(path, index=False)


def save_json(path, data):
    path.write_text(
        json.dumps(data, indent=2, default=str),
        encoding="utf-8"
    )


def cyid(value):
    return "`" + str(value).replace("`", "``") + "`"


def normalize(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def is_identifier(name):
    n = normalize(name)
    return any(hint in n for hint in ID_HINTS)


def is_clinical(name):
    n = normalize(name)
    return any(hint in n for hint in CLINICAL_HINTS)


def neo_read(driver, database, query, params=None):
    with driver.session(
        database=database,
        default_access_mode="READ"
    ) as session:
        result = session.run(query, params or {})
        rows = [record.data() for record in result]
        result.consume()
        return rows


def pg_connect(args):
    return psycopg.connect(
        host=args.pg_host,
        port=args.pg_port,
        user=args.pg_user,
        password=args.pg_password,
        dbname=args.pg_database,
        autocommit=True,
        connect_timeout=15,
        options=(
            "-c default_transaction_read_only=on "
            "-c statement_timeout=30000"
        )
    )


def graph_labels(driver, database):
    rows = neo_read(
        driver,
        database,
        "CALL db.labels() YIELD label RETURN label ORDER BY label"
    )
    return [row["label"] for row in rows]


def graph_properties(driver, database, label, sample):
    query = (
        f"MATCH (n:{cyid(label)}) "
        "WITH n LIMIT $sample "
        "UNWIND keys(n) AS property "
        "RETURN property, count(*) AS presence "
        "ORDER BY presence DESC"
    )
    return neo_read(
        driver, database, query, {"sample": sample}
    )


def graph_count(driver, database, label):
    query = (
        f"MATCH (n:{cyid(label)}) "
        "RETURN count(n) AS total"
    )
    return neo_read(driver, database, query)[0]["total"]


def pg_schema(conn):
    with conn.cursor() as cur:
        cur.execute("""
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
            ORDER BY c.table_name, c.ordinal_position
        """)
        return [
            {
                "table": row[0],
                "column": row[1],
                "data_type": row[2]
            }
            for row in cur.fetchall()
        ]


def phase_a(driver, database, labels):
    heading("10.6A - CONNECTIVITY BY NODE LABEL")
    results = []

    for label in labels:
        query = (
            f"MATCH (n:{cyid(label)}) "
            "RETURN "
            "count(n) AS total, "
            "count(CASE WHEN (n)--() THEN 1 END) AS connected, "
            "count(CASE WHEN NOT (n)--() THEN 1 END) AS isolated"
        )

        row = neo_read(driver, database, query)[0]

        total = int(row["total"])
        isolated = int(row["isolated"])

        results.append({
            "label": label,
            "total_nodes": total,
            "connected_nodes": int(row["connected"]),
            "isolated_nodes": isolated,
            "isolated_pct": round(
                100 * isolated / total, 3
            ) if total else 0
        })

        log(
            f"{label}: {isolated:,}/{total:,} isolated "
            f"({results[-1]['isolated_pct']}%)"
        )

    return results


def phase_b(driver, database):
    heading("10.6B - RELATIONSHIP COVERAGE")

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
            "relationship_type": row["relationship_type"],
            "target_labels": json.dumps(
                row["target_labels"]
            ),
            "relationship_count": row["relationship_count"]
        })

    log(f"Relationship patterns: {len(results)}")
    return results


def phase_c(driver, database, labels, conn, sample):
    heading("10.6C - IDENTIFIER INVENTORY")

    inventory = []
    graph_id_props = defaultdict(list)
    pg_id_columns = defaultdict(list)

    for label in labels:
        for row in graph_properties(
            driver, database, label, sample
        ):
            prop = row["property"]

            if is_identifier(prop):
                graph_id_props[label].append(prop)

            inventory.append({
                "database": "neo4j",
                "entity": label,
                "field": prop,
                "data_type": "",
                "identifier_candidate": is_identifier(prop),
                "sample_presence": row["presence"]
            })

    for row in pg_schema(conn):
        if is_identifier(row["column"]):
            pg_id_columns[row["table"]].append(
                row["column"]
            )

        inventory.append({
            "database": "postgresql",
            "entity": row["table"],
            "field": row["column"],
            "data_type": row["data_type"],
            "identifier_candidate": (
                is_identifier(row["column"])
            ),
            "sample_presence": ""
        })

    log(
        f"Graph entities with candidate identifiers: "
        f"{len(graph_id_props)}"
    )
    log(
        f"PostgreSQL tables with candidate identifiers: "
        f"{len(pg_id_columns)}"
    )

    return inventory, graph_id_props, pg_id_columns


def graph_id_sample(
    driver, database, label, prop, limit
):
    query = (
        f"MATCH (n:{cyid(label)}) "
        f"WHERE n.{cyid(prop)} IS NOT NULL "
        f"RETURN DISTINCT toString(n.{cyid(prop)}) AS value "
        "LIMIT $limit"
    )

    rows = neo_read(
        driver, database, query, {"limit": limit}
    )

    return {
        str(row["value"])
        for row in rows
        if row["value"] is not None
    }


def pg_id_sample(conn, table, column, limit):
    query = psql.SQL("""
        SELECT DISTINCT {col}::text
        FROM {table}
        WHERE {col} IS NOT NULL
        LIMIT %s
    """).format(
        col=psql.Identifier(column),
        table=psql.Identifier("public", table)
    )

    with conn.cursor() as cur:
        cur.execute(query, (limit,))
        return {
            str(row[0])
            for row in cur.fetchall()
        }


def candidate_score(label, table, gprop, pcol):
    score = 0

    if normalize(label) == normalize(table):
        score += 3

    if normalize(gprop) == normalize(pcol):
        score += 3

    if (
        is_clinical(label)
        and is_clinical(table)
    ):
        score += 1

    if (
        normalize(gprop).endswith("id")
        and normalize(pcol).endswith("id")
    ):
        score += 1

    return score


def phase_d(
    driver, database, conn,
    graph_id_props, pg_id_columns,
    sample_size, max_candidates
):
    heading("10.6D - CROSS-DATABASE IDENTIFIER MAPPING")

    candidate_pairs = []

    for label, props in graph_id_props.items():
        for table, columns in pg_id_columns.items():
            for gprop in props:
                for pcol in columns:
                    score = candidate_score(
                        label, table, gprop, pcol
                    )

                    if score >= 3:
                        candidate_pairs.append({
                            "graph_label": label,
                            "graph_property": gprop,
                            "postgres_table": table,
                            "postgres_column": pcol,
                            "name_score": score
                        })

    candidate_pairs.sort(
        key=lambda x: x["name_score"],
        reverse=True
    )

    candidate_pairs = candidate_pairs[:max_candidates]

    graph_cache = {}
    pg_cache = {}
    results = []

    for number, pair in enumerate(
        candidate_pairs, start=1
    ):
        gkey = (
            pair["graph_label"],
            pair["graph_property"]
        )
        pkey = (
            pair["postgres_table"],
            pair["postgres_column"]
        )

        try:
            if gkey not in graph_cache:
                graph_cache[gkey] = graph_id_sample(
                    driver, database,
                    *gkey, sample_size
                )

            if pkey not in pg_cache:
                pg_cache[pkey] = pg_id_sample(
                    conn, *pkey, sample_size
                )

            a = graph_cache[gkey]
            b = pg_cache[pkey]

            overlap = a & b

            results.append({
                **pair,
                "graph_sample_count": len(a),
                "postgres_sample_count": len(b),
                "overlap_count": len(overlap),
                "overlap_ratio_graph": (
                    len(overlap) / len(a) if a else 0
                ),
                "overlap_ratio_postgres": (
                    len(overlap) / len(b) if b else 0
                ),
                "sample_matching_values": json.dumps(
                    sorted(overlap)[:10]
                ),
                "status": (
                    "SAMPLE_OVERLAP_FOUND"
                    if overlap
                    else "NO_SAMPLE_OVERLAP"
                )
            })

        except Exception as error:
            results.append({
                **pair,
                "status": "ERROR",
                "error": str(error)[:500]
            })

        if number % 10 == 0:
            log(
                f"Mapping candidates tested: "
                f"{number}/{len(candidate_pairs)}"
            )

    log(
        f"Total mapping candidates tested: "
        f"{len(results)}"
    )

    return results


def phase_e(mappings):
    heading("10.6E - HYBRID QUERY CANDIDATES")

    results = []

    for mapping in mappings:
        if mapping.get("overlap_count", 0) <= 0:
            continue

        label = mapping["graph_label"]
        gprop = mapping["graph_property"]
        table = mapping["postgres_table"]
        pcol = mapping["postgres_column"]

        cypher = (
            f"MATCH (n:{cyid(label)}) "
            f"WHERE toString(n.{cyid(gprop)}) = $identifier "
            "RETURN properties(n) AS entity"
        )

        sql_query = psql.SQL("""
            SELECT *
            FROM {table}
            WHERE {column}::text = %(identifier)s
        """).format(
            table=psql.Identifier("public", table),
            column=psql.Identifier(pcol)
        ).as_string()

        results.append({
            "hybrid_id": (
                f"HQ{len(results) + 1:03d}"
            ),
            "graph_label": label,
            "postgres_table": table,
            "graph_identifier": gprop,
            "postgres_identifier": pcol,
            "cypher": cypher,
            "sql": sql_query,
            "observed_sample_overlap": (
                mapping["overlap_count"]
            ),
            "status": "CANDIDATE_NOT_SEMANTICALLY_VERIFIED",
            "notes": (
                "Identifiers overlap in samples, but "
                "clinical equivalence remains unverified."
            )
        })

    log(f"Hybrid query candidates: {len(results)}")
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

    parser.add_argument(
        "--sample-size",
        type=int,
        default=500
    )
    parser.add_argument(
        "--schema-sample",
        type=int,
        default=200
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=100
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
        args.sample_size,
        args.schema_sample,
        args.max_candidates
    ) < 1:
        parser.error(
            "All sample sizes must be positive."
        )

    heading(
        "PHASE 10.6 - HYBRID MAPPING AND CONNECTIVITY AUDIT"
    )

    output = (
        OUTPUT_ROOT
        / datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )
    )
    output.mkdir(parents=True, exist_ok=False)

    start = time.perf_counter()

    driver = GraphDatabase.driver(
        args.neo4j_uri,
        auth=(
            args.neo4j_user,
            args.neo4j_password
        )
    )

    try:
        driver.verify_connectivity()

        with pg_connect(args) as conn:
            labels = graph_labels(
                driver, args.neo4j_database
            )

            connectivity = phase_a(
                driver, args.neo4j_database, labels
            )
            save_csv(
                output / "label_connectivity.csv",
                connectivity
            )

            relationships = phase_b(
                driver, args.neo4j_database
            )
            save_csv(
                output / "relationship_coverage.csv",
                relationships
            )

            inventory, graph_ids, pg_ids = phase_c(
                driver, args.neo4j_database,
                labels, conn, args.schema_sample
            )
            save_csv(
                output / "identifier_inventory.csv",
                inventory
            )

            mappings = phase_d(
                driver, args.neo4j_database,
                conn, graph_ids, pg_ids,
                args.sample_size,
                args.max_candidates
            )
            save_csv(
                output / "identifier_mapping_candidates.csv",
                mappings
            )

            hybrid_queries = phase_e(mappings)
            save_csv(
                output / "hybrid_query_candidates.csv",
                hybrid_queries
            )

            report = {
                "phase": "10.6",
                "completed_at": datetime.now().isoformat(),
                "neo4j_database": args.neo4j_database,
                "postgres_database": args.pg_database,
                "graph_labels": len(labels),
                "graph_nodes": sum(
                    x["total_nodes"] for x in connectivity
                ),
                "isolated_nodes": sum(
                    x["isolated_nodes"] for x in connectivity
                ),
                "relationship_patterns": len(relationships),
                "identifier_candidates_tested": len(mappings),
                "identifier_overlaps_found": sum(
                    x.get("overlap_count", 0) > 0
                    for x in mappings
                ),
                "hybrid_query_candidates": len(hybrid_queries),
                "semantically_verified_mappings": 0,
                "duration_seconds": (
                    time.perf_counter() - start
                ),
                "read_only": True,
                "limitations": [
                    "Identifier matching is heuristic.",
                    "Sample overlap is not proof of equivalence.",
                    "No relationship mappings are asserted.",
                    "Graph isolation may be intentional.",
                    "No database data is modified."
                ]
            }

            save_json(
                output / "phase10_6_report.json",
                report
            )

            heading("PHASE 10.6 COMPLETE")
            log(json.dumps(report, indent=2))

            log("\nResults saved to:")
            log(str(output))

    finally:
        driver.close()


if __name__ == "__main__":
    main()
