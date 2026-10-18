
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import argparse
import getpass
import json
import os
import re
import time

import pandas as pd
import psycopg
from psycopg import sql as psql
from neo4j import GraphDatabase


ROOT = Path(__file__).resolve().parent.parent
PHASE10_ROOT = ROOT / "results" / "phase10_graph"
OUTPUT_ROOT = ROOT / "results" / "phase10_5_consistency"


def log(message=""):
    print(message, flush=True)


def heading(message):
    log("\n" + "=" * 76)
    log(message)
    log("=" * 76)


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


def save_csv(path, rows, columns=None):
    pd.DataFrame(
        rows,
        columns=columns if not rows else None
    ).to_csv(path, index=False)


def cypher_id(value):
    return "`" + str(value).replace("`", "``") + "`"


def normalize(value):
    return re.sub(
        r"[^a-z0-9]",
        "",
        str(value).lower()
    )


def latest_phase10_folder():
    if not PHASE10_ROOT.exists():
        raise FileNotFoundError(
            f"Phase 10 output not found: {PHASE10_ROOT}"
        )

    folders = sorted(
        folder
        for folder in PHASE10_ROOT.iterdir()
        if folder.is_dir()
        and (folder / "graph_label_counts.csv").exists()
    )

    if not folders:
        raise FileNotFoundError(
            "No completed Phase 10.1 label inventory found."
        )

    return folders[-1]


def neo4j_read(driver, database, query, params=None):
    with driver.session(
        database=database,
        default_access_mode="READ"
    ) as session:
        result = session.run(query, params or {})
        rows = [record.data() for record in result]
        result.consume()
        return rows


def postgres_connect(args):
    return psycopg.connect(
        host=args.pg_host,
        port=args.pg_port,
        dbname=args.pg_database,
        user=args.pg_user,
        password=args.pg_password,
        autocommit=True,
        connect_timeout=15,
        options=(
            "-c default_transaction_read_only=on "
            "-c statement_timeout=30000"
        )
    )


def postgres_schema(conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                c.table_schema,
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

        return [
            {
                "schema": row[0],
                "table": row[1],
                "column": row[2],
                "data_type": row[3]
            }
            for row in cur.fetchall()
        ]


def postgres_tables(conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """)

        return [
            row[0]
            for row in cur.fetchall()
        ]


def postgres_count(conn, table):
    statement = psql.SQL(
        "SELECT count(*) FROM {}"
    ).format(
        psql.Identifier("public", table)
    )

    with conn.cursor() as cur:
        cur.execute(statement)
        return int(cur.fetchone()[0])


def postgres_column_values(
    conn,
    table,
    column,
    limit
):
    statement = psql.SQL("""
        SELECT DISTINCT {column}::text
        FROM {table}
        WHERE {column} IS NOT NULL
        LIMIT %s
    """).format(
        column=psql.Identifier(column),
        table=psql.Identifier("public", table)
    )

    with conn.cursor() as cur:
        cur.execute(statement, (limit,))
        return [
            str(row[0])
            for row in cur.fetchall()
        ]


def graph_column_values(
    driver,
    database,
    label,
    property_name,
    limit
):
    query = (
        f"MATCH (n:{cypher_id(label)}) "
        f"WHERE n.{cypher_id(property_name)} IS NOT NULL "
        f"RETURN DISTINCT toString(n.{cypher_id(property_name)}) "
        "AS value LIMIT $limit"
    )

    rows = neo4j_read(
        driver,
        database,
        query,
        {"limit": limit}
    )

    return [
        str(row["value"])
        for row in rows
        if row.get("value") is not None
    ]


def graph_count(driver, database, label):
    query = (
        f"MATCH (n:{cypher_id(label)}) "
        "RETURN count(n) AS total"
    )

    rows = neo4j_read(
        driver,
        database,
        query
    )

    return int(rows[0]["total"])


def graph_properties(driver, database, label):
    query = (
        f"MATCH (n:{cypher_id(label)}) "
        "WITH n LIMIT 200 "
        "UNWIND keys(n) AS property "
        "RETURN DISTINCT property"
    )

    rows = neo4j_read(
        driver,
        database,
        query
    )

    return [
        row["property"]
        for row in rows
    ]


def choose_table(label, tables):
    exact = [
        table
        for table in tables
        if table.lower() == label.lower()
    ]

    if len(exact) == 1:
        return exact[0], "EXACT_NAME"

    normalized = [
        table
        for table in tables
        if normalize(table) == normalize(label)
    ]

    if len(normalized) == 1:
        return normalized[0], "NORMALIZED_NAME"

    return None, "NO_UNAMBIGUOUS_MATCH"


def choose_identifier(graph_props, pg_columns):
    candidates = [
        "id",
        "patient_id",
        "encounter_id",
        "condition_id",
        "medication_id",
        "procedure_id",
        "claim_id",
        "code",
        "rxcui"
    ]

    graph_map = {
        normalize(name): name
        for name in graph_props
    }

    pg_map = {
        normalize(name): name
        for name in pg_columns
    }

    for candidate in candidates:
        key = normalize(candidate)

        if key in graph_map and key in pg_map:
            return (
                graph_map[key],
                pg_map[key]
            )

    common = sorted(
        set(graph_map) & set(pg_map)
    )

    for key in common:
        if key.endswith("id"):
            return (
                graph_map[key],
                pg_map[key]
            )

    return None, None


def compare_identifiers(
    driver,
    database,
    conn,
    label,
    table,
    graph_property,
    pg_column,
    sample_size
):
    graph_values = graph_column_values(
        driver,
        database,
        label,
        graph_property,
        sample_size
    )

    pg_values = postgres_column_values(
        conn,
        table,
        pg_column,
        sample_size
    )

    graph_set = set(graph_values)
    pg_set = set(pg_values)

    intersection = graph_set & pg_set
    union = graph_set | pg_set

    return {
        "graph_sample_size": len(graph_set),
        "postgres_sample_size": len(pg_set),
        "sample_overlap_count": len(intersection),
        "sample_jaccard": (
            len(intersection) / len(union)
            if union else None
        ),
        "status": (
            "SAMPLE_OVERLAP_FOUND"
            if intersection
            else "NO_SAMPLE_OVERLAP"
        )
    }


def build_pairs(
    label,
    table,
    graph_property,
    pg_column,
    count_matches,
    identifier_result
):
    pairs = []

    cypher_count = (
        f"MATCH (n:{cypher_id(label)}) "
        "RETURN count(n) AS total"
    )

    sql_count = psql.SQL(
        "SELECT count(*) AS total FROM {}"
    ).format(
        psql.Identifier("public", table)
    ).as_string()

    pairs.append({
        "task": "entity_count",
        "graph_label": label,
        "postgres_table": table,
        "cypher": cypher_count,
        "sql": sql_count,
        "verification_status": (
            "COUNT_MATCH_ONLY"
            if count_matches
            else "COUNT_MISMATCH"
        ),
        "notes": (
            "Count agreement alone does not prove "
            "entity equivalence."
        )
    })

    if graph_property and pg_column:
        cypher_query = (
            f"MATCH (n:{cypher_id(label)}) "
            f"WHERE n.{cypher_id(graph_property)} IS NOT NULL "
            f"RETURN DISTINCT toString(n.{cypher_id(graph_property)}) "
            "AS entity_identifier LIMIT 100"
        )

        sql_query = psql.SQL("""
            SELECT DISTINCT {column}::text
                   AS entity_identifier
            FROM {table}
            WHERE {column} IS NOT NULL
            LIMIT 100
        """).format(
            column=psql.Identifier(pg_column),
            table=psql.Identifier("public", table)
        ).as_string()

        pairs.append({
            "task": "entity_identifier_sample",
            "graph_label": label,
            "postgres_table": table,
            "cypher": cypher_query,
            "sql": sql_query,
            "verification_status": (
                identifier_result.get(
                    "status",
                    "NOT_VERIFIED"
                )
            ),
            "notes": (
                "Independent LIMIT samples are not "
                "guaranteed to select the same records."
            )
        })

    return pairs


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

    if args.sample_size < 1:
        parser.error(
            "--sample-size must be positive."
        )

    heading(
        "PHASE 10.5 - STANDALONE CROSS-DATABASE VALIDATION"
    )

    phase10_dir = latest_phase10_folder()

    log(f"Using Phase 10 results: {phase10_dir}")

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

    labels_df = pd.read_csv(
        phase10_dir / "graph_label_counts.csv"
    )

    labels = labels_df["label"].tolist()

    driver = GraphDatabase.driver(
        args.neo4j_uri,
        auth=(
            args.neo4j_user,
            args.neo4j_password
        )
    )

    start = time.perf_counter()

    comparisons = []
    pairs = []
    errors = []
    schema = []

    try:
        driver.verify_connectivity()

        with postgres_connect(args) as conn:
            schema = postgres_schema(conn)
            tables = postgres_tables(conn)

            columns_by_table = defaultdict(list)

            for column in schema:
                columns_by_table[
                    column["table"]
                ].append(column["column"])

            heading("1. VALIDATING GRAPH LABELS")

            for number, label in enumerate(
                labels,
                start=1
            ):
                log(
                    f"[{number}/{len(labels)}] "
                    f"Checking {label}"
                )

                table, match_method = choose_table(
                    label,
                    tables
                )

                if not table:
                    comparisons.append({
                        "graph_label": label,
                        "postgres_table": "",
                        "status": (
                            "NO_UNAMBIGUOUS_TABLE_MATCH"
                        ),
                        "match_method": match_method
                    })
                    continue

                try:
                    n_graph = graph_count(
                        driver,
                        args.neo4j_database,
                        label
                    )

                    n_pg = postgres_count(
                        conn,
                        table
                    )

                    graph_props = graph_properties(
                        driver,
                        args.neo4j_database,
                        label
                    )

                    pg_columns = columns_by_table[table]

                    graph_property, pg_column = (
                        choose_identifier(
                            graph_props,
                            pg_columns
                        )
                    )

                    identifier_result = {}

                    if graph_property and pg_column:
                        identifier_result = (
                            compare_identifiers(
                                driver,
                                args.neo4j_database,
                                conn,
                                label,
                                table,
                                graph_property,
                                pg_column,
                                args.sample_size
                            )
                        )

                    count_matches = n_graph == n_pg

                    comparisons.append({
                        "graph_label": label,
                        "postgres_table": table,
                        "match_method": match_method,
                        "graph_count": n_graph,
                        "postgres_count": n_pg,
                        "count_difference": (
                            n_graph - n_pg
                        ),
                        "counts_match": count_matches,
                        "graph_identifier": graph_property,
                        "postgres_identifier": pg_column,
                        **identifier_result,
                        "status": (
                            "COUNT_MATCH"
                            if count_matches
                            else "COUNT_MISMATCH"
                        )
                    })

                    pairs.extend(
                        build_pairs(
                            label,
                            table,
                            graph_property,
                            pg_column,
                            count_matches,
                            identifier_result
                        )
                    )

                except Exception as error:
                    errors.append({
                        "graph_label": label,
                        "error": str(error)
                    })

                    log(f"  Error: {error}")

        heading("2. SAVING RESULTS")

        for number, pair in enumerate(
            pairs,
            start=1
        ):
            pair["pair_id"] = f"EQ{number:03d}"

        save_csv(
            output / "postgres_schema_inventory.csv",
            schema
        )

        save_csv(
            output / "cross_database_consistency.csv",
            comparisons
        )

        save_csv(
            output / "sql_cypher_query_pairs.csv",
            pairs
        )

        save_csv(
            output / "validation_errors.csv",
            errors
        )

        report = {
            "phase": "10.5",
            "completed_at": datetime.now().isoformat(),
            "neo4j_database": args.neo4j_database,
            "postgres_database": args.pg_database,
            "graph_labels_checked": len(labels),
            "table_matches": sum(
                bool(item.get("postgres_table"))
                for item in comparisons
            ),
            "count_matches": sum(
                item.get("counts_match") is True
                for item in comparisons
            ),
            "identifier_sample_overlap_found": sum(
                item.get("status") == "SAMPLE_OVERLAP_FOUND"
                or item.get("sample_overlap_count", 0) > 0
                for item in comparisons
            ),
            "proposed_query_pairs": len(pairs),
            "fully_verified_equivalent_pairs": 0,
            "errors": len(errors),
            "duration_seconds": (
                time.perf_counter() - start
            ),
            "read_only": True,
            "limitations": [
                "Label-table matching is name-based.",
                "Entity counts do not prove equivalence.",
                "Identifier overlap is based on independent samples.",
                "Relationship-level consistency is not validated.",
                "Query pairs are candidates for Phase 11.",
                "No clinical-result equivalence is claimed."
            ]
        }

        save_json(
            output / "phase10_5_report.json",
            report
        )

        heading("PHASE 10.5 COMPLETE")

        log(json.dumps(report, indent=2))

        log("\nResults saved to:")
        log(str(output))

    finally:
        driver.close()


if __name__ == "__main__":
    main()
