
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import re
import statistics
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from neo4j import GraphDatabase


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = ROOT / "results" / "phase10_graph"

EXPECTED_NODES = 551615
EXPECTED_RELATIONSHIPS = 804488

READ_TIMEOUT_SECONDS = 30
MAX_GRAPH_ROWS = 1000


def log(message=""):
    print(message, flush=True)


def heading(title):
    log("\n" + "=" * 76)
    log(title)
    log("=" * 76)


def json_safe(value):
    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [json_safe(x) for x in value]

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (datetime,)):
        return value.isoformat()

    if pd.isna(value) if not isinstance(
        value, (dict, list, tuple, set)
    ) else False:
        return None

    return value


def save_json(path, value):
    path.write_text(
        json.dumps(
            json_safe(value),
            indent=2,
            ensure_ascii=False,
            default=str
        ),
        encoding="utf-8"
    )


def save_csv(path, records, columns=None):
    if records:
        frame = pd.DataFrame(records)
    else:
        frame = pd.DataFrame(columns=columns or [])

    frame.to_csv(path, index=False)


def cypher_identifier(value):
    return "`" + str(value).replace("`", "``") + "`"


def sql_identifier(value):
    return '"' + str(value).replace('"', '""') + '"'


def fingerprint(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()[:16]


def execute_read(driver, database, query, parameters=None):
    with driver.session(
        database=database,
        default_access_mode="READ"
    ) as session:
        result = session.run(query, parameters or {})
        records = [record.data() for record in result]
        summary = result.consume()

    return records, summary


def scalar(driver, database, query, parameters=None):
    rows, _ = execute_read(
        driver,
        database,
        query,
        parameters
    )

    if not rows:
        return None

    return next(iter(rows[0].values()))


def graph_connection(args):
    driver = GraphDatabase.driver(
        args.neo4j_uri,
        auth=(
            args.neo4j_user,
            args.neo4j_password
        ),
        connection_timeout=15
    )

    driver.verify_connectivity()
    return driver


def pg_connection(args):
    return psycopg.connect(
        host=args.pg_host,
        port=args.pg_port,
        user=args.pg_user,
        password=args.pg_password,
        dbname=args.pg_database,
        connect_timeout=15,
        autocommit=True,
        options="-c statement_timeout=15000 "
                "-c default_transaction_read_only=on"
    )


def get_labels(driver, database):
    rows, _ = execute_read(
        driver,
        database,
        "CALL db.labels() YIELD label RETURN label ORDER BY label"
    )

    return [row["label"] for row in rows]


def get_relationship_types(driver, database):
    rows, _ = execute_read(
        driver,
        database,
        "CALL db.relationshipTypes() "
        "YIELD relationshipType "
        "RETURN relationshipType ORDER BY relationshipType"
    )

    return [row["relationshipType"] for row in rows]


def phase101(driver, args, output):
    heading("PHASE 10.1 - GRAPH STRUCTURE AND INTEGRITY")

    node_count = scalar(
        driver,
        args.neo4j_database,
        "MATCH (n) RETURN count(n) AS total"
    )

    relationship_count = scalar(
        driver,
        args.neo4j_database,
        "MATCH ()-[r]->() RETURN count(r) AS total"
    )

    labels = get_labels(
        driver,
        args.neo4j_database
    )

    relationship_types = get_relationship_types(
        driver,
        args.neo4j_database
    )

    label_counts = []

    for label in labels:
        query = (
            f"MATCH (n:{cypher_identifier(label)}) "
            "RETURN count(n) AS total"
        )

        count = scalar(
            driver,
            args.neo4j_database,
            query
        )

        label_counts.append({
            "label": label,
            "count": int(count)
        })

    relationship_counts = []

    for relationship_type in relationship_types:
        query = (
            "MATCH ()-[r:"
            + cypher_identifier(relationship_type)
            + "]->() RETURN count(r) AS total"
        )

        count = scalar(
            driver,
            args.neo4j_database,
            query
        )

        relationship_counts.append({
            "relationship_type": relationship_type,
            "count": int(count)
        })

    schema_rows, _ = execute_read(
        driver,
        args.neo4j_database,
        """
        MATCH (a)-[r]->(b)
        RETURN
            labels(a) AS source_labels,
            type(r) AS relationship_type,
            labels(b) AS target_labels,
            count(*) AS relationship_count
        ORDER BY relationship_count DESC
        LIMIT 200
        """
    )

    property_rows = []

    for label in labels:
        query = (
            f"MATCH (n:{cypher_identifier(label)}) "
            "WITH n LIMIT $sample "
            "UNWIND keys(n) AS property "
            "RETURN property, count(*) AS sampled_presence "
            "ORDER BY sampled_presence DESC"
        )

        rows, _ = execute_read(
            driver,
            args.neo4j_database,
            query,
            {"sample": args.schema_sample}
        )

        for row in rows:
            property_rows.append({
                "label": label,
                "property": row["property"],
                "sampled_presence": (
                    row["sampled_presence"]
                )
            })

    # Relationship endpoints always exist in Neo4j.
    # This checks nodes with no relationships instead.
    isolated_nodes = scalar(
        driver,
        args.neo4j_database,
        """
        MATCH (n)
        WHERE NOT (n)--()
        RETURN count(n) AS total
        """
    )

    report = {
        "phase": "10.1",
        "database": args.neo4j_database,
        "node_count": int(node_count),
        "relationship_count": int(relationship_count),
        "expected_node_count": EXPECTED_NODES,
        "expected_relationship_count": (
            EXPECTED_RELATIONSHIPS
        ),
        "node_count_matches_reference": (
            int(node_count) == EXPECTED_NODES
        ),
        "relationship_count_matches_reference": (
            int(relationship_count)
            == EXPECTED_RELATIONSHIPS
        ),
        "isolated_nodes": int(isolated_nodes),
        "labels": labels,
        "relationship_types": relationship_types,
        "notes": [
            "Reference counts are historical import counts.",
            "A count mismatch does not prove corruption.",
            "Isolated nodes are not necessarily invalid.",
            "Property coverage is sample-based.",
            "Neo4j relationships cannot have dangling endpoints."
        ]
    }

    save_json(
        output / "graph_integrity_report.json",
        report
    )

    save_csv(
        output / "graph_label_counts.csv",
        label_counts
    )

    save_csv(
        output / "graph_relationship_counts.csv",
        relationship_counts
    )

    save_csv(
        output / "graph_schema_patterns.csv",
        schema_rows
    )

    save_csv(
        output / "graph_property_inventory.csv",
        property_rows
    )

    log(f"Nodes: {node_count:,}")
    log(f"Relationships: {relationship_count:,}")
    log(f"Labels: {len(labels)}")
    log(f"Relationship types: {len(relationship_types)}")
    log(f"Isolated nodes: {isolated_nodes:,}")

    return {
        "report": report,
        "labels": labels,
        "relationship_types": relationship_types,
        "properties": property_rows,
        "patterns": schema_rows
    }


def phase102(driver, args, output, graph_info):
    heading("PHASE 10.2 - INDEX AND CONSTRAINT AUDIT")

    indexes = []
    constraints = []
    errors = []

    try:
        indexes, _ = execute_read(
            driver,
            args.neo4j_database,
            "SHOW INDEXES"
        )
    except Exception as error:
        errors.append(
            f"SHOW INDEXES: {error}"
        )

    try:
        constraints, _ = execute_read(
            driver,
            args.neo4j_database,
            "SHOW CONSTRAINTS"
        )
    except Exception as error:
        errors.append(
            f"SHOW CONSTRAINTS: {error}"
        )

    save_csv(
        output / "graph_index_audit.csv",
        indexes
    )

    save_csv(
        output / "graph_constraint_audit.csv",
        constraints
    )

    report = {
        "phase": "10.2",
        "index_count": len(indexes),
        "constraint_count": len(constraints),
        "errors": errors,
        "notes": [
            "Existing indexes and constraints are audited.",
            "No index or constraint is created or removed.",
            "Query-plan analysis is completed after workload generation."
        ]
    }

    save_json(
        output / "graph_index_report.json",
        report
    )

    log(f"Existing indexes: {len(indexes)}")
    log(f"Existing constraints: {len(constraints)}")

    return report


def classify_label(label):
    value = label.lower()

    if any(
        term in value
        for term in (
            "patient", "person"
        )
    ):
        return "patient"

    if any(
        term in value
        for term in (
            "diagnos", "condition", "disease"
        )
    ):
        return "diagnosis"

    if any(
        term in value
        for term in (
            "medicat", "drug", "prescription"
        )
    ):
        return "medication"

    if any(
        term in value
        for term in (
            "lab", "observation", "test"
        )
    ):
        return "laboratory"

    if any(
        term in value
        for term in (
            "encounter", "visit"
        )
    ):
        return "encounter"

    if any(
        term in value
        for term in (
            "procedure", "surgery"
        )
    ):
        return "procedure"

    if any(
        term in value
        for term in (
            "claim", "transaction"
        )
    ):
        return "administrative"

    return "other"


def generate_workloads(graph_info, args):
    workloads = []

    def add(category, name, query, parameters=None):
        if any(
            item["cypher"] == query
            for item in workloads
        ):
            return

        workloads.append({
            "query_id": (
                f"G{len(workloads) + 1:04d}"
            ),
            "category": category,
            "query_name": name,
            "fingerprint": fingerprint(query),
            "cypher": query,
            "parameters_json": json.dumps(
                parameters or {}
            ),
            "expected_result": "read_only",
            "schema_source": "discovered"
        })

    labels = graph_info["labels"]

    for label in labels:
        quoted = cypher_identifier(label)
        category = classify_label(label)

        add(
            category,
            f"Count {label}",
            f"MATCH (n:{quoted}) "
            "RETURN count(n) AS total"
        )

        add(
            category,
            f"Sample {label}",
            f"MATCH (n:{quoted}) "
            "RETURN properties(n) AS properties "
            "LIMIT 20"
        )

    for pattern in graph_info["patterns"][:30]:
        sources = pattern.get("source_labels") or []
        targets = pattern.get("target_labels") or []
        rel = pattern.get("relationship_type")

        if not sources or not targets or not rel:
            continue

        source_label = cypher_identifier(sources[0])
        target_label = cypher_identifier(targets[0])
        rel_type = cypher_identifier(rel)

        query = (
            f"MATCH (a:{source_label})"
            f"-[r:{rel_type}]->"
            f"(b:{target_label}) "
            "RETURN properties(a) AS source, "
            "type(r) AS relationship, "
            "properties(b) AS target "
            "LIMIT 25"
        )

        add(
            "relationship",
            f"{sources[0]} to {targets[0]} via {rel}",
            query
        )

        count_query = (
            f"MATCH (a:{source_label})"
            f"-[r:{rel_type}]->"
            f"(b:{target_label}) "
            "RETURN count(r) AS total"
        )

        add(
            "relationship",
            f"Count {rel} relationships",
            count_query
        )

    property_inventory = graph_info["properties"]

    for item in property_inventory:
        label = item["label"]
        property_name = item["property"]

        if not any(
            keyword in property_name.lower()
            for keyword in (
                "id", "code", "name", "date", "time"
            )
        ):
            continue

        query = (
            f"MATCH (n:{cypher_identifier(label)}) "
            f"WHERE n.{cypher_identifier(property_name)} "
            "IS NOT NULL "
            f"RETURN n.{cypher_identifier(property_name)} "
            "AS property_value "
            "LIMIT 20"
        )

        add(
            classify_label(label),
            f"{label}.{property_name} non-null sample",
            query
        )

    # Add a bounded, two-hop clinical traversal
    # for discovered relationship structures.
    for label in labels:
        if classify_label(label) != "patient":
            continue

        add(
            "clinical_traversal",
            f"Two-hop relationships from {label}",
            (
                f"MATCH (p:{cypher_identifier(label)})"
                "-[r1]-(n)-[r2]-(m) "
                "RETURN type(r1) AS first_relation, "
                "labels(n) AS intermediate_labels, "
                "type(r2) AS second_relation, "
                "labels(m) AS destination_labels "
                "LIMIT 50"
            )
        )

    return workloads[:args.max_workloads]


def phase103(driver, args, output, graph_info):
    heading("PHASE 10.3 - CYPHER WORKLOAD GENERATION")

    workloads = generate_workloads(
        graph_info,
        args
    )

    plans = []

    for number, workload in enumerate(
        workloads,
        start=1
    ):
        try:
            rows, summary = execute_read(
                driver,
                args.neo4j_database,
                "EXPLAIN " + workload["cypher"],
                json.loads(
                    workload["parameters_json"]
                )
            )

            plan = (
                str(summary.plan)
                if summary.plan is not None
                else ""
            )

            plans.append({
                "query_id": workload["query_id"],
                "query_name": workload["query_name"],
                "status": "ok",
                "plan": plan[:10000],
                "error": ""
            })

        except Exception as error:
            plans.append({
                "query_id": workload["query_id"],
                "query_name": workload["query_name"],
                "status": "error",
                "plan": "",
                "error": str(error)[:1000]
            })

        if number % 20 == 0:
            log(
                f"EXPLAIN checked: {number}/{len(workloads)}"
            )

    save_csv(
        output / "cypher_workloads.csv",
        workloads
    )

    save_csv(
        output / "graph_plan_analysis.csv",
        plans
    )

    log(f"Generated workloads: {len(workloads)}")
    log(
        "EXPLAIN successes: "
        f"{sum(x['status'] == 'ok' for x in plans)}"
    )

    return workloads, plans


def execute_benchmark_query(
    driver,
    database,
    query,
    parameters,
    max_rows
):
    start = time.perf_counter()

    with driver.session(
        database=database,
        default_access_mode="READ",
        fetch_size=250
    ) as session:
        result = session.run(query, parameters)

        count = 0

        for record in result:
            count += 1

            if count >= max_rows:
                break

        summary = result.consume()

    elapsed = (
        time.perf_counter() - start
    ) * 1000.0

    return {
        "latency_ms": elapsed,
        "rows_consumed": count,
        "result_available_after_ms": (
            summary.result_available_after
        ),
        "result_consumed_after_ms": (
            summary.result_consumed_after
        )
    }


def phase104(driver, args, output, workloads):
    heading("PHASE 10.4 - GRAPH WORKLOAD BENCHMARKING")

    samples = []

    for number, workload in enumerate(
        workloads,
        start=1
    ):
        parameters = json.loads(
            workload["parameters_json"]
        )

        for attempt in range(
            args.warmups + args.repetitions
        ):
            warmup = attempt < args.warmups

            try:
                result = execute_benchmark_query(
                    driver,
                    args.neo4j_database,
                    workload["cypher"],
                    parameters,
                    args.max_result_rows
                )

                if warmup:
                    continue

                samples.append({
                    "query_id": workload["query_id"],
                    "category": workload["category"],
                    "query_name": workload["query_name"],
                    "repetition": (
                        attempt - args.warmups + 1
                    ),
                    **result,
                    "status": "ok",
                    "error": ""
                })

            except Exception as error:
                if not warmup:
                    samples.append({
                        "query_id": workload["query_id"],
                        "category": workload["category"],
                        "query_name": workload["query_name"],
                        "repetition": (
                            attempt - args.warmups + 1
                        ),
                        "latency_ms": None,
                        "rows_consumed": None,
                        "result_available_after_ms": None,
                        "result_consumed_after_ms": None,
                        "status": "error",
                        "error": str(error)[:500]
                    })

                break

        if number % 10 == 0:
            log(
                f"Benchmarked: {number}/{len(workloads)}"
            )

            save_csv(
                output / "graph_execution_samples.csv",
                samples
            )

    save_csv(
        output / "graph_execution_samples.csv",
        samples
    )

    frame = pd.DataFrame(samples)
    summaries = []

    if not frame.empty:
        for query_id, group in frame.groupby("query_id"):
            successful = group[
                group["status"] == "ok"
            ]

            values = successful[
                "latency_ms"
            ].dropna().astype(float)

            summaries.append({
                "query_id": query_id,
                "category": group["category"].iloc[0],
                "query_name": group["query_name"].iloc[0],
                "successful_runs": len(values),
                "failed_runs": int(
                    (group["status"] == "error").sum()
                ),
                "mean_ms": (
                    float(values.mean())
                    if len(values) else None
                ),
                "p50_ms": (
                    float(values.median())
                    if len(values) else None
                ),
                "p95_ms": (
                    float(values.quantile(0.95))
                    if len(values) else None
                ),
                "max_ms": (
                    float(values.max())
                    if len(values) else None
                )
            })

    save_csv(
        output / "graph_performance_summary.csv",
        summaries
    )

    report = {
        "phase": "10.4",
        "workloads": len(workloads),
        "sample_count": len(samples),
        "successful_samples": sum(
            row["status"] == "ok"
            for row in samples
        ),
        "failed_samples": sum(
            row["status"] == "error"
            for row in samples
        ),
        "notes": [
            "Latency includes client/server round trip.",
            "P95 is exploratory with few repetitions.",
            "Queries may stop at max_result_rows.",
            "Graph benchmarks cannot be directly compared "
            "with PostgreSQL EXPLAIN execution times."
        ]
    }

    save_json(
        output / "graph_benchmark_report.json",
        report
    )

    log(
        f"Successful samples: "
        f"{report['successful_samples']}"
    )

    log(
        f"Failed samples: "
        f"{report['failed_samples']}"
    )

    return report


def postgres_schema(conn):
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT
                table_schema,
                table_name,
                column_name,
                data_type
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_type IS NULL
            ORDER BY table_name, ordinal_position
        """)

        return [
            {
                "schema": row[0],
                "table": row[1],
                "column": row[2],
                "data_type": row[3]
            }
            for row in cursor.fetchall()
        ]


def postgres_table_counts(conn):
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """)

        tables = [
            row[0]
            for row in cursor.fetchall()
        ]

    results = {}

    for table in tables:
        query = (
            "SELECT count(*) FROM public."
            + sql_identifier(table)
        )

        try:
            with conn.cursor() as cursor:
                cursor.execute(query)
                results[table] = int(
                    cursor.fetchone()[0]
                )

        except Exception as error:
            results[table] = {
                "error": str(error)[:300]
            }

    return results


def phase105(args, output, graph_info, workloads):
    heading("PHASE 10.5 - CROSS-DATABASE CONSISTENCY")

    comparisons = []
    query_pairs = []

    try:
        with pg_connection(args) as conn:
            schema = postgres_schema(conn)
            counts = postgres_table_counts(conn)

        table_names = sorted(counts.keys())

        for item in graph_info["report"]["labels"]:
            label = item

            candidates = [
                table
                for table in table_names
                if table.lower() == label.lower()
            ]

            if not candidates:
                candidates = [
                    table
                    for table in table_names
                    if (
                        table.lower().replace("_", "")
                        == label.lower().replace("_", "")
                    )
                ]

            if not candidates:
                comparisons.append({
                    "graph_label": label,
                    "postgres_table": "",
                    "graph_count": "",
                    "postgres_count": "",
                    "counts_match": "",
                    "status": "NO_DIRECT_TABLE_MATCH"
                })

                continue

            table = candidates[0]

            graph_count = next(
                (
                    row["count"]
                    for row in graph_info.get(
                        "label_counts", []
                    )
                    if row["label"] == label
                ),
                None
            )

            if graph_count is None:
                graph_count = next(
                    (
                        row["count"]
                        for row in pd.read_csv(
                            output / "graph_label_counts.csv"
                        ).to_dict("records")
                        if row["label"] == label
                    ),
                    None
                )

            pg_count = counts[table]

            if isinstance(pg_count, dict):
                status = "POSTGRES_COUNT_ERROR"
                matches = None
            else:
                matches = (
                    graph_count == pg_count
                )

                status = (
                    "COUNT_MATCH"
                    if matches
                    else "COUNT_MISMATCH"
                )

            comparisons.append({
                "graph_label": label,
                "postgres_table": table,
                "graph_count": graph_count,
                "postgres_count": pg_count,
                "counts_match": matches,
                "status": status
            })

            cypher = (
                f"MATCH (n:{cypher_identifier(label)}) "
                "RETURN count(n) AS total"
            )

            sql_query = (
                "SELECT count(*) AS total FROM public."
                + sql_identifier(table)
            )

            query_pairs.append({
                "pair_id": (
                    f"EQ{len(query_pairs) + 1:03d}"
                ),
                "task": "entity_count",
                "graph_label": label,
                "postgres_table": table,
                "cypher": cypher,
                "sql": sql_query,
                "status": (
                    "COUNT_MATCH_ONLY"
                    if matches
                    else "PROPOSED_NOT_VERIFIED"
                ),
                "notes": (
                    "Count equality does not establish "
                    "entity-level equivalence."
                )
            })

        save_csv(
            output / "postgres_schema_inventory.csv",
            schema
        )

        save_csv(
            output / "postgres_table_counts.csv",
            [
                {
                    "table": table,
                    "count": count
                }
                for table, count in counts.items()
            ]
        )

        pg_status = "connected"

    except Exception as error:
        pg_status = "error"

        log(
            "PostgreSQL verification unavailable: "
            + str(error)
        )

    save_csv(
        output / "cross_database_consistency.csv",
        comparisons
    )

    save_csv(
        output / "sql_cypher_query_pairs.csv",
        query_pairs
    )

    report = {
        "phase": "10.5",
        "postgres_status": pg_status,
        "comparison_count": len(comparisons),
        "proposed_query_pairs": len(query_pairs),
        "fully_verified_equivalent_pairs": 0,
        "notes": [
            "Table and label names are matched heuristically.",
            "Entity counts are checked where possible.",
            "Count equality does not prove data equivalence.",
            "Relationship-level equivalence is pending.",
            "Only verified equivalent pairs should be used "
            "for formal Phase 11 benchmarking."
        ]
    }

    save_json(
        output / "cross_database_report.json",
        report
    )

    log(f"PostgreSQL status: {pg_status}")
    log(f"Proposed SQL/Cypher pairs: {len(query_pairs)}")

    return report


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--neo4j-uri",
        default="bolt://localhost:7687"
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
        "--pg-password",
        default=os.getenv("PGPASSWORD")
    )

    parser.add_argument(
        "--ask-pg-password",
        action="store_true"
    )

    parser.add_argument(
        "--pg-database",
        default="adaptive_healthcare_scale_10k"
    )

    parser.add_argument(
        "--schema-sample",
        type=int,
        default=200
    )

    parser.add_argument(
        "--max-workloads",
        type=int,
        default=80
    )

    parser.add_argument(
        "--warmups",
        type=int,
        default=1
    )

    parser.add_argument(
        "--repetitions",
        type=int,
        default=3
    )

    parser.add_argument(
        "--max-result-rows",
        type=int,
        default=1000
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
        parser.error(
            "Neo4j password is required."
        )

    if not args.pg_password:
        parser.error(
            "PostgreSQL password is required."
        )

    if (
        args.schema_sample < 1
        or args.max_workloads < 1
        or args.repetitions < 1
        or args.warmups < 0
        or args.max_result_rows < 1
    ):
        parser.error(
            "Invalid benchmark arguments."
        )

    heading(
        "PHASE 10 - COMPLETE NEO4J GRAPH PIPELINE"
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

    start = time.perf_counter()

    driver = graph_connection(args)

    try:
        graph_info = phase101(
            driver,
            args,
            output
        )

        phase102(
            driver,
            args,
            output,
            graph_info
        )

        workloads, plans = phase103(
            driver,
            args,
            output,
            graph_info
        )

        benchmark_report = phase104(
            driver,
            args,
            output,
            workloads
        )

        consistency_report = phase105(
            args,
            output,
            graph_info,
            workloads
        )

        report = {
            "phase": "10",
            "completed_at": datetime.now().isoformat(),
            "duration_seconds": (
                time.perf_counter() - start
            ),
            "graph_nodes": (
                graph_info["report"]["node_count"]
            ),
            "graph_relationships": (
                graph_info["report"][
                    "relationship_count"
                ]
            ),
            "workloads_generated": len(workloads),
            "explain_successes": sum(
                item["status"] == "ok"
                for item in plans
            ),
            "benchmark": benchmark_report,
            "consistency": consistency_report,
            "read_only": True,
            "original_databases_modified": False
        }

        save_json(
            output / "phase10_final_report.json",
            report
        )

        heading("PHASE 10 COMPLETE")

        log(
            json.dumps(
                json_safe(report),
                indent=2,
                default=str
            )
        )

        log("\nResults saved to:")
        log(str(output))

    finally:
        driver.close()


if __name__ == "__main__":
    main()
