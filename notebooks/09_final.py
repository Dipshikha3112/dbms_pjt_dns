
import argparse
import csv
import getpass
import hashlib
import importlib.util
import json
import math
import random
import re
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg
from psycopg import sql


PRIMARY_DB = "adaptive_healthcare_phase9_benchmark"

PORTFOLIOS = {
    "composite_planner": [
        ("patient_procedure", ("patient_id", "procedure_id")),
        ("claim", ("payer_id", "patient_id")),
        ("lab_result", ("loinc_code", "patient_id")),
    ],
    "planner_gain": [
        ("patient_allergy", ("patient_id",)),
        ("patient_procedure", ("patient_id", "procedure_id")),
        ("patient_procedure", ("patient_id",)),
    ],
    "index_usage": [
        ("lab_result", ("patient_id", "result_time DESC")),
        ("lab_result", ("patient_id", "loinc_code")),
        ("encounter", ("patient_id", "provider_id")),
    ],
    "no_new_indexes": [],
}

PROTECTED = {
    "adaptive_healthcare_db",
    "adaptive_healthcare_scale_10k",
    "adaptive_healthcare_scale_50k",
}


def digest(value):
    return hashlib.sha256(
        str(value).encode("utf-8")
    ).hexdigest()[:16]


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def save_csv(path, rows):
    rows = list(rows)
    columns = list(dict.fromkeys(
        k for row in rows for k in row
    )) or ["status"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, obj):
    Path(path).write_text(
        json.dumps(obj, indent=2, default=str),
        encoding="utf-8"
    )


def safe_sql(query):
    query = str(query or "").strip().rstrip(";").strip()

    if not re.match(r"^SELECT\s", query, re.I):
        return False

    if ";" in query or "--" in query or "/*" in query:
        return False

    blocked = (
        r"\b(INSERT|UPDATE|DELETE|MERGE|CREATE|ALTER|DROP|"
        r"TRUNCATE|COPY|CALL|DO|EXECUTE|GRANT|REVOKE|"
        r"LOCK|SET|RESET|INTO|PG_SLEEP|NEXTVAL|SETVAL|"
        r"DBLINK|LO_IMPORT|LO_EXPORT)\b"
        r"|\bFOR\s+(UPDATE|SHARE)\b"
    )

    return not bool(re.search(blocked, query, re.I))


def load_workload(path, limit):
    df = pd.read_csv(path)

    if "query_text" not in df.columns:
        raise RuntimeError(
            "Workload CSV must contain query_text"
        )

    unique = {}

    for _, row in df.iterrows():
        query = str(row["query_text"]).strip().rstrip(";")

        if not safe_sql(query):
            continue

        if "safe" in df.columns:
            value = str(row["safe"]).lower()

            if value not in ("true", "1", "yes"):
                continue

        key = digest(query)

        if key not in unique:
            unique[key] = {
                "query_hash": key,
                "query_text": query,
                "frequency": 0,
                "query_family": str(
                    row.get("query_family", "")
                ),
                "query_name": str(
                    row.get("query_name", "")
                ),
            }

        unique[key]["frequency"] += 1

    workload = list(unique.values())

    if limit:
        workload = workload[:limit]

    if not workload:
        raise RuntimeError("No valid workload queries")

    return workload


def verify(conn, expected):
    if (
        expected in PROTECTED
        or not expected.startswith(
            "adaptive_healthcare_phase9_benchmark"
        )
    ):
        raise RuntimeError(
            "Protected or unauthorized database"
        )

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT current_database(),
                   current_setting('port')::integer
            """
        )

        database, internal_port = cur.fetchone()

    if database != expected or internal_port != 5432:
        raise RuntimeError(
            "Unexpected PostgreSQL target"
        )


def connect(args, database, password):
    if args.host not in ("localhost", "127.0.0.1"):
        raise RuntimeError("Remote hosts are prohibited")

    if args.port != 5433:
        raise RuntimeError(
            "Only Docker port 5433 is authorized"
        )

    conn = psycopg.connect(
        host=args.host,
        port=args.port,
        dbname=database,
        user=args.user,
        password=password,
        connect_timeout=10,
        autocommit=False,
    )

    verify(conn, database)

    return conn


def index_definition(table, columns, name):
    expressions = []

    for column in columns:
        match = re.fullmatch(
            r"([A-Za-z_]\w*)(?:\s+(ASC|DESC))?",
            column,
            re.I
        )

        if not match:
            raise ValueError(
                f"Invalid index column: {column}"
            )

        expression = sql.Identifier(match.group(1))

        if match.group(2):
            expression += sql.SQL(
                " " + match.group(2).upper()
            )

        expressions.append(expression)

    return sql.SQL(
        "CREATE INDEX {} ON public.{} ({})"
    ).format(
        sql.Identifier(name),
        sql.Identifier(table),
        sql.SQL(", ").join(expressions)
    )


def get_existing_indexes(conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT tablename, indexname, indexdef
            FROM pg_indexes
            WHERE schemaname = 'public'
            """
        )

        return [
            {
                "table": row[0],
                "name": row[1],
                "definition": row[2],
            }
            for row in cur.fetchall()
        ]


def explain_analyze(conn, query, timeout_ms):
    with conn.cursor() as cur:
        cur.execute("SAVEPOINT phase9_query")

        try:
            cur.execute(
                "SELECT set_config('statement_timeout', %s, true)",
                (f"{timeout_ms}ms",)
            )

            cur.execute(
                "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) "
                + query
            )

            result = cur.fetchone()[0]

            if isinstance(result, str):
                result = json.loads(result)

            report = result[0]

            cur.execute(
                "RELEASE SAVEPOINT phase9_query"
            )

            return {
                "execution_ms": float(
                    report["Execution Time"]
                ),
                "planning_ms": float(
                    report["Planning Time"]
                ),
                "total_cost": float(
                    report["Plan"]["Total Cost"]
                ),
                "plan": report["Plan"],
            }

        except Exception:
            cur.execute(
                "ROLLBACK TO SAVEPOINT phase9_query"
            )
            cur.execute(
                "RELEASE SAVEPOINT phase9_query"
            )
            raise


def relevant_workload(workload, portfolio, limit):
    if not portfolio:
        return workload[:limit]

    tables = {table for table, _ in portfolio}

    selected = []

    for record in workload:
        query = record["query_text"]

        for table in tables:
            pattern = (
                r"\b(?:FROM|JOIN)\s+"
                r"(?:public\.)?\"?"
                + re.escape(table)
                + r"\"?\b"
            )

            if re.search(pattern, query, re.I):
                selected.append(record)
                break

    return selected[:limit]


def create_portfolio(conn, portfolio, run_tag):
    created = []

    with conn.cursor() as cur:
        for position, (table, columns) in enumerate(
            portfolio, 1
        ):
            name = (
                f"p9_{run_tag}_{position}_"
                f"{digest((table, columns))[:7]}"
            )[:63]

            cur.execute(
                index_definition(table, columns, name)
            )

            created.append(name)

    return created


def physical_portfolio_benchmark(
    conn,
    workload,
    portfolio_name,
    portfolio,
    args,
    database,
):
    queries = relevant_workload(
        workload,
        portfolio,
        args.physical_queries
    )

    rows = []

    rng = random.Random(args.seed)

    for round_number in range(args.rounds):
        order = ["baseline", "indexed"]

        if round_number % 2:
            order.reverse()

        for condition in order:
            try:
                with conn.cursor() as cur:
                    cur.execute("BEGIN")

                    cur.execute(
                        "SELECT set_config('statement_timeout', %s, true)",
                        (f"{args.timeout_ms}ms",)
                    )

                if condition == "indexed" and portfolio:
                    create_portfolio(
                        conn,
                        portfolio,
                        digest(
                            (
                                database,
                                portfolio_name,
                                round_number
                            )
                        )[:8]
                    )

                query_order = list(queries)
                rng.shuffle(query_order)

                for item in query_order:
                    for iteration in range(
                        args.warmups + args.repeats
                    ):
                        try:
                            result = explain_analyze(
                                conn,
                                item["query_text"],
                                args.timeout_ms
                            )

                            if iteration < args.warmups:
                                continue

                            rows.append({
                                "database": database,
                                "portfolio": portfolio_name,
                                "round": round_number,
                                "condition": condition,
                                "query_hash": item["query_hash"],
                                "frequency": item["frequency"],
                                "iteration": iteration - args.warmups,
                                "execution_ms": result["execution_ms"],
                                "planning_ms": result["planning_ms"],
                                "planner_cost": result["total_cost"],
                                "status": "ok",
                            })

                        except Exception as exc:
                            rows.append({
                                "database": database,
                                "portfolio": portfolio_name,
                                "round": round_number,
                                "condition": condition,
                                "query_hash": item["query_hash"],
                                "iteration": iteration,
                                "status": "error",
                                "error": str(exc)[:400],
                            })

            finally:
                conn.rollback()

    return rows


def physical_summary(records):
    groups = defaultdict(list)

    for record in records:
        if record.get("status") == "ok":
            key = (
                record["database"],
                record["portfolio"],
                record["query_hash"],
                record["condition"],
            )

            groups[key].append(
                float(record["execution_ms"])
            )

    medians = {
        key: statistics.median(values)
        for key, values in groups.items()
    }

    results = []

    keys = {
        (db, portfolio, query)
        for db, portfolio, query, condition
        in medians
    }

    for db, portfolio, query in sorted(keys):
        baseline = medians.get(
            (db, portfolio, query, "baseline")
        )

        indexed = medians.get(
            (db, portfolio, query, "indexed")
        )

        if baseline is None or indexed is None:
            continue

        results.append({
            "database": db,
            "portfolio": portfolio,
            "query_hash": query,
            "baseline_median_ms": baseline,
            "indexed_median_ms": indexed,
            "gain_pct": (
                100 * (baseline - indexed) / baseline
                if baseline > 0 else None
            )
        })

    return results


def write_overhead_experiment(
    conn,
    portfolio_name,
    portfolio,
    args,
    database
):
    results = []

    for table, columns in portfolio:
        for condition in ("baseline", "indexed"):
            for repeat in range(args.write_repeats):
                try:
                    with conn.cursor() as cur:
                        cur.execute("BEGIN")

                        temp_name = (
                            "p9_write_"
                            + digest(
                                (
                                    database,
                                    portfolio_name,
                                    table,
                                    condition,
                                    repeat
                                )
                            )
                        )[:25]

                        cur.execute(
                            sql.SQL(
                                "CREATE TEMP TABLE {} ON COMMIT DROP "
                                "AS SELECT * FROM public.{} LIMIT {}"
                            ).format(
                                sql.Identifier(temp_name),
                                sql.Identifier(table),
                                sql.Literal(args.write_seed_rows)
                            )
                        )

                        if condition == "indexed":
                            name = (
                                "p9_temp_idx_"
                                + digest(
                                    (
                                        table,
                                        columns,
                                        repeat
                                    )
                                )
                            )[:20]

                            cur.execute(
                                sql.SQL(
                                    "CREATE INDEX {} ON {} ({})"
                                ).format(
                                    sql.Identifier(name),
                                    sql.Identifier(temp_name),
                                    sql.SQL(", ").join(
                                        sql.SQL("{} {}").format(
                                            sql.Identifier(
                                                column.split()[0]
                                            ),
                                            sql.SQL(
                                                "DESC"
                                                if column.upper().endswith(
                                                    "DESC"
                                                )
                                                else "ASC"
                                            )
                                        )
                                        for column in columns
                                    )
                                )
                            )

                        start = time.perf_counter()

                        cur.execute(
                            sql.SQL(
                                "INSERT INTO {} "
                                "SELECT * FROM public.{} LIMIT {}"
                            ).format(
                                sql.Identifier(temp_name),
                                sql.Identifier(table),
                                sql.Literal(args.write_insert_rows)
                            )
                        )

                        inserted = cur.rowcount

                        elapsed_ms = (
                            time.perf_counter() - start
                        ) * 1000

                        cur.execute(
                            sql.SQL(
                                "SELECT pg_total_relation_size(%s::regclass)"
                            ),
                            (temp_name,)
                        )

                        storage_bytes = cur.fetchone()[0]

                        results.append({
                            "database": database,
                            "portfolio": portfolio_name,
                            "table": table,
                            "condition": condition,
                            "repeat": repeat,
                            "inserted_rows": inserted,
                            "insert_ms": elapsed_ms,
                            "temporary_table_total_bytes": storage_bytes,
                            "status": "ok",
                        })

                except Exception as exc:
                    results.append({
                        "database": database,
                        "portfolio": portfolio_name,
                        "table": table,
                        "condition": condition,
                        "repeat": repeat,
                        "status": "error",
                        "error": str(exc)[:400],
                    })

                finally:
                    conn.rollback()

    return results


def write_summary(records):
    groups = defaultdict(list)

    for row in records:
        if row["status"] == "ok":
            groups[
                (
                    row["database"],
                    row["portfolio"],
                    row["table"],
                    row["condition"]
                )
            ].append(float(row["insert_ms"]))

    output = []

    keys = {
        (db, portfolio, table)
        for db, portfolio, table, condition
        in groups
    }

    for db, portfolio, table in sorted(keys):
        baseline = groups.get(
            (db, portfolio, table, "baseline"), []
        )

        indexed = groups.get(
            (db, portfolio, table, "indexed"), []
        )

        if not baseline or not indexed:
            continue

        before = statistics.median(baseline)
        after = statistics.median(indexed)

        output.append({
            "database": db,
            "portfolio": portfolio,
            "table": table,
            "baseline_insert_ms": before,
            "indexed_insert_ms": after,
            "insert_overhead_pct": (
                100 * (after - before) / before
                if before > 0 else None
            ),
        })

    return output


def inspect_v6_bundle(model_path):
    bundle = joblib.load(model_path)

    info = {
        "bundle_type": type(bundle).__name__,
        "status": "inspection_only",
        "models": [],
    }

    if isinstance(bundle, dict):
        info["bundle_keys"] = list(bundle.keys())

        for key, value in bundle.items():
            if hasattr(value, "predict"):
                info["models"].append({
                    "key": key,
                    "type": type(value).__name__,
                    "n_features_in": getattr(
                        value, "n_features_in_", None
                    ),
                    "feature_names_in": (
                        list(value.feature_names_in_)
                        if hasattr(value, "feature_names_in_")
                        else None
                    )
                })

    elif hasattr(bundle, "predict"):
        info["models"].append({
            "key": "root",
            "type": type(bundle).__name__,
            "n_features_in": getattr(
                bundle, "n_features_in_", None
            ),
            "feature_names_in": (
                list(bundle.feature_names_in_)
                if hasattr(bundle, "feature_names_in_")
                else None
            )
        })

    return info


def ai_ablation(
    args,
    output,
    physical_results,
    portfolio_results
):
    report = {
        "status": "blocked",
        "reason": "",
        "model_bundle": args.v6_bundle,
        "ai_ranking_executed": False,
        "causal_ai_gain_claimed": False,
    }

    if not args.v6_bundle:
        report["reason"] = "No V6 bundle supplied"
        return report

    path = Path(args.v6_bundle)

    if not path.is_file():
        report["reason"] = "V6 bundle not found"
        return report

    try:
        inspection = inspect_v6_bundle(path)
        save_json(
            output / "v6_model_inspection.json",
            inspection
        )

    except Exception as exc:
        report["reason"] = (
            "V6 bundle loading failed: "
            + str(exc)[:300]
        )
        return report

    report["model_inspected"] = True

    if not args.v6_feature_contract:
        report["reason"] = (
            "Exact V6 feature extraction and target "
            "transformation contract not supplied. "
            "Cannot scientifically compare AI rankings."
        )
        return report

    contract_path = Path(args.v6_feature_contract)

    if not contract_path.is_file():
        report["reason"] = (
            "Feature contract file does not exist"
        )
        return report

    contract = json.loads(
        contract_path.read_text(encoding="utf-8")
    )

    required = {
        "feature_names",
        "classifier_key",
        "regressor_key",
        "target_transform",
        "training_feature_version",
    }

    missing = required - set(contract)

    if missing:
        report["reason"] = (
            "Incomplete feature contract: "
            + ", ".join(sorted(missing))
        )
        return report

    if len(contract["feature_names"]) != 41:
        report["reason"] = (
            "Expected 41 V6 features; received "
            + str(len(contract["feature_names"]))
        )
        return report

    report["status"] = "contract_present_not_validated"
    report["reason"] = (
        "A feature-name manifest alone cannot establish "
        "training/inference parity. The original V6 feature "
        "extractor and target transformation must be tested "
        "against known training examples before AI ranking."
    )

    return report


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5433)
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password")

    parser.add_argument(
        "--databases",
        default=PRIMARY_DB
    )

    parser.add_argument("--workload", required=True)

    parser.add_argument(
        "--v6-bundle",
        default=""
    )

    parser.add_argument(
        "--v6-feature-contract",
        default=""
    )

    parser.add_argument(
        "--output",
        default="results/phase9_final"
    )

    parser.add_argument(
        "--physical-queries",
        type=int,
        default=12
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=4
    )

    parser.add_argument(
        "--warmups",
        type=int,
        default=2
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=8
    )

    parser.add_argument(
        "--write-repeats",
        type=int,
        default=5
    )

    parser.add_argument(
        "--write-seed-rows",
        type=int,
        default=1000
    )

    parser.add_argument(
        "--write-insert-rows",
        type=int,
        default=500
    )

    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=15000
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--confirm-physical-indexes",
        action="store_true"
    )

    args = parser.parse_args()

    databases = [
        value.strip()
        for value in args.databases.split(",")
        if value.strip()
    ]

    if not databases:
        raise RuntimeError("No databases specified")

    for database in databases:
        if (
            database in PROTECTED
            or not database.startswith(
                "adaptive_healthcare_phase9_benchmark"
            )
        ):
            raise RuntimeError(
                "Protected database in scale list: "
                + database
            )

    if not args.confirm_physical_indexes:
        raise RuntimeError(
            "Physical benchmarking requires "
            "--confirm-physical-indexes"
        )

    if min(
        args.physical_queries,
        args.rounds,
        args.repeats,
        args.write_repeats,
        args.write_seed_rows,
        args.write_insert_rows,
        args.timeout_ms
    ) < 1:
        raise ValueError("Experiment parameters must be positive")

    password = (
        args.password
        if args.password is not None
        else getpass.getpass(
            "Docker PostgreSQL password: "
        )
    )

    output = (
        Path(args.output)
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    output.mkdir(parents=True, exist_ok=True)

    workload = load_workload(
        args.workload,
        limit=None
    )

    physical_records = []
    write_records = []
    scale_records = []
    index_inventories = []

    for database in databases:
        print(
            "\nDATABASE:",
            database,
            flush=True
        )

        conn = connect(
            args,
            database,
            password
        )

        try:
            existing = get_existing_indexes(conn)

            for item in existing:
                index_inventories.append({
                    "database": database,
                    **item
                })

            conn.rollback()

            for name, portfolio in PORTFOLIOS.items():
                print(
                    "Benchmarking portfolio:",
                    name,
                    flush=True
                )

                if name == "no_new_indexes":
                    continue

                results = physical_portfolio_benchmark(
                    conn,
                    workload,
                    name,
                    portfolio,
                    args,
                    database
                )

                physical_records.extend(results)

                writes = write_overhead_experiment(
                    conn,
                    name,
                    portfolio,
                    args,
                    database
                )

                write_records.extend(writes)

            scale_records.append({
                "database": database,
                "status": "completed",
                "workload_sql_variants": len(workload),
            })

        except Exception as exc:
            scale_records.append({
                "database": database,
                "status": "failed",
                "error": str(exc)[:500]
            })

        finally:
            conn.rollback()
            conn.close()

    physical_aggregates = physical_summary(
        physical_records
    )

    write_aggregates = write_summary(
        write_records
    )

    save_csv(
        output / "existing_indexes.csv",
        index_inventories
    )

    save_csv(
        output / "physical_query_runs.csv",
        physical_records
    )

    save_csv(
        output / "physical_query_summary.csv",
        physical_aggregates
    )

    save_csv(
        output / "write_overhead_runs.csv",
        write_records
    )

    save_csv(
        output / "write_overhead_summary.csv",
        write_aggregates
    )

    save_csv(
        output / "scale_results.csv",
        scale_records
    )

    ai_report = ai_ablation(
        args,
        output,
        physical_records,
        physical_aggregates
    )

    save_json(
        output / "v6_ablation_status.json",
        ai_report
    )

    report = {
        "project": "Adaptive Healthcare Knowledge Database",
        "phase": "Phase 9 Integrated Validation",
        "timestamp": timestamp(),
        "databases_requested": databases,
        "databases_completed": sum(
            row["status"] == "completed"
            for row in scale_records
        ),
        "physical_query_executions": sum(
            row["status"] == "ok"
            for row in physical_records
        ),
        "physical_query_failures": sum(
            row["status"] != "ok"
            for row in physical_records
        ),
        "write_experiments": len(write_records),
        "ai_ablation_status": ai_report["status"],
        "multi_scale_independent": (
            len(databases) > 1
            and all(
                row["status"] == "completed"
                for row in scale_records
            )
        ),
        "automatic_deployment": False,
        "physical_indexes_committed": False,
        "limitations": [
            "V6 ablation requires verified feature parity.",
            "Physical query selection is table-relevance based.",
            "Physical timings are sensitive to caching.",
            "Write overhead is measured on temporary copies.",
            "Write tests measure inserts, not updates or deletes.",
            "Temporary tables do not reproduce every production constraint.",
            "Multi-scale validation requires separately prepared benchmark databases.",
            "No clinical accuracy claim is made.",
        ]
    }

    save_json(
        output / "final_report.json",
        report
    )

    print("\n" + "=" * 75)
    print("PHASE 9 INTEGRATED EXPERIMENT COMPLETE")
    print("=" * 75)

    for key, value in report.items():
        if key != "limitations":
            print(f"{key}: {value}")

    print("\nResults:", output)
    print("Physical index changes: ROLLBACK ONLY")
    print("Automatic deployment: DISABLED")


if __name__ == "__main__":
    main()
