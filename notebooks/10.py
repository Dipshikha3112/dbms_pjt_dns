
from pathlib import Path
from datetime import datetime
from collections import Counter
import argparse
import getpass
import json
import os
import re
import time

import joblib
import pandas as pd
import psycopg
import sqlglot

from sqlglot import exp


ROOT = Path(__file__).resolve().parent.parent

DATA = (
    ROOT / "results"
    / "phase8_2_expanded"
    / "phase82_training.csv"
)

MODEL = (
    ROOT / "results"
    / "phase8_2_expanded"
    / "model_results_final"
    / "phase82_final_sql_only.joblib"
)

RESULTS = ROOT / "results" / "phase9_optimizer"


def heading(text):
    print("\n" + "=" * 76)
    print(text)
    print("=" * 76, flush=True)


def normalize_sql(sql):
    sql = str(sql)

    sql = re.sub(
        r"/\*.*?\*/",
        " ",
        sql,
        flags=re.S
    )

    sql = re.sub(
        r"--[^\n]*",
        " ",
        sql
    )

    sql = re.sub(
        r"'(?:''|[^'])*'",
        " STR_VALUE ",
        sql
    )

    sql = re.sub(
        r"\b\d+(?:\.\d+)?\b",
        " NUM_VALUE ",
        sql
    )

    return re.sub(
        r"\s+",
        " ",
        sql
    ).strip().lower()


def extract_ast_features(sql):
    try:
        tree = sqlglot.parse_one(
            sql,
            read="postgres"
        )

        nodes = list(tree.walk())

        tables = [
            n.name.lower()
            for n in tree.find_all(exp.Table)
            if n.name
        ]

        columns = [
            n.name.lower()
            for n in tree.find_all(exp.Column)
            if n.name
        ]

        tokens = (
            ["table_" + x for x in tables]
            + ["column_" + x for x in columns]
            + [
                "node_" + type(n).__name__.lower()
                for n in nodes[:150]
            ]
        )

        return {
            "ast_text": " ".join(tokens) or "empty",
            "table_count": len(tables),
            "column_count": len(columns),
            "ast_depth": max(
                (
                    getattr(n, "depth", 0)
                    for n in nodes
                ),
                default=0
            )
        }

    except Exception:
        return {
            "ast_text": "parse_failed",
            "table_count": 0,
            "column_count": 0,
            "ast_depth": 0
        }


def prepare_model_input(sql):
    s = normalize_sql(sql)

    features = extract_ast_features(sql)

    features["sql_text"] = s
    features["sql_length"] = len(s)
    features["sql_words"] = len(s.split())

    patterns = {
        "join_count": r"\bjoin\b",
        "where_count": r"\bwhere\b",
        "group_count": r"\bgroup\s+by\b",
        "order_count": r"\border\s+by\b",
        "having_count": r"\bhaving\b",
        "subquery_count": r"\bselect\b"
    }

    for name, pattern in patterns.items():
        features[name] = len(
            re.findall(pattern, s)
        )

    features["subquery_count"] = max(
        0,
        features["subquery_count"] - 1
    )

    return pd.DataFrame([features])


def validate_read_only_sql(sql):
    try:
        statements = sqlglot.parse(
            sql,
            read="postgres"
        )

        if len(statements) != 1:
            return False

        tree = statements[0]

        if not isinstance(
            tree,
            (
                exp.Select,
                exp.Union,
                exp.Intersect,
                exp.Except
            )
        ):
            return False

        forbidden = (
            exp.Insert,
            exp.Update,
            exp.Delete,
            exp.Create,
            exp.Drop,
            exp.Alter,
            exp.Merge,
            exp.Command
        )

        if any(
            isinstance(node, forbidden)
            for node in tree.walk()
        ):
            return False

        if re.search(
            r"\bFOR\s+(UPDATE|SHARE|NO\s+KEY\s+UPDATE|KEY\s+SHARE)\b",
            sql,
            flags=re.I
        ):
            return False

        return True

    except Exception:
        return False


def load_workload(limit):
    heading("1. LOADING PHASE 8.2 WORKLOAD")

    if not DATA.exists():
        raise FileNotFoundError(DATA)

    df = pd.read_csv(
        DATA,
        low_memory=False
    )

    required = [
        "sql",
        "query_hash",
        "fingerprint_hash",
        "role_label",
        "source"
    ]

    for column in required:
        if column not in df.columns:
            raise ValueError(
                f"Missing column: {column}"
            )

    df = df.dropna(
        subset=required
    ).copy()

    df = df.drop_duplicates(
        "fingerprint_hash"
    )

    df = df.reset_index(drop=True)

    if limit > 0:
        df = df.head(limit).copy()

    print("Selected fingerprints:", len(df))
    print(
        "Role distribution:",
        df.role_label.value_counts().to_dict()
    )

    return df


def get_database_metadata(conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                n.nspname,
                c.relname,
                a.attname
            FROM pg_class c
            JOIN pg_namespace n
                ON n.oid = c.relnamespace
            JOIN pg_attribute a
                ON a.attrelid = c.oid
            WHERE c.relkind IN ('r', 'p')
              AND a.attnum > 0
              AND NOT a.attisdropped
              AND n.nspname NOT IN (
                  'pg_catalog',
                  'information_schema'
              )
        """)

        columns = {}

        for schema, table, column in cur.fetchall():
            columns.setdefault(
                (schema, table),
                set()
            ).add(column)

        cur.execute("""
            SELECT
                schemaname,
                tablename,
                indexname,
                indexdef
            FROM pg_indexes
            WHERE schemaname NOT IN (
                'pg_catalog',
                'information_schema'
            )
        """)

        indexes = {}

        for schema, table, name, definition in cur.fetchall():
            indexes.setdefault(
                (schema, table),
                []
            ).append({
                "name": name,
                "definition": definition
            })

    return columns, indexes


def get_explain_plan(conn, sql, timeout_ms):
    start = time.perf_counter()

    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute(
                "SET TRANSACTION READ ONLY"
            )

            cur.execute(
                "SELECT set_config("
                "'statement_timeout', %s, true)",
                (str(timeout_ms),)
            )

            cur.execute(
                "EXPLAIN (FORMAT JSON, "
                "VERBOSE TRUE, COSTS TRUE) "
                + sql
            )

            result = cur.fetchone()[0]

    elapsed = (
        time.perf_counter() - start
    ) * 1000

    if isinstance(result, str):
        result = json.loads(result)

    if isinstance(result, list):
        result = result[0]

    return result, elapsed


def walk_plan(node):
    yield node

    for child in node.get("Plans", []):
        yield from walk_plan(child)


def analyze_plan(plan_document):
    root = plan_document["Plan"]

    nodes = list(walk_plan(root))

    types = Counter(
        node.get("Node Type", "Unknown")
        for node in nodes
    )

    scans = []

    for node in nodes:
        node_type = node.get("Node Type", "")

        if node_type in [
            "Seq Scan",
            "Parallel Seq Scan",
            "Index Scan",
            "Index Only Scan",
            "Bitmap Heap Scan"
        ]:
            scans.append({
                "node_type": node_type,
                "schema": node.get(
                    "Schema",
                    "public"
                ),
                "table": node.get(
                    "Relation Name"
                ),
                "alias": node.get("Alias"),
                "estimated_rows": node.get(
                    "Plan Rows",
                    0
                ),
                "total_cost": node.get(
                    "Total Cost",
                    0
                ),
                "filter": node.get(
                    "Filter",
                    ""
                ),
                "index_condition": node.get(
                    "Index Cond",
                    ""
                )
            })

    return {
        "estimated_cost": root.get(
            "Total Cost",
            0
        ),
        "estimated_rows": root.get(
            "Plan Rows",
            0
        ),
        "total_nodes": len(nodes),
        "sequential_scans": (
            types["Seq Scan"]
            + types["Parallel Seq Scan"]
        ),
        "index_scans": (
            types["Index Scan"]
            + types["Index Only Scan"]
        ),
        "sort_operations": types["Sort"],
        "hash_joins": types["Hash Join"],
        "nested_loops": types["Nested Loop"],
        "node_types": dict(types),
        "scans": scans
    }


def quote_identifier(value):
    return '"' + str(value).replace(
        '"',
        '""'
    ) + '"'


def recommend_indexes(
    analysis,
    columns,
    indexes,
    min_rows
):
    recommendations = []

    for scan in analysis["scans"]:
        if scan["node_type"] not in [
            "Seq Scan",
            "Parallel Seq Scan"
        ]:
            continue

        schema = scan["schema"]
        table = scan["table"]

        if not table:
            continue

        key = (schema, table)

        if key not in columns:
            continue

        if scan["estimated_rows"] < min_rows:
            continue

        filter_text = scan["filter"]

        if not filter_text:
            continue

        candidates = re.findall(
            r'(?:"?[A-Za-z_]\w*"?\.)?'
            r'("?[\w]+"?)\s*'
            r'(?:=|>=|<=|>|<|\bIN\b|\bIS\b)',
            filter_text,
            flags=re.I
        )

        for candidate in set(candidates):
            column = candidate.strip('"')

            if column not in columns[key]:
                continue

            existing = indexes.get(
                key,
                []
            )

            # Conservative heuristic only.
            # Inspect definitions for possible existing
            # indexes before proposing a new one.
            already_indexed = any(
                re.search(
                    r"\(\s*\"?"
                    + re.escape(column)
                    + r"\"?\s*(?:,|\))",
                    index["definition"],
                    flags=re.I
                )
                for index in existing
            )

            if already_indexed:
                continue

            index_name = (
                f"idx_p9_{table}_{column}"
            )[:58]

            ddl = (
                "CREATE INDEX CONCURRENTLY "
                "IF NOT EXISTS "
                + quote_identifier(index_name)
                + " ON "
                + quote_identifier(schema)
                + "."
                + quote_identifier(table)
                + " ("
                + quote_identifier(column)
                + ");"
            )

            recommendations.append({
                "schema": schema,
                "table": table,
                "column": column,
                "scan_type": scan["node_type"],
                "estimated_rows": (
                    scan["estimated_rows"]
                ),
                "scan_cost": scan["total_cost"],
                "filter": filter_text,
                "suggested_ddl": ddl,
                "status": "NOT_EXECUTED"
            })

    return recommendations


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--db",
        default="adaptive_healthcare_scale_10k"
    )

    parser.add_argument(
        "--host",
        default="localhost"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=5432
    )

    parser.add_argument(
        "--user",
        default="postgres"
    )

    parser.add_argument(
        "--password",
        default=os.getenv("PGPASSWORD")
    )

    parser.add_argument(
        "--ask-password",
        action="store_true"
    )

    parser.add_argument(
        "--max-queries",
        type=int,
        default=70
    )

    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=10000
    )

    parser.add_argument(
        "--min-scan-rows",
        type=int,
        default=1000
    )

    args = parser.parse_args()

    if args.ask_password:
        args.password = getpass.getpass(
            "PostgreSQL password: "
        )

    if args.max_queries < 0:
        parser.error(
            "--max-queries must be >= 0"
        )

    if args.timeout_ms < 100:
        parser.error(
            "--timeout-ms must be >= 100"
        )

    heading(
        "PHASE 9 - AI-ASSISTED POSTGRESQL "
        "QUERY OPTIMIZATION ADVISOR"
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    out = (
        RESULTS
        / args.db
        / timestamp
    )

    out.mkdir(
        parents=True,
        exist_ok=True
    )

    workload = load_workload(
        args.max_queries
    )

    heading("2. LOADING PHASE 8.2 MODEL")

    model = None

    if MODEL.exists():
        model = joblib.load(MODEL)
        print("Loaded:", MODEL.name)
    else:
        print(
            "Model not found. "
            "Proceeding without predictions."
        )

    heading("3. CONNECTING TO POSTGRESQL")

    connection_args = {
        "host": args.host,
        "port": args.port,
        "dbname": args.db,
        "user": args.user,
        "autocommit": True,
        "connect_timeout": 15,
        "options": (
            "-c default_transaction_read_only=on"
        )
    }

    if args.password is not None:
        connection_args["password"] = (
            args.password
        )

    summaries = []
    all_scans = []
    all_recommendations = []
    plans = {}

    with psycopg.connect(
        **connection_args
    ) as conn:

        columns, indexes = (
            get_database_metadata(conn)
        )

        print(
            "Connected to:",
            args.db
        )

        heading(
            "4. ANALYZING POSTGRESQL QUERY PLANS"
        )

        total = len(workload)

        for i, row in enumerate(
            workload.itertuples(index=False),
            start=1
        ):
            sql = str(row.sql).strip().rstrip(";")

            fingerprint = str(
                row.fingerprint_hash
            )

            result = {
                "fingerprint_hash": fingerprint,
                "query_hash": str(
                    row.query_hash
                ),
                "role_label": str(
                    row.role_label
                ),
                "source": str(row.source),
                "predicted_role": None,
                "status": "skipped",
                "error": "",
                "explain_wall_ms": None,
                "estimated_cost": None,
                "estimated_rows": None,
                "total_nodes": None,
                "sequential_scans": None,
                "index_scans": None,
                "sort_operations": None,
                "hash_joins": None,
                "nested_loops": None,
                "recommendation_count": 0
            }

            if model is not None:
                try:
                    model_input = (
                        prepare_model_input(sql)
                    )

                    prediction = model.predict(
                        model_input
                    )[0]

                    result["predicted_role"] = (
                        str(prediction)
                    )

                except Exception as error:
                    result["error"] = (
                        "Prediction error: "
                        + str(error)[:150]
                    )

            if not validate_read_only_sql(sql):
                result["error"] = (
                    "Unsafe or unsupported SQL"
                )

                summaries.append(result)
                continue

            try:
                plan, elapsed = get_explain_plan(
                    conn,
                    sql,
                    args.timeout_ms
                )

                analysis = analyze_plan(plan)

                recommendations = (
                    recommend_indexes(
                        analysis,
                        columns,
                        indexes,
                        args.min_scan_rows
                    )
                )

                result.update({
                    "status": "ok",
                    "explain_wall_ms": round(
                        elapsed,
                        3
                    ),
                    "estimated_cost": (
                        analysis["estimated_cost"]
                    ),
                    "estimated_rows": (
                        analysis["estimated_rows"]
                    ),
                    "total_nodes": (
                        analysis["total_nodes"]
                    ),
                    "sequential_scans": (
                        analysis["sequential_scans"]
                    ),
                    "index_scans": (
                        analysis["index_scans"]
                    ),
                    "sort_operations": (
                        analysis["sort_operations"]
                    ),
                    "hash_joins": (
                        analysis["hash_joins"]
                    ),
                    "nested_loops": (
                        analysis["nested_loops"]
                    ),
                    "recommendation_count": (
                        len(recommendations)
                    )
                })

                plans[fingerprint] = plan

                for scan in analysis["scans"]:
                    all_scans.append({
                        "fingerprint_hash": fingerprint,
                        **scan
                    })

                for recommendation in recommendations:
                    all_recommendations.append({
                        "fingerprint_hash": fingerprint,
                        "role_label": str(
                            row.role_label
                        ),
                        **recommendation
                    })

            except Exception as error:
                result["status"] = "error"
                result["error"] = str(error)[:350]

            summaries.append(result)

            if i % 10 == 0 or i == total:
                successful = sum(
                    r["status"] == "ok"
                    for r in summaries
                )

                print(
                    f"Phase 9: {i}/{total} | "
                    f"Successful plans: {successful} | "
                    f"Index hypotheses: "
                    f"{len(all_recommendations)}",
                    flush=True
                )

    heading("5. GENERATING PHASE 9 REPORTS")

    summary_df = pd.DataFrame(
        summaries
    )

    summary_df.to_csv(
        out / "query_plan_summary.csv",
        index=False
    )

    scan_columns = [
        "fingerprint_hash",
        "node_type",
        "schema",
        "table",
        "alias",
        "estimated_rows",
        "total_cost",
        "filter",
        "index_condition"
    ]

    pd.DataFrame(
        all_scans,
        columns=scan_columns
    ).to_csv(
        out / "scan_diagnostics.csv",
        index=False
    )

    advice_columns = [
        "fingerprint_hash",
        "role_label",
        "schema",
        "table",
        "column",
        "scan_type",
        "estimated_rows",
        "scan_cost",
        "filter",
        "suggested_ddl",
        "status"
    ]

    advice_df = pd.DataFrame(
        all_recommendations,
        columns=advice_columns
    )

    advice_df.to_csv(
        out / "index_advice_per_query.csv",
        index=False
    )

    if not advice_df.empty:
        ranked = (
            advice_df.groupby(
                [
                    "schema",
                    "table",
                    "column",
                    "suggested_ddl"
                ],
                as_index=False
            )
            .agg(
                affected_fingerprints=(
                    "fingerprint_hash",
                    "nunique"
                ),
                maximum_scan_cost=(
                    "scan_cost",
                    "max"
                )
            )
            .sort_values(
                [
                    "affected_fingerprints",
                    "maximum_scan_cost"
                ],
                ascending=False
            )
        )

    else:
        ranked = pd.DataFrame(
            columns=[
                "schema",
                "table",
                "column",
                "suggested_ddl",
                "affected_fingerprints",
                "maximum_scan_cost"
            ]
        )

    ranked.to_csv(
        out / "ranked_index_hypotheses.csv",
        index=False
    )

    (
        out / "explain_plans.json"
    ).write_text(
        json.dumps(
            plans,
            indent=2,
            default=str
        ),
        encoding="utf-8"
    )

    successful = summary_df[
        summary_df["status"] == "ok"
    ]

    report = {
        "phase": "9",
        "database": args.db,
        "created_at": datetime.now().isoformat(),
        "selected_fingerprints": len(workload),
        "successful_explain": len(successful),
        "failed_or_skipped": (
            len(workload) - len(successful)
        ),
        "unique_index_hypotheses": (
            len(ranked)
        ),
        "total_recommendation_links": (
            len(advice_df)
        ),
        "model": (
            str(MODEL)
            if model is not None
            else None
        ),
        "safety": (
            "Read-only EXPLAIN without ANALYZE. "
            "No indexes created."
        ),
        "limitations": [
            "Planner costs are not milliseconds.",
            "EXPLAIN wall time is not query latency.",
            "Index recommendations are untested.",
            "Sequential scans can be optimal.",
            "Only one instance per fingerprint "
            "is analyzed.",
            "No actual performance improvement "
            "has yet been demonstrated.",
            "The classifier does not alter "
            "the PostgreSQL execution plan."
        ]
    }

    (
        out / "phase9_report.json"
    ).write_text(
        json.dumps(
            report,
            indent=2
        ),
        encoding="utf-8"
    )

    heading("PHASE 9 PROFILING COMPLETE")

    print(
        json.dumps(
            report,
            indent=2
        )
    )

    print("\nSaved results to:")
    print(out)

    print(
        "\nNo indexes were created. "
        "No schema changes were performed."
    )


if __name__ == "__main__":
    main()
