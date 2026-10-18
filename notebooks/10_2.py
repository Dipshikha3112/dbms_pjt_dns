
from pathlib import Path
from datetime import datetime
import argparse
import getpass
import json
import os
import re
import time

import numpy as np
import pandas as pd
import psycopg
import sqlglot

from psycopg import sql as pgsql
from sqlglot import exp


ROOT = Path(__file__).resolve().parent.parent

BENCH_DB = "adaptive_healthcare_phase9_benchmark"
SOURCE_DB = "adaptive_healthcare_scale_10k"

PHASE9A_ROOT = (
    ROOT / "results" / "phase9_optimizer" / SOURCE_DB
)
PHASE9B_ROOT = ROOT / "results" / "phase9b_benchmark"
WORKLOAD_PATH = (
    ROOT / "results" / "phase8_2_expanded"
    / "phase82_training.csv"
)
OUTPUT_ROOT = ROOT / "results" / "phase9c_validation"

INDEX_PREFIX = "p9c_exp_"


def log(message):
    print(message, flush=True)


def heading(message):
    log("\n" + "=" * 72)
    log(message)
    log("=" * 72)


def connect(args):
    options = {
        "host": args.host,
        "port": args.port,
        "user": args.user,
        "dbname": BENCH_DB,
        "autocommit": True,
        "connect_timeout": 20
    }

    if args.password is not None:
        options["password"] = args.password

    return psycopg.connect(**options)


def latest_folder(root, required_file):
    if not root.exists():
        raise FileNotFoundError(root)

    folders = sorted(
        folder
        for folder in root.iterdir()
        if folder.is_dir()
        and (folder / required_file).exists()
    )

    if not folders:
        raise FileNotFoundError(
            f"No {required_file} found in {root}"
        )

    return folders[-1]


def verify_database(conn):
    with conn.cursor() as cursor:
        cursor.execute("SELECT current_database()")

        if cursor.fetchone()[0] != BENCH_DB:
            raise RuntimeError(
                "Unsafe database target."
            )

        cursor.execute("""
            SELECT
                to_regclass('public.patient'),
                to_regclass('public.encounter'),
                to_regclass('public.diagnosis')
        """)

        if any(
            value is None
            for value in cursor.fetchone()
        ):
            raise RuntimeError(
                "Benchmark database is missing tables."
            )

        cursor.execute("""
            SELECT schemaname, indexname
            FROM pg_indexes
            WHERE indexname LIKE 'p9c_exp_%'
        """)

        leftovers = cursor.fetchall()

        if leftovers:
            raise RuntimeError(
                "Leftover Phase 9C indexes found: "
                f"{leftovers}. Inspect them before retrying."
            )

    log("Database verified: " + BENCH_DB)
    log("No leftover Phase 9C indexes.")


def load_inputs(args):
    heading("1. LOADING PREVIOUS EXPERIMENTS")

    folder_a = (
        Path(args.phase9a_dir)
        if args.phase9a_dir
        else latest_folder(
            PHASE9A_ROOT,
            "index_advice_per_query.csv"
        )
    )

    folder_b = (
        Path(args.phase9b_dir)
        if args.phase9b_dir
        else latest_folder(
            PHASE9B_ROOT,
            "index_benchmark_ranking.csv"
        )
    )

    ranking = pd.read_csv(
        folder_b / "index_benchmark_ranking.csv"
    ).fillna("")

    advice = pd.read_csv(
        folder_a / "index_advice_per_query.csv",
        dtype=str
    ).fillna("")

    workload = pd.read_csv(
        WORKLOAD_PATH,
        dtype=str,
        low_memory=False
    ).fillna("")

    if not args.all_candidates:
        if "queries_using_index" not in ranking:
            raise ValueError(
                "Missing queries_using_index in Phase 9B."
            )

        used = pd.to_numeric(
            ranking["queries_using_index"],
            errors="coerce"
        ).fillna(0)

        ranking = ranking[used > 0]

    ranking = ranking.head(args.max_candidates)

    if ranking.empty:
        raise RuntimeError(
            "No candidates selected."
        )

    log(f"Phase 9A: {folder_a}")
    log(f"Phase 9B: {folder_b}")
    log(f"Selected candidates: {len(ranking)}")

    return ranking, advice, workload


def valid_sql(statement):
    try:
        parsed = sqlglot.parse(
            statement,
            read="postgres"
        )

        if len(parsed) != 1:
            return False

        tree = parsed[0]

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
            r"\bFOR\s+(UPDATE|SHARE|NO\s+KEY\s+UPDATE)\b",
            statement,
            re.I
        ):
            return False

        return True

    except Exception:
        return False


def matching_queries(candidate, advice, workload, limit):
    matching = advice[
        (advice["schema"] == str(candidate["schema"]))
        & (advice["table"] == str(candidate["table"]))
        & (advice["column"] == str(candidate["column"]))
    ]

    fingerprints = set(
        matching["fingerprint_hash"].astype(str)
    )

    selected = workload[
        workload["fingerprint_hash"].astype(str).isin(
            fingerprints
        )
    ].drop_duplicates("fingerprint_hash")

    queries = []

    for _, row in selected.iterrows():
        statement = str(row["sql"]).strip().rstrip(";")

        if not valid_sql(statement):
            continue

        queries.append({
            "fingerprint_hash": str(
                row["fingerprint_hash"]
            ),
            "role_label": str(
                row.get("role_label", "")
            ),
            "sql": statement
        })

        if len(queries) >= limit:
            break

    return queries


def already_indexed(conn, schema, table, column):
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT 1
            FROM pg_index i
            JOIN pg_class t ON t.oid = i.indrelid
            JOIN pg_namespace n
                ON n.oid = t.relnamespace
            JOIN pg_class ix ON ix.oid = i.indexrelid
            JOIN pg_am am ON am.oid = ix.relam
            JOIN pg_attribute a
                ON a.attrelid = t.oid
               AND a.attnum = i.indkey[0]
            WHERE n.nspname = %s
              AND t.relname = %s
              AND a.attname = %s
              AND am.amname = 'btree'
              AND i.indisvalid
              AND i.indisready
              AND i.indpred IS NULL
              AND i.indexprs IS NULL
            LIMIT 1
        """, (schema, table, column))

        return cursor.fetchone() is not None


def explain_analyze(conn, statement, timeout_ms):
    with conn.transaction():
        with conn.cursor() as cursor:
            cursor.execute(
                "SET TRANSACTION READ ONLY"
            )

            cursor.execute(
                "SELECT set_config('statement_timeout', %s, true)",
                (str(timeout_ms),)
            )

            cursor.execute(
                "SELECT set_config('lock_timeout', '2000', true)"
            )

            cursor.execute(
                "EXPLAIN (ANALYZE, BUFFERS, "
                "FORMAT JSON, TIMING OFF) "
                + statement
            )

            document = cursor.fetchone()[0]

    if isinstance(document, str):
        document = json.loads(document)

    if isinstance(document, list):
        document = document[0]

    plan = document["Plan"]
    indexes = set()

    def traverse(node):
        if node.get("Index Name"):
            indexes.add(node["Index Name"])

        for child in node.get("Plans", []):
            traverse(child)

    traverse(plan)

    return {
        "execution_ms": float(
            document["Execution Time"]
        ),
        "planning_ms": float(
            document.get("Planning Time", 0)
        ),
        "shared_hit": int(
            plan.get("Shared Hit Blocks", 0)
        ),
        "shared_read": int(
            plan.get("Shared Read Blocks", 0)
        ),
        "temp_read": int(
            plan.get("Temp Read Blocks", 0)
        ),
        "temp_written": int(
            plan.get("Temp Written Blocks", 0)
        ),
        "indexes_used": sorted(indexes)
    }


def measure(
    conn,
    queries,
    args,
    candidate_id,
    cycle,
    position,
    stage,
    index_name
):
    rows = []

    for query in queries:
        total = args.warmups + args.repetitions

        for attempt in range(total):
            if attempt < args.warmups:
                is_warmup = True
            else:
                is_warmup = False

            try:
                result = explain_analyze(
                    conn,
                    query["sql"],
                    args.timeout_ms
                )

                if is_warmup:
                    continue

                used = (
                    index_name
                    in result["indexes_used"]
                )

                rows.append({
                    "candidate_id": candidate_id,
                    "fingerprint_hash": (
                        query["fingerprint_hash"]
                    ),
                    "role_label": query["role_label"],
                    "cycle": cycle,
                    "position": position,
                    "stage": stage,
                    "repetition": (
                        attempt - args.warmups + 1
                    ),
                    "execution_ms": (
                        result["execution_ms"]
                    ),
                    "planning_ms": (
                        result["planning_ms"]
                    ),
                    "shared_hit": result["shared_hit"],
                    "shared_read": result["shared_read"],
                    "temp_read": result["temp_read"],
                    "temp_written": (
                        result["temp_written"]
                    ),
                    "experimental_index_used": used,
                    "indexes_used": json.dumps(
                        result["indexes_used"]
                    ),
                    "status": "ok",
                    "error": ""
                })

            except Exception as error:
                rows.append({
                    "candidate_id": candidate_id,
                    "fingerprint_hash": (
                        query["fingerprint_hash"]
                    ),
                    "role_label": query["role_label"],
                    "cycle": cycle,
                    "position": position,
                    "stage": stage,
                    "repetition": (
                        attempt - args.warmups + 1
                    ),
                    "execution_ms": None,
                    "planning_ms": None,
                    "shared_hit": None,
                    "shared_read": None,
                    "temp_read": None,
                    "temp_written": None,
                    "experimental_index_used": False,
                    "indexes_used": "[]",
                    "status": "error",
                    "error": str(error)[:400]
                })

                break

    return rows


def create_index(conn, schema, table, column, name):
    statement = pgsql.SQL(
        "CREATE INDEX {} ON {} ({})"
    ).format(
        pgsql.Identifier(name),
        pgsql.Identifier(schema, table),
        pgsql.Identifier(column)
    )

    start = time.perf_counter()

    with conn.cursor() as cursor:
        cursor.execute(statement)

    elapsed = time.perf_counter() - start

    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT pg_relation_size(%s::regclass)",
            (f'"{schema}"."{name}"',)
        )

        size = cursor.fetchone()[0]

    return elapsed, size


def drop_index(conn, schema, name):
    statement = pgsql.SQL(
        "DROP INDEX IF EXISTS {}"
    ).format(
        pgsql.Identifier(schema, name)
    )

    with conn.cursor() as cursor:
        cursor.execute(statement)


def save_progress(output, samples, candidates):
    pd.DataFrame(samples).to_csv(
        output / "phase9c_execution_samples.csv",
        index=False
    )

    pd.DataFrame(candidates).to_csv(
        output / "phase9c_candidate_status.csv",
        index=False
    )


def bootstrap_ci(baseline, indexed, seed, iterations):
    rng = np.random.default_rng(seed)

    baseline = np.asarray(
        baseline,
        dtype=float
    )

    indexed = np.asarray(
        indexed,
        dtype=float
    )

    if len(baseline) < 2 or len(indexed) < 2:
        return np.nan, np.nan

    ratios = []

    for _ in range(iterations):
        b = rng.choice(
            baseline,
            size=len(baseline),
            replace=True
        )

        i = rng.choice(
            indexed,
            size=len(indexed),
            replace=True
        )

        ratios.append(
            np.median(b) / max(np.median(i), 1e-9)
        )

    return tuple(
        float(value)
        for value in np.quantile(
            ratios,
            [0.025, 0.975]
        )
    )


def generate_results(output, samples, candidates, args):
    heading("4. GENERATING VALIDATION RESULTS")

    df = pd.DataFrame(samples)
    comparisons = []

    if not df.empty:
        valid = df[df["status"] == "ok"]

        grouped = valid.groupby(
            ["candidate_id", "fingerprint_hash"]
        )

        for (candidate_id, fingerprint), group in grouped:
            before = group[
                group["stage"] == "baseline"
            ]

            after = group[
                group["stage"] == "indexed"
            ]

            b = before["execution_ms"].astype(float)
            a = after["execution_ms"].astype(float)

            expected = (
                2 * args.cycles * args.repetitions
            )

            all_rows = df[
                (df["candidate_id"] == candidate_id)
                & (df["fingerprint_hash"] == fingerprint)
            ]

            complete = (
                len(b) == expected
                and len(a) == expected
                and not (
                    all_rows["status"] == "error"
                ).any()
            )

            if complete:
                low, high = bootstrap_ci(
                    b,
                    a,
                    args.seed,
                    args.bootstrap
                )
            else:
                low, high = np.nan, np.nan

            comparisons.append({
                "candidate_id": candidate_id,
                "fingerprint_hash": fingerprint,
                "complete": complete,
                "baseline_samples": len(b),
                "indexed_samples": len(a),
                "baseline_p50_ms": (
                    float(b.median())
                    if len(b) else np.nan
                ),
                "indexed_p50_ms": (
                    float(a.median())
                    if len(a) else np.nan
                ),
                "baseline_p95_ms": (
                    float(b.quantile(0.95))
                    if len(b) else np.nan
                ),
                "indexed_p95_ms": (
                    float(a.quantile(0.95))
                    if len(a) else np.nan
                ),
                "speedup": (
                    float(b.median() / a.median())
                    if len(b) and len(a)
                    and a.median() > 0
                    else np.nan
                ),
                "ci95_low": low,
                "ci95_high": high,
                "index_use_rate": (
                    float(
                        after[
                            "experimental_index_used"
                        ].mean()
                    )
                    if len(after) else 0
                ),
                "baseline_shared_reads": (
                    float(before["shared_read"].median())
                    if len(before) else np.nan
                ),
                "indexed_shared_reads": (
                    float(after["shared_read"].median())
                    if len(after) else np.nan
                )
            })

    comparison_df = pd.DataFrame(comparisons)

    comparison_df.to_csv(
        output / "phase9c_query_comparisons.csv",
        index=False
    )

    decisions = []

    for candidate in candidates:
        if comparison_df.empty:
            matched = pd.DataFrame()
        else:
            matched = comparison_df[
                comparison_df["candidate_id"]
                == candidate["candidate_id"]
            ]

        complete = (
            matched[matched["complete"] == True]
            if not matched.empty
            else pd.DataFrame()
        )

        if (
            candidate["status"] != "completed"
            or complete.empty
        ):
            decision = "NEEDS_MORE_EVIDENCE"
            reason = "Incomplete or missing comparisons"

        elif len(complete) < args.min_queries:
            decision = "NEEDS_MORE_EVIDENCE"
            reason = "Too few independent fingerprints"

        elif (
            complete["index_use_rate"] == 0
        ).all():
            decision = "REJECT"
            reason = "Experimental index was never selected"

        elif (
            (
                complete["ci95_low"]
                >= args.accept_speedup
            ).all()
            and (
                complete["index_use_rate"]
                >= args.min_use_rate
            ).all()
        ):
            decision = "ACCEPT"
            reason = "Passed benefit and index-use thresholds"

        elif (
            complete["ci95_high"] < 1.0
        ).all():
            decision = "REJECT"
            reason = "Measured slowdown"

        else:
            decision = "NEEDS_MORE_EVIDENCE"
            reason = "Uncertain or mixed evidence"

        decisions.append({
            **candidate,
            "evaluated_fingerprints": len(complete),
            "median_speedup": (
                float(complete["speedup"].median())
                if not complete.empty
                else np.nan
            ),
            "mean_index_use_rate": (
                float(
                    complete["index_use_rate"].mean()
                )
                if not complete.empty
                else np.nan
            ),
            "decision": decision,
            "reason": reason
        })

    decision_df = pd.DataFrame(decisions)

    decision_df.to_csv(
        output / "index_decisions.csv",
        index=False
    )

    report = {
        "phase": "9C",
        "generated_at": datetime.now().isoformat(),
        "database": BENCH_DB,
        "candidates": len(candidates),
        "query_comparisons": len(comparison_df),
        "decisions": (
            decision_df["decision"]
            .value_counts()
            .to_dict()
        ),
        "configuration": {
            "cycles": args.cycles,
            "repetitions": args.repetitions,
            "warmups": args.warmups,
            "timeout_ms": args.timeout_ms,
            "bootstrap_iterations": args.bootstrap,
            "minimum_fingerprints": args.min_queries
        },
        "limitations": [
            "Measurements remain sensitive to caching.",
            "Bootstrap intervals assume independent samples.",
            "Execution repetitions within cycles are correlated.",
            "Only available query fingerprints are evaluated.",
            "Decisions are advisory, not automatic schema changes.",
            "No production database indexes are created."
        ]
    }

    (
        output / "phase9c_report.json"
    ).write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    heading("PHASE 9C COMPLETE")

    print(
        decision_df[
            [
                "candidate_id",
                "status",
                "evaluated_fingerprints",
                "median_speedup",
                "mean_index_use_rate",
                "decision"
            ]
        ].to_string(index=False)
    )

    log("\nResults saved to:")
    log(str(output))


def main():
    parser = argparse.ArgumentParser()

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
        "--phase9a-dir",
        default=None
    )

    parser.add_argument(
        "--phase9b-dir",
        default=None
    )

    parser.add_argument(
        "--all-candidates",
        action="store_true"
    )

    parser.add_argument(
        "--max-candidates",
        type=int,
        default=4
    )

    parser.add_argument(
        "--max-queries-per-index",
        type=int,
        default=3
    )

    parser.add_argument(
        "--cycles",
        type=int,
        default=3
    )

    parser.add_argument(
        "--repetitions",
        type=int,
        default=3
    )

    parser.add_argument(
        "--warmups",
        type=int,
        default=1
    )

    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=15000
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=2000
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--min-queries",
        type=int,
        default=2
    )

    parser.add_argument(
        "--accept-speedup",
        type=float,
        default=1.10
    )

    parser.add_argument(
        "--min-use-rate",
        type=float,
        default=0.50
    )

    args = parser.parse_args()

    if args.ask_password:
        args.password = getpass.getpass(
            "PostgreSQL password: "
        )

    if (
        min(
            args.cycles,
            args.repetitions,
            args.max_candidates,
            args.max_queries_per_index,
            args.bootstrap,
            args.min_queries
        ) < 1
        or args.warmups < 0
        or args.timeout_ms < 1000
    ):
        parser.error(
            "Invalid benchmark configuration."
        )

    heading(
        "PHASE 9C - ROBUST INDEX VALIDATION"
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

    with connect(args) as conn:
        verify_database(conn)

        ranking, advice, workload = load_inputs(args)

        samples = []
        candidates = []

        for number, (_, candidate) in enumerate(
            ranking.iterrows(),
            start=1
        ):
            schema = str(candidate["schema"])
            table = str(candidate["table"])
            column = str(candidate["column"])

            candidate_id = (
                f"{number:02d}_{table}_{column}"
            )

            index_name = (
                f"{INDEX_PREFIX}{number:02d}_"
                f"{table}_{column}"
            )[:60]

            heading(
                f"3. CANDIDATE {number}/{len(ranking)}: "
                f"{table}({column})"
            )

            record = {
                "candidate_id": candidate_id,
                "schema": schema,
                "table": table,
                "column": column,
                "index_name": index_name,
                "status": "pending",
                "query_count": 0,
                "mean_build_seconds": np.nan,
                "index_size_bytes": np.nan,
                "cleanup_status": "not_needed",
                "error": ""
            }

            candidates.append(record)

            try:
                if already_indexed(
                    conn,
                    schema,
                    table,
                    column
                ):
                    record["status"] = "already_indexed"
                    log("Skipped: index already exists.")
                    continue

                queries = matching_queries(
                    candidate,
                    advice,
                    workload,
                    args.max_queries_per_index
                )

                record["query_count"] = len(queries)

                if not queries:
                    record["status"] = "no_matching_queries"
                    log("No matching valid queries.")
                    continue

                build_times = []
                index_sizes = []

                for cycle in range(1, args.cycles + 1):
                    log(
                        f"\nCycle {cycle}/{args.cycles}: "
                        "A -> B -> B -> A"
                    )

                    created = False

                    try:
                        samples.extend(
                            measure(
                                conn, queries, args,
                                candidate_id, cycle,
                                1, "baseline", index_name
                            )
                        )

                        elapsed, size = create_index(
                            conn,
                            schema,
                            table,
                            column,
                            index_name
                        )

                        created = True
                        build_times.append(elapsed)
                        index_sizes.append(size)

                        for position in (2, 3):
                            samples.extend(
                                measure(
                                    conn, queries, args,
                                    candidate_id, cycle,
                                    position, "indexed",
                                    index_name
                                )
                            )

                    finally:
                        if created:
                            try:
                                drop_index(
                                    conn,
                                    schema,
                                    index_name
                                )

                                record["cleanup_status"] = (
                                    "removed"
                                )

                            except Exception as error:
                                record["status"] = (
                                    "cleanup_failed"
                                )

                                record["cleanup_status"] = (
                                    "FAILED"
                                )

                                record["error"] = str(error)

                                save_progress(
                                    output,
                                    samples,
                                    candidates
                                )

                                raise RuntimeError(
                                    "Index cleanup failed. "
                                    "Stop and inspect the "
                                    "benchmark database."
                                ) from error

                    samples.extend(
                        measure(
                            conn, queries, args,
                            candidate_id, cycle,
                            4, "baseline", index_name
                        )
                    )

                    save_progress(
                        output,
                        samples,
                        candidates
                    )

                record["mean_build_seconds"] = float(
                    np.mean(build_times)
                )

                record["index_size_bytes"] = int(
                    max(index_sizes)
                )

                record["status"] = "completed"

            except Exception as error:
                if record["status"] == "cleanup_failed":
                    raise

                record["status"] = "error"
                record["error"] = str(error)[:400]

                log(f"Candidate failed: {error}")

            finally:
                save_progress(
                    output,
                    samples,
                    candidates
                )

        generate_results(
            output,
            samples,
            candidates,
            args
        )


if __name__ == "__main__":
    main()
