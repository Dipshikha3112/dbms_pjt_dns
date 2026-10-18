
from pathlib import Path
from datetime import datetime
import argparse
import getpass
import json
import os
import re
import shutil
import subprocess
import time

import numpy as np
import pandas as pd
import psycopg
import sqlglot

from psycopg import sql as pgsql
from sqlglot import exp


ROOT = Path(__file__).resolve().parent.parent

SOURCE_DB = "adaptive_healthcare_scale_10k"
BENCH_DB = "adaptive_healthcare_phase9_benchmark"

WORKLOAD = (
    ROOT / "results"
    / "phase8_2_expanded"
    / "phase82_training.csv"
)

PHASE9_ROOT = (
    ROOT / "results"
    / "phase9_optimizer"
    / SOURCE_DB
)

OUTPUT_ROOT = (
    ROOT / "results"
    / "phase9b_benchmark"
)

INDEX_PREFIX = "p9b_exp_"


def heading(message):
    print("\n" + "=" * 76)
    print(message)
    print("=" * 76, flush=True)


def connection_parameters(args, database):
    params = {
        "host": args.host,
        "port": args.port,
        "user": args.user,
        "dbname": database,
        "autocommit": True,
        "connect_timeout": 20
    }

    if args.password is not None:
        params["password"] = args.password

    return params


def validate_query(query):
    try:
        statements = sqlglot.parse(
            query,
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
            r"\bFOR\s+"
            r"(UPDATE|SHARE|NO\s+KEY\s+UPDATE|KEY\s+SHARE)\b",
            query,
            flags=re.I
        ):
            return False

        return True

    except Exception:
        return False


def run_utility(command, password, log_path):
    environment = os.environ.copy()

    if password is not None:
        environment["PGPASSWORD"] = password

    with log_path.open(
        "a",
        encoding="utf-8"
    ) as log:
        result = subprocess.run(
            command,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            check=False
        )

    if result.returncode:
        raise RuntimeError(
            f"PostgreSQL utility failed. "
            f"See {log_path}"
        )


def create_database_copy(args, output):
    heading("1. CREATING ISOLATED DATABASE")

    dump_exe = shutil.which("pg_dump")
    restore_exe = shutil.which("pg_restore")

    if not dump_exe or not restore_exe:
        raise RuntimeError(
            "pg_dump and pg_restore are required. "
            "Add PostgreSQL bin directory to PATH."
        )

    with psycopg.connect(
        **connection_parameters(args, "postgres")
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM pg_database "
                "WHERE datname = %s",
                (BENCH_DB,)
            )

            if cur.fetchone():
                raise RuntimeError(
                    "Benchmark database already exists. "
                    "Use --reuse-benchmark."
                )

    dump_dir = Path(args.dump_dir)
    dump_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    dump_path = (
        dump_dir
        / f"phase9b_{output.name}.dump"
    )

    log_path = output / "database_copy_log.txt"

    print("Backing up source database...", flush=True)

    run_utility(
        [
            dump_exe,
            "-h", args.host,
            "-p", str(args.port),
            "-U", args.user,
            "-d", SOURCE_DB,
            "-Fc",
            "-f", str(dump_path)
        ],
        args.password,
        log_path
    )

    print(
        "Backup size:",
        round(
            dump_path.stat().st_size / 1024**3,
            3
        ),
        "GB"
    )

    with psycopg.connect(
        **connection_parameters(args, "postgres")
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(
                pgsql.SQL(
                    "CREATE DATABASE {}"
                ).format(
                    pgsql.Identifier(BENCH_DB)
                )
            )

    print("Restoring benchmark database...", flush=True)

    run_utility(
        [
            restore_exe,
            "-h", args.host,
            "-p", str(args.port),
            "-U", args.user,
            "-d", BENCH_DB,
            "--no-owner",
            "--no-acl",
            str(dump_path)
        ],
        args.password,
        log_path
    )

    if not args.keep_dump:
        dump_path.unlink(missing_ok=True)

    print("Benchmark database created successfully.")


def verify_database(args):
    heading("1. VERIFYING ISOLATED DATABASE")

    if args.db != BENCH_DB:
        raise RuntimeError(
            "Only the designated benchmark database "
            "can be modified."
        )

    with psycopg.connect(
        **connection_parameters(args, BENCH_DB)
    ) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    to_regclass('public.patient'),
                    to_regclass('public.encounter'),
                    to_regclass('public.lab_result')
            """)

            if any(
                value is None
                for value in cur.fetchone()
            ):
                raise RuntimeError(
                    "Required healthcare tables are missing."
                )

            cur.execute("""
                SELECT indexname
                FROM pg_indexes
                WHERE schemaname = 'public'
                  AND indexname LIKE 'p9b_exp_%'
            """)

            leftovers = [
                row[0]
                for row in cur.fetchall()
            ]

            if leftovers:
                raise RuntimeError(
                    "Experimental indexes already exist: "
                    + ", ".join(leftovers)
                    + ". Inspect and remove them from "
                    "the benchmark database before retrying."
                )

    print("Verified:", BENCH_DB)
    print("No leftover Phase 9B indexes.")


def load_inputs(args):
    heading("2. LOADING PHASE 9 DATA")

    if args.phase9a_dir:
        folder = Path(args.phase9a_dir)
    else:
        if not PHASE9_ROOT.exists():
            raise FileNotFoundError(PHASE9_ROOT)

        folders = sorted(
            path
            for path in PHASE9_ROOT.iterdir()
            if path.is_dir()
            and (
                path / "ranked_index_hypotheses.csv"
            ).exists()
        )

        if not folders:
            raise FileNotFoundError(
                "No Phase 9A results found."
            )

        folder = folders[-1]

    ranked_path = (
        folder / "ranked_index_hypotheses.csv"
    )

    advice_path = (
        folder / "index_advice_per_query.csv"
    )

    if not ranked_path.exists():
        raise FileNotFoundError(ranked_path)

    if not advice_path.exists():
        raise FileNotFoundError(advice_path)

    if not WORKLOAD.exists():
        raise FileNotFoundError(WORKLOAD)

    ranked = pd.read_csv(
        ranked_path,
        dtype=str
    ).fillna("")

    advice = pd.read_csv(
        advice_path,
        dtype=str
    ).fillna("")

    workload = pd.read_csv(
        WORKLOAD,
        dtype=str,
        low_memory=False
    ).dropna(
        subset=[
            "sql",
            "fingerprint_hash"
        ]
    )

    required_ranked = {
        "schema",
        "table",
        "column"
    }

    required_advice = {
        "schema",
        "table",
        "column",
        "fingerprint_hash"
    }

    if not required_ranked.issubset(ranked.columns):
        raise ValueError(
            "Ranked index CSV is missing columns."
        )

    if not required_advice.issubset(advice.columns):
        raise ValueError(
            "Index advice CSV is missing columns."
        )

    print("Phase 9A folder:", folder)
    print("Index candidates:", len(ranked))
    print("Workload instances:", len(workload))

    return ranked, advice, workload


def select_queries(candidate, advice, workload, maximum):
    mask = (
        (advice["schema"] == candidate["schema"])
        & (advice["table"] == candidate["table"])
        & (advice["column"] == candidate["column"])
    )

    fingerprints = set(
        advice.loc[
            mask,
            "fingerprint_hash"
        ]
    )

    selected = workload[
        workload["fingerprint_hash"].isin(
            fingerprints
        )
    ].drop_duplicates(
        "fingerprint_hash"
    )

    queries = []

    for row in selected.itertuples(index=False):
        query = str(row.sql).strip().rstrip(";")

        if not validate_query(query):
            continue

        queries.append({
            "fingerprint_hash": str(
                row.fingerprint_hash
            ),
            "role_label": str(
                row.role_label
            ),
            "sql": query
        })

        if len(queries) >= maximum:
            break

    return queries


def already_indexed(conn, schema, table, column):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 1
            FROM pg_index i
            JOIN pg_class t
              ON t.oid = i.indrelid
            JOIN pg_namespace n
              ON n.oid = t.relnamespace
            JOIN pg_class ix
              ON ix.oid = i.indexrelid
            JOIN pg_am am
              ON am.oid = ix.relam
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

        return cur.fetchone() is not None


def explain_analyze(conn, query, timeout_ms):
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute(
                "SET TRANSACTION READ ONLY"
            )

            cur.execute(
                "SELECT set_config(%s, %s, true)",
                (
                    "statement_timeout",
                    str(timeout_ms)
                )
            )

            cur.execute(
                "SELECT set_config(%s, %s, true)",
                (
                    "lock_timeout",
                    "2000"
                )
            )

            cur.execute(
                "EXPLAIN (ANALYZE, BUFFERS, "
                "FORMAT JSON, TIMING OFF) "
                + query
            )

            result = cur.fetchone()[0]

    if isinstance(result, str):
        result = json.loads(result)

    if isinstance(result, list):
        result = result[0]

    plan = result["Plan"]

    index_names = set()

    def walk(node):
        if node.get("Index Name"):
            index_names.add(node["Index Name"])

        for child in node.get("Plans", []):
            walk(child)

    walk(plan)

    return {
        "execution_ms": float(
            result["Execution Time"]
        ),
        "planning_ms": float(
            result.get("Planning Time", 0)
        ),
        "index_names": sorted(index_names),
        "plan_node": plan.get(
            "Node Type",
            "Unknown"
        )
    }


def measure_query(
    conn,
    query,
    args,
    candidate_id,
    stage,
    expected_index_name=None
):
    rows = []

    total = (
        args.warmups + args.repetitions
    )

    for attempt in range(total):
        is_warmup = (
            attempt < args.warmups
        )

        try:
            result = explain_analyze(
                conn,
                query["sql"],
                args.timeout_ms
            )

            if is_warmup:
                continue

            rows.append({
                "candidate_id": candidate_id,
                "fingerprint_hash": (
                    query["fingerprint_hash"]
                ),
                "role_label": (
                    query["role_label"]
                ),
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
                "plan_node": (
                    result["plan_node"]
                ),
                "indexes_used": json.dumps(
                    result["index_names"]
                ),
                "experimental_index_used": (
                    expected_index_name
                    in result["index_names"]
                    if expected_index_name
                    else False
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
                "role_label": (
                    query["role_label"]
                ),
                "stage": stage,
                "repetition": (
                    attempt - args.warmups + 1
                ),
                "execution_ms": None,
                "planning_ms": None,
                "plan_node": "",
                "indexes_used": "[]",
                "experimental_index_used": False,
                "status": "error",
                "error": str(error)[:400]
            })

            break

    return rows


def summarize_query(
    measurements,
    candidate_id,
    fingerprint,
    repetitions
):
    df = pd.DataFrame(
        [
            row
            for row in measurements
            if row["candidate_id"] == candidate_id
            and row["fingerprint_hash"] == fingerprint
        ]
    )

    result = {
        "candidate_id": candidate_id,
        "fingerprint_hash": fingerprint,
        "baseline_samples": 0,
        "indexed_samples": 0,
        "baseline_p50_ms": None,
        "indexed_p50_ms": None,
        "baseline_p95_ms": None,
        "indexed_p95_ms": None,
        "speedup": None,
        "p95_change_pct": None,
        "experimental_index_used": False,
        "status": "insufficient_samples"
    }

    if df.empty:
        return result

    before = df[
        (df["stage"] == "baseline")
        & (df["status"] == "ok")
    ]

    after = df[
        (df["stage"] == "indexed")
        & (df["status"] == "ok")
    ]

    result["baseline_samples"] = len(before)
    result["indexed_samples"] = len(after)

    if (
        len(before) != repetitions
        or len(after) != repetitions
    ):
        return result

    b = before["execution_ms"].astype(float)
    a = after["execution_ms"].astype(float)

    before_p50 = float(b.median())
    after_p50 = float(a.median())

    before_p95 = float(b.quantile(0.95))
    after_p95 = float(a.quantile(0.95))

    result.update({
        "baseline_p50_ms": before_p50,
        "indexed_p50_ms": after_p50,
        "baseline_p95_ms": before_p95,
        "indexed_p95_ms": after_p95,
        "speedup": (
            before_p50 / after_p50
            if after_p50 > 0
            else None
        ),
        "p95_change_pct": (
            100 * (
                after_p95 - before_p95
            ) / before_p95
            if before_p95 > 0
            else None
        ),
        "experimental_index_used": bool(
            after["experimental_index_used"].any()
        ),
        "status": "ok"
    })

    return result


def save_progress(
    output,
    measurements,
    summaries,
    candidates
):
    pd.DataFrame(measurements).to_csv(
        output / "execution_samples.csv",
        index=False
    )

    pd.DataFrame(summaries).to_csv(
        output / "per_fingerprint_results.csv",
        index=False
    )

    pd.DataFrame(candidates).to_csv(
        output / "candidate_results.csv",
        index=False
    )


def benchmark(args, output):
    ranked, advice, workload = load_inputs(args)

    heading("3. BENCHMARKING INDEX CANDIDATES")

    measurements = []
    summaries = []
    candidates = []

    selected_candidates = ranked.head(
        args.max_candidates
    )

    with psycopg.connect(
        **connection_parameters(args, BENCH_DB)
    ) as conn:

        for number, (_, candidate) in enumerate(
            selected_candidates.iterrows(),
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

            print(
                f"\nCandidate {number}/"
                f"{len(selected_candidates)}: "
                f"{schema}.{table}({column})",
                flush=True
            )

            record = {
                "candidate_id": candidate_id,
                "schema": schema,
                "table": table,
                "column": column,
                "index_name": index_name,
                "status": "pending",
                "query_count": 0,
                "index_build_seconds": None,
                "index_size_bytes": None,
                "cleanup_status": "not_needed",
                "error": ""
            }

            candidates.append(record)

            created = False

            try:
                if already_indexed(
                    conn,
                    schema,
                    table,
                    column
                ):
                    record["status"] = "already_indexed"
                    print(
                        "Skipped: existing B-tree index."
                    )
                    continue

                queries = select_queries(
                    candidate,
                    advice,
                    workload,
                    args.max_queries_per_index
                )

                record["query_count"] = len(queries)

                if not queries:
                    record["status"] = "no_matching_queries"
                    print(
                        "Skipped: no matching queries."
                    )
                    continue

                print("Measuring baseline...", flush=True)

                for query in queries:
                    measurements.extend(
                        measure_query(
                            conn,
                            query,
                            args,
                            candidate_id,
                            "baseline"
                        )
                    )

                # Correct PostgreSQL CREATE INDEX syntax:
                # index name is NOT schema-qualified.
                create_statement = pgsql.SQL(
                    "CREATE INDEX {} ON {} ({})"
                ).format(
                    pgsql.Identifier(index_name),
                    pgsql.Identifier(
                        schema,
                        table
                    ),
                    pgsql.Identifier(column)
                )

                print(
                    "Creating experimental index...",
                    flush=True
                )

                start = time.perf_counter()

                with conn.cursor() as cur:
                    cur.execute(create_statement)

                created = True

                record["index_build_seconds"] = round(
                    time.perf_counter() - start,
                    4
                )

                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT pg_relation_size(%s::regclass)",
                        (
                            f'"{schema}"."{index_name}"',
                        )
                    )

                    record["index_size_bytes"] = (
                        cur.fetchone()[0]
                    )

                print(
                    "Measuring indexed performance...",
                    flush=True
                )

                for query in queries:
                    measurements.extend(
                        measure_query(
                            conn,
                            query,
                            args,
                            candidate_id,
                            "indexed",
                            expected_index_name=index_name
                        )
                    )

                    summaries.append(
                        summarize_query(
                            measurements,
                            candidate_id,
                            query["fingerprint_hash"],
                            args.repetitions
                        )
                    )

                successful = sum(
                    row["status"] == "ok"
                    for row in summaries
                    if row["candidate_id"]
                    == candidate_id
                )

                if successful:
                    record["status"] = "completed"
                else:
                    record["status"] = (
                        "insufficient_samples"
                    )

            except Exception as error:
                record["status"] = "error"
                record["error"] = str(error)[:500]

                print(
                    "Candidate failed:",
                    record["error"],
                    flush=True
                )

            finally:
                if created:
                    print(
                        "Removing experimental index...",
                        flush=True
                    )

                    try:
                        # DROP INDEX accepts schema-qualified
                        # index names.
                        drop_statement = pgsql.SQL(
                            "DROP INDEX IF EXISTS {}"
                        ).format(
                            pgsql.Identifier(
                                schema,
                                index_name
                            )
                        )

                        with conn.cursor() as cur:
                            cur.execute(drop_statement)

                        record["cleanup_status"] = (
                            "removed"
                        )

                        print(
                            "Experimental index removed.",
                            flush=True
                        )

                    except Exception as error:
                        record["cleanup_status"] = (
                            "FAILED"
                        )

                        record["status"] = (
                            "cleanup_failed"
                        )

                        record["error"] += (
                            " | Cleanup error: "
                            + str(error)[:300]
                        )

                        print(
                            "CRITICAL: Index cleanup failed.",
                            flush=True
                        )

                save_progress(
                    output,
                    measurements,
                    summaries,
                    candidates
                )

                if record["cleanup_status"] == "FAILED":
                    raise RuntimeError(
                        "Stopping benchmark because "
                        "an experimental index remains. "
                        "Inspect the benchmark database."
                    )

    return candidates, summaries


def generate_report(output, candidates, summaries):
    heading("4. GENERATING FINAL RESULTS")

    details = pd.DataFrame(summaries)

    ranking = []

    for candidate in candidates:
        if details.empty:
            valid = pd.DataFrame()
        else:
            valid = details[
                (
                    details["candidate_id"]
                    == candidate["candidate_id"]
                )
                & (details["status"] == "ok")
            ]

        ranking.append({
            **candidate,
            "measured_queries": len(valid),
            "median_speedup": (
                float(valid["speedup"].median())
                if not valid.empty
                else None
            ),
            "median_p95_change_pct": (
                float(
                    valid["p95_change_pct"].median()
                )
                if not valid.empty
                else None
            ),
            "queries_using_index": (
                int(
                    valid["experimental_index_used"].sum()
                )
                if not valid.empty
                else 0
            )
        })

    ranking_df = pd.DataFrame(ranking)

    ranking_df = ranking_df.sort_values(
        "median_speedup",
        ascending=False,
        na_position="last"
    )

    ranking_df.to_csv(
        output / "index_benchmark_ranking.csv",
        index=False
    )

    report = {
        "phase": "9B",
        "created_at": datetime.now().isoformat(),
        "source_database": SOURCE_DB,
        "benchmark_database": BENCH_DB,
        "candidates_tested": len(candidates),
        "completed_candidates": sum(
            row["status"] == "completed"
            for row in candidates
        ),
        "successful_query_comparisons": (
            int(
                (details["status"] == "ok").sum()
            )
            if not details.empty
            else 0
        ),
        "measurement": (
            "PostgreSQL EXPLAIN ANALYZE "
            "with BUFFERS and TIMING OFF"
        ),
        "limitations": [
            "Baseline is measured before indexed execution.",
            "Cache warming and system load can bias results.",
            "Five repetitions are insufficient for robust P95.",
            "Only one SQL instance per fingerprint is tested.",
            "Indexes are evaluated individually.",
            "The test is on a copied 10K database.",
            "Index creation and removal affect only "
            "the isolated benchmark database.",
            "Speedup alone does not measure write overhead."
        ]
    }

    (
        output / "phase9b_report.json"
    ).write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    heading("PHASE 9B COMPLETE")

    print(
        json.dumps(
            report,
            indent=2
        )
    )

    print("\nIndex ranking:")

    print(
        ranking_df[
            [
                "candidate_id",
                "status",
                "measured_queries",
                "median_speedup",
                "median_p95_change_pct",
                "queries_using_index",
                "cleanup_status"
            ]
        ].to_string(index=False)
    )

    print("\nResults saved to:")
    print(output)

    print(
        "\nOriginal healthcare databases "
        "were not modified."
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--db",
        default=BENCH_DB
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
        "--reuse-benchmark",
        action="store_true"
    )

    parser.add_argument(
        "--dump-dir",
        default=r"D:\phase9b_backup"
    )

    parser.add_argument(
        "--keep-dump",
        action="store_true"
    )

    parser.add_argument(
        "--phase9a-dir",
        default=None
    )

    parser.add_argument(
        "--max-candidates",
        type=int,
        default=9
    )

    parser.add_argument(
        "--max-queries-per-index",
        type=int,
        default=3
    )

    parser.add_argument(
        "--repetitions",
        type=int,
        default=5
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

    args = parser.parse_args()

    if args.ask_password:
        args.password = getpass.getpass(
            "PostgreSQL password: "
        )

    if args.db != BENCH_DB:
        parser.error(
            "Only the isolated benchmark "
            "database is permitted."
        )

    if (
        args.max_candidates < 1
        or args.max_queries_per_index < 1
        or args.repetitions < 2
        or args.warmups < 0
        or args.timeout_ms < 1000
    ):
        parser.error(
            "Invalid benchmark configuration."
        )

    heading(
        "PHASE 9B - CONTROLLED POSTGRESQL "
        "INDEX BENCHMARKING"
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

    if args.reuse_benchmark:
        print(
            "Reusing existing isolated "
            "benchmark database."
        )
    else:
        create_database_copy(
            args,
            output
        )

    verify_database(args)

    candidates, summaries = benchmark(
        args,
        output
    )

    generate_report(
        output,
        candidates,
        summaries
    )


if __name__ == "__main__":
    main()
