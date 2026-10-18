
import argparse
import ast
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_WORKLOAD = (
    ROOT / "results" / "phase9_research_optimizer"
    / "20261017_185042_5b6ae098"
    / "workload_validation.csv"
)

DEFAULT_MODEL = (
    ROOT / "results" / "phase8_v6_validation"
    / "v6_improved_generalization"
    / "models" / "v6_improved_models.joblib"
)

DEFAULT_V6_SCRIPT = ROOT / "notebooks" / "08_v6_final.py"

BENCHMARK_DB = "adaptive_healthcare_phase9_benchmark"

CANDIDATES = [
    ("lab_result", ("patient_id", "result_time DESC")),
    ("lab_result", ("patient_id", "loinc_code", "result_time DESC")),
    ("encounter", ("patient_id", "start_time DESC")),
    ("encounter", ("provider_id", "start_time DESC")),
    ("diagnosis", ("patient_id", "diagnosis_id")),
    ("prescription", ("patient_id", "prescription_id")),
    ("prescription_item", ("prescription_id", "rxcui")),
    ("patient_procedure", ("patient_id", "procedure_id")),
    ("claim", ("patient_id", "claim_id")),
    ("claim", ("payer_id", "patient_id")),
    ("lab_result", ("loinc_code", "patient_id")),
    ("patient_allergy", ("patient_id",)),
    ("patient_procedure", ("patient_id",)),
    ("lab_result", ("loinc_code",)),
    ("claim", ("payer_id",)),
    ("encounter", ("patient_id",)),
    ("diagnosis", ("patient_id",)),
    ("prescription", ("patient_id",)),
    ("patient_procedure", ("procedure_id",)),
    ("patient_allergy", ("patient_id", "allergy_id")),
]


def name_of(candidate):
    return f"{candidate[0]}({', '.join(candidate[1])})"


def hash_text(value):
    return hashlib.sha256(
        str(value).encode("utf-8")
    ).hexdigest()[:20]


def fingerprint(sql):
    sql = str(sql).lower()
    sql = re.sub(r"'(?:''|[^'])*'", " ? ", sql)
    sql = re.sub(r"\b\d+(?:\.\d+)?\b", " ? ", sql)
    sql = re.sub(r"\s+", " ", sql)
    return hash_text(sql)


def safe_sql(sql):
    sql = str(sql).strip()

    if not re.match(r"(?is)^SELECT\b", sql):
        return False

    if ";" in sql:
        return False

    if re.search(
        r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|"
        r"TRUNCATE|COPY|CALL|DO|EXECUTE|VACUUM|"
        r"ANALYZE|LOCK|SET|RESET|INTO)\b",
        sql,
        re.I,
    ):
        return False

    if re.search(
        r"\bFOR\s+(UPDATE|SHARE)\b",
        sql,
        re.I,
    ):
        return False

    return True


def load_workload(path):
    df = pd.read_csv(path, low_memory=False)

    required = {
        "query_text",
        "query_family",
        "query_name",
    }

    if not required.issubset(df.columns):
        raise RuntimeError(
            f"Missing columns: {required - set(df.columns)}"
        )

    df = df.dropna(subset=["query_text"]).copy()
    df = df[df["query_text"].map(safe_sql)].copy()

    df["query_text"] = (
        df["query_text"]
        .astype(str)
        .str.strip()
    )

    df["query_id"] = df["query_text"].map(hash_text)
    df["fingerprint"] = df["query_text"].map(fingerprint)

    df["template"] = (
        df["query_family"].fillna("unknown").astype(str)
        + "::"
        + df["query_name"].fillna("unknown").astype(str)
    )

    df = df.drop_duplicates("query_id").reset_index(drop=True)

    if len(df) < 30:
        raise RuntimeError("Insufficient workload queries.")

    return df


def load_v6(script_path, model_path):
    tree = ast.parse(
        script_path.read_text(encoding="utf-8-sig")
    )

    namespace = {
        "np": np,
        "pd": pd,
        "re": re,
        "hashlib": hashlib,
    }

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "FEATURES"
                ):
                    namespace["FEATURES"] = ast.literal_eval(
                        node.value
                    )

        if (
            isinstance(node, ast.FunctionDef)
            and node.name in {
                "normalize_sql",
                "safe_log",
                "sql_features",
            }
        ):
            module = ast.Module(
                body=[node],
                type_ignores=[],
            )

            exec(
                compile(
                    module,
                    str(script_path),
                    "exec",
                ),
                namespace,
            )

    if "FEATURES" not in namespace:
        raise RuntimeError("V6 feature list missing.")

    if "sql_features" not in namespace:
        raise RuntimeError("V6 SQL extractor missing.")

    # Only load the trusted model trained locally.
    bundle = joblib.load(model_path)

    if list(bundle["features"]) != list(
        namespace["FEATURES"]
    ):
        raise RuntimeError("V6 feature mismatch.")

    return namespace, bundle


def nodes_of(plan):
    nodes = [plan]

    for child in plan.get("Plans", []):
        nodes.extend(nodes_of(child))

    return nodes


def feature_frame(sql, plan, namespace, scale):
    values = dict(namespace["sql_features"](sql))

    nodes = nodes_of(plan)

    cost = max(
        float(plan.get("Total Cost", 0)),
        0,
    )

    startup = max(
        float(plan.get("Startup Cost", 0)),
        0,
    )

    rows = max(
        float(plan.get("Plan Rows", 0)),
        0,
    )

    count = len(nodes)

    types = " ".join(
        str(node.get("Node Type", "")).lower()
        for node in nodes
    )

    values.update(
        {
            "log_planner_cost": np.log1p(cost),
            "log_startup_cost": np.log1p(startup),
            "log_estimated_rows": np.log1p(rows),
            "log_plan_nodes": np.log1p(count),
            "scale_log": np.log1p(scale),
            "planner_cost_per_row": cost / (rows + 1),
            "planner_cost_per_node": cost / (count + 1),
            "estimated_rows_per_node": rows / (count + 1),
            "join_cost_interaction": (
                values["join_count"] * np.log1p(cost)
            ),
            "filter_cost_interaction": (
                values["predicate_count"]
                * np.log1p(cost)
            ),
            "sort_indicator": int("sort" in types),
            "hash_join_indicator": int(
                "hash join" in types
            ),
            "nested_loop_indicator": int(
                "nested loop" in types
            ),
            "seq_scan_indicator": int(
                "seq scan" in types
            ),
            "index_scan_indicator": int(
                "index scan" in types
            ),
        }
    )

    missing = set(
        namespace["FEATURES"]
    ) - set(values)

    if missing:
        raise RuntimeError(
            f"Missing features: {sorted(missing)}"
        )

    return (
        pd.DataFrame([values])[
            namespace["FEATURES"]
        ]
        .replace([np.inf, -np.inf], np.nan)
    )


def v6_predict(sql, plan, namespace, bundle, scale):
    frame = feature_frame(
        sql,
        plan,
        namespace,
        scale,
    )

    x = bundle["imputer"].transform(frame)

    predicted_ms = float(
        np.expm1(
            np.clip(
                bundle["regressor"].predict(x)[0],
                0,
                25,
            )
        )
    )

    probability = float(
        bundle["classifier"].predict_proba(x)[0, 1]
    )

    return predicted_ms, probability


def connection(args):
    if args.database != BENCHMARK_DB:
        raise RuntimeError(
            "Only the isolated benchmark DB is allowed."
        )

    if args.host not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        raise RuntimeError("Only localhost is allowed.")

    if args.port != 5433:
        raise RuntimeError(
            "Expected Docker host port 5433."
        )

    conn = psycopg.connect(
        host=args.host,
        port=args.port,
        dbname=args.database,
        user=args.user,
        password=args.password,
        connect_timeout=8,
        autocommit=False,
    )

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT current_database()")

            if cur.fetchone()[0] != BENCHMARK_DB:
                raise RuntimeError("Wrong database.")

            cur.execute(
                "SELECT EXISTS("
                "SELECT 1 FROM pg_extension "
                "WHERE extname='hypopg')"
            )

            if not cur.fetchone()[0]:
                raise RuntimeError(
                    "HypoPG extension missing."
                )

            cur.execute(
                "SELECT set_config('statement_timeout', %s, false)",
                (f"{args.timeout_ms}ms",),
                )

        return conn

    except Exception:
        conn.close()
        raise


def get_candidates(cur):
    cur.execute(
        """
        SELECT table_name, column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
        """
    )

    available = {}

    for table, column in cur.fetchall():
        available.setdefault(table, set()).add(column)

    valid = []

    for table, columns in CANDIDATES:
        if all(
            column.split()[0]
            in available.get(table, set())
            for column in columns
        ):
            valid.append((table, columns))

    return valid


def guarded(cur, callback):
    cur.execute("SAVEPOINT m1_guard")

    try:
        value = callback()
        cur.execute("RELEASE SAVEPOINT m1_guard")
        return value, None

    except psycopg.Error as exc:
        cur.execute(
            "ROLLBACK TO SAVEPOINT m1_guard"
        )
        cur.execute(
            "RELEASE SAVEPOINT m1_guard"
        )
        return None, type(exc).__name__


def install_hypothetical(cur, portfolio, lookup):
    cur.execute("SELECT hypopg_reset()")

    for selected in portfolio:
        table, columns = lookup[selected]

        # Fixed, validated candidate allowlist.
        ddl = (
            f"CREATE INDEX ON public.{table} "
            f"({', '.join(columns)})"
        )

        cur.execute(
            "SELECT * FROM hypopg_create_index(%s)",
            (ddl,),
        )

    cur.execute(
        """
        SELECT indexrelid, index_name
        FROM hypopg_list_indexes
        """
    )

    return {
        str(name).lower()
        for _, name in cur.fetchall()
    }


def explain(cur, sql):
    cur.execute(
        "EXPLAIN (FORMAT JSON) " + sql
    )

    value = cur.fetchone()[0]

    if isinstance(value, str):
        value = json.loads(value)

    if isinstance(value, list):
        value = value[0]

    return value["Plan"]


def uses_hypothetical(plan, names):
    for node in nodes_of(plan):
        index_name = str(
            node.get("Index Name", "")
        ).lower()

        if index_name in names:
            return True

    return False


def evaluate_worker(
    args,
    portfolio,
    queries,
    lookup,
    namespace,
    bundle,
):
    """
    Each task owns an independent PostgreSQL connection.
    No HypoPG state is shared between workers.
    """

    conn = connection(args)

    try:
        with conn.cursor() as cur:
            names = install_hypothetical(
                cur,
                portfolio,
                lookup,
            )

            results = {}
            failures = []

            for row in queries.itertuples(index=False):
                plan, error = guarded(
                    cur,
                    lambda sql=row.query_text: explain(
                        cur,
                        sql,
                    ),
                )

                if error:
                    failures.append(
                        {
                            "query_id": row.query_id,
                            "error": error,
                        }
                    )
                    continue

                predicted_ms, probability = v6_predict(
                    row.query_text,
                    plan,
                    namespace,
                    bundle,
                    args.scale,
                )

                results[row.query_id] = {
                    "cost": float(plan["Total Cost"]),
                    "predicted_ms": predicted_ms,
                    "probability": probability,
                    "index_used": uses_hypothetical(
                        plan,
                        names,
                    ),
                }

            return results, failures

    finally:
        conn.rollback()
        conn.close()


def parallel_evaluate(
    executor,
    args,
    portfolios,
    queries,
    lookup,
    namespace,
    bundle,
    cache,
    stage,
):
    """
    Cache is accessed only by the coordinator thread.
    Worker threads never modify shared cache state.
    """

    unique = {}

    for portfolio in portfolios:
        key = tuple(sorted(portfolio))
        unique[key] = list(portfolio)

    output = {}
    futures = {}

    for key, portfolio in unique.items():
        if key in cache:
            output[key] = cache[key]
            continue

        future = executor.submit(
            evaluate_worker,
            args,
            portfolio,
            queries,
            lookup,
            namespace,
            bundle,
        )

        futures[future] = key

    total = len(futures)

    for completed, future in enumerate(
        as_completed(futures),
        1,
    ):
        key = futures[future]

        result = future.result()

        cache[key] = result
        output[key] = result

        print(
            f"    {stage}: {completed}/{total} "
            f"portfolios completed",
            flush=True,
        )

    return output


def split_data(
    df,
    seed,
    mode,
    test_fraction,
    validation_fraction,
):
    rng = np.random.default_rng(seed)

    if mode == "unseen_template":
        templates = np.array(
            sorted(df["template"].unique()),
            dtype=object,
        )

        rng.shuffle(templates)

        n_test = max(
            1,
            round(len(templates) * test_fraction),
        )

        n_validation = max(
            1,
            round(
                len(templates) * validation_fraction
            ),
        )

        if (
            n_test + n_validation
            >= len(templates)
        ):
            raise RuntimeError(
                "Insufficient templates."
            )

        test_templates = set(
            templates[:n_test]
        )

        validation_templates = set(
            templates[
                n_test:n_test + n_validation
            ]
        )

        train_templates = set(
            templates[
                n_test + n_validation:
            ]
        )

        train = df[
            df["template"].isin(train_templates)
        ].copy()

        validation = df[
            df["template"].isin(
                validation_templates
            )
        ].copy()

        test = df[
            df["template"].isin(test_templates)
        ].copy()

        # A fingerprint appearing under different template
        # names must not cross split boundaries.
        fingerprint_sets = [
            set(part["fingerprint"])
            for part in (train, validation, test)
        ]

        shared = (
            (fingerprint_sets[0] & fingerprint_sets[1])
            | (fingerprint_sets[0] & fingerprint_sets[2])
            | (fingerprint_sets[1] & fingerprint_sets[2])
        )

        train = train[
            ~train["fingerprint"].isin(shared)
        ]

        validation = validation[
            ~validation["fingerprint"].isin(shared)
        ]

        test = test[
            ~test["fingerprint"].isin(shared)
        ]

    else:
        # Allocate globally unique fingerprint groups,
        # then retain only templates represented in
        # all three partitions.
        fingerprints = np.array(
            sorted(df["fingerprint"].unique()),
            dtype=object,
        )

        rng.shuffle(fingerprints)

        n_test = max(
            1,
            round(len(fingerprints) * test_fraction),
        )

        n_validation = max(
            1,
            round(
                len(fingerprints)
                * validation_fraction
            ),
        )

        test_fps = set(
            fingerprints[:n_test]
        )

        validation_fps = set(
            fingerprints[
                n_test:n_test + n_validation
            ]
        )

        train_fps = set(
            fingerprints[
                n_test + n_validation:
            ]
        )

        train = df[
            df["fingerprint"].isin(train_fps)
        ].copy()

        validation = df[
            df["fingerprint"].isin(
                validation_fps
            )
        ].copy()

        test = df[
            df["fingerprint"].isin(test_fps)
        ].copy()

        shared_templates = (
            set(train["template"])
            & set(validation["template"])
            & set(test["template"])
        )

        train = train[
            train["template"].isin(
                shared_templates
            )
        ]

        validation = validation[
            validation["template"].isin(
                shared_templates
            )
        ]

        test = test[
            test["template"].isin(
                shared_templates
            )
        ]

    for a, b in [
        (train, validation),
        (train, test),
        (validation, test),
    ]:
        if set(a["query_id"]) & set(
            b["query_id"]
        ):
            raise RuntimeError(
                "Exact-query leakage."
            )

        if set(a["fingerprint"]) & set(
            b["fingerprint"]
        ):
            raise RuntimeError(
                "Fingerprint leakage."
            )

        if mode == "unseen_template":
            if set(a["template"]) & set(
                b["template"]
            ):
                raise RuntimeError(
                    "Template leakage."
                )

    if min(
        len(train),
        len(validation),
        len(test),
    ) == 0:
        raise RuntimeError(
            "Empty train/validation/test split. "
            "Try another seed or split mode."
        )

    return train, validation, test


def gain_pct(before, after, field):
    if set(before) != set(after):
        return np.nan

    baseline = sum(
        max(float(row[field]), 0)
        for row in before.values()
    )

    indexed = sum(
        max(float(row[field]), 0)
        for row in after.values()
    )

    return (
        100 * (baseline - indexed)
        / max(baseline, 1e-9)
    )


def require_complete(result, expected, context):
    rows, failures = result

    if failures or len(rows) != expected:
        raise RuntimeError(
            f"{context}: incomplete evaluation "
            f"({len(rows)}/{expected}, "
            f"{len(failures)} errors). "
            "Results are not comparable."
        )

    return rows


def parallel_greedy(
    executor,
    args,
    validation,
    shortlist,
    lookup,
    namespace,
    bundle,
    budget,
    method,
    cache,
):
    baseline_result = parallel_evaluate(
        executor,
        args,
        [[]],
        validation,
        lookup,
        namespace,
        bundle,
        cache,
        "validation baseline",
    )[()]

    current = require_complete(
        baseline_result,
        len(validation),
        "Validation baseline",
    )

    selected = []
    remaining = list(shortlist)
    steps = []

    field = (
        "predicted_ms"
        if method == "ai_v6"
        else "cost"
    )

    for round_number in range(1, budget + 1):
        if not remaining:
            break

        proposals = [
            selected + [candidate]
            for candidate in remaining
        ]

        evaluated = parallel_evaluate(
            executor,
            args,
            proposals,
            validation,
            lookup,
            namespace,
            bundle,
            cache,
            f"{method} round {round_number}",
        )

        options = []

        for candidate, portfolio in zip(
            remaining,
            proposals,
        ):
            key = tuple(sorted(portfolio))

            result = evaluated[key]

            rows, failures = result

            if failures or len(rows) != len(validation):
                continue

            marginal = gain_pct(
                current,
                rows,
                field,
            )

            planner_marginal = gain_pct(
                current,
                rows,
                "cost",
            )

            options.append(
                (
                    marginal,
                    planner_marginal,
                    candidate,
                    rows,
                )
            )

        if not options:
            break

        options.sort(
            key=lambda item: (
                -item[0],
                -item[1],
                item[2],
            )
        )

        best_gain, planner_gain, best, rows = (
            options[0]
        )

        if (
            not np.isfinite(best_gain)
            or best_gain <= 1e-7
        ):
            break

        selected.append(best)
        remaining.remove(best)
        current = rows

        steps.append(
            {
                "method": method,
                "round": round_number,
                "candidate": best,
                "marginal_gain_pct": best_gain,
                "marginal_planner_gain_pct": planner_gain,
            }
        )

        print(
            f"    {method}: selected {best} "
            f"(marginal {best_gain:.4f}%)",
            flush=True,
        )

    return selected, steps


def save_csv(path, rows):
    if rows:
        pd.DataFrame(rows).to_csv(
            path,
            index=False,
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workload",
        type=Path,
        default=DEFAULT_WORKLOAD,
    )

    parser.add_argument(
        "--v6-bundle",
        type=Path,
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--training-script",
        type=Path,
        default=DEFAULT_V6_SCRIPT,
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=5433,
    )

    parser.add_argument(
        "--database",
        default=BENCHMARK_DB,
    )

    parser.add_argument(
        "--user",
        default="postgres",
    )

    parser.add_argument(
        "--password",
        default=os.environ.get("PGPASSWORD"),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--start-seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--budget",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--shortlist",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--scale",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--split-mode",
        choices=[
            "both",
            "unseen_template",
            "within_template",
        ],
        default="both",
    )

    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=15000,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "results"
            / "phase9_m1_parallel_v4"
        ),
    )

    args = parser.parse_args()

    if not 1 <= args.workers <= 8:
        raise RuntimeError(
            "Use between 1 and 8 workers."
        )

    if args.seeds < 1:
        raise RuntimeError(
            "Seeds must be positive."
        )

    if args.budget < 1:
        raise RuntimeError(
            "Budget must be positive."
        )

    if args.shortlist < args.budget:
        raise RuntimeError(
            "Shortlist must be at least the budget."
        )

    if (
        args.holdout_fraction <= 0
        or args.validation_fraction <= 0
        or args.holdout_fraction
        + args.validation_fraction >= 0.8
    ):
        raise RuntimeError(
            "Invalid split fractions."
        )

    workload = load_workload(
        args.workload
    )

    namespace, bundle = load_v6(
        args.training_script,
        args.v6_bundle,
    )

    run_id = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S_%f")

    output = args.output / run_id
    output.mkdir(
        parents=True,
        exist_ok=False,
    )

    print("=" * 80)
    print("M1 V4 — PARALLEL AI INDEX OPTIMIZER")
    print("=" * 80)
    print("Unique SQL:", len(workload))
    print("V6 features:", len(namespace["FEATURES"]))
    print("Parallel workers:", args.workers)
    print("Portfolio budget:", args.budget)
    print("Output:", output)

    setup_conn = connection(args)

    try:
        with setup_conn.cursor() as cur:
            candidates = get_candidates(cur)
    finally:
        setup_conn.rollback()
        setup_conn.close()

    lookup = {
        name_of(candidate): candidate
        for candidate in candidates
    }

    if len(lookup) < args.budget:
        raise RuntimeError(
            "Not enough valid index candidates."
        )

    print("Valid candidates:", len(lookup))

    modes = (
        ["unseen_template", "within_template"]
        if args.split_mode == "both"
        else [args.split_mode]
    )

    all_results = []
    all_steps = []
    all_candidates = []
    all_splits = []

    experiment_start = time.perf_counter()

    # All workers are scoped to this executor.
    # Each evaluation creates its own DB connection.
    with ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:

        for mode in modes:
            for seed in range(
                args.start_seed,
                args.start_seed + args.seeds,
            ):
                train, validation, test = split_data(
                    workload,
                    seed,
                    mode,
                    args.holdout_fraction,
                    args.validation_fraction,
                )

                print("\n" + "=" * 80)
                print(
                    f"MODE: {mode} | SEED: {seed}"
                )
                print("=" * 80)

                print(
                    "Discovery:",
                    len(train),
                    "| Validation:",
                    len(validation),
                    "| Test:",
                    len(test),
                )

                all_splits.extend(
                    {
                        "mode": mode,
                        "seed": seed,
                        "partition": partition,
                        "query_id": row.query_id,
                        "fingerprint": row.fingerprint,
                        "template": row.template,
                    }
                    for partition, part in [
                        ("discovery", train),
                        ("validation", validation),
                        ("test", test),
                    ]
                    for row in part.itertuples(index=False)
                )

                # Separate cache for each dataset partition.
                # Never mix discovery, validation, and test.
                discovery_cache = {}
                validation_cache = {}
                test_cache = {}

                discovery_baseline_result = parallel_evaluate(
                    executor,
                    args,
                    [[]],
                    train,
                    lookup,
                    namespace,
                    bundle,
                    discovery_cache,
                    "discovery baseline",
                )[()]

                discovery_baseline = require_complete(
                    discovery_baseline_result,
                    len(train),
                    "Discovery baseline",
                )

                portfolios = [
                    [candidate]
                    for candidate in lookup
                ]

                discovery_results = parallel_evaluate(
                    executor,
                    args,
                    portfolios,
                    train,
                    lookup,
                    namespace,
                    bundle,
                    discovery_cache,
                    "candidate discovery",
                )

                singleton_scores = []

                for candidate in lookup:
                    key = (candidate,)

                    rows, failures = discovery_results[key]

                    if (
                        failures
                        or len(rows) != len(train)
                    ):
                        print(
                            "    Skipping incomplete candidate:",
                            candidate,
                        )
                        continue

                    planner_gain = gain_pct(
                        discovery_baseline,
                        rows,
                        "cost",
                    )

                    ai_gain = gain_pct(
                        discovery_baseline,
                        rows,
                        "predicted_ms",
                    )

                    used = sum(
                        int(row["index_used"])
                        for row in rows.values()
                    )

                    record = {
                        "mode": mode,
                        "seed": seed,
                        "candidate": candidate,
                        "planner_gain_pct": planner_gain,
                        "v6_predicted_gain_pct": ai_gain,
                        "hypothetical_index_used_queries": used,
                    }

                    singleton_scores.append(record)
                    all_candidates.append(record)

                if not singleton_scores:
                    raise RuntimeError(
                        "No valid discovery candidates."
                    )

                scores = pd.DataFrame(
                    singleton_scores
                )

                n = min(
                    args.shortlist,
                    len(scores),
                )

                planner_shortlist = (
                    scores.sort_values(
                        [
                            "planner_gain_pct",
                            "candidate",
                        ],
                        ascending=[False, True],
                    )
                    .head(n)["candidate"]
                    .tolist()
                )

                ai_shortlist = (
                    scores.sort_values(
                        [
                            "v6_predicted_gain_pct",
                            "candidate",
                        ],
                        ascending=[False, True],
                    )
                    .head(n)["candidate"]
                    .tolist()
                )

                shortlist = list(
                    dict.fromkeys(
                        planner_shortlist
                        + ai_shortlist
                    )
                )

                print(
                    "Combined shortlist:",
                    len(shortlist),
                )

                selected = {}

                for method in [
                    "ai_v6",
                    "planner_gain",
                ]:
                    portfolio, steps = parallel_greedy(
                        executor,
                        args,
                        validation,
                        shortlist,
                        lookup,
                        namespace,
                        bundle,
                        args.budget,
                        method,
                        validation_cache,
                    )

                    selected[method] = portfolio

                    all_steps.extend(
                        {
                            "mode": mode,
                            "seed": seed,
                            **step,
                        }
                        for step in steps
                    )

                rng = np.random.default_rng(seed)

                selected["random"] = list(
                    rng.choice(
                        sorted(lookup),
                        size=min(
                            args.budget,
                            len(lookup),
                        ),
                        replace=False,
                    )
                )

                selected["no_new_indexes"] = []

                print(
                    "AI portfolio:",
                    selected["ai_v6"],
                )

                print(
                    "Planner portfolio:",
                    selected["planner_gain"],
                )

                test_portfolios = list(
                    selected.values()
                )

                test_results = parallel_evaluate(
                    executor,
                    args,
                    test_portfolios,
                    test,
                    lookup,
                    namespace,
                    bundle,
                    test_cache,
                    "held-out evaluation",
                )

                baseline = require_complete(
                    test_results[()],
                    len(test),
                    "Held-out baseline",
                )

                changed = (
                    set(selected["ai_v6"])
                    != set(selected["planner_gain"])
                )

                for method, portfolio in selected.items():
                    key = tuple(sorted(portfolio))

                    rows = require_complete(
                        test_results[key],
                        len(test),
                        f"Held-out {method}",
                    )

                    planner_gain = gain_pct(
                        baseline,
                        rows,
                        "cost",
                    )

                    ai_gain = gain_pct(
                        baseline,
                        rows,
                        "predicted_ms",
                    )

                    improved = sum(
                        rows[q]["cost"]
                        < baseline[q]["cost"]
                        for q in baseline
                    )

                    used = sum(
                        int(row["index_used"])
                        for row in rows.values()
                    )

                    result = {
                        "mode": mode,
                        "seed": seed,
                        "method": method,
                        "selected_indexes": "; ".join(
                            portfolio
                        ),
                        "index_count": len(portfolio),
                        "heldout_queries": len(test),
                        "planner_gain_pct": planner_gain,
                        "v6_predicted_gain_pct": ai_gain,
                        "queries_improved": improved,
                        "hypothetical_index_used_queries": used,
                        "ai_changed_selection": changed,
                    }

                    all_results.append(result)

                    print(
                        f"  {method}: "
                        f"planner={planner_gain:.4f}% | "
                        f"V6={ai_gain:.4f}% | "
                        f"index-used={used}",
                        flush=True,
                    )

                # Checkpoint after each completed seed.
                save_csv(
                    output / "portfolio_comparison.csv",
                    all_results,
                )

                save_csv(
                    output / "selection_steps.csv",
                    all_steps,
                )

                save_csv(
                    output / "candidate_diagnostics.csv",
                    all_candidates,
                )

                save_csv(
                    output / "split_manifest.csv",
                    all_splits,
                )

                print(
                    "Elapsed seconds:",
                    round(
                        time.perf_counter()
                        - experiment_start,
                        1,
                    ),
                )

    results = pd.DataFrame(all_results)

    summary = (
        results.groupby(
            ["mode", "method"],
            as_index=False,
        )
        .agg(
            seeds=("seed", "nunique"),
            mean_planner_gain_pct=(
                "planner_gain_pct",
                "mean",
            ),
            median_planner_gain_pct=(
                "planner_gain_pct",
                "median",
            ),
            mean_v6_predicted_gain_pct=(
                "v6_predicted_gain_pct",
                "mean",
            ),
            mean_hypothetical_index_usage=(
                "hypothetical_index_used_queries",
                "mean",
            ),
        )
    )

    summary.to_csv(
        output / "method_summary.csv",
        index=False,
    )

    metadata = {
        "experiment": "M1 V4 parallel index selection",
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "database": BENCHMARK_DB,
        "workers": args.workers,
        "seeds": args.seeds,
        "start_seed": args.start_seed,
        "budget": args.budget,
        "shortlist": args.shortlist,
        "modes": modes,
        "unique_sql": len(workload),
        "candidate_count": len(lookup),
        "runtime_seconds": (
            time.perf_counter()
            - experiment_start
        ),
        "independent_worker_connections": True,
        "physical_indexes_created": False,
        "physical_indexes_committed": False,
        "physical_runtime_ablation": False,
        "limitations": [
            "HypoPG planner-cost results are estimates.",
            "V6 predictions are not observed execution times.",
            "Greedy optimization may find a local optimum.",
            "The model was not retrained in this experiment.",
            "No physical indexed-query runtime was measured.",
            "Parallel execution may not improve speed if "
            "the PostgreSQL container is resource-constrained.",
        ],
    }

    (
        output / "experiment_metadata.json"
    ).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("M1 V4 PARALLEL EXPERIMENT COMPLETE")
    print("=" * 80)

    print(
        summary.to_string(index=False)
    )

    print(
        "\nTotal execution time:",
        round(metadata["runtime_seconds"], 2),
        "seconds",
    )

    print("Results directory:", output)


if __name__ == "__main__":
    main()
