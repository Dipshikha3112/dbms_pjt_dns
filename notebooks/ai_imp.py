
import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import ai_omptimization as v4


TARGET_GAIN = 50.0
MAX_CANDIDATES = 32
MAX_GENERATED_PER_TABLE = 8
RANDOM_BASELINES = 5

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]

IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$", re.I)

SQL_KEYWORDS = {
    "where", "join", "left", "right", "inner", "outer",
    "full", "cross", "on", "group", "order", "limit",
    "having", "union", "offset", "as", "select"
}


def portfolio_key(portfolio):
    return tuple(sorted(portfolio))


def candidate_name(table, columns):
    return f"{table}({', '.join(columns)})"


def validate_identifier(value):
    return bool(IDENTIFIER.fullmatch(str(value)))


def get_schema(cur):
    cur.execute("""
        SELECT table_name, column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
    """)

    schema = defaultdict(set)

    for table, column in cur.fetchall():
        schema[table].add(column)

    return dict(schema)


def get_existing_indexes(cur):
    cur.execute("""
        SELECT
            t.relname,
            array_agg(
                lower(pg_get_indexdef(i.indexrelid, k.n, true))
                ORDER BY k.n
            ) AS key_columns
        FROM pg_index i
        JOIN pg_class t
          ON t.oid = i.indrelid
        JOIN pg_namespace ns
          ON ns.oid = t.relnamespace
        CROSS JOIN LATERAL
            generate_series(1, i.indnkeyatts) AS k(n)
        WHERE ns.nspname = 'public'
          AND i.indisvalid
          AND i.indisready
          AND i.indpred IS NULL
          AND i.indexprs IS NULL
          AND pg_get_indexdef(i.indexrelid)
              ILIKE 'CREATE%INDEX%USING btree%'
        GROUP BY t.relname, i.indexrelid
    """)

    indexes = defaultdict(list)

    for table, columns in cur.fetchall():
        indexes[table].append(tuple(columns))

    return dict(indexes)


def normalized_column(value):
    value = str(value).lower().strip()
    value = value.replace('"', "")
    value = re.sub(
        r"\s+(asc|desc)(\s+nulls\s+(first|last))?$",
        "",
        value
    )
    return value.strip()


def is_redundant(table, columns, existing):
    wanted = tuple(normalized_column(c) for c in columns)

    for physical in existing.get(table, []):
        current = tuple(
            normalized_column(c) for c in physical
        )

        if len(current) < len(wanted):
            continue

        if current[:len(wanted)] == wanted:
            return True

    return False


def extract_aliases(sql, schema):
    aliases = {}

    pattern = re.compile(
        r"\b(?:FROM|JOIN)\s+"
        r"(?:public\.)?"
        r"([a-z_][a-z0-9_]*)"
        r"(?:\s+(?:AS\s+)?([a-z_][a-z0-9_]*))?",
        re.I
    )

    for match in pattern.finditer(sql):
        table = match.group(1).lower()
        alias = match.group(2)

        if table not in schema:
            continue

        aliases[table] = table

        if alias:
            alias = alias.lower()

            if alias not in SQL_KEYWORDS:
                aliases[alias] = table

    return aliases


def extract_sql_columns(sql, schema):
    aliases = extract_aliases(sql, schema)

    found = defaultdict(set)

    pattern = re.compile(
        r"\b([a-z_][a-z0-9_]*)\."
        r"([a-z_][a-z0-9_]*)\b",
        re.I
    )

    for alias, column in pattern.findall(sql):
        table = aliases.get(alias.lower())
        column = column.lower()

        if table and column in schema[table]:
            found[table].add(column)

    return found


def discover_candidates(workload, schema, existing):
    frequency = Counter()
    pairs = Counter()

    for sql in workload["query_text"].astype(str):
        columns_by_table = extract_sql_columns(
            sql, schema
        )

        for table, columns in columns_by_table.items():
            columns = sorted(columns)

            for column in columns:
                frequency[(table, column)] += 1

            for first in columns:
                for second in columns:
                    if first == second:
                        continue

                    pairs[(table, first, second)] += 1

    candidates = {}

    for table, columns in v4.CANDIDATES:
        if table not in schema:
            continue

        names = [
            c.split()[0].lower() for c in columns
        ]

        if not all(c in schema[table] for c in names):
            continue

        if is_redundant(table, columns, existing):
            continue

        name = candidate_name(table, columns)
        candidates[name] = (table, tuple(columns))

    ranked_singles = frequency.most_common()

    table_counts = Counter()

    for (table, column), count in ranked_singles:
        if table_counts[table] >= MAX_GENERATED_PER_TABLE:
            continue

        if is_redundant(table, (column,), existing):
            continue

        name = candidate_name(table, (column,))

        candidates[name] = (table, (column,))
        table_counts[table] += 1

    for (table, first, second), count in pairs.most_common(40):
        if count < 2:
            continue

        columns = (first, second)

        if is_redundant(table, columns, existing):
            continue

        name = candidate_name(table, columns)
        candidates[name] = (table, columns)

    ranked = []

    for name, (table, columns) in candidates.items():
        base_columns = [
            normalized_column(c) for c in columns
        ]

        support = sum(
            frequency[(table, c)]
            for c in base_columns
        )

        if len(base_columns) >= 2:
            support += 2 * pairs[
                (table, base_columns[0], base_columns[1])
            ]

        ranked.append((support, name))

    ranked.sort(key=lambda x: (-x[0], x[1]))

    selected = [
        name for _, name in ranked[:MAX_CANDIDATES]
    ]

    return {
        name: candidates[name]
        for name in selected
    }, frequency


def evaluate(
    executor,
    args,
    portfolios,
    queries,
    lookup,
    namespace,
    bundle,
    cache,
    label
):
    return v4.parallel_evaluate(
        executor,
        args,
        portfolios,
        queries,
        lookup,
        namespace,
        bundle,
        cache,
        label
    )


def complete(result, expected, label):
    return v4.require_complete(
        result, expected, label
    )


def total_gain(before, after, field):
    return float(
        v4.gain_pct(before, after, field)
    )


def family_metrics(before, after, queries):
    family_queries = defaultdict(list)

    for row in queries.itertuples(index=False):
        family_queries[str(row.query_family)].append(
            row.query_id
        )

    gains = []
    covered = 0

    for family, query_ids in family_queries.items():
        baseline = sum(
            max(float(before[q]["cost"]), 0.0)
            for q in query_ids
        )

        optimized = sum(
            max(float(after[q]["cost"]), 0.0)
            for q in query_ids
        )

        if baseline <= 0:
            continue

        gain = 100.0 * (
            baseline - optimized
        ) / baseline

        gains.append(gain)

        if gain > 1e-6:
            covered += 1

    family_gain = (
        float(np.mean(gains)) if gains else 0.0
    )

    coverage = (
        covered / len(gains) if gains else 0.0
    )

    return family_gain, coverage


def score_portfolio(
    baseline,
    result,
    queries,
    alpha
):
    planner = total_gain(
        baseline, result, "cost"
    )

    ai = total_gain(
        baseline, result, "predicted_ms"
    )

    family_gain, coverage = family_metrics(
        baseline, result, queries
    )

    # Planner cost remains the primary objective.
    # The AI signal is auxiliary.
    planner_weight = 0.80 + 0.15 * alpha
    ai_weight = 0.15 * (1.0 - alpha)

    score = (
        planner_weight * planner
        + ai_weight * ai
        + 0.05 * family_gain
    )

    # Coverage is a small tie-breaking incentive,
    # not an artificial reported performance gain.
    score += 0.05 * coverage

    return {
        "score": float(score),
        "planner_gain_pct": float(planner),
        "v6_predicted_gain_pct": float(ai),
        "family_gain_pct": float(family_gain),
        "family_coverage": float(coverage)
    }


def evaluate_singletons(
    executor,
    args,
    discovery,
    lookup,
    namespace,
    bundle,
    cache,
    baseline
):
    portfolios = [[name] for name in lookup]

    evaluated = evaluate(
        executor,
        args,
        portfolios,
        discovery,
        lookup,
        namespace,
        bundle,
        cache,
        "candidate discovery"
    )

    rows = []

    for name in lookup:
        result = complete(
            evaluated[(name,)],
            len(discovery),
            f"candidate {name}"
        )

        metrics = score_portfolio(
            baseline, result, discovery, 1.0
        )

        usage = sum(
            int(r["index_used"])
            for r in result.values()
        )

        rows.append({
            "candidate": name,
            **metrics,
            "index_used_queries": usage
        })

    return pd.DataFrame(rows)


def build_shortlist(diagnostics, size):
    planner = diagnostics.sort_values(
        ["planner_gain_pct", "candidate"],
        ascending=[False, True]
    )["candidate"].tolist()

    ai = diagnostics.sort_values(
        ["v6_predicted_gain_pct", "candidate"],
        ascending=[False, True]
    )["candidate"].tolist()

    family = diagnostics.sort_values(
        ["family_gain_pct", "candidate"],
        ascending=[False, True]
    )["candidate"].tolist()

    pool = list(dict.fromkeys(
        planner[:size]
        + ai[:size]
        + family[:size]
    ))

    return pool


def greedy_portfolio(
    executor,
    args,
    discovery,
    candidates,
    lookup,
    namespace,
    bundle,
    cache,
    baseline,
    alpha
):
    selected = []
    steps = []
    best_score = score_portfolio(
        baseline, baseline, discovery, alpha
    )["score"]

    for round_number in range(
        1, args.budget + 1
    ):
        remaining = [
            c for c in candidates
            if c not in selected
        ]

        if not remaining:
            break

        proposals = [
            selected + [candidate]
            for candidate in remaining
        ]

        results = evaluate(
            executor,
            args,
            proposals,
            discovery,
            lookup,
            namespace,
            bundle,
            cache,
            f"greedy alpha={alpha} round={round_number}"
        )

        options = []

        for candidate, proposal in zip(
            remaining, proposals
        ):
            result = complete(
                results[portfolio_key(proposal)],
                len(discovery),
                f"greedy {candidate}"
            )

            metrics = score_portfolio(
                baseline,
                result,
                discovery,
                alpha
            )

            options.append((
                metrics["score"],
                metrics["planner_gain_pct"],
                candidate,
                metrics
            ))

        if not options:
            break

        options.sort(
            key=lambda x: (-x[0], -x[1], x[2])
        )

        score, planner, winner, metrics = options[0]

        if score <= best_score + 1e-7:
            break

        selected.append(winner)
        best_score = score

        steps.append({
            "alpha": alpha,
            "round": round_number,
            "candidate": winner,
            **metrics
        })

        print(
            f"  alpha={alpha:.2f} "
            f"round={round_number} "
            f"selected={winner} "
            f"discovery_gain={planner:.4f}%",
            flush=True
        )

    return selected, steps


def score_methods(
    executor,
    args,
    methods,
    queries,
    lookup,
    namespace,
    bundle,
    cache,
    label
):
    portfolios = list(methods.values()) + [[]]

    evaluated = evaluate(
        executor,
        args,
        portfolios,
        queries,
        lookup,
        namespace,
        bundle,
        cache,
        label
    )

    baseline = complete(
        evaluated[()],
        len(queries),
        f"{label} baseline"
    )

    output = []

    for method, portfolio in methods.items():
        result = complete(
            evaluated[portfolio_key(portfolio)],
            len(queries),
            f"{label} {method}"
        )

        metrics = score_portfolio(
            baseline,
            result,
            queries,
            1.0
        )

        usage = sum(
            int(row["index_used"])
            for row in result.values()
        )

        improved = sum(
            result[q]["cost"] < baseline[q]["cost"]
            for q in baseline
        )

        output.append({
            "method": method,
            "selected_indexes": "; ".join(portfolio),
            "index_count": len(portfolio),
            "evaluated_queries": len(queries),
            **metrics,
            "index_used_queries": usage,
            "improved_queries": improved
        })

    return pd.DataFrame(output)


def select_validation_winner(
    validation_scores,
    paths
):
    ranked = validation_scores.copy()

    ranked["alpha"] = ranked["method"].str.replace(
        "alpha_", "", regex=False
    ).astype(float)

    # Validation uses planner cost as the main criterion.
    # A lower number of indexes wins an exact tie.
    ranked = ranked.sort_values(
        [
            "planner_gain_pct",
            "family_gain_pct",
            "index_count",
            "alpha"
        ],
        ascending=[False, False, True, False]
    )

    winner = ranked.iloc[0]

    return (
        float(winner["alpha"]),
        paths[float(winner["alpha"])]
    )


def random_portfolios(lookup, budget, seed):
    rng = np.random.default_rng(seed)

    names = sorted(lookup)

    output = {}

    for i in range(RANDOM_BASELINES):
        selected = rng.choice(
            names,
            size=min(budget, len(names)),
            replace=False
        ).tolist()

        output[f"random_{i + 1}"] = selected

    return output


def save_csv(path, rows):
    if rows:
        pd.DataFrame(rows).to_csv(
            path, index=False
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workload",
        type=Path,
        default=v4.DEFAULT_WORKLOAD
    )

    parser.add_argument(
        "--v6-bundle",
        type=Path,
        default=v4.DEFAULT_MODEL
    )

    parser.add_argument(
        "--training-script",
        type=Path,
        default=v4.DEFAULT_V6_SCRIPT
    )

    parser.add_argument(
        "--host", default="127.0.0.1"
    )

    parser.add_argument(
        "--port", type=int, default=5433
    )

    parser.add_argument(
        "--database",
        default=v4.BENCHMARK_DB
    )

    parser.add_argument(
        "--user", default="postgres"
    )

    parser.add_argument(
        "--password",
        default=os.getenv("PGPASSWORD")
    )

    parser.add_argument(
        "--workers", type=int, default=3
    )

    parser.add_argument(
        "--seeds", type=int, default=1
    )

    parser.add_argument(
        "--start-seed", type=int, default=42
    )

    parser.add_argument(
        "--budget", type=int, default=3
    )

    parser.add_argument(
        "--shortlist", type=int, default=8
    )

    parser.add_argument(
        "--scale", type=int, default=1000
    )

    parser.add_argument(
        "--split-mode",
        choices=[
            "unseen_template",
            "within_template"
        ],
        default="unseen_template"
    )

    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.25
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20
    )

    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=15000
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            v4.ROOT
            / "results"
            / "phase9_m1_v6"
        )
    )

    args = parser.parse_args()

    if not 1 <= args.workers <= 8:
        raise ValueError("Workers must be 1 to 8")

    if args.seeds < 1:
        raise ValueError("Seeds must be positive")

    if args.budget < 1:
        raise ValueError("Budget must be positive")

    if args.shortlist < args.budget:
        raise ValueError(
            "Shortlist must be at least the budget"
        )

    if (
        args.holdout_fraction <= 0
        or args.validation_fraction <= 0
        or args.holdout_fraction
        + args.validation_fraction >= 0.8
    ):
        raise ValueError("Invalid split fractions")

    start = time.perf_counter()

    workload = v4.load_workload(args.workload)

    namespace, bundle = v4.load_v6(
        args.training_script,
        args.v6_bundle
    )

    conn = v4.connection(args)

    try:
        with conn.cursor() as cur:
            schema = get_schema(cur)
            existing = get_existing_indexes(cur)
    finally:
        conn.rollback()
        conn.close()

    lookup, frequency = discover_candidates(
        workload,
        schema,
        existing
    )

    if len(lookup) < args.budget:
        raise RuntimeError(
            "Not enough nonredundant candidates"
        )

    run_id = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S_%f")

    output = args.output / run_id
    output.mkdir(
        parents=True,
        exist_ok=False
    )

    all_comparisons = []
    all_candidates = []
    all_validation = []
    all_steps = []
    all_splits = []

    print("=" * 78)
    print("M1 V6 — COVERAGE-AWARE INDEX OPTIMIZER")
    print("=" * 78)
    print("Database:", args.database)
    print("Queries:", len(workload))
    print("Candidates:", len(lookup))
    print("Index budget:", args.budget)
    print("Target gain:", TARGET_GAIN, "%")
    print("Workers:", args.workers)
    print("Output:", output)
    print("Physical index creation: DISABLED")
    print("=" * 78)

    with ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:

        for seed in range(
            args.start_seed,
            args.start_seed + args.seeds
        ):
            discovery, validation, test = (
                v4.split_data(
                    workload,
                    seed,
                    args.split_mode,
                    args.holdout_fraction,
                    args.validation_fraction
                )
            )

            print(
                f"\nSEED={seed} "
                f"DISCOVERY={len(discovery)} "
                f"VALIDATION={len(validation)} "
                f"TEST={len(test)}",
                flush=True
            )

            for partition, frame in [
                ("discovery", discovery),
                ("validation", validation),
                ("test", test)
            ]:
                for row in frame.itertuples(index=False):
                    all_splits.append({
                        "seed": seed,
                        "partition": partition,
                        "query_id": row.query_id,
                        "fingerprint": row.fingerprint,
                        "template": row.template
                    })

            discovery_cache = {}
            validation_cache = {}
            test_cache = {}

            baseline_result = evaluate(
                executor,
                args,
                [[]],
                discovery,
                lookup,
                namespace,
                bundle,
                discovery_cache,
                "discovery baseline"
            )

            baseline = complete(
                baseline_result[()],
                len(discovery),
                "discovery baseline"
            )

            diagnostics = evaluate_singletons(
                executor,
                args,
                discovery,
                lookup,
                namespace,
                bundle,
                discovery_cache,
                baseline
            )

            diagnostics["seed"] = seed

            all_candidates.extend(
                diagnostics.to_dict("records")
            )

            candidate_pool = build_shortlist(
                diagnostics,
                args.shortlist
            )

            print(
                "Candidate shortlist:",
                len(candidate_pool),
                flush=True
            )

            paths = {}

            for alpha in ALPHAS:
                portfolio, steps = greedy_portfolio(
                    executor,
                    args,
                    discovery,
                    candidate_pool,
                    lookup,
                    namespace,
                    bundle,
                    discovery_cache,
                    baseline,
                    alpha
                )

                paths[alpha] = portfolio

                for step in steps:
                    all_steps.append({
                        "seed": seed,
                        **step
                    })

            validation_methods = {
                f"alpha_{alpha}": paths[alpha]
                for alpha in ALPHAS
            }

            validation_scores = score_methods(
                executor,
                args,
                validation_methods,
                validation,
                lookup,
                namespace,
                bundle,
                validation_cache,
                "validation"
            )

            validation_scores["seed"] = seed

            all_validation.extend(
                validation_scores.to_dict("records")
            )

            best_alpha, best_portfolio = (
                select_validation_winner(
                    validation_scores,
                    paths
                )
            )

            print(
                "Validation-selected alpha:",
                best_alpha
            )

            print(
                "Selected portfolio:",
                best_portfolio
            )

            planner_portfolio = paths[1.0]
            ai_portfolio = paths[0.0]

            planner_rank = diagnostics.sort_values(
                ["planner_gain_pct", "candidate"],
                ascending=[False, True]
            )["candidate"].tolist()

            methods = {
                "v6_coverage_aware": best_portfolio,
                "planner_greedy": planner_portfolio,
                "ai_v6_only": ai_portfolio,
                "discovery_topk":
                    planner_rank[:args.budget],
                **random_portfolios(
                    lookup, args.budget, seed
                ),
                "no_new_indexes": []
            }

            test_scores = score_methods(
                executor,
                args,
                methods,
                test,
                lookup,
                namespace,
                bundle,
                test_cache,
                "held-out test"
            )

            test_scores["seed"] = seed
            test_scores["split_mode"] = args.split_mode

            all_comparisons.extend(
                test_scores.to_dict("records")
            )

            print("\nHELD-OUT RESULTS")

            print(
                test_scores[
                    [
                        "method",
                        "planner_gain_pct",
                        "v6_predicted_gain_pct",
                        "index_used_queries"
                    ]
                ].to_string(index=False)
            )

            chosen = test_scores[
                test_scores["method"]
                == "v6_coverage_aware"
            ].iloc[0]

            achieved = float(
                chosen["planner_gain_pct"]
            )

            if achieved >= TARGET_GAIN:
                print(
                    f"\nTARGET ACHIEVED: {achieved:.4f}%"
                )
            else:
                print(
                    f"\nTARGET NOT ACHIEVED: "
                    f"{achieved:.4f}% "
                    f"(target {TARGET_GAIN:.1f}%)"
                )

            save_csv(
                output / "portfolio_comparison.csv",
                all_comparisons
            )

            save_csv(
                output / "candidate_diagnostics.csv",
                all_candidates
            )

            save_csv(
                output / "validation_alpha_tuning.csv",
                all_validation
            )

            save_csv(
                output / "selection_steps.csv",
                all_steps
            )

            save_csv(
                output / "split_manifest.csv",
                all_splits
            )

            print(
                "Elapsed seconds:",
                round(time.perf_counter() - start, 1)
            )

    results = pd.DataFrame(all_comparisons)

    summary = results.groupby(
        "method",
        as_index=False
    ).agg(
        seeds=("seed", "nunique"),
        mean_planner_gain_pct=(
            "planner_gain_pct", "mean"
        ),
        median_planner_gain_pct=(
            "planner_gain_pct", "median"
        ),
        mean_v6_predicted_gain_pct=(
            "v6_predicted_gain_pct", "mean"
        ),
        mean_index_usage=(
            "index_used_queries", "mean"
        )
    )

    summary.to_csv(
        output / "method_summary.csv",
        index=False
    )

    metadata = {
        "experiment": "M1 V6 coverage-aware optimizer",
        "database": args.database,
        "workload": str(args.workload),
        "seed_start": args.start_seed,
        "seed_count": args.seeds,
        "split_mode": args.split_mode,
        "budget": args.budget,
        "candidate_count": len(lookup),
        "target_planner_gain_pct": TARGET_GAIN,
        "runtime_seconds": round(
            time.perf_counter() - start, 2
        ),
        "physical_indexes_created": False,
        "physical_runtime_measured": False,
        "heldout_used_for_selection": False,
        "limitations": [
            "HypoPG costs are planner estimates",
            "V6 predictions are not observed runtimes",
            "A 50 percent gain is not guaranteed",
            "Candidate extraction uses conservative SQL parsing",
            "Existing-index filtering is heuristic",
            "Unseen-template generalization remains empirical"
        ]
    }

    (
        output / "experiment_metadata.json"
    ).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8"
    )

    print("\n" + "=" * 78)
    print("M1 V6 EXPERIMENT COMPLETE")
    print("=" * 78)
    print(summary.to_string(index=False))
    print("\nResults:", output)


if __name__ == "__main__":
    main()
