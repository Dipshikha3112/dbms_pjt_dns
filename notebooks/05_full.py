
from pathlib import Path
from datetime import datetime
import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results" / "phase5_scaling"
FINAL = BASE / "final_analysis"
FINAL.mkdir(parents=True, exist_ok=True)

SCALES = {
    "10K": {
        "database": "adaptive_healthcare_scale_10k",
        "file": BASE / "scale_10k" / "benchmark_results.csv",
    },
    "50K": {
        "database": "adaptive_healthcare_scale_50k",
        "file": BASE / "scale_50k" / "benchmark_results.csv",
    },
}

REQUIRED = {
    "query_hash",
    "workload_type",
    "query_family",
    "query_name",
    "complexity",
    "execution_time_ms",
    "status",
}

def save_csv(df, filename):
    path = FINAL / filename
    df.to_csv(path, index=False)
    print("Saved:", path)
    return path

def load_data():
    frames = {}

    for scale, config in SCALES.items():
        path = config["file"]

        if not path.exists():
            raise FileNotFoundError(
                f"Missing benchmark: {path}"
            )

        df = pd.read_csv(path)
        missing = REQUIRED - set(df.columns)

        if missing:
            raise ValueError(
                f"{scale} missing columns: {missing}"
            )

        df["execution_time_ms"] = pd.to_numeric(
            df["execution_time_ms"],
            errors="coerce"
        )

        df["status"] = (
            df["status"]
            .fillna("unknown")
            .astype(str)
            .str.lower()
            .str.strip()
        )

        df["scale"] = scale
        frames[scale] = df

        print(
            f"{scale}: {len(df)} queries loaded"
        )

    return frames

def latency_summary(df):
    successful = df[
        (df["status"] == "success")
        & df["execution_time_ms"].notna()
    ]

    times = successful["execution_time_ms"]

    return {
        "attempted": len(df),
        "successful": len(successful),
        "failed": len(df) - len(successful),
        "success_rate_percent": round(
            100 * len(successful) / len(df), 3
        ) if len(df) else 0,
        "mean_ms": float(times.mean()) if len(times) else None,
        "median_ms": float(times.median()) if len(times) else None,
        "p95_ms": float(times.quantile(.95)) if len(times) else None,
        "p99_ms": float(times.quantile(.99)) if len(times) else None,
        "max_ms": float(times.max()) if len(times) else None,
    }

def summarize_groups(df, column):
    successful = df[
        (df["status"] == "success")
        & df["execution_time_ms"].notna()
    ]

    result = (
        successful.groupby(column)["execution_time_ms"]
        .agg(
            queries="count",
            mean_ms="mean",
            median_ms="median",
            p95_ms=lambda s: s.quantile(.95),
            max_ms="max",
        )
        .reset_index()
    )

    return result

def compare_groups(a, b, column):
    left = summarize_groups(a, column)
    right = summarize_groups(b, column)

    result = left.merge(
        right,
        on=column,
        how="outer",
        suffixes=("_10k", "_50k")
    )

    for metric in ["mean", "median", "p95"]:
        old = result[f"{metric}_ms_10k"]
        new = result[f"{metric}_ms_50k"]

        result[f"{metric}_slowdown"] = (
            new / old.replace(0, np.nan)
        )

    return result.sort_values(
        "mean_slowdown",
        ascending=False,
        na_position="last"
    )

def compare_query_hashes(a, b):
    left = (
        a.groupby("query_hash")
        .agg(
            occurrences_10k=("query_hash", "size"),
            family_10k=("query_family", "first")
        )
        .reset_index()
    )

    right = (
        b.groupby("query_hash")
        .agg(
            occurrences_50k=("query_hash", "size"),
            family_50k=("query_family", "first")
        )
        .reset_index()
    )

    comparison = left.merge(
        right,
        on="query_hash",
        how="outer",
        indicator=True
    )

    matched = comparison[
        comparison["_merge"] == "both"
    ]

    same_frequency = (
        matched["occurrences_10k"]
        == matched["occurrences_50k"]
    ).all()

    return comparison, {
        "unique_hashes_10k": len(left),
        "unique_hashes_50k": len(right),
        "matched_hashes": len(matched),
        "identical_hash_sets": (
            len(matched) == len(left) == len(right)
        ),
        "matching_occurrence_counts": bool(
            same_frequency
            and len(matched) == len(left) == len(right)
        ),
    }

def paired_latency(a, b):
    left = a[
        (a["status"] == "success")
        & a["execution_time_ms"].notna()
    ].copy()

    right = b[
        (b["status"] == "success")
        & b["execution_time_ms"].notna()
    ].copy()

    # Pair repeated hashes by occurrence order.
    # This is not a true run-ID pairing, but avoids
    # a many-to-many merge on repeated hashes.
    left["occurrence"] = (
        left.groupby("query_hash").cumcount()
    )
    right["occurrence"] = (
        right.groupby("query_hash").cumcount()
    )

    paired = left[
        ["query_hash", "occurrence", "query_family",
         "execution_time_ms"]
    ].merge(
        right[
            ["query_hash", "occurrence",
             "execution_time_ms"]
        ],
        on=["query_hash", "occurrence"],
        how="inner",
        suffixes=("_10k", "_50k")
    )

    paired["slowdown"] = (
        paired["execution_time_ms_50k"]
        / paired["execution_time_ms_10k"].replace(
            0, np.nan
        )
    )

    return paired.sort_values(
        "slowdown",
        ascending=False
    )

def failure_analysis(a, b):
    combined = pd.concat([a, b], ignore_index=True)
    failed = combined[
        combined["status"] != "success"
    ].copy()

    if failed.empty:
        return failed, pd.DataFrame()

    summary = (
        failed.groupby(
            ["scale", "query_family", "status"]
        )
        .size()
        .reset_index(name="count")
    )

    return failed, summary

def make_plots(overall, family):
    metrics = ["mean_ms", "median_ms", "p95_ms"]
    names = ["Mean", "Median", "P95"]

    x = np.arange(len(metrics))
    width = .35

    values10 = [
        overall["10K"][m] or 0 for m in metrics
    ]
    values50 = [
        overall["50K"][m] or 0 for m in metrics
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        x - width / 2,
        values10,
        width,
        label="10K"
    )
    ax.bar(
        x + width / 2,
        values50,
        width,
        label="50K"
    )

    ax.set_xticks(x, names)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("10K vs 50K Query Latency")
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        FINAL / "latency_comparison.png",
        dpi=180
    )
    plt.close(fig)

    top = family.dropna(
        subset=["mean_slowdown"]
    ).head(12)

    if not top.empty:
        fig, ax = plt.subplots(
            figsize=(11, 6)
        )

        ax.barh(
            top["query_family"].iloc[::-1],
            top["mean_slowdown"].iloc[::-1]
        )

        ax.set_xlabel("Mean latency slowdown (x)")
        ax.set_title(
            "Query Families Most Affected by Scaling"
        )

        fig.tight_layout()
        fig.savefig(
            FINAL / "family_slowdown.png",
            dpi=180
        )
        plt.close(fig)

def inspect_databases():
    try:
        import psycopg2
    except ImportError:
        print(
            "psycopg2 unavailable; skipping database checks"
        )
        return {}

    password = os.environ.get("PGPASSWORD")

    if not password:
        print(
            "PGPASSWORD not set; skipping database checks"
        )
        return {}

    results = {}

    for scale, config in SCALES.items():
        try:
            conn = psycopg2.connect(
                host="localhost",
                port=5432,
                user="postgres",
                password=password,
                dbname=config["database"],
                connect_timeout=10
            )

            conn.set_session(
                readonly=True,
                autocommit=True
            )

            with conn.cursor() as cur:
                cur.execute(
                    "SET statement_timeout = '15000ms'"
                )

                cur.execute("""
                    SELECT pg_database_size(
                        current_database()
                    )
                """)
                size_bytes = cur.fetchone()[0]

                cur.execute("""
                    SELECT COUNT(*) FROM patient
                """)
                patients = cur.fetchone()[0]

                cur.execute("""
                    SELECT COUNT(*)
                    FROM pg_tables
                    WHERE schemaname = 'public'
                """)
                tables = cur.fetchone()[0]

                cur.execute("""
                    SELECT COUNT(*)
                    FROM pg_indexes
                    WHERE schemaname = 'public'
                """)
                indexes = cur.fetchone()[0]

                cur.execute("""
                    SELECT COUNT(*)
                    FROM pg_views
                    WHERE schemaname = 'public'
                """)
                views = cur.fetchone()[0]

            conn.close()

            results[scale] = {
                "database": config["database"],
                "size_bytes": size_bytes,
                "size_gb": round(
                    size_bytes / (1024 ** 3), 3
                ),
                "patients": patients,
                "tables": tables,
                "indexes": indexes,
                "views": views,
            }

            print(
                f"{scale}: {patients:,} patients, "
                f"{size_bytes / (1024 ** 3):.2f} GiB"
            )

        except Exception as exc:
            results[scale] = {
                "error": str(exc)
            }
            print(f"{scale} DB check failed: {exc}")

    return results

def prepare_100k():
    scaling_script = ROOT / "notebooks" / "05_scaling.py"
    synthea_jar = (
        ROOT / "tools" / "synthea"
        / "synthea-with-dependencies.jar"
    )

    if not scaling_script.exists():
        raise FileNotFoundError(scaling_script)

    if not synthea_jar.exists():
        raise FileNotFoundError(synthea_jar)

    print("\nPreparing 100K Synthea workspace.")
    print("This generates a very large dataset.")
    print("It does NOT import data into PostgreSQL.")

    command = [
        sys.executable,
        str(scaling_script),
        "--patients", "100000",
        "--synthea-jar", str(synthea_jar),
        "--prepare-only",
    ]

    subprocess.run(
        command,
        cwd=str(scaling_script.parent),
        check=True
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepare-100k",
        action="store_true",
        help="Generate and stage 100K Synthea data"
    )
    args = parser.parse_args()

    print("=" * 75)
    print("PHASE 5 - FINAL SCALING ANALYSIS")
    print("=" * 75)

    frames = load_data()
    a = frames["10K"]
    b = frames["50K"]

    overall = {
        "10K": latency_summary(a),
        "50K": latency_summary(b),
    }

    overall_df = pd.DataFrame([
        {"scale": scale, **metrics}
        for scale, metrics in overall.items()
    ])

    save_csv(
        overall_df,
        "overall_comparison.csv"
    )

    family = compare_groups(
        a, b, "query_family"
    )
    save_csv(
        family,
        "query_family_comparison.csv"
    )

    workload = compare_groups(
        a, b, "workload_type"
    )
    save_csv(
        workload,
        "workload_comparison.csv"
    )

    complexity = compare_groups(
        a, b, "complexity"
    )
    save_csv(
        complexity,
        "complexity_comparison.csv"
    )

    hashes, hash_info = compare_query_hashes(a, b)
    save_csv(
        hashes,
        "query_hash_comparison.csv"
    )

    paired = paired_latency(a, b)
    save_csv(
        paired,
        "paired_query_comparison.csv"
    )

    failures, failure_summary = failure_analysis(
        a, b
    )
    save_csv(
        failures,
        "failed_queries.csv"
    )
    save_csv(
        failure_summary,
        "failure_summary.csv"
    )

    make_plots(overall, family)

    db_checks = inspect_databases()

    report = {
        "phase": 5,
        "title": "ETL and Scaling",
        "generated_at": datetime.now().isoformat(),
        "overall": overall,
        "query_hash_validation": hash_info,
        "paired_successful_queries": len(paired),
        "database_checks": db_checks,
        "outputs": [
            p.name for p in FINAL.iterdir()
            if p.is_file()
        ],
        "status": (
            "10K_AND_50K_ANALYZED_100K_PENDING"
        ),
        "notes": [
            "Latency metrics use successful queries only.",
            "Query hashes are checked for workload consistency.",
            "Repeated queries are paired by occurrence order.",
            "The 100K target remains pending.",
            "Database checks are read-only.",
        ]
    }

    report_path = FINAL / "phase5_final_report.json"
    report_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    print("\n" + "=" * 75)
    print("OVERALL RESULTS")
    print("=" * 75)
    print(
        overall_df.round(3).to_string(
            index=False
        )
    )

    print("\nTOP 10 SLOWEST-SCALING QUERY FAMILIES")
    print(
        family[
            [
                "query_family",
                "mean_ms_10k",
                "mean_ms_50k",
                "mean_slowdown",
                "p95_slowdown"
            ]
        ].head(10).round(3).to_string(
            index=False
        )
    )

    print("\nWORKLOAD CONSISTENCY")
    print(json.dumps(hash_info, indent=2))

    print("\nFAILED QUERIES")
    if failure_summary.empty:
        print("None")
    else:
        print(failure_summary.to_string(index=False))

    print("\nFINAL REPORT:", report_path)
    print("Analysis completed without rerunning ETL.")

    if args.prepare_100k:
        prepare_100k()

if __name__ == "__main__":
    main()
