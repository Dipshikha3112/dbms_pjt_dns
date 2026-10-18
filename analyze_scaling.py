from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "results" / "phase5_scaling"

p10 = BASE / "scale_10k" / "benchmark_results.csv"
p50 = BASE / "scale_50k" / "benchmark_results.csv"

df10 = pd.read_csv(p10)
df50 = pd.read_csv(p50)

def analyze(df, scale):
    success = df[df["status"] == "success"].copy()
    grouped = success.groupby("query_family")["execution_time_ms"]

    summary = grouped.agg(
        queries="count",
        mean_ms="mean",
        median_ms="median",
        p95_ms=lambda x: x.quantile(0.95),
        max_ms="max"
    ).reset_index()

    summary["scale"] = scale
    return summary

s10 = analyze(df10, "10K")
s50 = analyze(df50, "50K")

comparison = s10.merge(
    s50,
    on="query_family",
    suffixes=("_10k", "_50k")
)

for metric in ["mean", "median", "p95"]:
    comparison[f"{metric}_slowdown"] = (
        comparison[f"{metric}_ms_50k"]
        / comparison[f"{metric}_ms_10k"].replace(0, float("nan"))
    )

comparison = comparison.sort_values(
    "mean_slowdown", ascending=False
)

output = BASE / "scaling_comparison.csv"
comparison.to_csv(output, index=False)

failures = pd.concat([
    df10.assign(scale="10K"),
    df50.assign(scale="50K")
])

failures = failures[failures["status"] != "success"]

failure_summary = (
    failures.groupby(["scale", "query_family", "status"])
    .size()
    .reset_index(name="count")
)

failure_summary.to_csv(
    BASE / "scaling_failures.csv",
    index=False
)

print("\n" + "=" * 75)
print("PHASE 5 - 10K VS 50K SCALING ANALYSIS")
print("=" * 75)

print("\nTOP 10 QUERY FAMILIES BY MEAN SLOWDOWN:\n")

print(
    comparison[
        ["query_family", "mean_ms_10k", "mean_ms_50k",
         "mean_slowdown", "p95_slowdown"]
    ].head(10).round(2).to_string(index=False)
)

print("\nFAILED QUERIES:\n")
print(failure_summary.to_string(index=False))

print("\nSaved:", output)
print("Saved:", BASE / "scaling_failures.csv")
