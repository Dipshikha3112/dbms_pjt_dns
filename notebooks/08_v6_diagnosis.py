
import argparse
import json
import re
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import (
    ks_2samp,
    mannwhitneyu,
    wilcoxon,
    binomtest,
    spearmanr
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
V6 = ROOT / "results" / "phase8_v6_validation"
V5 = ROOT / "results" / "phase8_v5"
P7 = ROOT / "results" / "phase7_monitoring"
OUT = V6 / "diagnostics_v6_1"
OUT.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)

ALIASES = {
    "fingerprint": [
        "fingerprint", "sql_fingerprint", "query_fingerprint",
        "fingerprint_id", "query_hash", "sql_hash"
    ],
    "scale": [
        "scale", "database_scale", "dataset_scale",
        "scale_label", "workload_scale"
    ],
    "session": [
        "session_id", "session", "run_id",
        "collection_session", "benchmark_session"
    ],
    "latency": [
        "latency_ms", "execution_time_ms", "elapsed_ms",
        "duration_ms", "runtime_ms", "actual_latency_ms",
        "query_latency_ms", "wall_latency_ms"
    ],
    "pred_latency": [
        "predicted_latency_ms", "pred_latency_ms",
        "latency_prediction_ms", "y_pred_ms",
        "regression_prediction_ms", "prediction_ms",
        "predicted_ms"
    ],
    "true_class": [
        "y_true", "true_class", "actual_class",
        "is_expensive", "actual_expensive",
        "expensive_actual", "label", "target_class",
        "classification_target"
    ],
    "pred_class": [
        "y_pred", "predicted_class", "pred_class",
        "predicted_expensive", "is_expensive_pred",
        "classification_prediction", "class_prediction"
    ],
    "probability": [
        "predicted_probability", "expensive_probability",
        "class_probability", "prob_expensive",
        "probability", "y_score"
    ],
    "status": [
        "status", "execution_status", "query_status",
        "success", "successful", "is_success"
    ],
    "error": [
        "error", "error_message", "exception",
        "failure_reason", "failure_message"
    ],
    "sql": [
        "sql", "sql_text", "query_sql", "query_text",
        "query", "statement"
    ],
    "planner_cost": [
        "planner_total_cost", "plan_total_cost",
        "total_cost", "estimated_cost", "planner_cost",
        "explain_total_cost", "root_total_cost"
    ],
    "planner_rows": [
        "planner_rows", "plan_rows", "estimated_rows",
        "plan_estimated_rows", "root_plan_rows",
        "root_rows"
    ],
    "join_count": [
        "join_count", "num_joins", "n_joins",
        "sql_join_count", "number_of_joins"
    ],
    "table_count": [
        "table_count", "num_tables", "n_tables",
        "number_of_tables"
    ],
    "plan_nodes": [
        "plan_node_count", "planner_node_count",
        "num_plan_nodes", "plan_nodes", "node_count"
    ]
}

def section(title):
    print("\n" + "=" * 84)
    print(title)
    print("=" * 84)

def norm(x):
    return re.sub(r"[^a-z0-9]+", "_", str(x).lower()).strip("_")

def find_col(df, kind):
    mapping = {norm(c): c for c in df.columns}
    for candidate in ALIASES[kind]:
        if norm(candidate) in mapping:
            return mapping[norm(candidate)]
    return None

def numeric(s):
    return pd.to_numeric(s, errors="coerce")

def save_json(name, data):
    path = OUT / name
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            x = float(x)
        if isinstance(x, float) and not np.isfinite(x):
            return None
        if isinstance(x, Path):
            return str(x)
        return x
    path.write_text(json.dumps(clean(data), indent=2), encoding="utf-8")

def save_csv(name, df):
    df.to_csv(OUT / name, index=False)

def all_csv(folder):
    if not folder.exists():
        return []
    return [
        p for p in folder.rglob("*.csv")
        if OUT not in p.parents
    ]

def load_csv(path):
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as e:
        print(f"[SKIP] {path.name}: {e}")
        return None

def inventory():
    section("1. FILE INVENTORY")
    rows = []
    for origin, folder in [("V6", V6), ("V5", V5), ("P7", P7)]:
        for p in all_csv(folder):
            try:
                df = load_csv(p)
                if df is None:
                    continue
                rows.append({
                    "origin": origin,
                    "path": str(p),
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": " | ".join(map(str, df.columns))
                })
                print(
                    f"{origin}: {p.relative_to(ROOT)} "
                    f"rows={len(df)} cols={len(df.columns)}"
                )
            except Exception as e:
                print("[ERROR]", p, e)
    result = pd.DataFrame(rows)
    if len(result):
        save_csv("00_file_inventory.csv", result)
    return result

def pick_v6(inventory_df, override=None):
    if override:
        p = Path(override).resolve()
        df = load_csv(p)
        if df is None:
            raise RuntimeError("Cannot read the selected V6 CSV")
        return df, p

    candidates = []
    if inventory_df.empty:
        raise RuntimeError("No CSV files found")

    for _, row in inventory_df.iterrows():
        if row["origin"] != "V6":
            continue

        p = Path(row["path"])
        df = load_csv(p)
        if df is None or len(df) == 0:
            continue

        has_fp = find_col(df, "fingerprint") is not None
        has_latency = find_col(df, "latency") is not None
        has_prediction = (
            find_col(df, "pred_class") is not None
            or find_col(df, "pred_latency") is not None
        )

        name = p.name.lower()
        score = (
            1000 * has_prediction
            + 200 * has_fp
            + 200 * has_latency
            + min(len(df), 10000) / 10
            + 100 * ("prediction" in name)
            + 80 * ("validation" in name)
            - 200 * ("summary" in name)
        )
        candidates.append((score, p, df))

    if not candidates:
        raise RuntimeError("No readable V6 CSV candidates found")

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, p, df = candidates[0]

    print("\nSelected V6 dataset:", p)
    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    return df, p

def pick_v5(inventory_df, override=None):
    if override:
        p = Path(override).resolve()
        return load_csv(p), p

    candidates = []
    for _, row in inventory_df.iterrows():
        if row["origin"] not in ("V5", "P7"):
            continue
        p = Path(row["path"])
        df = load_csv(p)
        if df is None or len(df) == 0:
            continue

        score = (
            200 * (find_col(df, "latency") is not None)
            + 100 * (find_col(df, "fingerprint") is not None)
            + 100 * (find_col(df, "planner_cost") is not None)
            + min(len(df), 10000) / 10
            + 30 * (row["origin"] == "P7")
        )
        candidates.append((score, p, df))

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, p, df = candidates[0]
    print("Selected original workload:", p)
    return df, p

def binary_series(s):
    def parse(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, (bool, np.bool_)):
            return int(v)
        t = str(v).strip().lower()
        if t in ("1", "1.0", "true", "yes", "expensive", "slow", "positive"):
            return 1
        if t in ("0", "0.0", "false", "no", "cheap", "fast", "negative"):
            return 0
        return np.nan
    return s.map(parse).astype(float)

def prepare(df, threshold):
    d = df.copy()

    for kind in ALIASES:
        col = find_col(d, kind)
        if col is not None:
            d["__" + kind] = d[col]

    if "__fingerprint" not in d:
        d["__fingerprint"] = (
            "missing_fingerprint_" + d.index.astype(str)
        )
        print("[WARNING] Fingerprint column absent: "
              "bootstrap clustering is not reliable.")

    d["__fingerprint"] = d["__fingerprint"].astype(str)

    if "__scale" not in d:
        d["__scale"] = "unknown"

    d["__scale"] = d["__scale"].astype(str)

    if "__session" not in d:
        d["__session"] = "unknown"

    if "__latency" in d:
        d["__latency"] = numeric(d["__latency"])

    if "__pred_latency" in d:
        d["__pred_latency"] = numeric(d["__pred_latency"])

    if "__true_class" in d:
        d["__true_class"] = binary_series(d["__true_class"])
    elif "__latency" in d and threshold is not None:
        d["__true_class"] = (
            d["__latency"] >= threshold
        ).astype(float)

    if "__pred_class" in d:
        d["__pred_class"] = binary_series(d["__pred_class"])

    if "__probability" in d:
        d["__probability"] = numeric(d["__probability"])

    if "__sql" in d:
        sql = d["__sql"].fillna("").astype(str)
        if "__join_count" not in d:
            d["__join_count"] = sql.str.count(r"\bJOIN\b", flags=re.I)
        if "__table_count" not in d:
            d["__table_count"] = (
                sql.str.count(r"\bFROM\b", flags=re.I)
                + sql.str.count(r"\bJOIN\b", flags=re.I)
            )
        d["__sql_length"] = sql.str.len()
        d["__where_count"] = sql.str.count(r"\bWHERE\b", flags=re.I)
        d["__group_count"] = sql.str.count(r"\bGROUP\s+BY\b", flags=re.I)

    for kind in ("planner_cost", "planner_rows", "join_count",
                 "table_count", "plan_nodes"):
        k = "__" + kind
        if k in d:
            d[k] = numeric(d[k])

    if "__status" in d:
        raw = d["__status"].astype(str).str.lower()
        d["__failed"] = raw.isin([
            "failed", "failure", "error", "false",
            "0", "timeout", "exception"
        ])
    else:
        d["__failed"] = False

    return d

def class_metrics(df):
    required = {"__true_class", "__pred_class"}
    if not required.issubset(df.columns):
        return {}

    x = df.dropna(subset=list(required))
    if len(x) == 0:
        return {}

    y = x["__true_class"].astype(int)
    p = x["__pred_class"].astype(int)

    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()

    return {
        "n": len(x),
        "accuracy": accuracy_score(y, p),
        "balanced_accuracy": balanced_accuracy_score(y, p),
        "macro_f1": f1_score(y, p, average="macro", zero_division=0),
        "precision_expensive": precision_score(y, p, zero_division=0),
        "recall_expensive": recall_score(y, p, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "actual_expensive_rate": y.mean(),
        "predicted_expensive_rate": p.mean(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }

def reg_metrics(df):
    if not {"__latency", "__pred_latency"}.issubset(df.columns):
        return {}

    x = df.dropna(subset=["__latency", "__pred_latency"]).copy()
    x = x[
        (x["__latency"] >= 0)
        & (x["__pred_latency"] >= 0)
    ]
    if len(x) == 0:
        return {}

    y = x["__latency"].to_numpy(float)
    p = x["__pred_latency"].to_numpy(float)

    result = {
        "n": len(x),
        "row_mae_ms": mean_absolute_error(y, p),
        "row_rmse_ms": np.sqrt(mean_squared_error(y, p)),
        "row_log_mae": np.mean(
            np.abs(np.log1p(y) - np.log1p(p))
        ),
        "row_r2": r2_score(y, p) if len(y) > 1 else np.nan,
        "median_absolute_error_ms": np.median(np.abs(y - p)),
        "median_q_error": np.median(
            np.maximum((y + 1) / (p + 1), (p + 1) / (y + 1))
        )
    }

    fp = x.groupby("__fingerprint", as_index=False)[
        ["__latency", "__pred_latency"]
    ].median()

    a = fp["__latency"].to_numpy(float)
    b = fp["__pred_latency"].to_numpy(float)

    result.update({
        "fingerprints": len(fp),
        "fingerprint_mae_ms": mean_absolute_error(a, b),
        "fingerprint_log_mae": np.mean(
            np.abs(np.log1p(a) - np.log1p(b))
        ),
        "fingerprint_r2": r2_score(a, b) if len(a) > 1 else np.nan
    })

    return result

def metrics(df):
    result = {"executions": len(df),
              "fingerprints": df["__fingerprint"].nunique()}
    result.update(class_metrics(df))
    result.update(reg_metrics(df))
    return result

def per_scale(df):
    section("2. PER-SCALE METRICS")
    rows = []
    for scale, g in df.groupby("__scale"):
        m = metrics(g)
        m["scale"] = scale
        rows.append(m)

    result = pd.DataFrame(rows)
    if len(result):
        cols = ["scale"] + [c for c in result if c != "scale"]
        result = result[cols]
        save_csv("01_per_scale_metrics.csv", result)
        print(result.to_string(index=False))

    return result

def plot_confusion(df):
    section("3. CONFUSION MATRICES AND CLASS DISTRIBUTIONS")

    if not {"__true_class", "__pred_class"}.issubset(df.columns):
        print("[MISSING] Actual or predicted classification labels.")
        return

    x = df.dropna(subset=["__true_class", "__pred_class"])
    if x.empty:
        print("[MISSING] No valid classification pairs.")
        return

    summary = []
    for scale, g in [("all", x)] + list(x.groupby("__scale")):
        y = g["__true_class"].astype(int)
        p = g["__pred_class"].astype(int)
        cm = confusion_matrix(y, p, labels=[0, 1])

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center", fontsize=14)

        ax.set_xticks([0, 1], ["Inexpensive", "Expensive"])
        ax.set_yticks([0, 1], ["Inexpensive", "Expensive"])
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Actual class")
        ax.set_title(f"Confusion matrix — {scale}")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(OUT / f"02_confusion_{scale}.png", dpi=220)
        plt.close(fig)

        summary.append({
            "scale": scale,
            "actual_inexpensive": int((y == 0).sum()),
            "actual_expensive": int((y == 1).sum()),
            "pred_inexpensive": int((p == 0).sum()),
            "pred_expensive": int((p == 1).sum()),
            "false_positive": int(((y == 0) & (p == 1)).sum()),
            "false_negative": int(((y == 1) & (p == 0)).sum())
        })

    save_csv("02_class_distributions.csv", pd.DataFrame(summary))
    print(pd.DataFrame(summary).to_string(index=False))

def failure_analysis(df, raw_inventory):
    section("4. FAILED EXECUTION ANALYSIS")

    candidates = []
    for _, row in raw_inventory.iterrows():
        if row["origin"] != "V6":
            continue
        p = Path(row["path"])
        x = load_csv(p)
        if x is None or x.empty:
            continue

        status = find_col(x, "status")
        error = find_col(x, "error")

        if status is None and error is None:
            continue

        d = prepare(x, None)
        mask = d["__failed"].copy()

        if "__error" in d:
            err = d["__error"].fillna("").astype(str)
            mask = mask | (
                err.str.strip().ne("")
                & ~err.str.lower().isin(["none", "nan", "null"])
            )

        failures = d[mask].copy()

        if len(failures):
            candidates.append((len(failures), p, failures))

    if not candidates:
        print("No individual failure records found in the available CSVs.")
        print("The collection summary reports 129 failures, "
              "but their causes cannot be established from that summary.")
        save_json("03_failure_audit.json", {
            "reported_failures": 129,
            "individual_failure_records_available": False
        })
        return

    candidates.sort(key=lambda z: z[0], reverse=True)
    _, path, f = candidates[0]
    print("Failure source:", path)
    print("Failure records found:", len(f))

    def category(value):
        s = str(value).lower()
        if "timeout" in s or "statement_timeout" in s:
            return "timeout"
        if "syntax" in s or "syntaxerror" in s:
            return "sql_syntax"
        if "does not exist" in s or "undefined" in s:
            return "missing_table_or_column"
        if "permission" in s or "read-only" in s:
            return "permission_or_read_only"
        if "connection" in s or "server closed" in s:
            return "connection"
        if "memory" in s or "resource" in s:
            return "resource"
        if s.strip() in ("", "nan", "none"):
            return "unknown"
        return "other"

    if "__error" in f:
        f["failure_category"] = f["__error"].map(category)
    else:
        f["failure_category"] = "unknown"

    save_csv("03_failed_executions.csv", f)
    counts = f.groupby(
        ["__scale", "failure_category"],
        dropna=False
    ).size().reset_index(name="count")
    save_csv("03_failure_categories.csv", counts)
    print(counts.to_string(index=False))

def distribution_shift(v5, v6):
    section("5. FEATURE DISTRIBUTION SHIFT")

    if v5 is None:
        print("[MISSING] Original V5 workload records.")
        return

    features = [
        "__latency", "__planner_cost", "__planner_rows",
        "__join_count", "__table_count", "__plan_nodes",
        "__sql_length", "__where_count", "__group_count"
    ]

    rows = []

    for feature in features:
        if feature not in v5 or feature not in v6:
            continue

        a = numeric(v5[feature]).dropna().to_numpy()
        b = numeric(v6[feature]).dropna().to_numpy()

        if len(a) < 5 or len(b) < 5:
            continue

        ks = ks_2samp(a, b)

        row = {
            "feature": feature,
            "v5_n": len(a),
            "v6_n": len(b),
            "v5_median": np.median(a),
            "v6_median": np.median(b),
            "v5_p90": np.quantile(a, .9),
            "v6_p90": np.quantile(b, .9),
            "median_ratio_v6_to_v5": (
                np.median(b) / np.median(a)
                if np.median(a) != 0 else np.nan
            ),
            "ks_statistic": ks.statistic,
            "ks_p_value_descriptive": ks.pvalue
        }
        rows.append(row)

        fig, ax = plt.subplots(figsize=(8, 5))
        aa = a[np.isfinite(a)]
        bb = b[np.isfinite(b)]

        if np.min(aa) >= 0 and np.min(bb) >= 0:
            aa = np.log1p(aa)
            bb = np.log1p(bb)
            xlabel = f"log1p({feature})"
        else:
            xlabel = feature

        ax.hist(aa, bins=35, alpha=.55, density=True, label="V5")
        ax.hist(bb, bins=35, alpha=.55, density=True, label="V6")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.set_title(f"V5 vs V6 distribution — {feature}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            OUT / f"04_shift_{feature.strip('_')}.png", dpi=220
        )
        plt.close(fig)

    if rows:
        result = pd.DataFrame(rows)
        save_csv("04_distribution_shift.csv", result)
        print(result.to_string(index=False))
        print("\nNOTE: KS p-values are descriptive because "
              "repeated SQL executions are not independent.")
    else:
        print("No matching numeric feature columns found.")

def baseline_analysis(v5, v6, threshold):
    section("6. FROZEN BASELINES")

    rows = []

    if "__true_class" in v6 and "__pred_class" in v6:
        test = v6.dropna(
            subset=["__true_class", "__pred_class"]
        ).copy()

        if len(test):
            rows.append({
                "task": "classification",
                "model": "V6 frozen Extra Trees",
                "metric": "accuracy",
                "value": accuracy_score(
                    test["__true_class"], test["__pred_class"]
                ),
                "baseline_status": "frozen_model"
            })

            if v5 is not None and "__latency" in v5:
                original = v5["__latency"].dropna()
                if len(original) and threshold is not None:
                    train_labels = (original >= threshold).astype(int)
                    majority = int(train_labels.mean() >= .5)
                    prediction = np.full(len(test), majority)

                    rows.append({
                        "task": "classification",
                        "model": f"V5 majority class ({majority})",
                        "metric": "accuracy",
                        "value": accuracy_score(
                            test["__true_class"], prediction
                        ),
                        "baseline_status": "frozen_from_original"
                    })

            majority_test = int(test["__true_class"].mean() >= .5)
            rows.append({
                "task": "classification",
                "model": "V6 majority class (oracle descriptive)",
                "metric": "accuracy",
                "value": max(
                    test["__true_class"].mean(),
                    1 - test["__true_class"].mean()
                ),
                "baseline_status": "test_derived_not_deployable"
            })

    if "__latency" in v6 and "__pred_latency" in v6:
        test = v6.dropna(
            subset=["__latency", "__pred_latency"]
        ).copy()
        test = test[
            (test["__latency"] >= 0)
            & (test["__pred_latency"] >= 0)
        ]

        if len(test):
            rows.append({
                "task": "regression",
                "model": "V6 frozen cost-residual",
                "metric": "row_mae_ms",
                "value": mean_absolute_error(
                    test["__latency"], test["__pred_latency"]
                ),
                "baseline_status": "frozen_model"
            })

            if v5 is not None and "__latency" in v5:
                train_y = v5["__latency"].dropna()
                if len(train_y):
                    constant = float(train_y.median())
                    rows.append({
                        "task": "regression",
                        "model": "V5 constant median",
                        "metric": "row_mae_ms",
                        "value": mean_absolute_error(
                            test["__latency"],
                            np.full(len(test), constant)
                        ),
                        "baseline_status": "frozen_from_original"
                    })

            rows.append({
                "task": "regression",
                "model": "V6 constant median (oracle descriptive)",
                "metric": "row_mae_ms",
                "value": mean_absolute_error(
                    test["__latency"],
                    np.full(len(test), test["__latency"].median())
                ),
                "baseline_status": "test_derived_not_deployable"
            })

    if rows:
        result = pd.DataFrame(rows)
        save_csv("05_baselines.csv", result)
        print(result.to_string(index=False))
    else:
        print("Prediction columns unavailable; baselines not evaluated.")

def cluster_bootstrap(df, B):
    section("7. FINGERPRINT-CLUSTERED BOOTSTRAP")

    groups = [
        g for _, g in df.groupby("__fingerprint", sort=False)
    ]

    if len(groups) < 10:
        print("Too few fingerprints for meaningful cluster bootstrap.")
        return

    print(f"Bootstrap iterations: {B}")
    print(f"Fingerprint clusters: {len(groups)}")

    results = []

    for scale, data in [("all", df)] + list(df.groupby("__scale")):
        grouped = [
            g for _, g in data.groupby("__fingerprint", sort=False)
        ]

        if len(grouped) < 10:
            continue

        point = metrics(data)
        draws = []

        for _ in range(B):
            selected = RNG.integers(
                0, len(grouped), size=len(grouped)
            )
            sample = pd.concat(
                [grouped[i] for i in selected],
                ignore_index=True
            )
            draws.append(metrics(sample))

        for metric in [
            "accuracy", "balanced_accuracy", "macro_f1",
            "row_mae_ms", "row_log_mae",
            "fingerprint_mae_ms", "fingerprint_log_mae"
        ]:
            if metric not in point:
                continue

            vals = np.array([
                d.get(metric, np.nan) for d in draws
            ], dtype=float)
            vals = vals[np.isfinite(vals)]

            if not len(vals):
                continue

            results.append({
                "scale": scale,
                "metric": metric,
                "point_estimate": point[metric],
                "ci95_lower": np.quantile(vals, .025),
                "ci95_upper": np.quantile(vals, .975),
                "bootstrap_iterations": B,
                "fingerprint_clusters": len(grouped)
            })

        print(f"Completed: {scale}")

    if results:
        table = pd.DataFrame(results)
        save_csv("06_cluster_bootstrap_ci.csv", table)
        print(table.to_string(index=False))

def paired_scale_analysis(df):
    section("8. PAIRED SCALE COMPARISONS")

    if "__latency" not in df:
        print("Latency unavailable.")
        return

    pivot = df.groupby(
        ["__fingerprint", "__scale"]
    )["__latency"].median().unstack()

    scales = list(pivot.columns)
    rows = []

    for i in range(len(scales)):
        for j in range(i + 1, len(scales)):
            a, b = scales[i], scales[j]
            paired = pivot[[a, b]].dropna()

            if len(paired) < 10:
                continue

            x = np.log1p(paired[a].clip(lower=0))
            y = np.log1p(paired[b].clip(lower=0))
            diff = y - x

            try:
                p = wilcoxon(diff).pvalue
            except Exception:
                p = np.nan

            rows.append({
                "scale_a": a,
                "scale_b": b,
                "paired_fingerprints": len(paired),
                "median_log_latency_difference_b_minus_a": diff.median(),
                "median_latency_ratio_b_to_a": np.median(
                    (paired[b] + 1) / (paired[a] + 1)
                ),
                "wilcoxon_p_exploratory": p
            })

    if rows:
        result = pd.DataFrame(rows)
        save_csv("07_paired_scale_tests.csv", result)
        print(result.to_string(index=False))
        print("These tests compare observed query latency across "
              "scales, not model superiority.")
    else:
        print("Insufficient shared fingerprints between scales.")

def fingerprint_error_table(df):
    section("9. WORST-GENERALIZING SQL FINGERPRINTS")

    if not {"__latency", "__pred_latency"}.issubset(df.columns):
        print("Regression prediction columns unavailable.")
        return

    x = df.dropna(
        subset=["__latency", "__pred_latency"]
    ).copy()

    if x.empty:
        return

    x["absolute_error_ms"] = (
        x["__latency"] - x["__pred_latency"]
    ).abs()

    x["log_error"] = (
        np.log1p(x["__latency"].clip(lower=0))
        - np.log1p(x["__pred_latency"].clip(lower=0))
    ).abs()

    result = x.groupby(
        ["__scale", "__fingerprint"],
        as_index=False
    ).agg(
        executions=("__latency", "size"),
        actual_median_ms=("__latency", "median"),
        predicted_median_ms=("__pred_latency", "median"),
        mean_absolute_error_ms=("absolute_error_ms", "mean"),
        mean_log_error=("log_error", "mean")
    )

    result = result.sort_values(
        "mean_absolute_error_ms", ascending=False
    )
    save_csv("08_worst_fingerprints.csv", result)
    print(result.head(20).to_string(index=False))

def plot_regression(df):
    if not {"__latency", "__pred_latency"}.issubset(df.columns):
        return

    x = df.dropna(
        subset=["__latency", "__pred_latency"]
    ).copy()
    x = x[
        (x["__latency"] >= 0)
        & (x["__pred_latency"] >= 0)
    ]
    if x.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    for scale, g in x.groupby("__scale"):
        ax.scatter(
            np.log1p(g["__latency"]),
            np.log1p(g["__pred_latency"]),
            s=9, alpha=.35, label=scale
        )

    upper = max(
        np.log1p(x["__latency"]).max(),
        np.log1p(x["__pred_latency"]).max()
    )
    ax.plot([0, upper], [0, upper], "--", color="black")
    ax.set_xlabel("log1p(actual latency in ms)")
    ax.set_ylabel("log1p(predicted latency in ms)")
    ax.set_title("Frozen regression: independent validation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "09_regression_scatter.png", dpi=240)
    plt.close(fig)

def publication_summary(v5, v6, threshold, inventory_df):
    section("10. RESEARCH SUMMARY")

    overall = metrics(v6)

    report = [
        "# Phase 8 V6.1 — Independent Validation Diagnostics",
        "",
        "## Experimental protocol",
        "",
        "- Frozen V5 models; no retraining performed by this script.",
        "- Analysis uses previously collected V6 records.",
        "- Repeated executions are clustered by SQL fingerprint.",
        "- Database access is not required.",
        "",
        "## Available data",
        "",
        f"- V6 records loaded: {len(v6)}",
        f"- Unique fingerprint identifiers: {v6['__fingerprint'].nunique()}",
        f"- Scales: {', '.join(sorted(v6['__scale'].unique()))}",
        f"- Classification threshold: {threshold} ms",
        "",
        "## Overall metrics",
        ""
    ]

    for key, value in overall.items():
        report.append(f"- {key}: {value}")

    report += [
        "",
        "## Interpretation safeguards",
        "",
        "1. Internal V5 cross-validation and V6 independent "
        "validation are not necessarily identical evaluation protocols.",
        "2. A V6-derived majority or median baseline is descriptive "
        "and cannot be presented as a frozen deployment baseline.",
        "3. Bootstrap intervals cluster repeated rows by fingerprint.",
        "4. Scale comparisons share query fingerprints and may also "
        "share sessions; interpret exploratory p-values cautiously.",
        "5. KS tests on repeated rows are descriptive, not "
        "independent-sample confirmatory tests.",
        "6. SQL feature names must have identical semantics in "
        "V5 and V6 before claiming feature distribution shift.",
        "7. Missing prediction records must not be treated "
        "as evidence of model success.",
        "8. Validation-derived analysis must not be used to "
        "retrain and then re-evaluate on the same held-out set.",
        "",
        "## Generated tables",
        "",
        "- 00_file_inventory.csv",
        "- 01_per_scale_metrics.csv",
        "- 02_class_distributions.csv",
        "- 03_failure_categories.csv, if failure rows exist",
        "- 04_distribution_shift.csv, if common features exist",
        "- 05_baselines.csv, if predictions exist",
        "- 06_cluster_bootstrap_ci.csv, if predictions exist",
        "- 07_paired_scale_tests.csv, if matched scales exist",
        "- 08_worst_fingerprints.csv, if regression predictions exist",
        ""
    ]

    (OUT / "RESEARCH_REPORT.md").write_text(
        "\n".join(report), encoding="utf-8"
    )

    save_json("overall_metrics.json", overall)

    print("\nOverall metrics:")
    print(json.dumps(overall, indent=2, default=str))
    print("\nAll output:", OUT)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v6-csv", default=None)
    parser.add_argument("--v5-csv", default=None)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument(
        "--threshold", type=float, default=33.25075
    )
    args = parser.parse_args()

    section("PHASE 8 V6.1 — RESEARCH DIAGNOSTICS")
    print("Project:", ROOT)
    print("V6:", V6)
    print("Output:", OUT)
    print("Retraining: NO")
    print("Database access: NO")

    inv = inventory()
    raw_v6, v6_path = pick_v6(inv, args.v6_csv)
    raw_v5, v5_path = pick_v5(inv, args.v5_csv)

    v6 = prepare(raw_v6, args.threshold)
    v5 = prepare(raw_v5, args.threshold) if raw_v5 is not None else None

    has_predictions = (
        "__pred_class" in v6
        or "__pred_latency" in v6
    )

    if not has_predictions:
        print("\n" + "!" * 84)
        print("WARNING: SELECTED V6 CSV HAS NO PREDICTION COLUMNS.")
        print("Prediction-dependent results cannot be calculated.")
        print("Inspect 00_file_inventory.csv and locate the "
              "row-level frozen-model prediction file.")
        print("!" * 84)

    per_scale(v6)
    plot_confusion(v6)
    failure_analysis(v6, inv)
    distribution_shift(v5, v6)
    baseline_analysis(v5, v6, args.threshold)

    if has_predictions:
        cluster_bootstrap(v6, args.bootstrap)

    paired_scale_analysis(v6)
    fingerprint_error_table(v6)
    plot_regression(v6)
    publication_summary(v5, v6, args.threshold, inv)

    section("FINISHED")
    print("Reports and figures saved to:")
    print(OUT)

if __name__ == "__main__":
    main()
