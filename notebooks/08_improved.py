
from pathlib import Path
from datetime import datetime, timezone
import json
import re

import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    mean_absolute_error, mean_squared_error,
    r2_score, silhouette_score
)
from xgboost import XGBClassifier, XGBRegressor


ROOT = Path(__file__).resolve().parent.parent
P7 = ROOT / "results" / "phase7_monitoring" / "10K"
P6 = ROOT / "results" / "phase6_workloads" / "combined_workload.csv"
OUT = ROOT / "results" / "phase8_1_improved"
OUT.mkdir(parents=True, exist_ok=True)

PLAN = [
    "total_cost", "estimated_rows", "plan_nodes",
    "max_plan_depth", "sequential_scans", "index_scans",
    "index_only_scans", "bitmap_index_scans",
    "bitmap_heap_scans", "nested_loop_joins",
    "hash_joins", "merge_joins", "sort_nodes",
    "aggregate_nodes", "filter_nodes",
    "parallel_workers_planned"
]

DIAGNOSTIC = [
    "actual_rows", "planning_time_ms",
    "shared_hit_blocks", "shared_read_blocks",
    "shared_dirtied_blocks", "shared_written_blocks",
    "local_hit_blocks", "local_read_blocks",
    "temp_read_blocks", "temp_written_blocks",
    "rows_removed_by_filter",
    "parallel_workers_launched"
]

SQL_TERMS = [
    "join", "where", "group by", "order by",
    "having", "distinct", "union", "exists",
    "between", "like", "limit", "case", "with"
]


def stage(name):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70, flush=True)


def latest_file():
    files = list(P7.glob("*/ai_training_features.csv"))
    if not files:
        raise FileNotFoundError("Phase 7 dataset not found")
    return max(files, key=lambda p: p.stat().st_mtime)


def extract_sql_features(text):
    s = str(text).lower()
    s = re.sub(r"'(?:''|[^'])*'", "?", s)
    s = re.sub(r"\b\d+(?:\.\d+)?\b", "?", s)

    features = {
        "sql_length": len(s),
        "sql_tokens": len(s.split()),
        "sql_predicates": len(
            re.findall(r"\band\b|\bor\b", s)
        ),
        "sql_table_refs": len(
            re.findall(r"\bfrom\b|\bjoin\b", s)
        ),
        "sql_literals": s.count("?")
    }

    for term in SQL_TERMS:
        key = "sql_" + term.replace(" ", "_")
        features[key] = len(
            re.findall(r"\b" + re.escape(term) + r"\b", s)
        )

    return features


def load_sql_mapping():
    if not P6.exists():
        print("WARNING: Phase 6 SQL file unavailable")
        return {}

    workload = pd.read_csv(P6)

    sql_col = next(
        (c for c in [
            "sql_query", "query_sql", "sql",
            "query_text", "query", "statement"
        ] if c in workload.columns),
        None
    )

    if sql_col is None or "query_hash" not in workload:
        print("WARNING: SQL mapping columns unavailable")
        return {}

    workload = workload.dropna(
        subset=["query_hash", sql_col]
    ).drop_duplicates("query_hash")

    return dict(zip(
        workload["query_hash"].astype(str),
        workload[sql_col].astype(str)
    ))


def prepare_data():
    stage("1. LOADING AND FEATURE ENGINEERING")

    source = latest_file()
    df = pd.read_csv(source)

    required = {
        "query_hash", "fingerprint_hash", "status",
        "workload_type", "query_family",
        "execution_time_ms"
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["query_hash"] = df["query_hash"].astype(str)
    df["fingerprint_hash"] = (
        df["fingerprint_hash"].astype(str)
    )
    df["status"] = df["status"].astype(str).str.lower()

    sql_map = load_sql_mapping()

    structure = {
        h: extract_sql_features(sql_map.get(h, ""))
        for h in df["query_hash"].unique()
    }

    structure_df = pd.DataFrame.from_dict(
        structure, orient="index"
    )
    structure_df.index.name = "query_hash"

    df = df.join(structure_df, on="query_hash")

    sql_features = list(structure_df.columns)

    for column in PLAN + DIAGNOSTIC + sql_features:
        if column not in df:
            df[column] = 0

        df[column] = (
            pd.to_numeric(df[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
            .clip(lower=0)
        )

    df["log_cost"] = np.log1p(df["total_cost"])
    df["log_estimated_rows"] = np.log1p(
        df["estimated_rows"]
    )

    df["join_count"] = df[
        ["nested_loop_joins", "hash_joins", "merge_joins"]
    ].sum(axis=1)

    df["scan_count"] = df[
        ["sequential_scans", "index_scans",
         "index_only_scans", "bitmap_index_scans"]
    ].sum(axis=1)

    pre_features = (
        sql_features + PLAN
        + ["log_cost", "log_estimated_rows",
           "join_count", "scan_count"]
    )

    diagnostic_features = pre_features + DIAGNOSTIC

    success = df[
        df["status"] == "success"
    ].copy()

    success["execution_time_ms"] = pd.to_numeric(
        success["execution_time_ms"],
        errors="coerce"
    )

    success = success[
        success["execution_time_ms"].notna()
    ].copy()

    df.to_csv(
        OUT / "all_query_features.csv",
        index=False
    )

    print("Total records:", len(df))
    print("Successful records:", len(success))
    print("Timeouts:", (df["status"] == "timeout").sum())
    print("SQL mappings:", len(sql_map))
    print("Pre-execution features:", len(pre_features))
    print("Diagnostic features:", len(diagnostic_features))

    return df, success, pre_features, diagnostic_features


def group_split(df):
    stage("2. LEAKAGE-AWARE TRAIN/TEST SPLIT")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=42
    )

    train_idx, test_idx = next(
        splitter.split(
            df, groups=df["query_hash"]
        )
    )

    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()

    overlap = len(
        set(train["query_hash"])
        & set(test["query_hash"])
    )

    print("Training:", len(train))
    print("Testing:", len(test))
    print("Overlapping SQL hashes:", overlap)

    assert overlap == 0

    return train, test


def classification(train, test, features, prefix):
    stage(f"3. CLASSIFICATION - {prefix}")

    train_labels = train["workload_type"].astype(str)
    test_labels = test["workload_type"].astype(str)

    classes = sorted(train_labels.unique())

    if len(classes) < 2 or (
        set(test_labels) - set(classes)
    ):
        print("Classification skipped: insufficient labels")
        return {"status": "skipped"}

    label_map = {
        label: i for i, label in enumerate(classes)
    }

    y_train = train_labels.map(label_map).to_numpy()
    y_test = test_labels.map(label_map).to_numpy()

    X_train = train[features]
    X_test = test[features]

    baseline = DummyClassifier(
        strategy="most_frequent"
    )
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=5,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=4
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "macro_f1": float(
            f1_score(
                y_test, predictions,
                average="macro", zero_division=0
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_test, predictions,
                average="weighted", zero_division=0
            )
        ),
        "majority_baseline_accuracy": float(
            accuracy_score(y_test, baseline_predictions)
        )
    }

    report = classification_report(
        y_test, predictions,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).T.to_csv(
        OUT / f"{prefix}_classification_report.csv"
    )

    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    importance.to_csv(
        OUT / f"{prefix}_classification_importance.csv",
        index=False
    )

    joblib.dump({
        "model": model,
        "features": features,
        "classes": classes
    }, OUT / f"{prefix}_classifier.joblib")

    print(json.dumps(metrics, indent=2))

    return metrics


def latency_prediction(train, test, features, prefix):
    stage(f"4. LATENCY PREDICTION - {prefix}")

    X_train = train[features]
    X_test = test[features]

    y_train = train["execution_time_ms"].to_numpy()
    y_test = test["execution_time_ms"].to_numpy()

    baseline = DummyRegressor(strategy="median")
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)

    model = XGBRegressor(
        n_estimators=350,
        max_depth=3,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=5,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=4
    )

    model.fit(X_train, np.log1p(y_train))

    predictions = np.clip(
        np.expm1(model.predict(X_test)),
        0, None
    )

    expensive_mask = (
        y_test >= np.quantile(y_test, 0.75)
    )

    metrics = {
        "mae_ms": float(
            mean_absolute_error(y_test, predictions)
        ),
        "rmse_ms": float(
            np.sqrt(mean_squared_error(y_test, predictions))
        ),
        "median_ae_ms": float(
            np.median(np.abs(y_test - predictions))
        ),
        "r2": float(
            r2_score(y_test, predictions)
        ),
        "baseline_mae_ms": float(
            mean_absolute_error(y_test, baseline_pred)
        ),
        "expensive_query_mae_ms": float(
            mean_absolute_error(
                y_test[expensive_mask],
                predictions[expensive_mask]
            )
        )
    }

    pd.DataFrame({
        "query_hash": test["query_hash"].values,
        "query_family": test["query_family"].values,
        "actual_ms": y_test,
        "predicted_ms": predictions,
        "absolute_error_ms": np.abs(
            y_test - predictions
        )
    }).to_csv(
        OUT / f"{prefix}_latency_predictions.csv",
        index=False
    )

    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    importance.to_csv(
        OUT / f"{prefix}_latency_importance.csv",
        index=False
    )

    joblib.dump({
        "model": model,
        "features": features,
        "target_transform": "log1p"
    }, OUT / f"{prefix}_latency_model.joblib")

    print(json.dumps(metrics, indent=2))

    return metrics


def clustering(df, features):
    stage("5. IMPROVED QUERY CLUSTERING")

    grouped = (
        df.groupby("fingerprint_hash")[features]
        .median()
    )

    X = StandardScaler().fit_transform(
        np.log1p(grouped.clip(lower=0))
    )

    results = []
    best = None

    for k in range(2, min(9, len(grouped))):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20
        )

        labels = model.fit_predict(X)

        if len(set(labels)) < 2:
            continue

        score = silhouette_score(X, labels)

        results.append({
            "clusters": k,
            "silhouette": score
        })

        print(f"k={k}, silhouette={score:.4f}")

        if best is None or score > best[0]:
            best = (score, k, labels)

    pd.DataFrame(results).to_csv(
        OUT / "cluster_selection.csv",
        index=False
    )

    if best is None:
        return {"status": "no_valid_clusters"}

    score, k, labels = best

    grouped["cluster"] = labels

    grouped.reset_index().to_csv(
        OUT / "fingerprint_clusters.csv",
        index=False
    )

    return {
        "clusters": int(k),
        "silhouette": float(score)
    }


def optimization_priorities(df):
    stage("6. TIMEOUT-AWARE OPTIMIZATION PRIORITIES")

    data = df.copy()

    data["is_timeout"] = (
        data["status"] == "timeout"
    ).astype(int)

    data["is_failure"] = (
        data["status"] != "success"
    ).astype(int)

    data["is_success"] = (
        data["status"] == "success"
    ).astype(int)

    data["execution_time_ms"] = pd.to_numeric(
        data["execution_time_ms"],
        errors="coerce"
    )

    summary = (
        data.groupby(
            ["query_hash", "query_family"],
            dropna=False
        )
        .agg(
            executions=("query_hash", "size"),
            timeouts=("is_timeout", "sum"),
            failures=("is_failure", "sum"),
            successes=("is_success", "sum"),
            mean_latency_ms=("execution_time_ms", "mean"),
            mean_seq_scans=("sequential_scans", "mean"),
            mean_buffer_reads=("shared_read_blocks", "mean")
        )
        .reset_index()
    )

    summary["observed_total_success_ms"] = (
        summary["mean_latency_ms"].fillna(0)
        * summary["successes"]
    )

    summary = summary.sort_values(
        ["timeouts", "observed_total_success_ms"],
        ascending=[False, False]
    )

    summary["priority_rank"] = np.arange(
        1, len(summary) + 1
    )

    summary["recommendation"] = np.where(
        summary["timeouts"] > 0,
        "HIGH PRIORITY: Investigate query timeout",
        np.where(
            summary["mean_seq_scans"] > 0,
            "Investigate sequential scans and indexes",
            "Inspect joins, predicates and query plan"
        )
    )

    summary.to_csv(
        OUT / "phase9_priorities_including_timeouts.csv",
        index=False
    )

    print("Unique query hashes:", len(summary))
    print("Total timeouts:", summary["timeouts"].sum())

    print(
        summary[
            ["query_family", "timeouts",
             "observed_total_success_ms", "recommendation"]
        ].head(12).to_string(index=False)
    )

    return {
        "candidate_hashes": len(summary),
        "timeouts_included": int(
            summary["timeouts"].sum()
        )
    }


def main():
    stage("PHASE 8.1 - ADVANCED AI WORKLOAD LEARNER")

    raw, success, pre, diagnostic = prepare_data()

    train, test = group_split(success)

    results = {
        "phase": "8.1",
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "records": len(raw),
        "successful": len(success),
        "train": len(train),
        "test": len(test),
        "baseline_phase8": {
            "classification_accuracy": 0.5957,
            "weighted_f1": 0.5846,
            "latency_mae_ms": 355.888,
            "latency_rmse_ms": 770.531,
            "latency_r2": 0.3595
        },
        "models": {}
    }

    for name, features in [
        ("pre_execution", pre),
        ("diagnostic", diagnostic)
    ]:
        cls = classification(
            train, test, features, name
        )

        reg = latency_prediction(
            train, test, features, name
        )

        results["models"][name] = {
            "classification": cls,
            "latency": reg
        }

    results["clustering"] = clustering(
        success, pre
    )

    results["phase9"] = optimization_priorities(raw)

    results["limitations"] = [
        "Only one monitoring run is available.",
        "No validated temporal demand forecast.",
        "Post-execution diagnostic features cannot be used for pre-execution decisions.",
        "Train/test split prevents identical SQL hashes from overlapping.",
        "Identical normalized fingerprints may still appear in both sets.",
        "Original Phase 8 metrics may use a different feature set and must not be treated as a controlled model comparison.",
        "Timed-out query latencies are censored, not assumed to be zero."
    ]

    report = OUT / "phase81_report.json"

    report.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8"
    )

    stage("PHASE 8.1 EXECUTION COMPLETE")

    print("Output directory:", OUT)
    print("Report:", report)


if __name__ == "__main__":
    main()
