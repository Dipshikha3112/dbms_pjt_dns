
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import argparse
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
)

from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent

PHASE7 = (
    ROOT / "results" / "phase7_monitoring" / "10K"
)

OUTPUT = ROOT / "results" / "phase8_ai_workload"
OUTPUT.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "total_cost",
    "estimated_rows",
    "actual_rows",
    "planning_time_ms",
    "plan_nodes",
    "max_plan_depth",
    "sequential_scans",
    "index_scans",
    "index_only_scans",
    "bitmap_index_scans",
    "bitmap_heap_scans",
    "nested_loop_joins",
    "hash_joins",
    "merge_joins",
    "sort_nodes",
    "aggregate_nodes",
    "filter_nodes",
    "join_filter_nodes",
    "rows_removed_by_filter",
    "shared_hit_blocks",
    "shared_read_blocks",
    "shared_dirtied_blocks",
    "shared_written_blocks",
    "local_hit_blocks",
    "local_read_blocks",
    "temp_read_blocks",
    "temp_written_blocks",
    "parallel_workers_planned",
    "parallel_workers_launched",
]

TARGET = "execution_time_ms"


def section(title):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75, flush=True)


def latest_phase7_file():
    matches = list(
        PHASE7.glob("*/ai_training_features.csv")
    )

    if not matches:
        raise FileNotFoundError(
            "No Phase 7 training dataset found in "
            f"{PHASE7}"
        )

    return max(
        matches,
        key=lambda p: p.stat().st_mtime
    )


def load_dataset(path):
    section("1. LOADING PHASE 7 DATA")

    df = pd.read_csv(path)

    required = {
        "query_hash",
        "fingerprint_hash",
        "query_family",
        "workload_type",
        "status",
        TARGET,
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print("Dataset:", path)
    print("Total records:", len(df))
    print(
        "Unique SQL hashes:",
        df["query_hash"].nunique()
    )
    print(
        "Unique fingerprints:",
        df["fingerprint_hash"].nunique()
    )

    success = df[
        df["status"].astype(str).str.lower()
        == "success"
    ].copy()

    success[TARGET] = pd.to_numeric(
        success[TARGET], errors="coerce"
    )

    success = success[
        success[TARGET].notna()
        & (success[TARGET] >= 0)
    ].copy()

    if len(success) < 30:
        raise ValueError(
            "Not enough successful query executions"
        )

    print("Successful records:", len(success))
    print(
        "Excluded failures/timeouts:",
        len(df) - len(success)
    )

    return df, success


def engineer_features(df):
    section("2. FEATURE ENGINEERING")

    result = df.copy()

    for column in FEATURES:
        if column not in result.columns:
            print(
                "Missing feature, filling zero:",
                column
            )
            result[column] = 0

        result[column] = pd.to_numeric(
            result[column],
            errors="coerce"
        )

        result[column] = (
            result[column]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
            .clip(lower=0)
        )

    result["scan_count"] = (
        result["sequential_scans"]
        + result["index_scans"]
        + result["index_only_scans"]
        + result["bitmap_index_scans"]
    )

    result["join_count"] = (
        result["nested_loop_joins"]
        + result["hash_joins"]
        + result["merge_joins"]
    )

    result["buffer_total"] = (
        result["shared_hit_blocks"]
        + result["shared_read_blocks"]
        + result["temp_read_blocks"]
        + result["temp_written_blocks"]
    )

    result["read_hit_ratio"] = (
        result["shared_hit_blocks"]
        / (
            result["shared_hit_blocks"]
            + result["shared_read_blocks"]
            + 1
        )
    )

    result["estimate_error_ratio"] = (
        (result["actual_rows"] + 1)
        / (result["estimated_rows"] + 1)
    )

    result["log_total_cost"] = np.log1p(
        result["total_cost"]
    )

    result["log_estimated_rows"] = np.log1p(
        result["estimated_rows"]
    )

    result["log_buffer_total"] = np.log1p(
        result["buffer_total"]
    )

    engineered = [
        "scan_count",
        "join_count",
        "buffer_total",
        "read_hit_ratio",
        "estimate_error_ratio",
        "log_total_cost",
        "log_estimated_rows",
        "log_buffer_total",
    ]

    model_features = FEATURES + engineered

    result.to_csv(
        OUTPUT / "engineered_features.csv",
        index=False
    )

    print("Feature count:", len(model_features))

    return result, model_features


def split_by_query_hash(df):
    """
    Group-aware split prevents identical query hashes
    from appearing in both train and test.
    """

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=42
    )

    train_idx, test_idx = next(
        splitter.split(
            df,
            groups=df["query_hash"].astype(str)
        )
    )

    return (
        df.iloc[train_idx].copy(),
        df.iloc[test_idx].copy()
    )


def cluster_queries(df, features):
    section("3. QUERY CLUSTERING")

    # Cluster distinct fingerprints, not repeated
    # executions, to reduce frequency domination.
    grouped = (
        df.groupby("fingerprint_hash")[features]
        .median()
    )

    if len(grouped) < 3:
        print("Insufficient templates for clustering")
        return df, None, None, None

    scaler = StandardScaler()

    X = scaler.fit_transform(
        np.log1p(grouped.clip(lower=0))
    )

    best_score = -1
    best_k = None
    best_model = None

    max_k = min(8, len(grouped) - 1)

    for k in range(2, max_k + 1):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(X)

        if len(set(labels)) < 2:
            continue

        score = silhouette_score(X, labels)

        print(
            f"k={k} silhouette={score:.4f}"
        )

        if score > best_score:
            best_score = score
            best_k = k
            best_model = model

    if best_model is None:
        return df, None, None, None

    grouped["cluster"] = best_model.labels_

    mapping = grouped["cluster"].to_dict()

    df = df.copy()
    df["cluster"] = (
        df["fingerprint_hash"]
        .map(mapping)
        .astype(int)
    )

    grouped.reset_index().to_csv(
        OUTPUT / "fingerprint_clusters.csv",
        index=False
    )

    cluster_summary = (
        df.groupby("cluster")
        .agg(
            queries=("query_hash", "size"),
            fingerprints=("fingerprint_hash", "nunique"),
            mean_latency_ms=(TARGET, "mean"),
            median_latency_ms=(TARGET, "median"),
            mean_cost=("total_cost", "mean"),
            mean_seq_scans=("sequential_scans", "mean"),
        )
        .reset_index()
    )

    cluster_summary.to_csv(
        OUTPUT / "cluster_summary.csv",
        index=False
    )

    joblib.dump(
        {
            "model": best_model,
            "scaler": scaler,
            "features": features,
        },
        OUTPUT / "query_cluster_model.joblib"
    )

    print("Selected clusters:", best_k)
    print("Silhouette score:", round(best_score, 4))

    return df, best_k, best_score, cluster_summary


def classify_workload(train, test, features):
    section("4. XGBOOST WORKLOAD CLASSIFICATION")

    y_train_raw = train["workload_type"].astype(str)
    y_test_raw = test["workload_type"].astype(str)

    encoder = LabelEncoder()
    encoder.fit(y_train_raw)

    unseen = set(y_test_raw) - set(encoder.classes_)

    if unseen:
        print(
            "Classification skipped: unseen test labels",
            unseen
        )
        return None

    if len(encoder.classes_) < 2:
        print("Classification skipped: one class")
        return None

    X_train = train[features]
    X_test = test[features]

    y_train = encoder.transform(y_train_raw)
    y_test = encoder.transform(y_test_raw)

    model = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=4
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test, predictions
    )

    weighted_f1 = f1_score(
        y_test, predictions,
        average="weighted",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test, predictions,
        average="macro",
        zero_division=0
    )

    report = classification_report(
        y_test,
        predictions,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_,
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).transpose().to_csv(
        OUTPUT / "classification_report.csv"
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=np.arange(len(encoder.classes_))
    )

    pd.DataFrame(
        cm,
        index=encoder.classes_,
        columns=encoder.classes_
    ).to_csv(
        OUTPUT / "confusion_matrix.csv"
    )

    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    }).sort_values(
        "importance", ascending=False
    )

    importance.to_csv(
        OUTPUT / "classification_feature_importance.csv",
        index=False
    )

    joblib.dump(
        {
            "model": model,
            "label_encoder": encoder,
            "features": features,
        },
        OUTPUT / "workload_classifier.joblib"
    )

    print("Accuracy:", round(accuracy, 4))
    print("Weighted F1:", round(weighted_f1, 4))
    print("Macro F1:", round(macro_f1, 4))

    return {
        "accuracy": float(accuracy),
        "weighted_f1": float(weighted_f1),
        "macro_f1": float(macro_f1),
        "classes": encoder.classes_.tolist(),
    }


def predict_latency(train, test, features):
    section("5. XGBOOST LATENCY PREDICTION")

    X_train = train[features]
    X_test = test[features]

    # Log-transform helps with heavy-tailed query times.
    y_train = np.log1p(train[TARGET])
    y_test = test[TARGET].to_numpy()

    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=4
    )

    model.fit(X_train, y_train)

    predictions = np.expm1(
        model.predict(X_test)
    )

    predictions = np.clip(
        predictions, 0, None
    )

    mae = mean_absolute_error(
        y_test, predictions
    )

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(
        y_test, predictions
    )

    median_ae = float(
        np.median(
            np.abs(y_test - predictions)
        )
    )

    pd.DataFrame({
        "query_hash": test["query_hash"].values,
        "query_family": test["query_family"].values,
        "actual_latency_ms": y_test,
        "predicted_latency_ms": predictions,
        "absolute_error_ms": np.abs(
            y_test - predictions
        ),
    }).to_csv(
        OUTPUT / "latency_predictions.csv",
        index=False
    )

    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    }).sort_values(
        "importance", ascending=False
    )

    importance.to_csv(
        OUTPUT / "latency_feature_importance.csv",
        index=False
    )

    joblib.dump(
        {
            "model": model,
            "features": features,
            "target_transform": "log1p",
        },
        OUTPUT / "latency_predictor.joblib"
    )

    print("MAE (ms):", round(mae, 3))
    print("RMSE (ms):", round(rmse, 3))
    print("Median AE (ms):", round(median_ae, 3))
    print("R2:", round(r2, 4))

    return {
        "mae_ms": float(mae),
        "rmse_ms": float(rmse),
        "median_absolute_error_ms": median_ae,
        "r2": float(r2),
    }


def demand_baseline(all_records):
    section("6. WORKLOAD DEMAND BASELINE")

    # A single monitoring run is insufficient for
    # real temporal forecasting.
    counts = (
        all_records.groupby(
            ["fingerprint_hash", "workload_type"],
            dropna=False
        )
        .size()
        .reset_index(name="observed_frequency")
    )

    total = counts["observed_frequency"].sum()

    counts["observed_probability"] = (
        counts["observed_frequency"] / total
    )

    # A hypothetical next batch of 950 queries,
    # assuming the same workload distribution.
    counts["expected_next_950_queries"] = (
        counts["observed_probability"] * 950
    )

    counts.to_csv(
        OUTPUT / "demand_baseline.csv",
        index=False
    )

    print(
        "Saved observed-frequency demand baseline."
    )
    print(
        "Temporal forecasting not claimed."
    )

    return {
        "method": "empirical_frequency_baseline",
        "observations": int(total),
        "forecast_validated": False,
    }


def phase9_candidates(df):
    section("7. PHASE 9 OPTIMIZATION PRIORITIES")

    # Aggregate by query hash to avoid selecting
    # individual repeated executions as candidates.
    summary = (
        df.groupby(
            ["query_hash", "query_family"],
            dropna=False
        )
        .agg(
            executions=("query_hash", "size"),
            mean_latency_ms=(TARGET, "mean"),
            mean_cost=("total_cost", "mean"),
            mean_seq_scans=("sequential_scans", "mean"),
            mean_index_scans=("index_scans", "mean"),
            mean_buffer_reads=("shared_read_blocks", "mean"),
            mean_buffer_hits=("shared_hit_blocks", "mean"),
            mean_joins=("join_count", "mean"),
        )
        .reset_index()
    )

    summary["estimated_total_time_ms"] = (
        summary["executions"]
        * summary["mean_latency_ms"]
    )

    summary = summary.sort_values(
        "estimated_total_time_ms",
        ascending=False
    )

    summary["priority_rank"] = np.arange(
        1, len(summary) + 1
    )

    summary["recommendation_hint"] = np.where(
        summary["mean_seq_scans"] > 0,
        "Investigate sequential scans and predicates",
        np.where(
            summary["mean_joins"] > 0,
            "Investigate join strategy and join indexes",
            "Inspect plan and buffer usage"
        )
    )

    summary.to_csv(
        OUTPUT / "phase9_optimization_priorities.csv",
        index=False
    )

    print("Candidates:", len(summary))
    print(
        summary[
            [
                "query_family",
                "mean_latency_ms",
                "estimated_total_time_ms",
                "recommendation_hint",
            ]
        ].head(10).round(2).to_string(index=False)
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=None,
        help="Path to Phase 7 ai_training_features.csv"
    )

    args = parser.parse_args()

    input_path = (
        Path(args.input)
        if args.input
        else latest_phase7_file()
    )

    raw, success = load_dataset(input_path)

    df, features = engineer_features(success)

    df, best_k, silhouette, clusters = (
        cluster_queries(df, features)
    )

    train, test = split_by_query_hash(df)

    print("\nTrain records:", len(train))
    print("Test records:", len(test))
    print(
        "Overlapping query hashes:",
        len(
            set(train["query_hash"])
            & set(test["query_hash"])
        )
    )

    classification = classify_workload(
        train, test, features
    )

    latency = predict_latency(
        train, test, features
    )

    demand = demand_baseline(raw)

    phase9_candidates(df)

    report = {
        "phase": 8,
        "title": "AI Workload Learner",
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "source": str(input_path),
        "records_total": len(raw),
        "records_successful": len(df),
        "unique_query_hashes": int(
            df["query_hash"].nunique()
        ),
        "unique_fingerprints": int(
            df["fingerprint_hash"].nunique()
        ),
        "feature_count": len(features),
        "train_records": len(train),
        "test_records": len(test),
        "split_method": "GroupShuffleSplit by query_hash",
        "cluster_count": best_k,
        "silhouette_score": silhouette,
        "classification": classification,
        "latency_prediction": latency,
        "demand_baseline": demand,
        "limitations": [
            "Single monitoring run.",
            "Only successful queries used for ML.",
            "Timeouts excluded from latency regression.",
            "Fingerprint clusters are descriptive.",
            "No validated temporal demand forecast.",
            "Model uses post-execution plan features.",
            "Not yet a pre-execution cost predictor.",
            "No database optimization is applied.",
        ],
    }

    report_path = OUTPUT / "phase8_report.json"

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            default=str
        ),
        encoding="utf-8"
    )

    section("PHASE 8 COMPLETE")

    print("Training records:", len(train))
    print("Testing records:", len(test))
    print("Clusters:", best_k)
    print("Classification:", classification)
    print("Latency:", latency)
    print("\nOutput directory:", OUTPUT)
    print("Report:", report_path)


if __name__ == "__main__":
    main()
