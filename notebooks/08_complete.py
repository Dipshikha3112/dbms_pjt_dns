
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import sys
import traceback
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

DEFAULT_PHASE6 = RESULTS / "phase6_workloads" / "combined_workload.csv"
DEFAULT_PHASE7 = (
    RESULTS / "phase7_monitoring" / "phase7_query_monitoring.csv"
)

LOGGER = logging.getLogger("phase8")

SQL_KEYWORDS = [
    "SELECT", "FROM", "WHERE", "JOIN", "LEFT JOIN",
    "RIGHT JOIN", "INNER JOIN", "FULL JOIN",
    "GROUP BY", "ORDER BY", "HAVING", "DISTINCT",
    "UNION", "EXISTS", "BETWEEN", "LIKE", "LIMIT",
    "OFFSET", "CASE", "WHEN", "WITH", "IN",
    "IS NULL", "COUNT", "SUM", "AVG", "MIN", "MAX",
    "OVER", "PARTITION BY", "COALESCE", "CAST",
]

TIMESTAMP_ALIASES = [
    "monitored_at", "timestamp", "executed_at",
    "execution_timestamp", "started_at", "start_time",
    "event_time", "datetime", "query_time",
]

LATENCY_ALIASES = [
    "execution_time_ms", "wall_time_ms", "latency_ms",
    "duration_ms", "elapsed_ms",
]

LABEL_ALIASES = [
    "workload_type", "role_label", "query_family",
]

SQL_ALIASES = [
    "sql", "query", "sql_text", "query_text",
]

RANDOM_STATE = 42


def setup_logging(out: Path):
    out.mkdir(parents=True, exist_ok=True)

    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s"
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    LOGGER.addHandler(console)

    file_handler = logging.FileHandler(
        out / "phase8.log", encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    LOGGER.addHandler(file_handler)


def save_json(path: Path, data):
    def convert(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj) if np.isfinite(obj) else None
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        return obj

    with path.open("w", encoding="utf-8") as f:
        json.dump(convert(data), f, indent=2, ensure_ascii=False)


def pick_column(df: pd.DataFrame, aliases):
    mapping = {str(c).lower().strip(): c for c in df.columns}

    for alias in aliases:
        if alias in mapping:
            return mapping[alias]

    return None


def hash_text(text: str):
    return hashlib.sha256(
        text.encode("utf-8", errors="replace")
    ).hexdigest()[:20]


def normalize_sql(sql):
    text = str(sql or "")
    text = re.sub(r"--[^\n]*", " ", text)
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"'(?:''|[^'])*'", " ? ", text)
    text = re.sub(r'"(?:""|[^"])*"', " IDENT ", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\b", " ? ", text)
    text = re.sub(r"\s+", " ", text).strip().upper()
    return text


def keyword_count(sql: str, keyword: str):
    pattern = r"\b" + re.escape(keyword).replace(
        r"\ ", r"\s+"
    ) + r"\b"
    return len(re.findall(pattern, sql))


def sql_features(sql):
    original = str(sql or "")
    s = normalize_sql(original)

    tokens = re.findall(r"[A-Z_][A-Z_0-9]*|\?|[^\s]", s)

    result = {
        "sql_length": len(original),
        "normalized_length": len(s),
        "token_count": len(tokens),
        "unique_token_count": len(set(tokens)),
        "question_mark_count": s.count("?"),
        "parentheses_count": s.count("("),
        "comma_count": s.count(","),
        "comparison_count": len(
            re.findall(r"(?<![<>!])=(?!=)|>=|<=|<>|!=|>|<", s)
        ),
        "logical_and_count": keyword_count(s, "AND"),
        "logical_or_count": keyword_count(s, "OR"),
        "subquery_select_count": max(
            0, keyword_count(s, "SELECT") - 1
        ),
        "join_count": keyword_count(s, "JOIN"),
        "table_reference_estimate": (
            keyword_count(s, "FROM") +
            keyword_count(s, "JOIN")
        ),
    }

    for keyword in SQL_KEYWORDS:
        name = "kw_" + keyword.lower().replace(" ", "_")
        result[name] = keyword_count(s, keyword)

    return result


def feature_frame(sql_series):
    records = [sql_features(s) for s in sql_series.fillna("")]
    return pd.DataFrame(records).fillna(0).astype(float)


def valid_label(value):
    if pd.isna(value):
        return "unknown"
    text = str(value).strip().lower()
    return text if text else "unknown"


def load_phase6(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Phase 6 CSV not found: {path}")

    df = pd.read_csv(path, low_memory=False)

    sql_col = pick_column(df, SQL_ALIASES)
    label_col = pick_column(df, LABEL_ALIASES)

    if sql_col is None:
        raise ValueError(
            "Phase 6 CSV must contain an SQL/query column."
        )

    if label_col is None:
        raise ValueError(
            "Phase 6 CSV must contain workload_type or role_label."
        )

    df = df.copy()
    df["sql_input"] = df[sql_col].fillna("").astype(str)
    df["label"] = df[label_col].map(valid_label)

    df = df[df["sql_input"].str.strip().ne("")].copy()
    df = df[df["label"].ne("unknown")].copy()

    if df.empty:
        raise ValueError("No valid Phase 6 SQL records.")

    df["normalized_sql"] = df["sql_input"].map(normalize_sql)
    df["sql_hash_computed"] = df["normalized_sql"].map(hash_text)

    if "fingerprint_hash" in df.columns:
        groups = df["fingerprint_hash"].fillna("").astype(str)
        groups = groups.where(
            groups.str.strip().ne(""),
            df["sql_hash_computed"],
        )
    else:
        groups = df["sql_hash_computed"]

    df["validation_group"] = groups

    return df.reset_index(drop=True)


def safe_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
    }


def make_classifier():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=250,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


def run_clustering(X, base_df, out):
    LOGGER.info("Running workload clustering")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)

    unique_rows = len(np.unique(scaled, axis=0))
    max_k = min(9, len(X) - 1, unique_rows)

    candidates = []

    for k in range(2, max_k + 1):
        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=10,
        )
        labels = model.fit_predict(scaled)

        if len(np.unique(labels)) < 2:
            continue

        score = silhouette_score(
            scaled,
            labels,
            sample_size=min(len(scaled), 1000),
            random_state=RANDOM_STATE,
        )

        candidates.append({
            "k": k,
            "silhouette": float(score),
            "inertia": float(model.inertia_),
        })

    if not candidates:
        return {"status": "INSUFFICIENT_CLUSTER_VARIATION"}

    best_k = max(
        candidates, key=lambda item: item["silhouette"]
    )["k"]

    final = KMeans(
        n_clusters=best_k,
        random_state=RANDOM_STATE,
        n_init=20,
    )

    cluster_labels = final.fit_predict(scaled)

    result = base_df[[
        "label", "validation_group", "sql_hash_computed"
    ]].copy()
    result["cluster"] = cluster_labels

    result.to_csv(out / "clusters.csv", index=False)

    summary = (
        result.groupby(["cluster", "label"])
        .size()
        .reset_index(name="records")
    )
    summary.to_csv(
        out / "cluster_label_distribution.csv", index=False
    )

    pd.DataFrame(candidates).to_csv(
        out / "cluster_selection.csv", index=False
    )

    plt.figure(figsize=(8, 5))
    plt.plot(
        [x["k"] for x in candidates],
        [x["silhouette"] for x in candidates],
        marker="o",
    )
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette score")
    plt.title("Workload clustering model selection")
    plt.tight_layout()
    plt.savefig(
        out / "cluster_silhouette.png", dpi=160
    )
    plt.close()

    joblib.dump(
        {"scaler": scaler, "kmeans": final,
         "features": list(X.columns)},
        out / "clustering_model.joblib",
    )

    return {
        "status": "COMPLETED",
        "best_k": best_k,
        "silhouette": max(
            x["silhouette"] for x in candidates
        ),
        "cluster_sizes": result["cluster"]
            .value_counts().sort_index().to_dict(),
    }


def run_classification(X, df, out):
    LOGGER.info("Running grouped workload classification")

    y = df["label"].astype(str)
    groups = df["validation_group"].astype(str)

    if y.nunique() < 2 or groups.nunique() < 3:
        return {"status": "INSUFFICIENT_CLASS_OR_GROUP_VARIATION"}

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups)
    )

    train_classes = set(y.iloc[train_idx])
    test_classes = set(y.iloc[test_idx])

    if len(train_classes) < 2:
        return {
            "status": "INVALID_TRAIN_SPLIT",
            "reason": "Fewer than two training classes",
        }

    model = make_classifier()
    model.fit(X.iloc[train_idx], y.iloc[train_idx])

    predicted = model.predict(X.iloc[test_idx])

    metrics = safe_metrics(
        y.iloc[test_idx], predicted
    )

    labels = sorted(set(y) | set(predicted))

    report = classification_report(
        y.iloc[test_idx],
        predicted,
        labels=labels,
        zero_division=0,
        output_dict=True,
    )

    save_json(
        out / "classification_report.json", report
    )

    predictions = pd.DataFrame({
        "actual": y.iloc[test_idx].to_numpy(),
        "predicted": predicted,
        "group": groups.iloc[test_idx].to_numpy(),
        "sql_hash": df.iloc[test_idx][
            "sql_hash_computed"
        ].to_numpy(),
    })

    predictions.to_csv(
        out / "classification_predictions.csv",
        index=False,
    )

    matrix = confusion_matrix(
        y.iloc[test_idx],
        predicted,
        labels=labels,
    )

    pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    ).to_csv(out / "confusion_matrix.csv")

    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, aspect="auto")
    plt.colorbar()
    plt.xticks(
        range(len(labels)), labels, rotation=45, ha="right"
    )
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.title("Grouped holdout confusion matrix")
    plt.tight_layout()
    plt.savefig(
        out / "confusion_matrix.png", dpi=160
    )
    plt.close()

    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.named_steps[
            "model"
        ].feature_importances_,
    }).sort_values("importance", ascending=False)

    importance.to_csv(
        out / "classification_feature_importance.csv",
        index=False,
    )

    joblib.dump(
        {
            "model": model,
            "feature_columns": list(X.columns),
            "classes": list(model.classes_),
            "validation": "grouped_holdout",
        },
        out / "classification_holdout_model.joblib",
    )

    final_model = make_classifier()
    final_model.fit(X, y)

    joblib.dump(
        {
            "model": final_model,
            "feature_columns": list(X.columns),
            "classes": list(final_model.classes_),
            "validation": "trained_on_all_phase6_records",
        },
        out / "classification_deployment_model.joblib",
    )

    return {
        "status": "COMPLETED",
        "training_rows": len(train_idx),
        "testing_rows": len(test_idx),
        "training_groups": groups.iloc[
            train_idx
        ].nunique(),
        "testing_groups": groups.iloc[
            test_idx
        ].nunique(),
        "training_classes": sorted(train_classes),
        "testing_classes": sorted(test_classes),
        "missing_test_classes": sorted(
            set(y) - test_classes
        ),
        **metrics,
    }


def run_grouped_cv(X, df, out):
    LOGGER.info("Running grouped cross-validation")

    y = df["label"].astype(str)
    groups = df["validation_group"].astype(str)

    n_splits = min(5, groups.nunique())

    if n_splits < 2:
        return {"status": "INSUFFICIENT_GROUPS"}

    splitter = GroupKFold(n_splits=n_splits)
    records = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(X, y, groups), start=1
    ):
        if y.iloc[train_idx].nunique() < 2:
            continue

        model = make_classifier()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        pred = model.predict(X.iloc[test_idx])

        row = {
            "fold": fold,
            "train_rows": len(train_idx),
            "test_rows": len(test_idx),
            "test_classes": y.iloc[
                test_idx
            ].nunique(),
            **safe_metrics(y.iloc[test_idx], pred),
        }
        records.append(row)

    if not records:
        return {"status": "NO_VALID_FOLDS"}

    pd.DataFrame(records).to_csv(
        out / "grouped_cv_folds.csv", index=False
    )

    return {
        "status": "COMPLETED",
        "folds": len(records),
        "mean_macro_f1": float(
            np.mean([r["macro_f1"] for r in records])
        ),
        "std_macro_f1": float(
            np.std([r["macro_f1"] for r in records])
        ),
        "mean_accuracy": float(
            np.mean([r["accuracy"] for r in records])
        ),
    }


def parse_timestamps(series):
    try:
        parsed = pd.to_datetime(
            series, errors="coerce", utc=True,
            format="mixed",
        )
    except (TypeError, ValueError):
        parsed = pd.to_datetime(
            series, errors="coerce", utc=True
        )
    return parsed


def load_monitoring(path: Path):
    LOGGER.info("Loading Phase 7 monitoring")

    if not path.exists():
        raise FileNotFoundError(
            f"Phase 7 monitoring file not found: {path}"
        )

    df = pd.read_csv(path, low_memory=False)

    time_col = pick_column(df, TIMESTAMP_ALIASES)
    latency_col = pick_column(df, LATENCY_ALIASES)
    label_col = pick_column(df, LABEL_ALIASES)
    sql_col = pick_column(df, SQL_ALIASES)

    if time_col is None:
        raise ValueError(
            "No timestamp column found. Expected monitored_at."
        )

    df["event_timestamp"] = parse_timestamps(df[time_col])

    valid = df["event_timestamp"].notna()
    dropped = int((~valid).sum())

    df = df.loc[valid].copy()
    df = df.sort_values(
        "event_timestamp", kind="stable"
    ).reset_index(drop=True)

    if df.empty:
        raise ValueError("No parseable monitoring timestamps.")

    if latency_col is not None:
        df["latency_ms"] = pd.to_numeric(
            df[latency_col], errors="coerce"
        )
        df.loc[
            (df["latency_ms"] < 0) |
            (~np.isfinite(df["latency_ms"])),
            "latency_ms",
        ] = np.nan
    else:
        df["latency_ms"] = np.nan

    if label_col is not None:
        df["workload_label"] = df[
            label_col
        ].map(valid_label)
    else:
        df["workload_label"] = "unknown"

    if sql_col is not None:
        df["sql_input"] = df[
            sql_col
        ].fillna("").astype(str)
    else:
        df["sql_input"] = ""

    span = (
        df["event_timestamp"].iloc[-1] -
        df["event_timestamp"].iloc[0]
    ).total_seconds()

    metadata = {
        "status": "SOURCE_SELECTED",
        "source": str(path),
        "records": len(df),
        "dropped_invalid_timestamps": dropped,
        "first_timestamp": df[
            "event_timestamp"
        ].iloc[0],
        "last_timestamp": df[
            "event_timestamp"
        ].iloc[-1],
        "duration_seconds": span,
        "duration_minutes": span / 60,
        "timestamp_column": time_col,
        "latency_column": latency_col,
        "label_column": label_col,
        "sql_column": sql_col,
        "distinct_timestamps": df[
            "event_timestamp"
        ].nunique(),
    }

    return df, metadata


def temporal_bins(df, seconds):
    start = df["event_timestamp"].iloc[0].floor(
        f"{seconds}s"
    )
    end = df["event_timestamp"].iloc[-1].ceil(
        f"{seconds}s"
    )

    index = pd.date_range(
        start=start,
        end=end,
        freq=f"{seconds}s",
        tz="UTC",
    )

    counts = (
        df.set_index("event_timestamp")
        .resample(f"{seconds}s")
        .size()
        .reindex(index, fill_value=0)
    )

    result = pd.DataFrame({
        "timestamp": counts.index,
        "query_count": counts.to_numpy(),
    })

    return result


def make_lag_dataset(values, lags):
    rows = []
    targets = []
    positions = []

    for t in range(lags, len(values)):
        history = values[t-lags:t]

        row = list(history[::-1])
        row.extend([
            float(np.mean(history)),
            float(np.std(history)),
            float(np.min(history)),
            float(np.max(history)),
            float(history[-1] - history[0]),
        ])

        rows.append(row)
        targets.append(values[t])
        positions.append(t)

    columns = [
        f"lag_{i}" for i in range(1, lags + 1)
    ] + [
        "rolling_mean",
        "rolling_std",
        "rolling_min",
        "rolling_max",
        "rolling_change",
    ]

    return (
        pd.DataFrame(rows, columns=columns),
        np.asarray(targets, dtype=float),
        np.asarray(positions),
    )


def forecast_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return {
        "mae": mean_absolute_error(actual, predicted),
        "rmse": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "r2": (
            r2_score(actual, predicted)
            if len(actual) > 1 else None
        ),
    }


def run_forecasting(df, out, bin_seconds=5):
    LOGGER.info("Running real-timestamp workload forecasting")

    bins = temporal_bins(df, bin_seconds)
    bins.to_csv(
        out / "temporal_query_counts.csv",
        index=False,
    )

    values = bins["query_count"].to_numpy(dtype=float)

    if len(values) < 45:
        return {
            "status": "INSUFFICIENT_BINS",
            "bins": len(values),
            "minimum_bins": 45,
        }

    lags = min(10, max(4, len(values) // 10))

    X, y, positions = make_lag_dataset(values, lags)

    split = int(len(X) * 0.75)

    if split < 20 or len(X) - split < 10:
        return {
            "status": "INSUFFICIENT_TRAIN_TEST_WINDOWS"
        }

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y[:split]
    y_test = y[split:]

    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=8,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    predicted = np.maximum(
        0, model.predict(X_test)
    )

    persistence = values[positions[split:] - 1]

    rolling = np.asarray([
        np.mean(values[t-lags:t])
        for t in positions[split:]
    ])

    baseline_zero = np.zeros_like(y_test)

    scores = {
        "random_forest": forecast_metrics(
            y_test, predicted
        ),
        "persistence": forecast_metrics(
            y_test, persistence
        ),
        "rolling_mean": forecast_metrics(
            y_test, rolling
        ),
        "zero_baseline": forecast_metrics(
            y_test, baseline_zero
        ),
    }

    pred_frame = pd.DataFrame({
        "timestamp": bins.iloc[
            positions[split:]
        ]["timestamp"].to_numpy(),
        "actual_queries": y_test,
        "rf_prediction": predicted,
        "persistence_prediction": persistence,
        "rolling_mean_prediction": rolling,
        "zero_prediction": baseline_zero,
    })

    pred_frame.to_csv(
        out / "forecast_test_predictions.csv",
        index=False,
    )

    plt.figure(figsize=(11, 5))
    plt.plot(
        np.arange(len(y_test)),
        y_test,
        label="Actual",
        marker=".",
    )
    plt.plot(
        np.arange(len(y_test)),
        predicted,
        label="Random forest",
    )
    plt.plot(
        np.arange(len(y_test)),
        rolling,
        label="Rolling mean baseline",
    )
    plt.xlabel("Chronological test bin")
    plt.ylabel("Queries per bin")
    plt.title(
        f"Within-session query forecasting ({bin_seconds}s bins)"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        out / "forecast_comparison.png", dpi=160
    )
    plt.close()

    baseline_maes = {
        name: result["mae"]
        for name, result in scores.items()
        if name != "random_forest"
    }

    best_baseline = min(
        baseline_maes,
        key=baseline_maes.get,
    )

    rf_mae = scores["random_forest"]["mae"]
    best_mae = baseline_maes[best_baseline]

    improvement = (
        100 * (best_mae - rf_mae) / best_mae
        if best_mae > 0 else None
    )

    joblib.dump(
        {
            "model": model,
            "lags": lags,
            "bin_seconds": bin_seconds,
            "training_scope": "single_short_benchmark_session",
        },
        out / "short_term_forecaster.joblib",
    )

    save_json(
        out / "forecast_metrics.json",
        scores,
    )

    return {
        "status": "COMPLETED_WITHIN_SESSION",
        "bin_seconds": bin_seconds,
        "total_bins": len(bins),
        "training_bins": len(y_train),
        "testing_bins": len(y_test),
        "lags": lags,
        "model_scores": scores,
        "best_baseline": best_baseline,
        "rf_beats_best_baseline": bool(
            rf_mae < best_mae
        ),
        "mae_improvement_over_best_baseline_pct":
            improvement,
        "limitations": [
            "Single benchmark session only",
            "Not a daily or weekly demand forecast",
            "Autocorrelation may reflect benchmark execution order",
            "Results require replication across independent sessions",
        ],
    }


def psi_statistic(reference, current, bins=8):
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]

    if len(reference) < 15 or len(current) < 15:
        return None

    if np.unique(reference).size < 2:
        return None

    edges = np.unique(
        np.quantile(
            reference,
            np.linspace(0, 1, bins + 1),
        )
    )

    if len(edges) < 3:
        return None

    edges[0] = -np.inf
    edges[-1] = np.inf

    p = np.histogram(
        reference, bins=edges
    )[0].astype(float)

    q = np.histogram(
        current, bins=edges
    )[0].astype(float)

    epsilon = 1e-6

    p = np.maximum(p / max(p.sum(), 1), epsilon)
    q = np.maximum(q / max(q.sum(), 1), epsilon)

    p /= p.sum()
    q /= q.sum()

    return float(
        np.sum((q - p) * np.log(q / p))
    )


def js_divergence(labels_a, labels_b):
    a = Counter(map(str, labels_a))
    b = Counter(map(str, labels_b))

    categories = sorted(set(a) | set(b))

    if not categories:
        return None

    p = np.asarray(
        [a.get(x, 0) for x in categories],
        dtype=float,
    )
    q = np.asarray(
        [b.get(x, 0) for x in categories],
        dtype=float,
    )

    p /= p.sum()
    q /= q.sum()

    m = (p + q) / 2

    def kl(x, y):
        mask = x > 0
        return np.sum(
            x[mask] * np.log2(x[mask] / y[mask])
        )

    return float(
        0.5 * kl(p, m) + 0.5 * kl(q, m)
    )


def run_drift(df, out):
    LOGGER.info("Running within-session drift diagnostics")

    if len(df) < 80:
        return {
            "status": "INSUFFICIENT_EVENTS",
            "events": len(df),
        }

    timestamps = df["event_timestamp"]

    midpoint = timestamps.iloc[0] + (
        timestamps.iloc[-1] - timestamps.iloc[0]
    ) / 2

    first = df[timestamps < midpoint].copy()
    second = df[timestamps >= midpoint].copy()

    if min(len(first), len(second)) < 30:
        return {
            "status": "INSUFFICIENT_EVENTS_PER_TIME_WINDOW",
            "first_window_events": len(first),
            "second_window_events": len(second),
        }

    records = []

    if (
        first["latency_ms"].notna().sum() >= 15 and
        second["latency_ms"].notna().sum() >= 15
    ):
        score = psi_statistic(
            first["latency_ms"],
            second["latency_ms"],
        )
        records.append({
            "feature": "latency_ms",
            "metric": "PSI",
            "score": score,
            "reference_mean": first[
                "latency_ms"
            ].mean(),
            "current_mean": second[
                "latency_ms"
            ].mean(),
        })

    label_score = js_divergence(
        first["workload_label"],
        second["workload_label"],
    )

    records.append({
        "feature": "workload_label",
        "metric": "Jensen_Shannon_divergence_bits",
        "score": label_score,
        "reference_mean": None,
        "current_mean": None,
    })

    if df["sql_input"].str.strip().ne("").any():
        sql_X = feature_frame(df["sql_input"])

        midpoint_index = len(first)

        for feature in sql_X.columns:
            score = psi_statistic(
                sql_X[feature].iloc[:midpoint_index],
                sql_X[feature].iloc[midpoint_index:],
            )

            if score is not None:
                records.append({
                    "feature": feature,
                    "metric": "PSI",
                    "score": score,
                    "reference_mean": sql_X[
                        feature
                    ].iloc[:midpoint_index].mean(),
                    "current_mean": sql_X[
                        feature
                    ].iloc[midpoint_index:].mean(),
                })

    result = pd.DataFrame(records)

    if result.empty:
        return {
            "status": "NO_COMPARABLE_FEATURES"
        }

    result["heuristic_flag"] = result.apply(
        lambda r: (
            "POTENTIAL_SHIFT"
            if r["metric"] == "PSI"
            and pd.notna(r["score"])
            and r["score"] >= 0.2
            else "DESCRIPTIVE_ONLY"
            if r["metric"] != "PSI"
            else "NO_LARGE_PSI_FLAG"
        ),
        axis=1,
    )

    result.to_csv(
        out / "temporal_drift_metrics.csv",
        index=False,
    )

    first_counts = first[
        "workload_label"
    ].value_counts(normalize=True)

    second_counts = second[
        "workload_label"
    ].value_counts(normalize=True)

    labels = sorted(
        set(first_counts.index) |
        set(second_counts.index)
    )

    composition = pd.DataFrame({
        "workload_label": labels,
        "first_half_proportion": [
            first_counts.get(x, 0) for x in labels
        ],
        "second_half_proportion": [
            second_counts.get(x, 0) for x in labels
        ],
    })

    composition.to_csv(
        out / "workload_composition_shift.csv",
        index=False,
    )

    positions = np.arange(len(labels))
    width = 0.38

    plt.figure(figsize=(10, 5))
    plt.bar(
        positions - width / 2,
        composition["first_half_proportion"],
        width,
        label="First time window",
    )
    plt.bar(
        positions + width / 2,
        composition["second_half_proportion"],
        width,
        label="Second time window",
    )
    plt.xticks(
        positions, labels, rotation=35, ha="right"
    )
    plt.ylabel("Proportion of executions")
    plt.title("Within-session workload composition")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        out / "workload_composition_drift.png",
        dpi=160,
    )
    plt.close()

    return {
        "status": "COMPLETED_DIAGNOSTIC_ONLY",
        "first_window_events": len(first),
        "second_window_events": len(second),
        "midpoint_utc": midpoint,
        "js_divergence_bits": label_score,
        "psi_features_evaluated": int(
            (result["metric"] == "PSI").sum()
        ),
        "psi_features_flagged": int(
            (result["heuristic_flag"] ==
             "POTENTIAL_SHIFT").sum()
        ),
        "limitations": [
            "PSI threshold is heuristic, not a significance test",
            "No established long-term reference distribution",
            "Workload order may be controlled by benchmark design",
            "No demonstrated long-term concept drift",
        ],
    }


def run_latency_model(df, out):
    LOGGER.info("Running SQL-only latency prediction")

    usable = df[
        df["latency_ms"].notna() &
        df["sql_input"].str.strip().ne("")
    ].copy()

    if len(usable) < 60:
        return {
            "status": "INSUFFICIENT_SQL_LATENCY_RECORDS",
            "records": len(usable),
        }

    usable["normalized_sql"] = usable[
        "sql_input"
    ].map(normalize_sql)

    usable["group"] = usable[
        "normalized_sql"
    ].map(hash_text)

    if usable["group"].nunique() < 3:
        return {
            "status": "INSUFFICIENT_SQL_GROUPS"
        }

    X = feature_frame(usable["sql_input"])

    y = np.log1p(
        usable["latency_ms"].to_numpy()
    )

    groups = usable["group"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups)
    )

    model = HistGradientBoostingRegressor(
        max_iter=180,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )

    model.fit(X.iloc[train_idx], y[train_idx])

    predicted = np.maximum(
        0,
        np.expm1(model.predict(X.iloc[test_idx])),
    )

    actual = np.expm1(y[test_idx])

    baseline_value = float(
        np.median(np.expm1(y[train_idx]))
    )

    baseline = np.full_like(
        actual, baseline_value
    )

    metrics = {
        "model": forecast_metrics(actual, predicted),
        "training_median_baseline": forecast_metrics(
            actual, baseline
        ),
    }

    predictions = pd.DataFrame({
        "actual_ms": actual,
        "predicted_ms": predicted,
        "baseline_ms": baseline,
        "sql_group": groups.iloc[
            test_idx
        ].to_numpy(),
    })

    predictions.to_csv(
        out / "latency_predictions.csv",
        index=False,
    )

    joblib.dump(
        {
            "model": model,
            "features": list(X.columns),
            "target_transform": "log1p_latency_ms",
            "validation": "grouped_sql_holdout",
        },
        out / "latency_model.joblib",
    )

    save_json(
        out / "latency_metrics.json",
        metrics,
    )

    return {
        "status": "COMPLETED",
        "training_rows": len(train_idx),
        "testing_rows": len(test_idx),
        "metrics": metrics,
        "beats_median_baseline": bool(
            metrics["model"]["mae"] <
            metrics["training_median_baseline"]["mae"]
        ),
        "limitation": (
            "SQL-only features cannot capture all runtime "
            "effects such as caching, data size, or hardware load"
        ),
    }


def run_optimization_ranking(df, out):
    LOGGER.info("Generating optimization risk ranking")

    if "fingerprint_hash" in df.columns:
        key = df[
            "fingerprint_hash"
        ].fillna("").astype(str)
    else:
        key = df[
            "sql_input"
        ].map(normalize_sql).map(hash_text)

    temp = pd.DataFrame({
        "fingerprint_hash": key,
        "latency_ms": df["latency_ms"],
        "workload_label": df["workload_label"],
    })

    temp["fingerprint_hash"] = temp[
        "fingerprint_hash"
    ].replace("", "unknown")

    if "status" in df.columns:
        temp["status"] = df["status"].fillna(
            "unknown"
        ).astype(str)
    else:
        temp["status"] = "unknown"

    result = temp.groupby(
        "fingerprint_hash", dropna=False
    ).agg(
        executions=("latency_ms", "size"),
        mean_latency_ms=("latency_ms", "mean"),
        median_latency_ms=("latency_ms", "median"),
        p95_latency_ms=(
            "latency_ms",
            lambda x: (
                x.quantile(0.95)
                if x.notna().any() else np.nan
            ),
        ),
        missing_latency_records=(
            "latency_ms",
            lambda x: x.isna().sum(),
        ),
    ).reset_index()

    result["observed_latency_burden_ms"] = (
        result["executions"] *
        result["mean_latency_ms"].fillna(0)
    )

    result["priority_score"] = (
        result["observed_latency_burden_ms"].rank(
            pct=True
        ) * 0.65 +
        result["p95_latency_ms"].fillna(0).rank(
            pct=True
        ) * 0.25 +
        result["executions"].rank(
            pct=True
        ) * 0.10
    )

    result = result.sort_values(
        "priority_score", ascending=False
    )

    result["priority_rank"] = np.arange(
        1, len(result) + 1
    )

    result.to_csv(
        out / "optimization_priorities.csv",
        index=False,
    )

    return {
        "status": "COMPLETED",
        "fingerprints": len(result),
        "interpretation": (
            "Descriptive ranking for candidate investigation; "
            "not an index recommendation or measured speedup"
        ),
    }


def run_monitoring_visualizations(df, out):
    LOGGER.info("Creating monitoring visualizations")

    plt.figure(figsize=(11, 5))
    plt.plot(
        df["event_timestamp"],
        df["latency_ms"],
        linewidth=0.8,
    )
    plt.xlabel("Recorded execution time (UTC)")
    plt.ylabel("Execution latency (ms)")
    plt.title("Phase 7 query latency over time")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(
        out / "latency_timeline.png", dpi=160
    )
    plt.close()

    counts = df["workload_label"].value_counts()

    plt.figure(figsize=(9, 5))
    plt.bar(counts.index, counts.values)
    plt.xticks(rotation=35, ha="right")
    plt.xlabel("Workload label")
    plt.ylabel("Executions")
    plt.title("Observed workload composition")
    plt.tight_layout()
    plt.savefig(
        out / "workload_distribution.png",
        dpi=160,
    )
    plt.close()

    return {
        "status": "COMPLETED",
        "figures": [
            "latency_timeline.png",
            "workload_distribution.png",
        ],
    }


def run_stage(name, fn, summary, out):
    try:
        result = fn()

        summary["components"][name] = result

        LOGGER.info(
            "%s => %s",
            name,
            result.get("status", "UNKNOWN"),
        )

        return result

    except Exception as exc:
        LOGGER.exception("Stage failed: %s", name)

        summary["components"][name] = {
            "status": "FAILED",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }

        summary["errors"].append({
            "stage": name,
            "error": str(exc),
        })

        save_json(
            out / "summary_partial.json",
            summary,
        )

        return None


def write_research_report(summary, out):
    components = summary["components"]

    def status(name):
        return components.get(
            name, {}
        ).get("status", "NOT_RUN")

    lines = [
        "# Phase 8 — AI Workload Intelligence",
        "",
        "## Experimental setting",
        "",
        f"- Phase 6 source: {summary['phase6_source']}",
        f"- Phase 7 source: {summary['phase7_source']}",
        "- Database access: none",
        "- Database modifications: none",
        "- Random seed: 42",
        "",
        "## Component status",
        "",
    ]

    for name in [
        "dataset",
        "clustering",
        "classification",
        "cross_validation",
        "monitoring",
        "latency_prediction",
        "optimization_ranking",
        "forecasting",
        "temporal_drift",
        "visualizations",
    ]:
        lines.append(f"- {name}: {status(name)}")

    lines += [
        "",
        "## Scientific interpretation",
        "",
        "SQL classification is evaluated using grouped splits.",
        "A successful classifier is not a SQL/Cypher/vector router.",
        "",
        "Forecasting uses a chronological holdout from one "
        "short monitoring session.",
        "It is not a validated daily or weekly demand predictor.",
        "",
        "Drift analysis compares time-separated portions of "
        "the same session.",
        "PSI and Jensen-Shannon divergence are descriptive "
        "diagnostics, not proof of long-term concept drift.",
        "",
        "Optimization priorities are candidate rankings, "
        "not deployed physical-design changes.",
        "",
        "## Safety",
        "",
        "No PostgreSQL or Neo4j connection is created.",
        "No schema, index, graph, or clinical data is modified.",
        "",
        "## Limitations",
        "",
        "- Single short benchmark monitoring session.",
        "- Potential scripted workload-order effects.",
        "- No independent multi-day temporal validation.",
        "- No demonstrated adaptive closed-loop optimization.",
        "- No demonstrated clinical decision accuracy.",
        "- No demonstrated hybrid-engine routing policy.",
        "",
    ]

    (out / "research_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Complete read-only Phase 8 workload intelligence"
        )
    )

    parser.add_argument(
        "--phase6",
        default=str(DEFAULT_PHASE6),
    )
    parser.add_argument(
        "--phase7",
        default=str(DEFAULT_PHASE7),
    )
    parser.add_argument(
        "--bin-seconds",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--mode",
        choices=["all", "static", "temporal"],
        default="all",
    )

    args = parser.parse_args()

    if args.bin_seconds < 1:
        parser.error("--bin-seconds must be positive")

    stamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S_%f")

    out = RESULTS / "phase8_complete" / stamp
    setup_logging(out)

    summary = {
        "phase": 8,
        "status": "RUNNING",
        "started_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "phase6_source": args.phase6,
        "phase7_source": args.phase7,
        "output_directory": str(out),
        "mode": args.mode,
        "components": {},
        "errors": [],
        "database_modifications": "NONE",
    }

    LOGGER.info("=" * 72)
    LOGGER.info("PHASE 8: COMPLETE AI WORKLOAD INTELLIGENCE")
    LOGGER.info("=" * 72)
    LOGGER.info("Output directory: %s", out)
    LOGGER.info("No database connections will be made")

    if args.mode in ("all", "static"):
        try:
            LOGGER.info("Loading Phase 6 SQL workload")

            workload = load_phase6(Path(args.phase6))
            X = feature_frame(workload["sql_input"])

            X.to_csv(
                out / "sql_feature_matrix.csv",
                index=False,
            )

            workload[[
                "label",
                "sql_hash_computed",
                "validation_group",
            ]].to_csv(
                out / "workload_metadata.csv",
                index=False,
            )

            summary["components"]["dataset"] = {
                "status": "COMPLETED",
                "rows": len(workload),
                "features": X.shape[1],
                "classes": workload[
                    "label"
                ].value_counts().to_dict(),
                "unique_groups": workload[
                    "validation_group"
                ].nunique(),
            }

            run_stage(
                "clustering",
                lambda: run_clustering(
                    X, workload, out
                ),
                summary, out,
            )

            run_stage(
                "classification",
                lambda: run_classification(
                    X, workload, out
                ),
                summary, out,
            )

            run_stage(
                "cross_validation",
                lambda: run_grouped_cv(
                    X, workload, out
                ),
                summary, out,
            )

        except Exception as exc:
            LOGGER.exception("Static pipeline failed")
            summary["errors"].append({
                "stage": "static_pipeline",
                "error": str(exc),
            })
            summary["components"]["dataset"] = {
                "status": "FAILED",
                "error": str(exc),
            }

    if args.mode in ("all", "temporal"):
        monitoring = None

        try:
            monitoring, metadata = load_monitoring(
                Path(args.phase7)
            )

            summary["components"][
                "monitoring"
            ] = metadata

            save_json(
                out / "monitoring_metadata.json",
                metadata,
            )

        except Exception as exc:
            LOGGER.exception(
                "Monitoring integration failed"
            )
            summary["errors"].append({
                "stage": "monitoring",
                "error": str(exc),
            })
            summary["components"]["monitoring"] = {
                "status": "FAILED",
                "error": str(exc),
            }

        if monitoring is not None:
            run_stage(
                "latency_prediction",
                lambda: run_latency_model(
                    monitoring, out
                ),
                summary, out,
            )

            run_stage(
                "optimization_ranking",
                lambda: run_optimization_ranking(
                    monitoring, out
                ),
                summary, out,
            )

            run_stage(
                "forecasting",
                lambda: run_forecasting(
                    monitoring,
                    out,
                    args.bin_seconds,
                ),
                summary, out,
            )

            run_stage(
                "temporal_drift",
                lambda: run_drift(
                    monitoring, out
                ),
                summary, out,
            )

            run_stage(
                "visualizations",
                lambda: run_monitoring_visualizations(
                    monitoring, out
                ),
                summary, out,
            )

    summary["finished_utc"] = datetime.now(
        timezone.utc
    ).isoformat()

    if summary["errors"]:
        summary["status"] = "COMPLETED_WITH_ERRORS"
    else:
        summary["status"] = "COMPLETED_WITH_LIMITATIONS"

    write_research_report(summary, out)
    save_json(out / "summary.json", summary)

    print("\n" + "=" * 72)
    print("PHASE 8 FINAL SUMMARY")
    print("=" * 72)
    print("Status:", summary["status"])
    print("Output:", out)

    for name, value in summary["components"].items():
        print(
            f"{name:24s} => "
            f"{value.get('status', 'UNKNOWN')}"
        )

    print("Errors:", len(summary["errors"]))
    print("Database modifications: NONE")
    print("=" * 72)

    if summary["errors"]:
        print(
            "\nInspect phase8.log and summary.json "
            "for error details."
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
