
#!/usr/bin/env python3
"""
PHASE 8: AI WORKLOAD LEARNING AND TEMPORAL INTELLIGENCE
Adaptive Healthcare Knowledge Database

Modules:
  8.1  Workload discovery and data validation
  8.2  SQL normalization and fingerprinting
  8.3  Structural feature extraction
  8.4  Unsupervised workload clustering
  8.5  Supervised workload classification
  8.6  Grouped cross-validation
  8.7  Phase 7 monitoring integration
  8.8  Chronological workload demand forecasting
  8.9  Temporal workload drift detection
  8.10 Research reporting and reproducibility

READ-ONLY:
  No database connections.
  No database modifications.
  No existing models overwritten.
  Results saved in a new timestamped directory.

Run:
  python -u 08_full.py --mode all

Optional explicit monitoring file:
  python -u 08_full.py --mode all --temporal-csv PATH

Optional real timestamp column:
  python -u 08_full.py --mode all --timestamp-column executed_at
"""

import argparse
import hashlib
import json
import logging
import math
import re
import sys
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
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    silhouette_score,
)
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
)
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

PHASE6_FILE = (
    RESULTS
    / "phase6_workloads"
    / "combined_workload.csv"
)

OUTPUT_ROOT = RESULTS / "phase8_full"

RANDOM_STATE = 42

SQL_ALIASES = [
    "sql", "sql_text", "query", "query_text",
    "statement", "query_sql", "raw_sql"
]

LABEL_ALIASES = [
    "workload_label", "workload_type",
    "workload", "category", "role",
    "persona", "query_category"
]

TIMESTAMP_ALIASES = [
    "timestamp", "execution_timestamp",
    "executed_at", "execution_time",
    "start_time", "started_at",
    "event_time", "query_time",
    "created_at", "datetime",
    "start_timestamp", "run_timestamp",
]

LATENCY_ALIASES = [
    "latency_ms", "execution_time_ms",
    "execution_ms", "duration_ms",
    "elapsed_ms", "runtime_ms",
    "total_time_ms"
]

FEATURE_NAMES = [
    "sql_length",
    "token_count",
    "select_count",
    "join_count",
    "left_join_count",
    "where_count",
    "group_count",
    "order_count",
    "having_count",
    "limit_count",
    "distinct_count",
    "union_count",
    "cte_count",
    "subquery_count",
    "aggregate_count",
    "case_count",
    "exists_count",
    "in_count",
    "between_count",
    "like_count",
    "comparison_count",
    "logical_count",
    "parenthesis_count",
    "nesting_depth",
    "parameter_count",
]

LOG = logging.getLogger("phase8")


# ============================================================
# UTILITIES
# ============================================================

def setup_logging(out):
    out.mkdir(parents=True, exist_ok=True)

    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s"
    )

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    LOG.addHandler(stream)

    file_handler = logging.FileHandler(
        out / "execution.log",
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    LOG.addHandler(file_handler)


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {
            str(k): to_jsonable(v)
            for k, v in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, (Path, pd.Timestamp)):
        return str(obj)

    if isinstance(obj, float) and not math.isfinite(obj):
        return None

    return obj


def save_json(path, data):
    path.write_text(
        json.dumps(
            to_jsonable(data),
            indent=2,
            default=str
        ),
        encoding="utf-8"
    )


def save_csv(path, df):
    df.to_csv(path, index=False)


def find_column(df, aliases):
    columns = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for name in aliases:
        if name.lower() in columns:
            return columns[name.lower()]

    return None


def sha256(text):
    return hashlib.sha256(
        str(text).encode("utf-8")
    ).hexdigest()


def safe_read_csv(path):
    return pd.read_csv(
        path,
        low_memory=False
    )


# ============================================================
# 8.1 WORKLOAD INGESTION
# ============================================================

def load_phase6(path):
    LOG.info("Loading Phase 6 workload: %s", path)

    if not path.exists():
        raise FileNotFoundError(
            f"Phase 6 workload not found: {path}"
        )

    df = safe_read_csv(path)

    sql_col = find_column(df, SQL_ALIASES)

    if sql_col is None:
        raise ValueError(
            "Phase 6 workload has no recognized SQL column. "
            f"Columns: {list(df.columns)}"
        )

    label_col = find_column(df, LABEL_ALIASES)

    result = pd.DataFrame()

    result["sql"] = (
        df[sql_col]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    if label_col is not None:
        result["label"] = (
            df[label_col]
            .astype("string")
            .str.lower()
            .str.strip()
        )
    else:
        result["label"] = pd.Series(
            pd.NA,
            index=result.index,
            dtype="string"
        )

    result = result[
        result["sql"].str.len() > 0
    ].reset_index(drop=True)

    LOG.info(
        "Loaded %d valid SQL records",
        len(result)
    )

    return result


# ============================================================
# 8.2 SQL FINGERPRINTING
# ============================================================

def normalize_sql(sql):
    sql = str(sql)

    # Heuristic normalization, not a complete SQL parser.
    sql = re.sub(
        r"/\*.*?\*/", " ", sql,
        flags=re.DOTALL
    )

    sql = re.sub(
        r"--[^\r\n]*", " ", sql
    )

    sql = re.sub(
        r"'(?:''|[^'])*'", "?",
        sql
    )

    sql = re.sub(
        r"\$\d+", "?", sql
    )

    sql = re.sub(
        r"\b\d+(?:\.\d+)?\b", "?",
        sql
    )

    sql = sql.lower()

    sql = re.sub(
        r"\s+", " ", sql
    )

    return sql.strip().rstrip(";")


def fingerprint_sql(sql):
    return sha256(normalize_sql(sql))


def add_fingerprints(df):
    df = df.copy()

    df["normalized_sql"] = (
        df["sql"].map(normalize_sql)
    )

    df["fingerprint"] = (
        df["normalized_sql"].map(sha256)
    )

    df["exact_hash"] = (
        df["sql"].map(sha256)
    )

    return df


# ============================================================
# 8.3 STRUCTURAL FEATURE ENGINEERING
# ============================================================

def count(sql, pattern):
    return len(
        re.findall(
            pattern,
            sql,
            flags=re.IGNORECASE
        )
    )


def max_depth(sql):
    depth = 0
    maximum = 0

    for char in sql:
        if char == "(":
            depth += 1
            maximum = max(maximum, depth)
        elif char == ")":
            depth = max(0, depth - 1)

    return maximum


def extract_features(sql):
    s = normalize_sql(sql)

    tokens = re.findall(
        r"\b[a-z_][a-z_0-9]*\b", s
    )

    return {
        "sql_length": len(s),
        "token_count": len(tokens),
        "select_count": count(s, r"\bselect\b"),
        "join_count": count(s, r"\bjoin\b"),
        "left_join_count": count(
            s, r"\bleft\s+join\b"
        ),
        "where_count": count(s, r"\bwhere\b"),
        "group_count": count(
            s, r"\bgroup\s+by\b"
        ),
        "order_count": count(
            s, r"\border\s+by\b"
        ),
        "having_count": count(s, r"\bhaving\b"),
        "limit_count": count(s, r"\blimit\b"),
        "distinct_count": count(
            s, r"\bdistinct\b"
        ),
        "union_count": count(s, r"\bunion\b"),
        "cte_count": count(s, r"^\s*with\b"),
        "subquery_count": max(
            0, count(s, r"\bselect\b") - 1
        ),
        "aggregate_count": count(
            s, r"\b(count|sum|avg|min|max)\s*\("
        ),
        "case_count": count(s, r"\bcase\b"),
        "exists_count": count(s, r"\bexists\b"),
        "in_count": count(s, r"\bin\s*\("),
        "between_count": count(
            s, r"\bbetween\b"
        ),
        "like_count": count(
            s, r"\b(like|ilike)\b"
        ),
        "comparison_count": count(
            s, r"<=|>=|<>|!=|(?<![<>!=])=(?!=)|<|>"
        ),
        "logical_count": count(
            s, r"\b(and|or|not)\b"
        ),
        "parenthesis_count": s.count("("),
        "nesting_depth": max_depth(s),
        "parameter_count": s.count("?"),
    }


def build_features(df):
    LOG.info("Extracting SQL features")

    features = pd.DataFrame(
        [
            extract_features(sql)
            for sql in df["sql"]
        ],
        index=df.index
    )

    return features[FEATURE_NAMES].astype(float)


# ============================================================
# 8.4 WORKLOAD CLUSTERING
# ============================================================

def run_clustering(df, features, out, max_k=6):
    LOG.info("Running workload clustering")

    if len(features) < 4:
        return {"status": "INSUFFICIENT_DATA"}

    unique_count = len(
        features.drop_duplicates()
    )

    upper = min(
        max_k,
        unique_count,
        len(features) - 1
    )

    if upper < 2:
        return {"status": "INSUFFICIENT_VARIATION"}

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    evaluations = []

    for k in range(2, upper + 1):
        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=10
        )

        labels = model.fit_predict(X)

        if len(np.unique(labels)) < 2:
            continue

        score = silhouette_score(
            X,
            labels,
            sample_size=min(1000, len(X)),
            random_state=RANDOM_STATE
        )

        evaluations.append({
            "k": k,
            "silhouette": float(score),
            "inertia": float(model.inertia_)
        })

    if not evaluations:
        return {"status": "NO_VALID_CLUSTERING"}

    best = max(
        evaluations,
        key=lambda row: row["silhouette"]
    )

    model = KMeans(
        n_clusters=best["k"],
        random_state=RANDOM_STATE,
        n_init=20
    )

    labels = model.fit_predict(X)

    clustered = df.copy()
    clustered["cluster_id"] = labels

    save_csv(
        out / "clustered_workloads.csv",
        clustered
    )

    save_csv(
        out / "clustering_evaluation.csv",
        pd.DataFrame(evaluations)
    )

    joblib.dump(
        {
            "scaler": scaler,
            "model": model,
            "features": FEATURE_NAMES
        },
        out / "clustering_model.joblib"
    )

    counts = (
        clustered["cluster_id"]
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        counts.index.astype(str),
        counts.values
    )
    ax.set_title("Workload Clusters")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("SQL executions")
    fig.tight_layout()
    fig.savefig(
        out / "cluster_distribution.png",
        dpi=150
    )
    plt.close(fig)

    return {
        "status": "COMPLETED",
        "clusters": int(best["k"]),
        "silhouette": float(best["silhouette"]),
        "cluster_sizes": counts.to_dict()
    }


# ============================================================
# 8.5 SUPERVISED CLASSIFICATION
# ============================================================

def run_classification(df, features, out):
    LOG.info("Training workload classifier")

    valid = (
        df["label"].notna()
        & df["label"].astype(str).ne("")
    )

    labeled = df.loc[valid]
    X = features.loc[valid]

    if len(labeled) < 30:
        return {
            "status": "SKIPPED",
            "reason": "Insufficient labeled samples"
        }

    y = labeled["label"].astype(str).to_numpy()
    groups = labeled["fingerprint"].to_numpy()

    if len(np.unique(y)) < 2:
        return {
            "status": "SKIPPED",
            "reason": "Only one workload class"
        }

    if len(np.unique(groups)) < 3:
        return {
            "status": "SKIPPED",
            "reason": "Insufficient fingerprint groups"
        }

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=RANDOM_STATE
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups)
    )

    if len(np.unique(y[train_idx])) < 2:
        return {
            "status": "SKIPPED",
            "reason": "Training split has one class"
        }

    if set(groups[train_idx]) & set(groups[test_idx]):
        raise RuntimeError(
            "SQL fingerprint leakage detected"
        )

    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(
        X.iloc[train_idx],
        y[train_idx]
    )

    predictions = model.predict(
        X.iloc[test_idx]
    )

    metrics = {
        "accuracy": float(
            accuracy_score(y[test_idx], predictions)
        ),
        "macro_f1": float(
            f1_score(
                y[test_idx],
                predictions,
                average="macro",
                zero_division=0
            )
        ),
        "weighted_f1": float(
            f1_score(
                y[test_idx],
                predictions,
                average="weighted",
                zero_division=0
            )
        )
    }

    save_json(
        out / "classification_report.json",
        classification_report(
            y[test_idx],
            predictions,
            zero_division=0,
            output_dict=True
        )
    )

    labels = sorted(
        set(y[test_idx]) | set(predictions)
    )

    cm = confusion_matrix(
        y[test_idx],
        predictions,
        labels=labels
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(cm)
    fig.colorbar(image, ax=ax)
    ax.set_xticks(
        range(len(labels)),
        labels,
        rotation=45,
        ha="right"
    )
    ax.set_yticks(
        range(len(labels)),
        labels
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Workload Classification")
    fig.tight_layout()
    fig.savefig(
        out / "classification_confusion_matrix.png",
        dpi=150
    )
    plt.close(fig)

    # Refit final artifact on all labeled data,
    # AFTER obtaining the held-out evaluation.
    model.fit(X, y)

    joblib.dump(
        {
            "model": model,
            "features": FEATURE_NAMES,
            "classes": sorted(np.unique(y)),
            "holdout_metrics": metrics
        },
        out / "workload_classifier.joblib"
    )

    return {
        "status": "COMPLETED",
        "labeled_rows": int(len(labeled)),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "fingerprint_leakage": False,
        "metrics": metrics
    }


# ============================================================
# 8.6 GROUPED CROSS-VALIDATION
# ============================================================

def run_grouped_cv(df, features, out, folds=5):
    LOG.info("Running grouped cross-validation")

    valid = (
        df["label"].notna()
        & df["label"].astype(str).ne("")
    )

    labeled = df.loc[valid]
    X = features.loc[valid]

    y = labeled["label"].astype(str).to_numpy()
    groups = labeled["fingerprint"].to_numpy()

    n_splits = min(
        folds,
        len(np.unique(groups))
    )

    if n_splits < 2 or len(np.unique(y)) < 2:
        return {"status": "SKIPPED"}

    splitter = GroupKFold(
        n_splits=n_splits
    )

    rows = []

    for fold, (tr, te) in enumerate(
        splitter.split(X, y, groups),
        start=1
    ):
        if len(np.unique(y[tr])) < 2:
            continue

        model = RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE + fold,
            n_jobs=-1
        )

        model.fit(X.iloc[tr], y[tr])

        pred = model.predict(X.iloc[te])

        rows.append({
            "fold": fold,
            "accuracy": accuracy_score(y[te], pred),
            "macro_f1": f1_score(
                y[te], pred,
                average="macro",
                zero_division=0
            ),
            "train_rows": len(tr),
            "test_rows": len(te)
        })

    if not rows:
        return {"status": "SKIPPED"}

    results = pd.DataFrame(rows)

    save_csv(
        out / "grouped_cross_validation.csv",
        results
    )

    return {
        "status": "COMPLETED",
        "folds": len(results),
        "mean_accuracy": float(
            results["accuracy"].mean()
        ),
        "mean_macro_f1": float(
            results["macro_f1"].mean()
        ),
        "std_macro_f1": float(
            results["macro_f1"].std(ddof=0)
        )
    }


# ============================================================
# 8.7 PHASE 7 MONITORING INTEGRATION
# ============================================================

def discover_monitoring_files(explicit_path=None):
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()

        if not path.is_file():
            raise FileNotFoundError(path)

        return [path]

    candidates = []

    for directory in RESULTS.iterdir():
        if not directory.is_dir():
            continue

        name = directory.name.lower()

        if (
            "phase7" not in name
            and "phase_7" not in name
            and "monitor" not in name
        ):
            continue

        candidates.extend(
            directory.rglob("*.csv")
        )

    return sorted(set(candidates))


def inspect_monitoring_file(
    path,
    timestamp_override=None
):
    try:
        df = safe_read_csv(path)
    except Exception as exc:
        return None, {
            "path": str(path),
            "status": "READ_ERROR",
            "reason": str(exc)
        }

    if df.empty:
        return None, {
            "path": str(path),
            "status": "EMPTY"
        }

    if timestamp_override:
        ts_col = find_column(
            df, [timestamp_override]
        )
    else:
        ts_col = find_column(
            df, TIMESTAMP_ALIASES
        )

    if ts_col is None:
        return None, {
            "path": str(path),
            "status": "NO_TIMESTAMP_COLUMN",
            "columns": list(df.columns)
        }

    # Numeric epochs are intentionally not auto-interpreted.
    # Their units (seconds/ms/us) must be confirmed first.
    timestamps = pd.to_datetime(
        df[ts_col],
        errors="coerce",
        utc=True
    )

    valid = timestamps.notna()

    sql_col = find_column(df, SQL_ALIASES)
    label_col = find_column(df, LABEL_ALIASES)
    latency_col = find_column(df, LATENCY_ALIASES)

    result = pd.DataFrame({
        "timestamp": timestamps
    })

    result["sql"] = (
        df[sql_col].fillna("").astype(str)
        if sql_col is not None
        else ""
    )

    result["label"] = (
        df[label_col].astype("string")
        if label_col is not None
        else pd.Series(
            pd.NA,
            index=df.index,
            dtype="string"
        )
    )

    result["latency_ms"] = (
        pd.to_numeric(
            df[latency_col],
            errors="coerce"
        )
        if latency_col is not None
        else np.nan
    )

    result = result.loc[valid].copy()
    result = result.sort_values("timestamp")

    audit = {
        "path": str(path),
        "status": (
            "USABLE" if len(result) > 0
            else "NO_VALID_TIMESTAMPS"
        ),
        "original_rows": int(len(df)),
        "timestamped_rows": int(len(result)),
        "timestamp_column": str(ts_col),
        "sql_column": (
            str(sql_col) if sql_col else None
        ),
        "label_column": (
            str(label_col) if label_col else None
        ),
        "latency_column": (
            str(latency_col) if latency_col else None
        ),
        "start": (
            str(result["timestamp"].min())
            if len(result) else None
        ),
        "end": (
            str(result["timestamp"].max())
            if len(result) else None
        )
    }

    return result, audit


def load_monitoring(
    explicit_path,
    timestamp_override,
    out
):
    LOG.info("Discovering Phase 7 monitoring logs")

    files = discover_monitoring_files(
        explicit_path
    )

    audits = []
    usable = []

    for path in files:
        df, audit = inspect_monitoring_file(
            path,
            timestamp_override
        )

        audits.append(audit)

        if (
            df is not None
            and not df.empty
        ):
            # Do not combine unrelated experiment files:
            # overlapping benchmark runs or duplicated
            # exports can create artificial demand.
            usable.append((path, df, audit))

    save_json(
        out / "monitoring_discovery.json",
        audits
    )

    if not usable:
        return None, {
            "status": "NO_USABLE_TEMPORAL_SOURCE",
            "files_inspected": len(files),
            "reason": (
                "No Phase 7 CSV with parseable timestamps "
                "was found. Inspect monitoring_discovery.json."
            )
        }

    # Choose a single source rather than blindly concatenating
    # multiple runs with potentially overlapping timestamps.
    usable.sort(
        key=lambda item: (
            len(item[1]),
            (
                item[1]["timestamp"].max()
                - item[1]["timestamp"].min()
            ).total_seconds()
        ),
        reverse=True
    )

    path, df, audit = usable[0]

    save_csv(
        out / "selected_temporal_events.csv",
        df
    )

    return df, {
        "status": "SOURCE_SELECTED",
        "selected_file": str(path),
        "valid_rows": len(df),
        "files_inspected": len(files),
        "timestamp_column": audit["timestamp_column"],
        "start": audit["start"],
        "end": audit["end"],
        "note": (
            "A timestamped source is not automatically "
            "a representative longitudinal workload."
        )
    }


# ============================================================
# TEMPORAL SOURCE QUALITY
# ============================================================

def temporal_quality(df):
    if df is None or df.empty:
        return {
            "valid": False,
            "reason": "No temporal events"
        }

    timestamps = df["timestamp"].dropna()

    if len(timestamps) < 40:
        return {
            "valid": False,
            "reason": "Fewer than 40 timestamped events"
        }

    start = timestamps.min()
    end = timestamps.max()

    span = end - start
    hours = span.total_seconds() / 3600

    return {
        "valid": True,
        "events": int(len(timestamps)),
        "start": str(start),
        "end": str(end),
        "duration_hours": hours,
        "duration_days": hours / 24,
        "unique_timestamps": int(
            timestamps.nunique()
        )
    }


def choose_frequency(duration_hours):
    if duration_hours >= 24 * 14:
        return "D"

    if duration_hours >= 48:
        return "h"

    if duration_hours >= 12:
        return "30min"

    return "5min"


# ============================================================
# 8.8 TEMPORAL DEMAND FORECASTING
# ============================================================

def make_lag_dataset(values, lag):
    X, y = [], []

    for index in range(lag, len(values)):
        history = values[index-lag:index]

        X.append(
            list(history) + [
                float(np.mean(history)),
                float(np.std(history)),
                float(np.min(history)),
                float(np.max(history))
            ]
        )

        y.append(values[index])

    return (
        np.asarray(X, dtype=float),
        np.asarray(y, dtype=float)
    )


def run_forecasting(
    events,
    out,
    min_duration_hours=24,
    forecast_steps=12
):
    LOG.info("Running chronological workload forecasting")

    quality = temporal_quality(events)

    if not quality["valid"]:
        return {
            "status": "INSUFFICIENT_TEMPORAL_EVIDENCE",
            "quality": quality
        }

    if quality["duration_hours"] < min_duration_hours:
        return {
            "status": "INSUFFICIENT_TEMPORAL_EVIDENCE",
            "reason": (
                "Observation duration is too short "
                "for the configured forecasting experiment"
            ),
            "quality": quality
        }

    frequency = choose_frequency(
        quality["duration_hours"]
    )

    series = (
        events
        .set_index("timestamp")
        .sort_index()
        .resample(frequency)
        .size()
        .astype(float)
    )

    save_csv(
        out / "temporal_demand_series.csv",
        series.rename(
            "query_count"
        ).reset_index()
    )

    if len(series) < 48:
        return {
            "status": "INSUFFICIENT_TEMPORAL_EVIDENCE",
            "reason": "Fewer than 48 temporal bins",
            "bins": len(series),
            "frequency": frequency
        }

    if series.nunique() < 2:
        return {
            "status": "INSUFFICIENT_VARIATION",
            "frequency": frequency
        }

    values = series.to_numpy()

    lag = min(
        12,
        max(4, len(values) // 10)
    )

    X, y = make_lag_dataset(
        values, lag
    )

    if len(X) < 30:
        return {
            "status": "INSUFFICIENT_TEMPORAL_EVIDENCE",
            "reason": "Insufficient lagged examples"
        }

    # Strict chronological split.
    split = int(len(X) * 0.8)

    X_train = X[:split]
    X_test = X[split:]

    y_train = y[:split]
    y_test = y[split:]

    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = np.maximum(
        0,
        model.predict(X_test)
    )

    # Persistence baseline:
    # next count = most recently observed count.
    persistence = X_test[:, lag - 1]

    # Historical mean baseline.
    mean_baseline = X_test[:, lag:lag+1].ravel()

    metrics = {}

    for name, pred in [
        ("random_forest", predictions),
        ("persistence", persistence),
        ("rolling_mean", mean_baseline)
    ]:
        metrics[name] = {
            "mae": float(
                mean_absolute_error(y_test, pred)
            ),
            "rmse": float(
                np.sqrt(
                    mean_squared_error(y_test, pred)
                )
            )
        }

    baseline_mae = metrics["persistence"]["mae"]
    model_mae = metrics["random_forest"]["mae"]

    metrics["improvement_vs_persistence_pct"] = (
        100 * (baseline_mae - model_mae) / baseline_mae
        if baseline_mae > 0
        else None
    )

    # Save chronological holdout predictions.
    test_positions = series.index[
        lag + split:
    ]

    save_csv(
        out / "forecast_holdout_predictions.csv",
        pd.DataFrame({
            "timestamp": test_positions,
            "actual": y_test,
            "random_forest": predictions,
            "persistence": persistence,
            "rolling_mean": mean_baseline
        })
    )

    # Final fit on the full historical sequence.
    model.fit(X, y)

    history = list(values)
    future_predictions = []

    for _ in range(forecast_steps):
        recent = np.asarray(
            history[-lag:],
            dtype=float
        )

        row = np.asarray(
            list(recent) + [
                recent.mean(),
                recent.std(),
                recent.min(),
                recent.max()
            ],
            dtype=float
        ).reshape(1, -1)

        prediction = max(
            0.0,
            float(model.predict(row)[0])
        )

        future_predictions.append(prediction)
        history.append(prediction)

    future_index = pd.date_range(
        start=series.index[-1],
        periods=forecast_steps + 1,
        freq=frequency
    )[1:]

    save_csv(
        out / "future_workload_forecast.csv",
        pd.DataFrame({
            "timestamp": future_index,
            "predicted_query_count": future_predictions
        })
    )

    joblib.dump(
        {
            "model": model,
            "lag": lag,
            "frequency": frequency,
            "metrics": metrics
        },
        out / "workload_forecaster.joblib"
    )

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        series.index,
        series.values,
        label="Observed demand"
    )

    ax.plot(
        future_index,
        future_predictions,
        linestyle="--",
        label="Forecast"
    )

    ax.set_title("Chronological Workload Forecast")
    ax.set_ylabel("Queries per time bin")
    ax.legend()

    fig.autofmt_xdate()
    fig.tight_layout()

    fig.savefig(
        out / "workload_forecast.png",
        dpi=150
    )
    plt.close(fig)

    save_json(
        out / "forecast_metrics.json",
        metrics
    )

    return {
        "status": "COMPLETED",
        "frequency": frequency,
        "time_bins": len(series),
        "lag": lag,
        "train_examples": len(X_train),
        "test_examples": len(X_test),
        "metrics": metrics,
        "quality": quality,
        "limitation": (
            "One chronological holdout is preliminary. "
            "Publication evaluation should add rolling-origin "
            "validation and multiple workload traces."
        )
    }


# ============================================================
# 8.9 TEMPORAL DRIFT DETECTION
# ============================================================

def psi_score(reference, current, bins=10):
    reference = np.asarray(
        reference, dtype=float
    )

    current = np.asarray(
        current, dtype=float
    )

    reference = reference[
        np.isfinite(reference)
    ]

    current = current[
        np.isfinite(current)
    ]

    if len(reference) < 20 or len(current) < 20:
        return float("nan")

    edges = np.unique(
        np.quantile(
            reference,
            np.linspace(0, 1, bins + 1)
        )
    )

    if len(edges) < 3:
        return float("nan")

    edges[0] = -np.inf
    edges[-1] = np.inf

    a = np.histogram(
        reference, bins=edges
    )[0].astype(float)

    b = np.histogram(
        current, bins=edges
    )[0].astype(float)

    eps = 1e-6

    a = (a + eps) / (a.sum() + eps * len(a))
    b = (b + eps) / (b.sum() + eps * len(b))

    return float(
        np.sum(
            (b - a) * np.log(b / a)
        )
    )


def js_divergence(reference, current):
    a = pd.Series(reference).fillna("unknown")
    b = pd.Series(current).fillna("unknown")

    categories = sorted(
        set(a.astype(str))
        | set(b.astype(str))
    )

    p = (
        a.astype(str)
        .value_counts(normalize=True)
        .reindex(categories, fill_value=0)
        .to_numpy(dtype=float)
    )

    q = (
        b.astype(str)
        .value_counts(normalize=True)
        .reindex(categories, fill_value=0)
        .to_numpy(dtype=float)
    )

    m = 0.5 * (p + q)

    def kl(x, y):
        mask = x > 0
        return np.sum(
            x[mask] * np.log2(
                x[mask] / y[mask]
            )
        )

    return float(
        0.5 * kl(p, m)
        + 0.5 * kl(q, m)
    )


def run_temporal_drift(
    events,
    out,
    min_duration_hours=24
):
    LOG.info("Running chronological drift detection")

    quality = temporal_quality(events)

    if not quality["valid"]:
        return {
            "status": "INSUFFICIENT_TEMPORAL_EVIDENCE",
            "quality": quality
        }

    if quality["duration_hours"] < min_duration_hours:
        return {
            "status": "INSUFFICIENT_TEMPORAL_EVIDENCE",
            "reason": "Insufficient observation duration",
            "quality": quality
        }

    events = events.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    midpoint = len(events) // 2

    reference = events.iloc[:midpoint].copy()
    current = events.iloc[midpoint:].copy()

    if (
        len(reference) < 20
        or len(current) < 20
    ):
        return {
            "status": "INSUFFICIENT_TEMPORAL_EVIDENCE"
        }

    # Check actual chronological separation.
    if (
        reference["timestamp"].max()
        >= current["timestamp"].min()
    ):
        return {
            "status": "INSUFFICIENT_TEMPORAL_EVIDENCE",
            "reason": (
                "Reference and current partitions "
                "do not have strictly separated timestamps"
            )
        }

    # Avoid claiming temporal SQL-feature drift
    # when the monitoring file has no SQL statements.
    has_sql = (
        events["sql"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
        >= 40
    )

    rows = []

    if has_sql:
        ref_features = pd.DataFrame(
            [
                extract_features(sql)
                for sql in reference["sql"]
            ]
        )

        cur_features = pd.DataFrame(
            [
                extract_features(sql)
                for sql in current["sql"]
            ]
        )

        for feature in FEATURE_NAMES:
            score = psi_score(
                ref_features[feature],
                cur_features[feature]
            )

            rows.append({
                "feature": feature,
                "metric": "PSI",
                "value": score,
                "heuristic_flag": (
                    bool(score >= 0.2)
                    if math.isfinite(score)
                    else None
                )
            })

    if events["label"].notna().sum() >= 40:
        score = js_divergence(
            reference["label"],
            current["label"]
        )

        rows.append({
            "feature": "workload_label",
            "metric": "Jensen_Shannon_divergence",
            "value": score,
            "heuristic_flag": None
        })

    if events["latency_ms"].notna().sum() >= 40:
        ref_latency = (
            reference["latency_ms"]
            .dropna()
        )

        cur_latency = (
            current["latency_ms"]
            .dropna()
        )

        score = psi_score(
            ref_latency,
            cur_latency
        )

        rows.append({
            "feature": "latency_ms",
            "metric": "PSI",
            "value": score,
            "heuristic_flag": (
                bool(score >= 0.2)
                if math.isfinite(score)
                else None
            )
        })

    if not rows:
        return {
            "status": "INSUFFICIENT_FEATURE_EVIDENCE",
            "reason": (
                "Monitoring contains timestamps but no "
                "usable SQL, workload labels, or latency"
            )
        }

    drift_df = pd.DataFrame(rows)

    save_csv(
        out / "temporal_drift_metrics.csv",
        drift_df
    )

    flags = drift_df[
        drift_df["heuristic_flag"] == True
    ]

    result = {
        "status": "COMPLETED",
        "reference_rows": len(reference),
        "current_rows": len(current),
        "reference_start": str(
            reference["timestamp"].min()
        ),
        "reference_end": str(
            reference["timestamp"].max()
        ),
        "current_start": str(
            current["timestamp"].min()
        ),
        "current_end": str(
            current["timestamp"].max()
        ),
        "features_evaluated": len(rows),
        "heuristic_flags": len(flags),
        "quality": quality,
        "limitations": [
            "PSI >= 0.2 is a heuristic, not a significance test.",
            "Distribution differences do not prove concept drift.",
            "False-alarm and detection-delay experiments remain necessary."
        ]
    }

    save_json(
        out / "temporal_drift_summary.json",
        result
    )

    return result


# ============================================================
# 8.10 DATASET SUMMARY
# ============================================================

def summarize_dataset(df, out):
    counts = (
        df["fingerprint"]
        .value_counts()
    )

    fingerprint_summary = (
        df.groupby("fingerprint")
        .agg(
            executions=("sql", "size"),
            example_sql=("sql", "first"),
            unique_exact_sql=("exact_hash", "nunique")
        )
        .reset_index()
        .sort_values(
            "executions",
            ascending=False
        )
    )

    save_csv(
        out / "fingerprint_summary.csv",
        fingerprint_summary
    )

    return {
        "rows": int(len(df)),
        "unique_exact_sql": int(
            df["exact_hash"].nunique()
        ),
        "unique_fingerprints": int(
            df["fingerprint"].nunique()
        ),
        "labels": (
            df["label"]
            .fillna("unlabeled")
            .value_counts()
            .to_dict()
        ),
        "largest_fingerprint_count": (
            int(counts.max())
            if len(counts) else 0
        )
    }


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "all",
            "classification",
            "clustering",
            "forecasting",
            "drift"
        ],
        default="all"
    )

    parser.add_argument(
        "--input",
        default=str(PHASE6_FILE)
    )

    parser.add_argument(
        "--temporal-csv",
        default=None
    )

    parser.add_argument(
        "--timestamp-column",
        default=None
    )

    parser.add_argument(
        "--clusters",
        type=int,
        default=6
    )

    parser.add_argument(
        "--forecast-steps",
        type=int,
        default=12
    )

    parser.add_argument(
        "--min-duration-hours",
        type=float,
        default=24
    )

    args = parser.parse_args()

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S_%f")

    out = OUTPUT_ROOT / timestamp
    setup_logging(out)

    LOG.info("=" * 72)
    LOG.info("PHASE 8: FULL AI WORKLOAD LEARNING")
    LOG.info("=" * 72)

    summary = {
        "phase": 8,
        "status": "STARTED",
        "output": str(out),
        "mode": args.mode,
        "components": {},
        "errors": [],
        "database_modifications": "NONE",
        "research_warning": (
            "Completion of software execution does not "
            "establish clinical validity or publication readiness."
        )
    }

    try:
        needs_sql = args.mode in (
            "all",
            "classification",
            "clustering"
        )

        if needs_sql:
            df = load_phase6(
                Path(args.input)
            )

            df = add_fingerprints(df)
            features = build_features(df)

            save_csv(
                out / "engineered_workloads.csv",
                pd.concat(
                    [
                        df,
                        features.add_prefix("feature_")
                    ],
                    axis=1
                )
            )

            summary["components"]["dataset"] = (
                summarize_dataset(df, out)
            )

            if args.mode in ("all", "clustering"):
                summary["components"]["clustering"] = (
                    run_clustering(
                        df,
                        features,
                        out,
                        max_k=args.clusters
                    )
                )

            if args.mode in ("all", "classification"):
                summary["components"]["classification"] = (
                    run_classification(
                        df,
                        features,
                        out
                    )
                )

                summary["components"]["cross_validation"] = (
                    run_grouped_cv(
                        df,
                        features,
                        out
                    )
                )

        needs_temporal = args.mode in (
            "all",
            "forecasting",
            "drift"
        )

        if needs_temporal:
            events, source_result = load_monitoring(
                args.temporal_csv,
                args.timestamp_column,
                out
            )

            summary["components"]["monitoring"] = (
                source_result
            )

            if events is None:
                missing = {
                    "status": "INSUFFICIENT_TEMPORAL_EVIDENCE",
                    "reason": (
                        "No usable monitoring events found. "
                        "See monitoring_discovery.json."
                    )
                }

                if args.mode in ("all", "forecasting"):
                    summary["components"]["forecasting"] = (
                        missing
                    )

                if args.mode in ("all", "drift"):
                    summary["components"]["temporal_drift"] = (
                        missing
                    )

            else:
                if args.mode in ("all", "forecasting"):
                    summary["components"]["forecasting"] = (
                        run_forecasting(
                            events,
                            out,
                            min_duration_hours=(
                                args.min_duration_hours
                            ),
                            forecast_steps=(
                                args.forecast_steps
                            )
                        )
                    )

                if args.mode in ("all", "drift"):
                    summary["components"]["temporal_drift"] = (
                        run_temporal_drift(
                            events,
                            out,
                            min_duration_hours=(
                                args.min_duration_hours
                            )
                        )
                    )

        statuses = [
            value.get("status")
            for value in summary["components"].values()
            if isinstance(value, dict)
            and "status" in value
        ]

        if any(
            status != "COMPLETED"
            and status != "SOURCE_SELECTED"
            for status in statuses
        ):
            summary["status"] = (
                "COMPLETED_WITH_LIMITATIONS"
            )
        else:
            summary["status"] = "COMPLETED"

    except Exception as exc:
        LOG.exception("Phase 8 failed")

        summary["status"] = "FAILED"

        summary["errors"].append({
            "type": type(exc).__name__,
            "message": str(exc)
        })

    finally:
        save_json(
            out / "summary.json",
            summary
        )

        print()
        print("=" * 72)
        print("PHASE 8 FINAL SUMMARY")
        print("=" * 72)

        print("Status:", summary["status"])
        print("Output:", out)

        for name, result in (
            summary["components"].items()
        ):
            print(
                name,
                "=>",
                result.get(
                    "status", "COMPLETED"
                )
            )

        print(
            "Database modifications: NONE"
        )
        print(
            "Errors:",
            len(summary["errors"])
        )
        print("=" * 72)

    if summary["status"] == "FAILED":
        sys.exit(1)


if __name__ == "__main__":
    main()
