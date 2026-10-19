
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn

from catboost import CatBoostClassifier, CatBoostRegressor
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    LogisticRegression,
    Ridge,
    HuberRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVR

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CSV = (
    ROOT
    / "results"
    / "phase7_monitoring"
    / "phase7_query_monitoring.csv"
)

BASE_OUTPUT = ROOT / "results" / "phase8_v3"

SEED = 42

# Features extracted from SQL text.
# These do not require executing the query.
STRUCTURAL = [
    "sql_length",
    "token_count",
    "join_count",
    "where_count",
    "group_count",
    "order_count",
    "having_count",
    "limit_count",
    "distinct_count",
    "aggregate_count",
    "subquery_count",
    "predicate_count",
    "table_count",
    "has_cte",
    "has_union",
    "has_window",
    "has_exists",
    "has_between",
    "has_like",
    "has_in",
    "has_or",
    "has_offset",
    "has_case",
    "has_cast",
    "has_outer_join",
    "has_left_join",
    "has_right_join",
    "has_cross_join",
    "has_count",
    "has_sum",
    "has_avg",
    "has_max",
    "has_min",
    "has_date_filter",
    "has_null_check",
]

# These may be available before execution from
# a plain EXPLAIN, depending on the monitoring schema.
PLANNER_CANDIDATES = [
    "estimated_cost",
    "startup_cost",
    "estimated_rows",
    "plan_node_count",
]

# Never use these as inputs for pre-execution
# latency prediction.
FORBIDDEN_PREEXECUTION = {
    "execution_time_ms",
    "planning_time_ms",
    "actual_rows",
    "actual_total_time",
    "actual_startup_time",
    "shared_hit_blocks",
    "shared_read_blocks",
    "shared_dirtied_blocks",
    "shared_written_blocks",
    "temp_read_blocks",
    "temp_written_blocks",
    "actual_loops",
    "workload_type",
    "query_family",
    "query_name",
    "expected_use",
    "fingerprint",
    "fingerprint_hash",
    "session_id",
    "timestamp",
}

def now():
    return datetime.now(timezone.utc).isoformat()

def json_safe(obj):
    if isinstance(obj, dict):
        return {
            str(k): json_safe(v)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        return x if np.isfinite(x) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            json_safe(value),
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

def read_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8")
    )

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(block)
    return digest.hexdigest()

def normalized_sql(value):
    text = str(value or "")
    text = re.sub(
        r"/\*.*?\*/",
        " ",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(r"--[^\n]*", " ", text)
    text = re.sub(
        r"'(?:''|[^'])*'",
        " ? ",
        text,
    )
    text = re.sub(
        r"\b\d+(?:\.\d+)?\b",
        " ? ",
        text,
    )
    text = re.sub(
        r"\s+",
        " ",
        text,
    )
    return text.strip().lower()

def make_fingerprint(sql):
    return hashlib.sha256(
        normalized_sql(sql).encode("utf-8")
    ).hexdigest()

def find_column(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None

def load_data(paths):
    frames = []
    sources = []

    for item in paths:
        path = Path(item).expanduser().resolve()

        if path.is_dir():
            files = sorted(path.glob("*.csv"))
        else:
            files = [path]

        for file in files:
            if not file.exists():
                raise FileNotFoundError(file)

            frame = pd.read_csv(
                file,
                low_memory=False,
            )

            sql_col = find_column(
                frame,
                ["sql", "query_sql", "sql_text", "query_text"],
            )

            latency_col = find_column(
                frame,
                [
                    "execution_time_ms",
                    "latency_ms",
                    "duration_ms",
                    "execution_ms",
                ],
            )

            if sql_col is None or latency_col is None:
                if path.is_dir():
                    continue
                raise ValueError(
                    f"{file}: missing SQL or latency column. "
                    f"Columns: {list(frame.columns)}"
                )

            frame["sql"] = (
                frame[sql_col].fillna("").astype(str)
            )

            frame["execution_time_ms"] = (
                pd.to_numeric(
                    frame[latency_col],
                    errors="coerce",
                )
            )

            label_col = find_column(
                frame,
                [
                    "workload_type",
                    "workload_class",
                    "user_role",
                    "query_category",
                ],
            )

            if label_col:
                frame["target_class"] = (
                    frame[label_col]
                    .fillna("unknown")
                    .astype(str)
                    .str.strip()
                    .str.lower()
                )
            else:
                frame["target_class"] = "unknown"

            fingerprint_col = find_column(
                frame,
                [
                    "fingerprint_hash",
                    "fingerprint",
                    "query_fingerprint",
                ],
            )

            if fingerprint_col:
                frame["fingerprint"] = (
                    frame[fingerprint_col]
                    .fillna("")
                    .astype(str)
                )
            else:
                frame["fingerprint"] = ""

            missing = (
                frame["fingerprint"].str.strip() == ""
            )

            frame.loc[missing, "fingerprint"] = (
                frame.loc[missing, "sql"]
                .map(make_fingerprint)
            )

            timestamp_col = find_column(
                frame,
                [
                    "monitored_at",
                    "timestamp",
                    "executed_at",
                    "recorded_at",
                ],
            )

            if timestamp_col:
                frame["timestamp"] = pd.to_datetime(
                    frame[timestamp_col],
                    utc=True,
                    errors="coerce",
                )
            else:
                frame["timestamp"] = pd.NaT

            if "session_id" not in frame.columns:
                frame["session_id"] = file.stem

            frame["session_id"] = (
                frame["session_id"]
                .fillna(file.stem)
                .astype(str)
            )

            if "query_family" not in frame.columns:
                frame["query_family"] = "unknown"

            frame["source_file"] = str(file)
            frame["source_row"] = np.arange(len(frame))

            frames.append(frame)

            sources.append({
                "file": str(file),
                "sha256": sha256_file(file),
                "rows": int(len(frame)),
            })

    if not frames:
        raise RuntimeError(
            "No valid monitoring CSV files found."
        )

    data = pd.concat(
        frames,
        ignore_index=True,
    )

    data = data[
        data["sql"].str.strip().ne("")
    ].copy()

    data["execution_time_ms"] = (
        pd.to_numeric(
            data["execution_time_ms"],
            errors="coerce",
        )
    )

    data = data[
        data["execution_time_ms"].notna()
        & np.isfinite(data["execution_time_ms"])
        & (data["execution_time_ms"] >= 0)
    ].copy()

    if len(data) < 30:
        raise RuntimeError(
            "Insufficient valid monitoring records."
        )

    return data.reset_index(drop=True), sources

def build_features(df, use_planner=True):
    out = df.copy()

    out["normalized_sql"] = (
        out["sql"].map(normalized_sql)
    )

    q = out["normalized_sql"]

    out["sql_length"] = q.str.len()
    out["token_count"] = (
        q.str.split().str.len()
    )

    count_patterns = {
        "join_count": r"\bjoin\b",
        "where_count": r"\bwhere\b",
        "group_count": r"\bgroup\s+by\b",
        "order_count": r"\border\s+by\b",
        "having_count": r"\bhaving\b",
        "limit_count": r"\blimit\b",
        "distinct_count": r"\bdistinct\b",
        "aggregate_count": (
            r"\b(?:count|sum|avg|min|max)\s*\("
        ),
        "subquery_count": r"\(\s*select\b",
        "predicate_count": (
            r"(?<![<>!])=(?!=)|>=|<=|<>|!=|"
            r"\blike\b|\bin\b|\bbetween\b"
        ),
    }

    for name, pattern in count_patterns.items():
        out[name] = q.str.count(pattern)

    out["table_count"] = (
        q.str.count(r"\bfrom\b")
        + q.str.count(r"\bjoin\b")
    )

    flag_patterns = {
        "has_cte": r"^\s*with\b",
        "has_union": r"\bunion\b",
        "has_window": r"\bover\s*\(",
        "has_exists": r"\bexists\b",
        "has_between": r"\bbetween\b",
        "has_like": r"\blike\b",
        "has_in": r"\bin\s*\(",
        "has_or": r"\bor\b",
        "has_offset": r"\boffset\b",
        "has_case": r"\bcase\b",
        "has_cast": r"\bcast\s*\(",
        "has_outer_join": r"\bouter\s+join\b",
        "has_left_join": r"\bleft\s+join\b",
        "has_right_join": r"\bright\s+join\b",
        "has_cross_join": r"\bcross\s+join\b",
        "has_count": r"\bcount\s*\(",
        "has_sum": r"\bsum\s*\(",
        "has_avg": r"\bavg\s*\(",
        "has_max": r"\bmax\s*\(",
        "has_min": r"\bmin\s*\(",
        "has_date_filter": (
            r"\bdate\b|\btimestamp\b|"
            r"\binterval\b|\bextract\s*\("
        ),
        "has_null_check": r"\bis\s+(?:not\s+)?null\b",
    }

    for name, pattern in flag_patterns.items():
        out[name] = (
            q.str.contains(
                pattern,
                regex=True,
            ).astype(int)
        )

    # SQL text is used only as an input feature.
    # It is never replaced with the target label.
    out["model_text"] = out["normalized_sql"]

    numeric = list(STRUCTURAL)
    planner_available = []

    if use_planner:
        for col in PLANNER_CANDIDATES:
            if col in out.columns:
                converted = pd.to_numeric(
                    out[col],
                    errors="coerce",
                )

                if converted.notna().any():
                    out[col] = converted
                    planner_available.append(col)
                    numeric.append(col)

    if set(numeric) & FORBIDDEN_PREEXECUTION:
        raise RuntimeError(
            "Feature leakage detected."
        )

    for col in numeric:
        out[col] = pd.to_numeric(
            out[col],
            errors="coerce",
        )

    return out, numeric, planner_available

def data_audit(df, sources, planner):
    ambiguous = (
        df.groupby("fingerprint")["target_class"]
        .nunique()
    )

    timestamps = df["timestamp"].dropna()

    duration = (
        (
            timestamps.max() - timestamps.min()
        ).total_seconds()
        if len(timestamps)
        else None
    )

    return {
        "records": len(df),
        "fingerprints": df["fingerprint"].nunique(),
        "classes": df["target_class"].value_counts().to_dict(),
        "sessions": df["session_id"].nunique(),
        "valid_timestamps": int(
            df["timestamp"].notna().sum()
        ),
        "duration_seconds": duration,
        "ambiguous_fingerprints": int(
            (ambiguous > 1).sum()
        ),
        "planner_features": planner,
        "planner_feature_caution": (
            "Only genuine pre-execution EXPLAIN "
            "estimates may be used at inference time."
        ),
        "label_warning": (
            "User-role labels can be ambiguous "
            "for identical SQL fingerprints."
        ),
        "sources": sources,
    }

def preprocessor(
    numeric,
    representation="word",
):
    transforms = [
        (
            "numeric",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value=0,
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]),
            numeric,
        )
    ]

    if representation == "word":
        transforms.append(
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    max_features=3000,
                    sublinear_tf=True,
                    token_pattern=r"(?u)\b\w+\b",
                ),
                "model_text",
            )
        )

    elif representation == "char":
        transforms.append(
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(3, 5),
                    max_features=4000,
                    sublinear_tf=True,
                    min_df=1,
                ),
                "model_text",
            )
        )

    return ColumnTransformer(
        transforms,
        sparse_threshold=0.3,
    )

def classification_candidates(numeric):
    models = {}

    def add(name, estimator, representation):
        models[name] = Pipeline([
            (
                "features",
                preprocessor(
                    numeric,
                    representation,
                ),
            ),
            ("model", estimator),
        ])

    add(
        "majority_baseline",
        DummyClassifier(strategy="most_frequent"),
        "numeric",
    )

    add(
        "logistic_word_tfidf",
        LogisticRegression(
            max_iter=2500,
            class_weight="balanced",
            random_state=SEED,
        ),
        "word",
    )

    add(
        "logistic_char_tfidf",
        LogisticRegression(
            max_iter=2500,
            class_weight="balanced",
            random_state=SEED,
        ),
        "char",
    )

    add(
        "linear_svm_word",
        LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=SEED,
        ),
        "word",
    )

    add(
        "linear_svm_char",
        LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=SEED,
        ),
        "char",
    )

    add(
        "random_forest_numeric",
        RandomForestClassifier(
            n_estimators=220,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        ),
        "numeric",
    )

    add(
        "random_forest_word",
        RandomForestClassifier(
            n_estimators=220,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        ),
        "word",
    )

    add(
        "extra_trees_numeric",
        ExtraTreesClassifier(
            n_estimators=220,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
        "numeric",
    )

    add(
        "hist_gradient_boosting",
        HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=SEED,
        ),
        "numeric",
    )

    add(
        "catboost_numeric",
        CatBoostClassifier(
            iterations=350,
            depth=4,
            learning_rate=0.04,
            loss_function="MultiClass",
            verbose=False,
            random_seed=SEED,
            thread_count=4,
            allow_writing_files=False,
        ),
        "numeric",
    )

    return models

def regression_candidates(numeric):
    models = {}

    def add(name, estimator, representation):
        models[name] = Pipeline([
            (
                "features",
                preprocessor(
                    numeric,
                    representation,
                ),
            ),
            ("model", estimator),
        ])

    add(
        "median_baseline",
        DummyRegressor(strategy="median"),
        "numeric",
    )

    add(
        "ridge_word",
        Ridge(alpha=50.0),
        "word",
    )

    add(
        "ridge_char",
        Ridge(alpha=100.0),
        "char",
    )

    add(
        "huber_numeric",
        HuberRegressor(
            epsilon=1.5,
            alpha=1.0,
            max_iter=500,
        ),
        "numeric",
    )

    add(
        "random_forest_numeric",
        RandomForestRegressor(
            n_estimators=220,
            min_samples_leaf=2,
            max_features=0.9,
            random_state=SEED,
            n_jobs=-1,
        ),
        "numeric",
    )

    add(
        "random_forest_word",
        RandomForestRegressor(
            n_estimators=220,
            min_samples_leaf=2,
            random_state=SEED,
            n_jobs=-1,
        ),
        "word",
    )

    add(
        "extra_trees_numeric",
        ExtraTreesRegressor(
            n_estimators=220,
            min_samples_leaf=2,
            random_state=SEED,
            n_jobs=-1,
        ),
        "numeric",
    )

    add(
        "hist_gradient_boosting",
        HistGradientBoostingRegressor(
            max_iter=220,
            max_leaf_nodes=15,
            learning_rate=0.04,
            l2_regularization=2.0,
            random_state=SEED,
        ),
        "numeric",
    )

    add(
        "catboost_numeric",
        CatBoostRegressor(
            iterations=450,
            depth=4,
            learning_rate=0.035,
            loss_function="RMSE",
            verbose=False,
            random_seed=SEED,
            thread_count=4,
            allow_writing_files=False,
        ),
        "numeric",
    )

    add(
        "svr_rbf_numeric",
        SVR(
            kernel="rbf",
            C=5.0,
            epsilon=0.15,
            gamma="scale",
        ),
        "numeric",
    )

    return models

def grouped_holdout(df, test_size=0.25):
    groups = df["fingerprint"].astype(str)

    if groups.nunique() < 8:
        raise RuntimeError(
            "At least eight unique fingerprints "
            "are required for grouped evaluation."
        )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=SEED + 300,
    )

    tr, te = next(
        splitter.split(
            df,
            groups=groups,
        )
    )

    assert set(groups.iloc[tr]).isdisjoint(
        set(groups.iloc[te])
    )

    return tr, te

def grouped_cv(df, folds=4):
    groups = df["fingerprint"].astype(str)

    splitter = GroupShuffleSplit(
        n_splits=folds,
        test_size=0.25,
        random_state=SEED + 20,
    )

    yield from splitter.split(
        df,
        groups=groups,
    )

def safe_log_target(values):
    arr = np.asarray(values, dtype=float)
    return np.log1p(np.maximum(0, arr))

def invert_log_target(values, ceiling_ms):
    # Avoid the extremely large back-transformed
    # predictions seen in the earlier Ridge model.
    values = np.asarray(values, dtype=float)

    upper = np.log1p(ceiling_ms)

    return np.expm1(
        np.clip(values, 0, upper)
    )

def regression_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return {
        "mae_ms": float(
            mean_absolute_error(actual, predicted)
        ),
        "median_ae_ms": float(
            median_absolute_error(actual, predicted)
        ),
        "rmse_ms": float(
            np.sqrt(
                mean_squared_error(actual, predicted)
            )
        ),
        "r2": float(
            r2_score(actual, predicted)
        ) if len(actual) > 1 else None,
        "wape_pct": float(
            100 * np.abs(actual - predicted).sum()
            / max(np.abs(actual).sum(), 1e-9)
        ),
    }

def classification_metrics(actual, predicted):
    return {
        "accuracy": float(
            accuracy_score(actual, predicted)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(actual, predicted)
        ),
        "macro_f1": float(
            f1_score(
                actual,
                predicted,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                actual,
                predicted,
                average="weighted",
                zero_division=0,
            )
        ),
    }

def evaluate_classification(
    df,
    numeric,
    output,
    folds,
):
    data = df[
        df["target_class"].ne("unknown")
    ].copy().reset_index(drop=True)

    if data["target_class"].nunique() < 2:
        return {
            "status": "INSUFFICIENT_CLASSES",
        }, None

    outer_train_idx, outer_test_idx = (
        grouped_holdout(data)
    )

    train = data.iloc[
        outer_train_idx
    ].reset_index(drop=True)

    test = data.iloc[
        outer_test_idx
    ].reset_index(drop=True)

    models = classification_candidates(numeric)

    cv_rows = []
    errors = []

    print(
        "\nCLASSIFICATION — GROUPED MODEL COMPARISON"
    )

    for name, definition in models.items():
        fold_scores = []

        for fold, (tr, va) in enumerate(
            grouped_cv(train, folds),
            start=1,
        ):
            model = clone(definition)

            try:
                model.fit(
                    train.iloc[tr],
                    train.iloc[tr]["target_class"],
                )

                pred = model.predict(
                    train.iloc[va]
                )

                metrics = classification_metrics(
                    train.iloc[va]["target_class"],
                    pred,
                )

                row = {
                    "model": name,
                    "fold": fold,
                    **metrics,
                }

                cv_rows.append(row)
                fold_scores.append(metrics["macro_f1"])

            except Exception as exc:
                errors.append({
                    "task": "classification",
                    "model": name,
                    "fold": fold,
                    "error": str(exc)[:500],
                })

        if fold_scores:
            print(
                f"{name:28s} "
                f"Macro F1={np.mean(fold_scores):.4f}"
            )

    cv_df = pd.DataFrame(cv_rows)

    cv_df.to_csv(
        output / "classification_cv.csv",
        index=False,
    )

    if cv_df.empty:
        return {
            "status": "ALL_MODELS_FAILED",
            "errors": errors,
        }, None

    leaderboard = (
        cv_df.groupby("model")
        .agg(
            macro_f1=("macro_f1", "mean"),
            accuracy=("accuracy", "mean"),
            balanced_accuracy=(
                "balanced_accuracy", "mean"
            ),
            std_macro_f1=("macro_f1", "std"),
            successful_folds=("fold", "count"),
        )
        .reset_index()
    )

    leaderboard = leaderboard[
        leaderboard["successful_folds"] == folds
    ].copy()

    leaderboard = leaderboard.sort_values(
        ["macro_f1", "balanced_accuracy"],
        ascending=False,
    )

    leaderboard.to_csv(
        output / "classification_leaderboard.csv",
        index=False,
    )

    if leaderboard.empty:
        return {
            "status": "NO_COMPLETE_CV_MODEL",
            "errors": errors,
        }, None

    winner = str(
        leaderboard.iloc[0]["model"]
    )

    # Final outer test is evaluated only once,
    # after model selection on the training set.
    final_model = clone(models[winner])

    final_model.fit(
        train,
        train["target_class"],
    )

    predictions = final_model.predict(test)

    final_metrics = classification_metrics(
        test["target_class"],
        predictions,
    )

    baseline = clone(
        models["majority_baseline"]
    )

    baseline.fit(
        train,
        train["target_class"],
    )

    baseline_predictions = baseline.predict(test)

    baseline_metrics = classification_metrics(
        test["target_class"],
        baseline_predictions,
    )

    classes = sorted(
        set(train["target_class"])
        | set(test["target_class"])
    )

    cm = confusion_matrix(
        test["target_class"],
        predictions,
        labels=classes,
    )

    fig, ax = plt.subplots(figsize=(8, 7))

    image = ax.imshow(
        cm,
        cmap="Blues",
    )

    fig.colorbar(image, ax=ax)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))

    ax.set_xticklabels(
        classes,
        rotation=35,
    )

    ax.set_yticklabels(classes)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax.set_title(
        "Unseen SQL Fingerprints — Final Holdout"
    )

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
            )

    fig.tight_layout()

    fig.savefig(
        output / "classification_confusion.png",
        dpi=180,
    )

    plt.close(fig)

    detailed = classification_report(
        test["target_class"],
        predictions,
        labels=classes,
        output_dict=True,
        zero_division=0,
    )

    prediction_df = pd.DataFrame({
        "fingerprint": test["fingerprint"].to_numpy(),
        "actual": test["target_class"].to_numpy(),
        "predicted": predictions,
    })

    prediction_df.to_csv(
        output / "classification_holdout_predictions.csv",
        index=False,
    )

    report = {
        "status": "EVALUATED",
        "winner": winner,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_fingerprints": (
            train["fingerprint"].nunique()
        ),
        "test_fingerprints": (
            test["fingerprint"].nunique()
        ),
        "group_overlap": False,
        "cv_winner_macro_f1": float(
            leaderboard.iloc[0]["macro_f1"]
        ),
        "outer_test_metrics": final_metrics,
        "outer_test_majority_baseline": baseline_metrics,
        "outer_test_macro_f1_gain": (
            final_metrics["macro_f1"]
            - baseline_metrics["macro_f1"]
        ),
        "per_class_report": detailed,
        "selection_rule": (
            "Inner grouped-CV macro F1, "
            "balanced accuracy as tiebreaker"
        ),
        "limitations": [
            "Only one independent outer holdout.",
            "Small number of distinct SQL fingerprints.",
            "User-role labels may not be identifiable from SQL.",
            "No calibrated probability claims.",
        ],
        "errors": errors,
    }

    write_json(
        output / "classification_report.json",
        report,
    )

    return report, final_model

def fit_predict_regression(
    model,
    train,
    test,
    ceiling_ms,
):
    model.fit(
        train,
        safe_log_target(
            train["execution_time_ms"]
        ),
    )

    predicted_log = model.predict(test)

    return invert_log_target(
        predicted_log,
        ceiling_ms,
    )

def regression_cv(
    train,
    numeric,
    output,
    folds,
):
    models = regression_candidates(numeric)

    cv_rows = []
    errors = []

    print(
        "\nREGRESSION — UNSEEN QUERY MODEL COMPARISON"
    )

    for name, definition in models.items():
        fold_maes = []

        for fold, (tr, va) in enumerate(
            grouped_cv(train, folds),
            start=1,
        ):
            tr_df = train.iloc[tr]
            va_df = train.iloc[va]

            # The ceiling comes from the training
            # fold only, never from validation.
            ceiling = max(
                1000.0,
                float(
                    tr_df["execution_time_ms"].max()
                ) * 5,
            )

            model = clone(definition)

            try:
                predictions = fit_predict_regression(
                    model,
                    tr_df,
                    va_df,
                    ceiling,
                )

                metrics = regression_metrics(
                    va_df["execution_time_ms"],
                    predictions,
                )

                cv_rows.append({
                    "model": name,
                    "fold": fold,
                    **metrics,
                })

                fold_maes.append(
                    metrics["mae_ms"]
                )

            except Exception as exc:
                errors.append({
                    "task": "regression",
                    "model": name,
                    "fold": fold,
                    "error": str(exc)[:500],
                })

        if fold_maes:
            print(
                f"{name:28s} "
                f"MAE={np.mean(fold_maes):.3f} ms"
            )

    cv_df = pd.DataFrame(cv_rows)

    cv_df.to_csv(
        output / "regression_cv.csv",
        index=False,
    )

    if cv_df.empty:
        return None, None, errors

    leaderboard = (
        cv_df.groupby("model")
        .agg(
            mae_ms=("mae_ms", "mean"),
            rmse_ms=("rmse_ms", "mean"),
            r2=("r2", "mean"),
            std_mae_ms=("mae_ms", "std"),
            successful_folds=("fold", "count"),
        )
        .reset_index()
    )

    leaderboard = leaderboard[
        leaderboard["successful_folds"] == folds
    ].sort_values(
        ["mae_ms", "rmse_ms"],
        ascending=True,
    )

    leaderboard.to_csv(
        output / "regression_leaderboard.csv",
        index=False,
    )

    if leaderboard.empty:
        return None, None, errors

    winner = str(
        leaderboard.iloc[0]["model"]
    )

    return winner, models[winner], errors

def conformal_quantile(errors, alpha=0.10):
    errors = np.asarray(
        errors,
        dtype=float,
    )

    errors = errors[np.isfinite(errors)]

    if len(errors) < 10:
        return None

    n = len(errors)

    level = min(
        1.0,
        np.ceil((n + 1) * (1 - alpha)) / n,
    )

    try:
        return float(
            np.quantile(
                errors,
                level,
                method="higher",
            )
        )
    except TypeError:
        return float(
            np.quantile(
                errors,
                level,
                interpolation="higher",
            )
        )

def unseen_latency_experiment(
    df,
    numeric,
    output,
    folds,
    alpha,
):
    train_idx, test_idx = grouped_holdout(df)

    development = df.iloc[
        train_idx
    ].reset_index(drop=True)

    final_test = df.iloc[
        test_idx
    ].reset_index(drop=True)

    winner, definition, errors = regression_cv(
        development,
        numeric,
        output,
        folds,
    )

    if winner is None:
        return {
            "status": "NO_VALID_MODEL",
            "errors": errors,
        }, None

    # Reserve independent query groups for
    # calibration, distinct from the final test.
    proper_idx, calibration_idx = (
        grouped_holdout(development, test_size=0.25)
    )

    proper = development.iloc[
        proper_idx
    ].reset_index(drop=True)

    calibration = development.iloc[
        calibration_idx
    ].reset_index(drop=True)

    model = clone(definition)

    ceiling = max(
        1000.0,
        float(proper["execution_time_ms"].max()) * 5,
    )

    calibration_pred = fit_predict_regression(
        model,
        proper,
        calibration,
        ceiling,
    )

    calibration_errors = np.abs(
        calibration["execution_time_ms"].to_numpy()
        - calibration_pred
    )

    radius = conformal_quantile(
        calibration_errors,
        alpha,
    )

    # Keep the same properly trained model for
    # final evaluation to preserve calibration.
    final_predictions = invert_log_target(
        model.predict(final_test),
        ceiling,
    )

    metrics = regression_metrics(
        final_test["execution_time_ms"],
        final_predictions,
    )

    baseline = DummyRegressor(
        strategy="median"
    )

    baseline.fit(
        proper[numeric],
        proper["execution_time_ms"],
    )

    baseline_pred = baseline.predict(
        final_test[numeric]
    )

    baseline_metrics = regression_metrics(
        final_test["execution_time_ms"],
        baseline_pred,
    )

    if radius is not None:
        lower = np.maximum(
            0,
            final_predictions - radius,
        )

        upper = final_predictions + radius

        actual = final_test[
            "execution_time_ms"
        ].to_numpy()

        coverage = float(
            np.mean(
                (actual >= lower)
                & (actual <= upper)
            )
        )

        mean_width = float(
            np.mean(upper - lower)
        )

    else:
        lower = np.full(
            len(final_test),
            np.nan,
        )
        upper = lower.copy()
        coverage = None
        mean_width = None

    predictions = pd.DataFrame({
        "fingerprint": (
            final_test["fingerprint"].to_numpy()
        ),
        "actual_ms": (
            final_test["execution_time_ms"].to_numpy()
        ),
        "predicted_ms": final_predictions,
        "lower_ms": lower,
        "upper_ms": upper,
        "absolute_error_ms": np.abs(
            final_test["execution_time_ms"].to_numpy()
            - final_predictions
        ),
    })

    predictions.to_csv(
        output / "unseen_latency_predictions.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(
        predictions["actual_ms"],
        predictions["predicted_ms"],
        alpha=0.55,
        s=22,
    )

    maximum = max(
        float(predictions["actual_ms"].max()),
        float(predictions["predicted_ms"].max()),
        1.0,
    )

    ax.plot(
        [0, maximum],
        [0, maximum],
        linestyle="--",
        color="gray",
    )

    ax.set_xlabel("Actual latency (ms)")
    ax.set_ylabel("Predicted latency (ms)")
    ax.set_title(
        "Unseen Query Latency Prediction"
    )

    fig.tight_layout()

    fig.savefig(
        output / "unseen_latency_scatter.png",
        dpi=180,
    )

    plt.close(fig)

    report = {
        "status": "EVALUATED",
        "winner": winner,
        "proper_train_rows": len(proper),
        "calibration_rows": len(calibration),
        "test_rows": len(final_test),
        "proper_train_fingerprints": (
            proper["fingerprint"].nunique()
        ),
        "calibration_fingerprints": (
            calibration["fingerprint"].nunique()
        ),
        "test_fingerprints": (
            final_test["fingerprint"].nunique()
        ),
        "group_overlap": False,
        "metrics": metrics,
        "median_baseline": baseline_metrics,
        "mae_improvement_pct": (
            100
            * (
                baseline_metrics["mae_ms"]
                - metrics["mae_ms"]
            )
            / max(
                baseline_metrics["mae_ms"],
                1e-9,
            )
        ),
        "prediction_interval": {
            "method": "split_conformal_absolute_residual",
            "nominal_coverage": 1 - alpha,
            "empirical_coverage": coverage,
            "mean_width_ms": mean_width,
            "radius_ms": radius,
            "calibration_size": len(calibration),
            "assumption": (
                "Approximate exchangeability between "
                "calibration and future test queries; "
                "may not hold across query families "
                "or different database scales."
            ),
        },
        "errors": errors,
    }

    write_json(
        output / "unseen_latency_report.json",
        report,
    )

    return report, {
        "model": model,
        "ceiling_ms": ceiling,
        "numeric_features": numeric,
        "training_mode": "unseen_fingerprint",
    }

def chronological_split(df, fraction=0.25):
    valid = df.dropna(
        subset=["timestamp"]
    ).copy()

    valid = valid.sort_values(
        ["timestamp", "source_row"],
        kind="mergesort",
    ).reset_index(drop=True)

    if len(valid) < 50:
        return None, None, "Too few timestamped records."

    if valid["timestamp"].nunique() < 10:
        return (
            None,
            None,
            "Too few distinct timestamps.",
        )

    boundary = max(
        1,
        int(len(valid) * (1 - fraction)),
    )

    cutoff = valid.iloc[
        boundary
    ]["timestamp"]

    train = valid[
        valid["timestamp"] < cutoff
    ].copy()

    test = valid[
        valid["timestamp"] >= cutoff
    ].copy()

    if len(train) < 30 or len(test) < 10:
        return (
            None,
            None,
            "Insufficient rows after timestamp split.",
        )

    if not (
        train["timestamp"].max()
        < test["timestamp"].min()
    ):
        return (
            None,
            None,
            "Temporal overlap detected.",
        )

    return (
        train.reset_index(drop=True),
        test.reset_index(drop=True),
        None,
    )

def fingerprint_history(train, test):
    # Historical statistics are computed exclusively
    # from the earlier training observations.
    history = (
        train.groupby("fingerprint")
        .agg(
            history_count=(
                "execution_time_ms", "size"
            ),
            history_median_ms=(
                "execution_time_ms", "median"
            ),
            history_mean_ms=(
                "execution_time_ms", "mean"
            ),
            history_std_ms=(
                "execution_time_ms", "std"
            ),
            history_p90_ms=(
                "execution_time_ms",
                lambda s: s.quantile(0.90),
            ),
        )
        .reset_index()
    )

    result = test.merge(
        history,
        on="fingerprint",
        how="left",
        validate="many_to_one",
    )

    return result, history

def known_query_experiment(
    df,
    numeric,
    output,
    alpha,
):
    train, test, error = chronological_split(df)

    if error:
        report = {
            "status": "INSUFFICIENT_TEMPORAL_DATA",
            "reason": error,
        }

        write_json(
            output / "known_latency_report.json",
            report,
        )

        return report, None

    known_fingerprints = set(
        train["fingerprint"]
    )

    known_test = test[
        test["fingerprint"].isin(
            known_fingerprints
        )
    ].copy()

    if len(known_test) < 20:
        report = {
            "status": "INSUFFICIENT_KNOWN_QUERY_TEST",
            "known_test_rows": len(known_test),
        }

        write_json(
            output / "known_latency_report.json",
            report,
        )

        return report, None

    # Within the earlier training period, reserve
    # a later chronological calibration segment.
    proper, calibration, error = (
        chronological_split(
            train,
            fraction=0.25,
        )
    )

    if error:
        report = {
            "status": "INSUFFICIENT_CALIBRATION_DATA",
            "reason": error,
        }

        write_json(
            output / "known_latency_report.json",
            report,
        )

        return report, None

    known_calibration = calibration[
        calibration["fingerprint"].isin(
            set(proper["fingerprint"])
        )
    ].copy()

    if len(known_calibration) < 10:
        report = {
            "status": "INSUFFICIENT_CALIBRATION_DATA",
            "calibration_rows": len(known_calibration),
        }

        write_json(
            output / "known_latency_report.json",
            report,
        )

        return report, None

    # Candidate A: per-fingerprint historical median.
    cal_hist, proper_stats = fingerprint_history(
        proper,
        known_calibration,
    )

    test_hist, train_stats = fingerprint_history(
        train,
        known_test,
    )

    # Candidate B: structural CatBoost trained on
    # earlier observations.
    # Candidate C: structural Random Forest.
    # Candidate D: structural Extra Trees.
    definitions = {
        "historical_median": None,
        "catboost_numeric": (
            regression_candidates(numeric)[
                "catboost_numeric"
            ]
        ),
        "random_forest_numeric": (
            regression_candidates(numeric)[
                "random_forest_numeric"
            ]
        ),
        "extra_trees_numeric": (
            regression_candidates(numeric)[
                "extra_trees_numeric"
            ]
        ),
    }

    calibration_results = {}
    errors = []

    for name, definition in definitions.items():
        try:
            if name == "historical_median":
                predicted = (
                    cal_hist["history_median_ms"]
                    .fillna(
                        proper["execution_time_ms"].median()
                    )
                    .to_numpy()
                )
            else:
                ceiling = max(
                    1000.0,
                    float(
                        proper["execution_time_ms"].max()
                    ) * 5,
                )

                model = clone(definition)

                predicted = fit_predict_regression(
                    model,
                    proper,
                    known_calibration,
                    ceiling,
                )

            metrics = regression_metrics(
                known_calibration["execution_time_ms"],
                predicted,
            )

            calibration_results[name] = {
                **metrics,
                "predictions": predicted,
            }

        except Exception as exc:
            errors.append({
                "model": name,
                "error": str(exc)[:500],
            })

    if not calibration_results:
        report = {
            "status": "ALL_MODELS_FAILED",
            "errors": errors,
        }

        write_json(
            output / "known_latency_report.json",
            report,
        )

        return report, None

    winner = min(
        calibration_results,
        key=lambda name: (
            calibration_results[name]["mae_ms"],
            calibration_results[name]["rmse_ms"],
        ),
    )

    # For the final known-query test, the model
    # selection is fixed using earlier calibration.
    # The structural model is refitted on all
    # observations before the final test cutoff.
    if winner == "historical_median":
        final_predictions = (
            test_hist["history_median_ms"]
            .fillna(
                train["execution_time_ms"].median()
            )
            .to_numpy()
        )

        final_model = None

    else:
        final_model = clone(
            definitions[winner]
        )

        final_ceiling = max(
            1000.0,
            float(
                train["execution_time_ms"].max()
            ) * 5,
        )

        final_predictions = fit_predict_regression(
            final_model,
            train,
            known_test,
            final_ceiling,
        )

    # Interval calibration uses the earlier
    # proper/calibration split. After refitting,
    # its coverage is empirical only; formal
    # split-conformal guarantees do not apply.
    cal_pred = calibration_results[
        winner
    ]["predictions"]

    residuals = np.abs(
        known_calibration[
            "execution_time_ms"
        ].to_numpy()
        - cal_pred
    )

    radius = conformal_quantile(
        residuals,
        alpha,
    )

    if radius is not None:
        lower = np.maximum(
            0,
            final_predictions - radius,
        )
        upper = final_predictions + radius

        actual = known_test[
            "execution_time_ms"
        ].to_numpy()

        coverage = float(
            np.mean(
                (actual >= lower)
                & (actual <= upper)
            )
        )

    else:
        lower = np.full(
            len(known_test),
            np.nan,
        )
        upper = lower.copy()
        coverage = None

    metrics = regression_metrics(
        known_test["execution_time_ms"],
        final_predictions,
    )

    # Strong deployment baseline:
    # previously observed median for each query.
    history_predictions = (
        test_hist["history_median_ms"]
        .fillna(
            train["execution_time_ms"].median()
        )
        .to_numpy()
    )

    history_metrics = regression_metrics(
        known_test["execution_time_ms"],
        history_predictions,
    )

    # Another baseline: global median of
    # all prior observations.
    global_predictions = np.full(
        len(known_test),
        float(
            train["execution_time_ms"].median()
        ),
    )

    global_metrics = regression_metrics(
        known_test["execution_time_ms"],
        global_predictions,
    )

    predictions = pd.DataFrame({
        "fingerprint": (
            known_test["fingerprint"].to_numpy()
        ),
        "timestamp": (
            known_test["timestamp"].astype(str).to_numpy()
        ),
        "actual_ms": (
            known_test["execution_time_ms"].to_numpy()
        ),
        "predicted_ms": final_predictions,
        "historical_median_ms": history_predictions,
        "lower_ms": lower,
        "upper_ms": upper,
    })

    predictions.to_csv(
        output / "known_latency_predictions.csv",
        index=False,
    )

    calibration_summary = {
        name: {
            key: value
            for key, value in metrics.items()
            if key != "predictions"
        }
        for name, metrics in calibration_results.items()
    }

    report = {
        "status": "EVALUATED",
        "winner": winner,
        "train_rows": len(train),
        "test_rows": len(known_test),
        "test_fingerprints": (
            known_test["fingerprint"].nunique()
        ),
        "train_latest_timestamp": str(
            train["timestamp"].max()
        ),
        "test_earliest_timestamp": str(
            known_test["timestamp"].min()
        ),
        "temporal_leakage": False,
        "model_selection_calibration": (
            calibration_summary
        ),
        "metrics": metrics,
        "historical_median_baseline": history_metrics,
        "global_median_baseline": global_metrics,
        "mae_gain_over_history_pct": (
            100
            * (
                history_metrics["mae_ms"]
                - metrics["mae_ms"]
            )
            / max(
                history_metrics["mae_ms"],
                1e-9,
            )
        ),
        "prediction_interval": {
            "method": (
                "earlier_residual_quantile; "
                "heuristic after model refitting"
            ),
            "nominal_coverage": 1 - alpha,
            "empirical_coverage": coverage,
            "radius_ms": radius,
        },
        "limitations": [
            "Historical observations span a short period.",
            "Training and testing may share one workload session.",
            "Calibration errors may not represent future shifts.",
            "Refitted-model intervals have no formal coverage guarantee.",
        ],
        "errors": errors,
    }

    write_json(
        output / "known_latency_report.json",
        report,
    )

    artifact = {
        "model": final_model,
        "historical_statistics": train_stats,
        "global_median_ms": float(
            train["execution_time_ms"].median()
        ),
        "winner": winner,
        "numeric_features": numeric,
        "training_mode": "known_fingerprint",
    }

    if winner != "historical_median":
        artifact["ceiling_ms"] = final_ceiling

    return report, artifact

def build_phase9_priorities(
    df,
    known_report,
    unseen_report,
    output,
):
    grouped = (
        df.groupby("fingerprint")
        .agg(
            observed_frequency=(
                "execution_time_ms", "size"
            ),
            median_latency_ms=(
                "execution_time_ms", "median"
            ),
            p90_latency_ms=(
                "execution_time_ms",
                lambda s: s.quantile(0.90),
            ),
            mean_latency_ms=(
                "execution_time_ms", "mean"
            ),
            query_family=(
                "query_family", "first"
            ),
        )
        .reset_index()
    )

    grouped["observed_total_time_ms"] = (
        grouped["observed_frequency"]
        * grouped["mean_latency_ms"]
    )

    grouped["priority_score"] = (
        grouped["observed_total_time_ms"]
    )

    grouped = grouped.sort_values(
        "priority_score",
        ascending=False,
    )

    grouped["priority_rank"] = np.arange(
        1,
        len(grouped) + 1,
    )

    grouped["optimization_recommendation"] = (
        "EXPLAIN_REVIEW"
    )

    grouped.to_csv(
        output / "phase9_priorities_v3.csv",
        index=False,
    )

    return {
        "status": "EXPORTED",
        "fingerprints": len(grouped),
        "priority_method": (
            "Observed total execution time per "
            "fingerprint; transparent non-ML baseline"
        ),
        "known_query_model": known_report.get(
            "winner"
        ),
        "unseen_query_model": unseen_report.get(
            "winner"
        ),
        "important": (
            "Ranking is not a measured prediction "
            "of index benefit. Phase 9 must validate "
            "index recommendations with controlled "
            "benchmarks."
        ),
        "file": str(
            output / "phase9_priorities_v3.csv"
        ),
    }

def recommendations(
    classification,
    unseen,
    known,
):
    items = []

    if classification.get("status") == "EVALUATED":
        score = classification[
            "outer_test_metrics"
        ]["macro_f1"]

        baseline = classification[
            "outer_test_majority_baseline"
        ]["macro_f1"]

        items.append({
            "task": "workload_classification",
            "selected_model": classification["winner"],
            "outer_macro_f1": score,
            "baseline_macro_f1": baseline,
            "suitable_for_phase9": False,
            "reason": (
                "User-role prediction is not directly "
                "required for index selection. "
                "Use only if independently validated "
                "and requester identity is unavailable."
            ),
        })

    if unseen.get("status") == "EVALUATED":
        gain = unseen[
            "mae_improvement_pct"
        ]

        suitable = (
            gain > 5
            and unseen["metrics"]["r2"] > 0
        )

        items.append({
            "task": "unseen_query_latency",
            "selected_model": unseen["winner"],
            "mae_gain_pct": gain,
            "suitable_for_phase9": suitable,
            "reason": (
                "Candidate for advisory prioritization "
                "only; verify performance across "
                "additional query groups and "
                "database sizes."
                if suitable
                else
                "No sufficient demonstrated advantage "
                "over the median baseline. Do not "
                "use as an optimization decision gate."
            ),
        })

    if known.get("status") == "EVALUATED":
        gain = known[
            "mae_gain_over_history_pct"
        ]

        suitable = (
            gain >= 0
            and known["metrics"]["mae_ms"]
            <= known[
                "historical_median_baseline"
            ]["mae_ms"]
        )

        items.append({
            "task": "known_query_latency",
            "selected_model": known["winner"],
            "mae_gain_over_history_pct": gain,
            "suitable_for_phase9": suitable,
            "reason": (
                "Use as a candidate advisory signal "
                "alongside frequency and measured "
                "query costs; validate on additional "
                "chronological sessions."
                if suitable
                else
                "Historical fingerprint median "
                "remains the safer baseline."
            ),
        })

    return {
        "recommendations": items,
        "overall_guidance": (
            "Do not automatically create indexes "
            "from ML predictions. Phase 9 must "
            "measure benefits, regressions, "
            "index size and write overhead."
        ),
    }

def plot_leaderboards(output):
    configs = [
        (
            "classification_leaderboard.csv",
            "macro_f1",
            "Classification Macro F1",
            "classification_comparison.png",
            False,
        ),
        (
            "regression_leaderboard.csv",
            "mae_ms",
            "Unseen Query CV MAE (ms)",
            "regression_comparison.png",
            True,
        ),
    ]

    for filename, metric, title, image_name, ascending in configs:
        path = output / filename

        if not path.exists():
            continue

        data = pd.read_csv(path)

        if data.empty or metric not in data.columns:
            continue

        data = data.sort_values(
            metric,
            ascending=ascending,
        ).head(12)

        fig, ax = plt.subplots(
            figsize=(10, max(5, len(data) * 0.45))
        )

        ax.barh(
            data["model"],
            data[metric],
        )

        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel(metric)

        fig.tight_layout()
        fig.savefig(
            output / image_name,
            dpi=180,
        )
        plt.close(fig)

def acceptance_report(
    classification,
    unseen,
    known,
    phase9,
):
    checks = {
        "classification_outer_test": (
            classification.get("status") == "EVALUATED"
        ),
        "unseen_latency_outer_test": (
            unseen.get("status") == "EVALUATED"
        ),
        "known_latency_temporal_test": (
            known.get("status") == "EVALUATED"
        ),
        "phase9_priorities_exported": (
            phase9.get("status") == "EXPORTED"
        ),
        "classification_group_separation": (
            classification.get("group_overlap") is False
        ),
        "regression_group_separation": (
            unseen.get("group_overlap") is False
        ),
        "known_query_time_separation": (
            known.get("temporal_leakage") is False
        ),
    }

    return {
        "status": (
            "IMPLEMENTATION_CHECKS_PASSED"
            if all(checks.values())
            else "PARTIAL_VALIDATION"
        ),
        "passed": sum(checks.values()),
        "total": len(checks),
        "checks": checks,
        "warning": (
            "Passing implementation checks does "
            "not establish model generalization "
            "or measured Phase 9 optimization benefit."
        ),
    }

def save_research_report(
    output,
    audit,
    classification,
    unseen,
    known,
    advice,
    acceptance,
):
    lines = [
        "# Phase 8 V3 — Model Comparison Research Report",
        "",
        f"Generated: {now()}",
        "",
        "## Dataset",
        f"- Records: {audit['records']}",
        f"- Unique fingerprints: {audit['fingerprints']}",
        f"- Sessions: {audit['sessions']}",
        f"- Ambiguous fingerprints: {audit['ambiguous_fingerprints']}",
        "",
        "## Unseen-query classification",
        f"- Status: {classification.get('status')}",
        f"- Selected model: {classification.get('winner')}",
    ]

    if classification.get("outer_test_metrics"):
        m = classification["outer_test_metrics"]
        lines += [
            f"- Outer accuracy: {m['accuracy']:.4f}",
            f"- Outer macro F1: {m['macro_f1']:.4f}",
        ]

    lines += [
        "",
        "## Unseen-query latency",
        f"- Status: {unseen.get('status')}",
        f"- Selected model: {unseen.get('winner')}",
    ]

    if unseen.get("metrics"):
        m = unseen["metrics"]
        lines += [
            f"- Outer MAE: {m['mae_ms']:.4f} ms",
            f"- Outer RMSE: {m['rmse_ms']:.4f} ms",
            f"- Outer R2: {m['r2']:.4f}",
            f"- MAE gain over baseline: {unseen['mae_improvement_pct']:.2f}%",
        ]

    lines += [
        "",
        "## Known-query latency",
        f"- Status: {known.get('status')}",
        f"- Selected model: {known.get('winner')}",
    ]

    if known.get("metrics"):
        m = known["metrics"]
        lines += [
            f"- Chronological MAE: {m['mae_ms']:.4f} ms",
            f"- Chronological RMSE: {m['rmse_ms']:.4f} ms",
            f"- Gain over historical median: {known['mae_gain_over_history_pct']:.2f}%",
        ]

    lines += [
        "",
        "## Phase 9 recommendations",
    ]

    for item in advice["recommendations"]:
        lines += [
            f"- {item['task']}: {item['selected_model']}",
            f"  Suitable: {item['suitable_for_phase9']}",
            f"  Reason: {item['reason']}",
        ]

    lines += [
        "",
        "## Acceptance",
        f"- {acceptance['status']}",
        f"- Checks: {acceptance['passed']}/{acceptance['total']}",
        "",
        "## Limitations",
        "- Only a small number of SQL fingerprints are available.",
        "- Role labels may overlap for identical SQL.",
        "- Known-query temporal evaluation uses limited monitoring history.",
        "- Prediction intervals require stable data distributions.",
        "- Phase 9 index benefit has not been established by this script.",
        "- No production database was modified.",
    ]

    (output / "research_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

def run(args):
    started = time.perf_counter()

    run_id = datetime.now(
        timezone.utc
    ).strftime("p8v3_%Y%m%d_%H%M%S_%f")

    output = BASE_OUTPUT / "runs" / run_id

    output.mkdir(
        parents=True,
        exist_ok=False,
    )

    print("=" * 80)
    print("PHASE 8 V3 — ADVANCED MODEL COMPARISON")
    print("=" * 80)
    print("Run:", run_id)
    print("Output:", output)

    df, sources = load_data(args.input)

    df, numeric, planner = build_features(
        df,
        use_planner=not args.sql_only,
    )

    audit = data_audit(
        df,
        sources,
        planner,
    )

    write_json(
        output / "data_audit.json",
        audit,
    )

    write_json(
        output / "experiment_config.json",
        {
            "run_id": run_id,
            "seed": SEED,
            "cv_folds": args.folds,
            "alpha": args.alpha,
            "numeric_features": numeric,
            "planner_features": planner,
            "sql_only": args.sql_only,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
                "sklearn": sklearn.__version__,
            },
            "source_files": sources,
        },
    )

    print(
        f"\nRecords: {len(df)}"
        f"\nFingerprints: {df['fingerprint'].nunique()}"
        f"\nSessions: {df['session_id'].nunique()}"
        f"\nFeatures: {len(numeric)}"
    )

    print("\n[1/5] Classification comparison")

    classification, classifier = (
        evaluate_classification(
            df,
            numeric,
            output,
            args.folds,
        )
    )

    print("\n[2/5] Unseen-query latency comparison")

    unseen, unseen_model = (
        unseen_latency_experiment(
            df,
            numeric,
            output,
            args.folds,
            args.alpha,
        )
    )

    print("\n[3/5] Known-query chronological comparison")

    known, known_model = known_query_experiment(
        df,
        numeric,
        output,
        args.alpha,
    )

    print("\n[4/5] Phase 9 integration and recommendations")

    phase9 = build_phase9_priorities(
        df,
        known,
        unseen,
        output,
    )

    advice = recommendations(
        classification,
        unseen,
        known,
    )

    write_json(
        output / "phase9_integration.json",
        phase9,
    )

    write_json(
        output / "recommendations.json",
        advice,
    )

    print("\n[5/5] Research reports and model artifacts")

    if classifier is not None:
        joblib.dump(
            classifier,
            output / "classification_model.joblib",
        )

    if unseen_model is not None:
        joblib.dump(
            unseen_model,
            output / "unseen_latency_model.joblib",
        )

    if known_model is not None:
        joblib.dump(
            known_model,
            output / "known_latency_model.joblib",
        )

    acceptance = acceptance_report(
        classification,
        unseen,
        known,
        phase9,
    )

    write_json(
        output / "acceptance_tests.json",
        acceptance,
    )

    plot_leaderboards(output)

    save_research_report(
        output,
        audit,
        classification,
        unseen,
        known,
        advice,
        acceptance,
    )

    summary = {
        "run_id": run_id,
        "status": "COMPLETED",
        "records": len(df),
        "fingerprints": df["fingerprint"].nunique(),
        "classification_winner": (
            classification.get("winner")
        ),
        "classification_accuracy": (
            classification.get(
                "outer_test_metrics", {}
            ).get("accuracy")
        ),
        "classification_macro_f1": (
            classification.get(
                "outer_test_metrics", {}
            ).get("macro_f1")
        ),
        "unseen_latency_winner": (
            unseen.get("winner")
        ),
        "unseen_latency_mae_ms": (
            unseen.get(
                "metrics", {}
            ).get("mae_ms")
        ),
        "unseen_latency_r2": (
            unseen.get(
                "metrics", {}
            ).get("r2")
        ),
        "known_latency_winner": (
            known.get("winner")
        ),
        "known_latency_mae_ms": (
            known.get(
                "metrics", {}
            ).get("mae_ms")
        ),
        "known_latency_status": (
            known.get("status")
        ),
        "acceptance": acceptance,
        "elapsed_seconds": round(
            time.perf_counter() - started,
            2,
        ),
        "output_directory": str(output),
    }

    write_json(
        output / "summary.json",
        summary,
    )

    print("\n" + "=" * 80)
    print("PHASE 8 V3 — FINAL RESULTS")
    print("=" * 80)

    print(
        json.dumps(
            json_safe(summary),
            indent=2,
        )
    )

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Phase 8 V3: CatBoost, SVM, "
            "Random Forest, robust regression, "
            "grouped and chronological evaluation"
        )
    )

    parser.add_argument(
        "--input",
        nargs="+",
        default=[str(DEFAULT_CSV)],
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--sql-only",
        action="store_true",
        help=(
            "Exclude planner features; "
            "use only SQL-derived features."
        ),
    )

    args = parser.parse_args()

    if args.folds < 2:
        parser.error(
            "--folds must be at least 2"
        )

    if not (0 < args.alpha < 1):
        parser.error(
            "--alpha must be between 0 and 1"
        )

    run(args)

if __name__ == "__main__":
    main()
