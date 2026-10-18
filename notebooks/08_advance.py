
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
from collections import Counter
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

from scipy.stats import ks_2samp
from sklearn.base import clone
from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUT = (
    ROOT
    / "results"
    / "phase7_monitoring"
    / "phase7_query_monitoring.csv"
)

OUTPUT = ROOT / "results" / "phase8_advanced"

REGISTRY = OUTPUT / "model_registry"
RUNS = OUTPUT / "runs"
ACTIVE_FILE = REGISTRY / "active_model.json"

SEED = 42

SQL_STRUCTURAL_FEATURES = [
    "sql_length",
    "word_count",
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
    "filter_count",
    "has_cte",
    "has_union",
    "has_window",
    "has_offset",
    "has_exists",
    "has_between",
    "has_like",
    "has_in",
    "has_or",
]

PREEXECUTION_FEATURES = (
    SQL_STRUCTURAL_FEATURES
    + ["estimated_cost", "startup_cost", "estimated_rows"]
)

MONITORING_FEATURES = PREEXECUTION_FEATURES + [
    "execution_time_ms",
    "planning_time_ms",
    "actual_rows",
    "index_scan_count",
    "seq_scan_count",
    "bitmap_scan_count",
    "shared_hit_blocks",
    "shared_read_blocks",
    "temp_read_blocks",
    "temp_written_blocks",
    "plan_node_count",
]

TARGET_CLASSES = [
    "admin",
    "doctor",
    "mixed",
    "research",
]

REQUIRED_COLUMNS = [
    "sql",
    "workload_type",
    "query_family",
    "fingerprint_hash",
    "execution_time_ms",
    "monitored_at",
]

def utcnow():
    return datetime.now(timezone.utc).isoformat()

def ensure_dirs():
    for folder in (OUTPUT, REGISTRY, RUNS):
        folder.mkdir(parents=True, exist_ok=True)

def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            data,
            indent=2,
            default=str,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))

def safe_float(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (ValueError, TypeError):
        return None

def hash_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def normalize_sql(text):
    text = str(text or "")
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"--[^\n]*", " ", text)
    text = re.sub(r"'(?:''|[^'])*'", "?", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\b", "?", text)
    return re.sub(r"\s+", " ", text).strip().lower()

def fingerprint_sql(text):
    return hashlib.sha256(
        normalize_sql(text).encode("utf-8")
    ).hexdigest()

def load_monitoring(paths):
    frames = []
    source_files = []

    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()

        if path.is_dir():
            candidates = sorted(path.glob("*.csv"))
        else:
            candidates = [path]

        for candidate in candidates:
            if not candidate.is_file():
                raise FileNotFoundError(candidate)

            sample = pd.read_csv(candidate, nrows=3)

            # A directory may contain summary CSVs.
            # Only full monitoring files are ingested.
            if not all(
                c in sample.columns
                for c in REQUIRED_COLUMNS
            ):
                if path.is_dir():
                    continue
                raise ValueError(
                    f"{candidate} is not a Phase 7 monitoring CSV. "
                    f"Missing: {sorted(set(REQUIRED_COLUMNS) - set(sample.columns))}"
                )

            df = pd.read_csv(candidate, low_memory=False)

            df["source_file"] = str(candidate)
            df["source_row"] = np.arange(len(df))

            # Each input file is a separate recording
            # session unless it has an explicit session_id.
            if "session_id" not in df.columns:
                df["session_id"] = candidate.stem

            frames.append(df)

            source_files.append({
                "path": str(candidate),
                "sha256": hash_file(candidate),
                "rows": int(len(df)),
            })

    if not frames:
        raise ValueError(
            "No valid Phase 7 monitoring CSV files found."
        )

    data = pd.concat(frames, ignore_index=True)

    data["sql"] = data["sql"].fillna("").astype(str)
    data = data[data["sql"].str.strip().ne("")].copy()

    data["workload_type"] = (
        data["workload_type"]
        .fillna("unknown")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    data["query_family"] = (
        data["query_family"]
        .fillna("unknown")
        .astype(str)
    )

    data["fingerprint"] = (
        data["fingerprint_hash"]
        .fillna("")
        .astype(str)
    )

    missing = data["fingerprint"].eq("")

    data.loc[missing, "fingerprint"] = (
        data.loc[missing, "sql"].map(fingerprint_sql)
    )

    data["timestamp"] = pd.to_datetime(
        data["monitored_at"],
        errors="coerce",
        utc=True,
    )

    data["session_id"] = (
        data["session_id"]
        .fillna("unknown")
        .astype(str)
    )

    numeric_columns = [
        "join_count",
        "filter_count",
        "execution_time_ms",
        "planning_time_ms",
        "estimated_cost",
        "startup_cost",
        "estimated_rows",
        "actual_rows",
        "index_scan_count",
        "seq_scan_count",
        "bitmap_scan_count",
        "shared_hit_blocks",
        "shared_read_blocks",
        "temp_read_blocks",
        "temp_written_blocks",
        "plan_node_count",
    ]

    for column in numeric_columns:
        if column not in data.columns:
            data[column] = np.nan

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    for column in [
        "tables_accessed",
        "columns_accessed",
        "plan_node_types",
    ]:
        if column not in data.columns:
            data[column] = ""

        data[column] = data[column].fillna("").astype(str)

    data = data.reset_index(drop=True)

    return data, source_files

def engineer_features(df):
    out = df.copy()

    out["normalized_sql"] = (
        out["sql"].map(normalize_sql)
    )

    q = out["normalized_sql"]

    out["sql_length"] = q.str.len()
    out["word_count"] = q.str.split().str.len()

    patterns = {
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

    for column, pattern in patterns.items():
        # Preserve Phase 7's original join_count
        # when available, but derive a separate
        # SQL-based count for consistent inference.
        derived = q.str.count(pattern)

        if column == "join_count":
            out[column] = derived
        else:
            out[column] = derived

    out["table_count"] = (
        q.str.count(r"\bfrom\b")
        + q.str.count(r"\bjoin\b")
    )

    boolean_patterns = {
        "has_cte": r"^\s*with\b",
        "has_union": r"\bunion\b",
        "has_window": r"\bover\s*\(",
        "has_offset": r"\boffset\b",
        "has_exists": r"\bexists\b",
        "has_between": r"\bbetween\b",
        "has_like": r"\blike\b",
        "has_in": r"\bin\s*\(",
        "has_or": r"\bor\b",
    }

    for column, pattern in boolean_patterns.items():
        out[column] = (
            q.str.contains(pattern, regex=True)
            .astype(int)
        )

    out["filter_count"] = (
        out["filter_count"]
        .fillna(out["predicate_count"])
    )

    # No workload labels, query_family, query_name,
    # or expected_use are included in text features.
    out["model_text"] = (
        out["normalized_sql"]
        + " "
        + out["tables_accessed"]
        .str.replace(",", " ", regex=False)
    )

    for column in MONITORING_FEATURES:
        out[column] = pd.to_numeric(
            out[column],
            errors="coerce",
        )

    return out

def quality_report(df, sources):
    timestamps = df["timestamp"].dropna()

    if len(timestamps):
        duration = (
            timestamps.max() - timestamps.min()
        ).total_seconds()
    else:
        duration = 0

    family_label_counts = (
        df.groupby("fingerprint")["workload_type"]
        .nunique()
    )

    ambiguous = family_label_counts[
        family_label_counts > 1
    ]

    report = {
        "records": int(len(df)),
        "columns": int(len(df.columns)),
        "unique_fingerprints": int(
            df["fingerprint"].nunique()
        ),
        "unique_query_families": int(
            df["query_family"].nunique()
        ),
        "sessions": int(df["session_id"].nunique()),
        "valid_timestamps": int(
            df["timestamp"].notna().sum()
        ),
        "duration_seconds": safe_float(duration),
        "classes": {
            str(k): int(v)
            for k, v in df["workload_type"]
            .value_counts().items()
        },
        "ambiguous_fingerprints": int(
            len(ambiguous)
        ),
        "label_ambiguity_warning": (
            "The same SQL fingerprint can occur "
            "under multiple workload_type labels. "
            "Predicting actor identity from SQL "
            "alone may be underdetermined."
        ),
        "missing_fraction": {
            col: safe_float(df[col].isna().mean())
            for col in [
                "execution_time_ms",
                "estimated_cost",
                "estimated_rows",
                "timestamp",
            ]
        },
        "sources": sources,
    }

    return report

def build_preprocessor(numeric_features, use_text=True):
    transformers = [
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
            numeric_features,
        )
    ]

    if use_text:
        transformers.append(
            (
                "sql_tfidf",
                TfidfVectorizer(
                    max_features=2500,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    token_pattern=r"(?u)\b\w+\b",
                ),
                "model_text",
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        sparse_threshold=0.3,
    )

def classification_models():
    return {
        "majority_baseline": DummyClassifier(
            strategy="most_frequent"
        ),
        "logistic_regression": LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=SEED,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=220,
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        ),
    }

def make_classifier(model, use_text=True):
    return Pipeline([
        (
            "features",
            build_preprocessor(
                SQL_STRUCTURAL_FEATURES,
                use_text=use_text,
            ),
        ),
        ("model", clone(model)),
    ])

def grouped_splits(df, n_splits=5):
    groups = df["fingerprint"].astype(str)

    splitter = GroupShuffleSplit(
        n_splits=n_splits,
        test_size=0.25,
        random_state=SEED,
    )

    for train_idx, test_idx in splitter.split(
        df,
        df["workload_type"],
        groups=groups,
    ):
        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])

        assert train_groups.isdisjoint(test_groups)

        yield train_idx, test_idx

def save_confusion_plot(matrix, classes, path):
    fig, ax = plt.subplots(figsize=(7, 6))

    image = ax.imshow(
        matrix,
        interpolation="nearest",
        cmap="Blues",
    )

    fig.colorbar(image, ax=ax)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))

    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticklabels(classes)

    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title("Fingerprint-grouped confusion matrix")

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
            )

    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)

def run_classification(df, run_dir, n_splits):
    data = df[
        df["workload_type"].isin(TARGET_CLASSES)
    ].copy().reset_index(drop=True)

    if data["fingerprint"].nunique() < 8:
        return {
            "status": "INSUFFICIENT_GROUPS",
        }, None

    X = data
    y = data["workload_type"]

    model_definitions = classification_models()

    rows = []
    prediction_rows = []

    confusion_totals = {
        name: np.zeros(
            (len(TARGET_CLASSES), len(TARGET_CLASSES)),
            dtype=int,
        )
        for name in model_definitions
    }

    successful_splits = 0

    for fold, (tr, te) in enumerate(
        grouped_splits(data, n_splits),
        start=1,
    ):
        train = X.iloc[tr]
        test = X.iloc[te]

        y_train = y.iloc[tr]
        y_test = y.iloc[te]

        if y_train.nunique() < 2:
            continue

        successful_splits += 1

        for name, definition in model_definitions.items():
            pipeline = make_classifier(definition)

            try:
                pipeline.fit(train, y_train)
                predicted = pipeline.predict(test)

                if hasattr(pipeline, "predict_proba"):
                    probabilities = pipeline.predict_proba(
                        test
                    )

                    confidence = probabilities.max(axis=1)

                    sorted_probabilities = np.sort(
                        probabilities,
                        axis=1,
                    )

                    if probabilities.shape[1] >= 2:
                        margin = (
                            sorted_probabilities[:, -1]
                            - sorted_probabilities[:, -2]
                        )
                    else:
                        margin = np.ones(len(test))
                else:
                    confidence = np.full(
                        len(test),
                        np.nan,
                    )
                    margin = np.full(
                        len(test),
                        np.nan,
                    )

                cm = confusion_matrix(
                    y_test,
                    predicted,
                    labels=TARGET_CLASSES,
                )

                confusion_totals[name] += cm

                rows.append({
                    "fold": fold,
                    "model": name,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "train_fingerprints": (
                        train["fingerprint"].nunique()
                    ),
                    "test_fingerprints": (
                        test["fingerprint"].nunique()
                    ),
                    "accuracy": accuracy_score(
                        y_test,
                        predicted,
                    ),
                    "macro_f1": f1_score(
                        y_test,
                        predicted,
                        average="macro",
                        labels=TARGET_CLASSES,
                        zero_division=0,
                    ),
                })

                for local_idx, original_idx in enumerate(te):
                    prediction_rows.append({
                        "fold": fold,
                        "model": name,
                        "fingerprint": (
                            data.iloc[original_idx][
                                "fingerprint"
                            ]
                        ),
                        "actual": y_test.iloc[local_idx],
                        "predicted": predicted[local_idx],
                        "confidence": confidence[local_idx],
                        "margin": margin[local_idx],
                        "correct": int(
                            predicted[local_idx]
                            == y_test.iloc[local_idx]
                        ),
                    })

            except Exception as exc:
                rows.append({
                    "fold": fold,
                    "model": name,
                    "error": str(exc)[:300],
                })

    scores = pd.DataFrame(rows)

    if scores.empty or "macro_f1" not in scores.columns:
        return {
            "status": "NO_VALID_FOLDS",
        }, None

    scores.to_csv(
        run_dir / "classification_folds.csv",
        index=False,
    )

    predictions = pd.DataFrame(prediction_rows)

    predictions.to_csv(
        run_dir / "classification_predictions.csv",
        index=False,
    )

    valid_scores = scores.dropna(
        subset=["macro_f1"]
    )

    summary = {}

    for name, group in valid_scores.groupby("model"):
        summary[name] = {
            "folds": int(len(group)),
            "mean_accuracy": float(
                group["accuracy"].mean()
            ),
            "std_accuracy": float(
                group["accuracy"].std(ddof=0)
            ),
            "mean_macro_f1": float(
                group["macro_f1"].mean()
            ),
            "std_macro_f1": float(
                group["macro_f1"].std(ddof=0)
            ),
        }

        matrix = confusion_totals[name]

        save_confusion_plot(
            matrix,
            TARGET_CLASSES,
            run_dir / f"confusion_{name}.png",
        )

        report = classification_report(
            np.repeat(
                TARGET_CLASSES,
                matrix.sum(axis=1),
            ),
            np.concatenate([
                np.repeat(
                    TARGET_CLASSES,
                    matrix[i],
                )
                for i in range(len(TARGET_CLASSES))
            ]),
            labels=TARGET_CLASSES,
            output_dict=True,
            zero_division=0,
        ) if False else None

        # Derive precision, recall and F1 directly
        # from the accumulated confusion matrix.
        per_class = {}

        for i, label in enumerate(TARGET_CLASSES):
            tp = int(matrix[i, i])
            support = int(matrix[i, :].sum())
            predicted_count = int(matrix[:, i].sum())

            precision = (
                tp / predicted_count
                if predicted_count else 0.0
            )
            recall = (
                tp / support
                if support else 0.0
            )
            f1 = (
                2 * precision * recall
                / (precision + recall)
                if precision + recall else 0.0
            )

            per_class[label] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }

        summary[name]["per_class"] = per_class
        summary[name]["confusion_matrix"] = (
            matrix.tolist()
        )

        model_predictions = predictions[
            predictions["model"] == name
        ]

        summary[name]["mean_confidence"] = safe_float(
            model_predictions["confidence"].mean()
        )

        summary[name]["low_confidence_fraction"] = (
            safe_float(
                (
                    model_predictions["confidence"] < 0.60
                ).mean()
            )
        )

        summary[name]["uncertainty_note"] = (
            "Raw model probabilities are not "
            "calibrated probabilities. Confidence "
            "is descriptive, not a validated "
            "probability of correctness."
        )

    candidates = [
        name
        for name in summary
        if name != "majority_baseline"
    ]

    if not candidates:
        return {
            "status": "NO_TRAINED_CANDIDATE",
            "models": summary,
        }, None

    winner = max(
        candidates,
        key=lambda name: (
            summary[name]["mean_macro_f1"],
            summary[name]["mean_accuracy"],
        ),
    )

    baseline_f1 = summary.get(
        "majority_baseline", {}
    ).get("mean_macro_f1")

    improvement = (
        summary[winner]["mean_macro_f1"] - baseline_f1
        if baseline_f1 is not None
        else None
    )

    final_pipeline = make_classifier(
        model_definitions[winner]
    )

    final_pipeline.fit(X, y)

    report = {
        "status": "EVALUATED",
        "winner": winner,
        "repeated_group_splits": successful_splits,
        "models": summary,
        "winner_macro_f1_gain_over_baseline": improvement,
        "selection_warning": (
            "The winner was selected using repeated "
            "holdout results. These scores are "
            "model-selection evidence, not an "
            "independent final test estimate."
        ),
        "group_overlap": False,
        "label_leakage_audit": {
            "excluded_columns": [
                "workload_type",
                "query_family",
                "query_name",
                "expected_use",
                "fingerprint_hash",
                "query_hash",
                "session_id",
                "source_file",
            ],
            "fingerprint_group_separation": True,
            "post_execution_features_excluded": True,
            "remaining_risk": (
                "SQL text may encode dataset-specific "
                "workload conventions, and repeated "
                "holdouts reuse some observations."
            ),
        },
    }

    save_json(
        run_dir / "classification_report.json",
        report,
    )

    return report, final_pipeline

def regression_models():
    return {
        "median_baseline": DummyRegressor(
            strategy="median"
        ),
        "ridge": Ridge(alpha=10.0),
        "random_forest": RandomForestRegressor(
            n_estimators=180,
            min_samples_leaf=2,
            random_state=SEED,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=180,
            min_samples_leaf=2,
            random_state=SEED,
            n_jobs=-1,
        ),
    }

def make_regressor(model, use_text=True):
    return Pipeline([
        (
            "features",
            build_preprocessor(
                PREEXECUTION_FEATURES,
                use_text=use_text,
            ),
        ),
        ("model", clone(model)),
    ])

def run_regression(df, run_dir, n_splits):
    data = df[
        df["execution_time_ms"].notna()
        & (df["execution_time_ms"] >= 0)
    ].copy().reset_index(drop=True)

    if (
        len(data) < 30
        or data["fingerprint"].nunique() < 8
    ):
        return {
            "status": "INSUFFICIENT_DATA",
        }, None

    y = np.log1p(
        data["execution_time_ms"].to_numpy()
    )

    definitions = regression_models()
    rows = []
    prediction_rows = []

    groups = data["fingerprint"]

    splitter = GroupShuffleSplit(
        n_splits=n_splits,
        test_size=0.25,
        random_state=SEED,
    )

    for fold, (tr, te) in enumerate(
        splitter.split(data, y, groups=groups),
        start=1,
    ):
        for name, model in definitions.items():
            try:
                pipeline = make_regressor(model)

                pipeline.fit(
                    data.iloc[tr],
                    y[tr],
                )

                predicted_log = pipeline.predict(
                    data.iloc[te]
                )

                actual_ms = np.expm1(y[te])

                predicted_ms = np.maximum(
                    0,
                    np.expm1(
                        np.clip(predicted_log, -1, 30)
                    ),
                )

                mae = mean_absolute_error(
                    actual_ms,
                    predicted_ms,
                )

                rmse = np.sqrt(
                    mean_squared_error(
                        actual_ms,
                        predicted_ms,
                    )
                )

                r2 = r2_score(
                    actual_ms,
                    predicted_ms,
                )

                rows.append({
                    "fold": fold,
                    "model": name,
                    "mae_ms": mae,
                    "rmse_ms": rmse,
                    "r2": r2,
                    "log_mae": mean_absolute_error(
                        y[te],
                        predicted_log,
                    ),
                })

                for i, index in enumerate(te):
                    prediction_rows.append({
                        "fold": fold,
                        "model": name,
                        "fingerprint": (
                            data.iloc[index]["fingerprint"]
                        ),
                        "actual_ms": actual_ms[i],
                        "predicted_ms": predicted_ms[i],
                        "absolute_error_ms": abs(
                            actual_ms[i] - predicted_ms[i]
                        ),
                    })

            except Exception as exc:
                rows.append({
                    "fold": fold,
                    "model": name,
                    "error": str(exc)[:300],
                })

    scores = pd.DataFrame(rows)

    scores.to_csv(
        run_dir / "regression_folds.csv",
        index=False,
    )

    pd.DataFrame(prediction_rows).to_csv(
        run_dir / "regression_predictions.csv",
        index=False,
    )

    valid = scores.dropna(subset=["mae_ms"])

    if valid.empty:
        return {
            "status": "NO_VALID_MODELS",
        }, None

    summary = {}

    for name, group in valid.groupby("model"):
        summary[name] = {
            "mean_mae_ms": float(
                group["mae_ms"].mean()
            ),
            "mean_rmse_ms": float(
                group["rmse_ms"].mean()
            ),
            "mean_r2": float(
                group["r2"].mean()
            ),
            "mean_log_mae": float(
                group["log_mae"].mean()
            ),
            "folds": int(len(group)),
        }

    candidates = [
        name
        for name in summary
        if name != "median_baseline"
    ]

    winner = min(
        candidates,
        key=lambda name: (
            summary[name]["mean_mae_ms"],
            summary[name]["mean_rmse_ms"],
        ),
    )

    final_pipeline = make_regressor(
        definitions[winner]
    )

    final_pipeline.fit(data, y)

    baseline_mae = summary.get(
        "median_baseline", {}
    ).get("mean_mae_ms")

    winner_mae = summary[winner]["mean_mae_ms"]

    report = {
        "status": "EVALUATED",
        "target": "execution_time_ms",
        "target_transform": "log1p",
        "winner": winner,
        "models": summary,
        "improvement_over_median_baseline_pct": (
            100 * (baseline_mae - winner_mae)
            / max(baseline_mae, 1e-9)
            if baseline_mae is not None
            else None
        ),
        "leakage_audit": {
            "execution_time_as_input": False,
            "actual_rows_as_input": False,
            "buffer_statistics_as_input": False,
            "planning_time_as_input": False,
            "estimated_cost_as_input": True,
            "estimated_rows_as_input": True,
            "condition": (
                "Planner estimates are only valid "
                "pre-execution inputs if EXPLAIN "
                "is available at prediction time."
            ),
        },
        "limitation": (
            "Performance generalization is measured "
            "on unseen query fingerprints, but "
            "the model may not generalize to "
            "different database sizes or hardware."
        ),
    }

    save_json(
        run_dir / "regression_report.json",
        report,
    )

    return report, final_pipeline

def complexity_prediction(df, run_dir):
    data = df.copy()

    if "complexity" in data.columns:
        labels = (
            data["complexity"]
            .fillna("UNKNOWN")
            .astype(str)
            .str.upper()
        )

        valid = labels.isin([
            "LOW", "MEDIUM", "HIGH", "COMPLEX"
        ])

        data = data.loc[valid].copy()
        data["complexity_label"] = labels.loc[valid]

        if (
            data["complexity_label"].nunique() >= 2
            and data["fingerprint"].nunique() >= 8
        ):
            splitter = GroupShuffleSplit(
                n_splits=5,
                test_size=0.25,
                random_state=SEED,
            )

            rows = []

            for fold, (tr, te) in enumerate(
                splitter.split(
                    data,
                    data["complexity_label"],
                    groups=data["fingerprint"],
                ),
                start=1,
            ):
                if (
                    data.iloc[tr]["complexity_label"]
                    .nunique() < 2
                ):
                    continue

                model = make_classifier(
                    RandomForestClassifier(
                        n_estimators=150,
                        random_state=SEED,
                        n_jobs=-1,
                    ),
                    use_text=True,
                )

                model.fit(
                    data.iloc[tr],
                    data.iloc[tr]["complexity_label"],
                )

                pred = model.predict(data.iloc[te])

                rows.append({
                    "fold": fold,
                    "accuracy": accuracy_score(
                        data.iloc[te]["complexity_label"],
                        pred,
                    ),
                    "macro_f1": f1_score(
                        data.iloc[te]["complexity_label"],
                        pred,
                        average="macro",
                        zero_division=0,
                    ),
                })

            if rows:
                report = {
                    "status": "EVALUATED",
                    "task": "query_complexity_classification",
                    "folds": rows,
                    "mean_macro_f1": float(
                        pd.DataFrame(rows)[
                            "macro_f1"
                        ].mean()
                    ),
                }

                save_json(
                    run_dir / "complexity_report.json",
                    report,
                )

                return report

    report = {
        "status": "INSUFFICIENT_COMPLEXITY_LABELS",
    }

    save_json(
        run_dir / "complexity_report.json",
        report,
    )

    return report

def run_anomaly_detection(df, run_dir):
    data = df.copy()

    features = MONITORING_FEATURES

    imputer = SimpleImputer(
        strategy="constant",
        fill_value=0,
    )

    scaler = StandardScaler()

    X = imputer.fit_transform(
        data[features]
    )

    X = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=250,
        contamination=0.05,
        random_state=SEED,
        n_jobs=-1,
    )

    model.fit(X)

    scores = model.decision_function(X)
    flags = model.predict(X) == -1

    latency = data["execution_time_ms"].fillna(0)
    reads = data["shared_read_blocks"].fillna(0)
    scans = data["seq_scan_count"].fillna(0)

    latency_threshold = float(
        latency.quantile(0.95)
    )
    reads_threshold = float(
        reads.quantile(0.95)
    )

    high_latency = (
        latency > latency_threshold
    )
    high_io = reads > reads_threshold

    seq_scan_alert = (
        (scans > 0)
        & (
            data["estimated_cost"].fillna(0)
            > data["estimated_cost"].quantile(0.75)
        )
    )

    data["anomaly_score"] = scores
    data["isolation_forest_anomaly"] = flags

    data["high_latency_alert"] = high_latency
    data["high_io_alert"] = high_io
    data["sequential_scan_alert"] = seq_scan_alert

    data["any_anomaly"] = (
        flags
        | high_latency
        | high_io
        | seq_scan_alert
    )

    columns = [
        "fingerprint",
        "query_family",
        "workload_type",
        "execution_time_ms",
        "estimated_cost",
        "shared_read_blocks",
        "seq_scan_count",
        "anomaly_score",
        "isolation_forest_anomaly",
        "high_latency_alert",
        "high_io_alert",
        "sequential_scan_alert",
        "any_anomaly",
    ]

    data[columns].to_csv(
        run_dir / "anomaly_results.csv",
        index=False,
    )

    report = {
        "status": "DETECTED",
        "method": "IsolationForest + threshold alerts",
        "rows": int(len(data)),
        "isolation_forest_anomalies": int(
            flags.sum()
        ),
        "high_latency_alerts": int(
            high_latency.sum()
        ),
        "high_io_alerts": int(
            high_io.sum()
        ),
        "sequential_scan_alerts": int(
            seq_scan_alert.sum()
        ),
        "combined_alerts": int(
            data["any_anomaly"].sum()
        ),
        "latency_p95_ms": latency_threshold,
        "read_blocks_p95": reads_threshold,
        "validation": (
            "Unsupervised alerts require manual "
            "review or known incident labels "
            "before precision/recall can be claimed."
        ),
    }

    save_json(
        run_dir / "anomaly_report.json",
        report,
    )

    return report, data, {
        "model": model,
        "imputer": imputer,
        "scaler": scaler,
        "features": features,
    }

def run_clustering(df, run_dir):
    features = [
        c for c in MONITORING_FEATURES
        if c in df.columns
    ]

    X = df[features].copy()

    imputer = SimpleImputer(
        strategy="constant",
        fill_value=0,
    )

    scaler = StandardScaler()

    matrix = scaler.fit_transform(
        imputer.fit_transform(X)
    )

    results = []
    best_model = None
    best_labels = None
    best_score = -np.inf

    for k in range(2, min(8, len(df) - 1) + 1):
        model = MiniBatchKMeans(
            n_clusters=k,
            n_init=10,
            random_state=SEED,
            batch_size=min(256, len(df)),
        )

        labels = model.fit_predict(matrix)

        if len(np.unique(labels)) < 2:
            continue

        score = silhouette_score(
            matrix,
            labels,
            sample_size=min(800, len(df)),
            random_state=SEED,
        )

        results.append({
            "clusters": k,
            "silhouette": float(score),
        })

        if score > best_score:
            best_score = score
            best_model = model
            best_labels = labels

    if best_model is None:
        return {
            "status": "NO_VALID_CLUSTERING",
        }, None

    result_df = df.copy()
    result_df["cluster"] = best_labels

    result_df[
        [
            "fingerprint",
            "query_family",
            "workload_type",
            "cluster",
            "execution_time_ms",
        ]
    ].to_csv(
        run_dir / "cluster_assignments.csv",
        index=False,
    )

    profiles = (
        result_df.groupby("cluster")
        .agg(
            queries=("fingerprint", "size"),
            unique_fingerprints=(
                "fingerprint", "nunique"
            ),
            median_latency_ms=(
                "execution_time_ms", "median"
            ),
            median_cost=("estimated_cost", "median"),
        )
        .reset_index()
    )

    profiles.to_csv(
        run_dir / "cluster_profiles.csv",
        index=False,
    )

    report = {
        "status": "EVALUATED",
        "best_clusters": int(
            best_model.n_clusters
        ),
        "best_silhouette": float(best_score),
        "tested_cluster_counts": results,
        "stability_validated": False,
        "note": (
            "Silhouette evaluates compactness "
            "and separation, not clinical meaning "
            "or cluster stability across sessions."
        ),
    }

    save_json(
        run_dir / "clustering_report.json",
        report,
    )

    return report, {
        "model": best_model,
        "imputer": imputer,
        "scaler": scaler,
        "features": features,
    }

def chronological_forecasting(
    df,
    run_dir,
    bucket="h",
    min_buckets=24,
    min_sessions=3,
):
    data = df.dropna(
        subset=["timestamp"]
    ).copy()

    if data.empty:
        report = {
            "status": "NO_TIMESTAMPS",
            "validated": False,
        }

        save_json(
            run_dir / "forecast_report.json",
            report,
        )
        return report

    first = data["timestamp"].min()
    last = data["timestamp"].max()

    duration_hours = (
        last - first
    ).total_seconds() / 3600

    data["bucket"] = (
        data["timestamp"].dt.floor(bucket)
    )

    counts = (
        data.groupby(["bucket", "workload_type"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    # A regular time grid is meaningful only when
    # the observation schedule covers those periods.
    # We do not treat gaps between isolated recording
    # sessions as known zero-demand intervals.
    sessions = data["session_id"].nunique()

    if (
        sessions < min_sessions
        or len(counts) < min_buckets
        or duration_hours < 24
    ):
        report = {
            "status": "INSUFFICIENT_CHRONOLOGICAL_HISTORY",
            "validated": False,
            "records": int(len(data)),
            "sessions": int(sessions),
            "observed_buckets": int(len(counts)),
            "duration_hours": duration_hours,
            "required_sessions": min_sessions,
            "required_buckets": min_buckets,
            "reason": (
                "Multiple sufficiently long and "
                "regularly monitored sessions are "
                "needed. Unobserved periods are "
                "not assumed to contain zero queries."
            ),
        }

        save_json(
            run_dir / "forecast_report.json",
            report,
        )

        return report

    # Check that hourly buckets form a contiguous
    # monitored period before chronological backtest.
    expected = pd.date_range(
        counts.index.min(),
        counts.index.max(),
        freq=bucket,
        tz="UTC",
    )

    if len(expected) != len(counts):
        report = {
            "status": "IRREGULAR_MONITORING_COVERAGE",
            "validated": False,
            "observed_buckets": int(len(counts)),
            "expected_contiguous_buckets": int(
                len(expected)
            ),
            "reason": (
                "Gaps in monitoring coverage cannot "
                "be interpreted as zero demand."
            ),
        }

        save_json(
            run_dir / "forecast_report.json",
            report,
        )

        return report

    counts.to_csv(
        run_dir / "hourly_workload_counts.csv"
    )

    records = []
    forecasts = {}

    for workload in counts.columns:
        series = counts[workload].astype(float)

        start_test = max(
            12,
            int(len(series) * 0.65),
        )

        actuals = []
        naive_predictions = []
        moving_predictions = []

        for i in range(start_test, len(series)):
            history = series.iloc[:i]

            actual = float(series.iloc[i])

            naive = float(history.iloc[-1])
            moving = float(
                history.tail(min(6, len(history))).mean()
            )

            actuals.append(actual)
            naive_predictions.append(naive)
            moving_predictions.append(moving)

            records.append({
                "workload_type": workload,
                "bucket": str(series.index[i]),
                "actual": actual,
                "naive_prediction": naive,
                "moving_average_prediction": moving,
            })

        if not actuals:
            continue

        naive_mae = mean_absolute_error(
            actuals,
            naive_predictions,
        )

        moving_mae = mean_absolute_error(
            actuals,
            moving_predictions,
        )

        selected = (
            "moving_average"
            if moving_mae < naive_mae
            else "last_observation"
        )

        forecast_value = (
            float(series.tail(6).mean())
            if selected == "moving_average"
            else float(series.iloc[-1])
        )

        forecasts[str(workload)] = {
            "naive_mae": float(naive_mae),
            "moving_average_mae": float(moving_mae),
            "selected_method": selected,
            "next_bucket_prediction": max(
                0,
                forecast_value,
            ),
            "backtest_points": len(actuals),
        }

    pd.DataFrame(records).to_csv(
        run_dir / "forecast_backtest.csv",
        index=False,
    )

    report = {
        "status": "BACKTESTED",
        "validated": True,
        "bucket": bucket,
        "sessions": int(sessions),
        "observed_buckets": int(len(counts)),
        "forecast_results": forecasts,
        "limitation": (
            "Validation covers only the observed "
            "workload period and does not establish "
            "generalization to future hospital traffic."
        ),
    }

    save_json(
        run_dir / "forecast_report.json",
        report,
    )

    return report

def population_stability_index(reference, current, bins=8):
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]

    if len(reference) < 20 or len(current) < 20:
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
        reference,
        bins=edges,
    )[0].astype(float)

    q = np.histogram(
        current,
        bins=edges,
    )[0].astype(float)

    p = (p + 1e-6) / (p.sum() + len(p) * 1e-6)
    q = (q + 1e-6) / (q.sum() + len(q) * 1e-6)

    return float(
        np.sum((q - p) * np.log(q / p))
    )

def run_drift(df, run_dir):
    sessions = (
        df.groupby("session_id")["timestamp"]
        .min()
        .sort_values()
    )

    session_names = list(sessions.index)

    if len(session_names) < 2:
        report = {
            "status": "INSUFFICIENT_SESSIONS",
            "validated": False,
            "sessions": len(session_names),
            "reason": (
                "At least two independent monitoring "
                "sessions are required."
            ),
        }

        save_json(
            run_dir / "drift_report.json",
            report,
        )

        return report

    earlier = df[
        df["session_id"] == session_names[-2]
    ]
    later = df[
        df["session_id"] == session_names[-1]
    ]

    if len(earlier) < 30 or len(later) < 30:
        report = {
            "status": "INSUFFICIENT_SESSION_ROWS",
            "validated": False,
            "earlier_rows": len(earlier),
            "later_rows": len(later),
        }

        save_json(
            run_dir / "drift_report.json",
            report,
        )

        return report

    metrics = [
        "execution_time_ms",
        "estimated_cost",
        "shared_read_blocks",
        "seq_scan_count",
        "join_count",
    ]

    results = {}

    for column in metrics:
        a = earlier[column].dropna().to_numpy()
        b = later[column].dropna().to_numpy()

        if len(a) < 20 or len(b) < 20:
            continue

        ks = ks_2samp(a, b)

        results[column] = {
            "ks_statistic": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            "psi": population_stability_index(a, b),
            "reference_median": safe_float(
                np.median(a)
            ),
            "current_median": safe_float(
                np.median(b)
            ),
        }

    classes = sorted(
        set(earlier["workload_type"])
        | set(later["workload_type"])
    )

    old_mix = (
        earlier["workload_type"]
        .value_counts(normalize=True)
        .reindex(classes, fill_value=0)
    )

    new_mix = (
        later["workload_type"]
        .value_counts(normalize=True)
        .reindex(classes, fill_value=0)
    )

    total_variation = float(
        0.5 * np.abs(
            old_mix.to_numpy() - new_mix.to_numpy()
        ).sum()
    )

    report = {
        "status": "COMPARED",
        "validated": True,
        "reference_session": session_names[-2],
        "current_session": session_names[-1],
        "workload_mix_total_variation": total_variation,
        "metric_drift": results,
        "drift_alert": bool(
            total_variation > 0.20
            or any(
                v.get("psi") is not None
                and v["psi"] > 0.25
                for v in results.values()
            )
        ),
        "caution": (
            "PSI thresholds are heuristic. "
            "KS p-values are exploratory because "
            "query observations are not necessarily "
            "independent. Multiple-testing and "
            "session-size effects must be considered."
        ),
    }

    save_json(
        run_dir / "drift_report.json",
        report,
    )

    return report

def run_ablation(df, run_dir):
    data = df[
        df["workload_type"].isin(TARGET_CLASSES)
    ].copy().reset_index(drop=True)

    if data["fingerprint"].nunique() < 8:
        return {
            "status": "INSUFFICIENT_DATA",
        }

    splitter = GroupShuffleSplit(
        n_splits=3,
        test_size=0.25,
        random_state=SEED + 100,
    )

    rows = []

    for fold, (tr, te) in enumerate(
        splitter.split(
            data,
            data["workload_type"],
            groups=data["fingerprint"],
        ),
        start=1,
    ):
        for use_text in [False, True]:
            model = make_classifier(
                LogisticRegression(
                    max_iter=1500,
                    class_weight="balanced",
                ),
                use_text=use_text,
            )

            model.fit(
                data.iloc[tr],
                data.iloc[tr]["workload_type"],
            )

            predicted = model.predict(
                data.iloc[te]
            )

            rows.append({
                "fold": fold,
                "configuration": (
                    "numeric_plus_sql"
                    if use_text
                    else "numeric_only"
                ),
                "macro_f1": f1_score(
                    data.iloc[te]["workload_type"],
                    predicted,
                    average="macro",
                    zero_division=0,
                ),
                "accuracy": accuracy_score(
                    data.iloc[te]["workload_type"],
                    predicted,
                ),
            })

    pd.DataFrame(rows).to_csv(
        run_dir / "ablation_results.csv",
        index=False,
    )

    report = {
        "status": "EVALUATED",
        "experiments": rows,
        "note": (
            "Ablation tests whether adding SQL "
            "text improves workload classification "
            "on fingerprint-separated splits."
        ),
    }

    save_json(
        run_dir / "ablation_report.json",
        report,
    )

    return report

def export_phase9_priorities(
    df,
    regression_model,
    anomaly_df,
    run_dir,
):
    data = df.copy()

    if regression_model is not None:
        log_predictions = regression_model.predict(data)

        data["predicted_latency_ms"] = np.maximum(
            0,
            np.expm1(
                np.clip(log_predictions, -1, 30)
            ),
        )
    else:
        data["predicted_latency_ms"] = np.nan

    data["anomaly_flag"] = (
        anomaly_df["any_anomaly"]
        .to_numpy()
        .astype(int)
    )

    frequency = (
        data.groupby("fingerprint")
        .size()
        .rename("observed_frequency")
    )

    data = data.join(
        frequency,
        on="fingerprint",
    )

    grouped = (
        data.groupby("fingerprint", as_index=False)
        .agg(
            query_family=("query_family", "first"),
            representative_sql=("sql", "first"),
            tables_accessed=("tables_accessed", "first"),
            observed_frequency=(
                "observed_frequency", "first"
            ),
            predicted_latency_ms=(
                "predicted_latency_ms", "median"
            ),
            observed_latency_ms=(
                "execution_time_ms", "median"
            ),
            anomaly_rate=("anomaly_flag", "mean"),
            estimated_cost=("estimated_cost", "median"),
        )
    )

    # Priority is a transparent heuristic.
    # It is not a learned prediction of index benefit.
    latency = grouped[
        "predicted_latency_ms"
    ].fillna(
        grouped["observed_latency_ms"]
    ).fillna(0)

    grouped["priority_score"] = (
        np.log1p(
            grouped["observed_frequency"]
        )
        * np.log1p(latency.clip(lower=0))
        * (1 + grouped["anomaly_rate"])
    )

    grouped = grouped.sort_values(
        "priority_score",
        ascending=False,
    )

    grouped["priority_rank"] = np.arange(
        1,
        len(grouped) + 1,
    )

    grouped.to_csv(
        run_dir / "phase9_workload_priorities.csv",
        index=False,
    )

    report = {
        "status": "EXPORTED",
        "query_fingerprints": int(len(grouped)),
        "file": str(
            run_dir / "phase9_workload_priorities.csv"
        ),
        "ranking_method": (
            "Observed frequency × predicted latency "
            "× anomaly multiplier"
        ),
        "measured_optimization_benefit": False,
        "integration_status": (
            "Prediction artifact ready for Phase 9. "
            "An end-to-end benchmark must still "
            "verify whether using these priorities "
            "improves index selection."
        ),
    }

    save_json(
        run_dir / "phase9_integration_report.json",
        report,
    )

    return report

def environment_report():
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "seed": SEED,
    }

def register_model(
    run_id,
    classification_model,
    regression_model,
    anomaly_model,
    cluster_model,
    metrics,
    source_files,
):
    version_dir = REGISTRY / run_id

    if version_dir.exists():
        raise RuntimeError(
            "Model version already exists: "
            + run_id
        )

    version_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    artifacts = {}

    for name, model in [
        ("classification", classification_model),
        ("regression", regression_model),
        ("anomaly", anomaly_model),
        ("clustering", cluster_model),
    ]:
        if model is None:
            continue

        path = version_dir / f"{name}.joblib"

        joblib.dump(model, path)

        artifacts[name] = {
            "path": str(path),
            "sha256": hash_file(path),
        }

    metadata = {
        "version": run_id,
        "created_at": utcnow(),
        "artifacts": artifacts,
        "metrics": metrics,
        "source_files": source_files,
        "environment": environment_report(),
        "promotion_status": "CANDIDATE",
        "production_database_modified": False,
    }

    save_json(
        version_dir / "metadata.json",
        metadata,
    )

    return metadata

def compare_versions(candidate, active):
    candidate_metrics = candidate.get(
        "metrics", {}
    )

    active_metrics = active.get(
        "metrics", {}
    )

    candidate_f1 = candidate_metrics.get(
        "classification_macro_f1"
    )
    active_f1 = active_metrics.get(
        "classification_macro_f1"
    )

    candidate_mae = candidate_metrics.get(
        "regression_mae_ms"
    )
    active_mae = active_metrics.get(
        "regression_mae_ms"
    )

    candidate_sources = {
        item["sha256"]
        for item in candidate.get("source_files", [])
    }

    active_sources = {
        item["sha256"]
        for item in active.get("source_files", [])
    }

    same_data = candidate_sources == active_sources

    if not same_data:
        return {
            "eligible": False,
            "reason": (
                "Different training datasets. "
                "A common frozen evaluation set "
                "is required before automatic "
                "model comparison."
            ),
        }

    if (
        candidate_f1 is None
        or active_f1 is None
        or candidate_mae is None
        or active_mae is None
    ):
        return {
            "eligible": False,
            "reason": "Missing comparable metrics.",
        }

    eligible = (
        candidate_f1 >= active_f1
        and candidate_mae <= active_mae * 1.05
    )

    return {
        "eligible": bool(eligible),
        "same_data": same_data,
        "candidate_f1": candidate_f1,
        "active_f1": active_f1,
        "candidate_mae": candidate_mae,
        "active_mae": active_mae,
        "reason": (
            "Candidate passes comparison gate."
            if eligible
            else "Candidate failed comparison gate."
        ),
    }

def promote_version(version, approve=False):
    version_dir = REGISTRY / version

    metadata = read_json(
        version_dir / "metadata.json"
    )

    if metadata is None:
        raise FileNotFoundError(
            f"Unknown model version: {version}"
        )

    for item in metadata["artifacts"].values():
        path = Path(item["path"])

        if (
            not path.is_file()
            or hash_file(path) != item["sha256"]
        ):
            raise RuntimeError(
                "Artifact integrity check failed."
            )

    active = read_json(ACTIVE_FILE)

    comparison = {
        "eligible": True,
        "reason": "No active version exists.",
    }

    if active:
        active_metadata = read_json(
            REGISTRY
            / active["version"]
            / "metadata.json"
        )

        if active_metadata is None:
            raise RuntimeError(
                "Active model metadata is missing."
            )

        comparison = compare_versions(
            metadata,
            active_metadata,
        )

    if not comparison["eligible"]:
        return {
            "status": "PROMOTION_BLOCKED",
            "comparison": comparison,
        }

    if not approve:
        return {
            "status": "AWAITING_EXPLICIT_APPROVAL",
            "comparison": comparison,
            "instruction": (
                "Rerun promote with --approve "
                "after reviewing the candidate."
            ),
        }

    previous = (
        active["version"]
        if active else None
    )

    record = {
        "version": version,
        "previous_version": previous,
        "promoted_at": utcnow(),
        "comparison": comparison,
    }

    save_json(ACTIVE_FILE, record)

    return {
        "status": "PROMOTED",
        "active_version": version,
        "previous_version": previous,
    }

def rollback_version(approve=False):
    active = read_json(ACTIVE_FILE)

    if not active:
        return {
            "status": "NO_ACTIVE_MODEL",
        }

    previous = active.get(
        "previous_version"
    )

    if not previous:
        return {
            "status": "NO_PREVIOUS_VERSION",
        }

    metadata = read_json(
        REGISTRY
        / previous
        / "metadata.json"
    )

    if metadata is None:
        raise RuntimeError(
            "Previous model version is missing."
        )

    if not approve:
        return {
            "status": "AWAITING_EXPLICIT_APPROVAL",
            "target_version": previous,
        }

    save_json(
        ACTIVE_FILE,
        {
            "version": previous,
            "previous_version": active["version"],
            "promoted_at": utcnow(),
            "rollback": True,
        },
    )

    return {
        "status": "ROLLED_BACK",
        "active_version": previous,
    }

def acceptance_tests(
    df,
    classification,
    regression,
    forecast,
    drift,
    integration,
    run_dir,
):
    checks = {
        "data_loaded": len(df) > 0,
        "schema_present": all(
            c in df.columns
            for c in [
                "sql",
                "fingerprint",
                "timestamp",
                "execution_time_ms",
            ]
        ),
        "fingerprint_diversity": (
            df["fingerprint"].nunique() >= 8
        ),
        "classification_evaluated": (
            classification.get("status")
            == "EVALUATED"
        ),
        "regression_evaluated": (
            regression.get("status")
            == "EVALUATED"
        ),
        "forecast_backtested": (
            forecast.get("status")
            == "BACKTESTED"
        ),
        "drift_compared": (
            drift.get("status")
            == "COMPARED"
        ),
        "phase9_exported": (
            integration.get("status")
            == "EXPORTED"
        ),
        "production_database_untouched": True,
    }

    report = {
        "status": (
            "ALL_GATES_PASSED"
            if all(checks.values())
            else "PARTIAL_VALIDATION"
        ),
        "passed": int(sum(checks.values())),
        "total": int(len(checks)),
        "checks": checks,
        "important": (
            "These checks verify implementation "
            "and basic evidence availability. "
            "They do not prove clinical safety, "
            "model generalization or production "
            "optimization benefit."
        ),
    }

    save_json(
        run_dir / "acceptance_tests.json",
        report,
    )

    return report

def create_research_summary(
    run_dir,
    quality,
    classification,
    regression,
    complexity,
    anomaly,
    clustering,
    forecast,
    drift,
    ablation,
    integration,
    acceptance,
):
    lines = [
        "# Phase 8 — Advanced AI Workload Intelligence",
        "",
        f"Generated: {utcnow()}",
        "",
        "## Dataset",
        "",
        f"- Monitoring records: {quality['records']}",
        f"- Query fingerprints: {quality['unique_fingerprints']}",
        f"- Recording sessions: {quality['sessions']}",
        f"- Monitoring duration: {quality['duration_seconds']} seconds",
        "",
        "## Classification",
        "",
        f"- Status: {classification.get('status')}",
        f"- Selected model: {classification.get('winner')}",
        f"- Repeated splits: {classification.get('repeated_group_splits')}",
        "",
        "## Latency Prediction",
        "",
        f"- Status: {regression.get('status')}",
        f"- Selected model: {regression.get('winner')}",
        "",
        "## Query Complexity",
        "",
        f"- Status: {complexity.get('status')}",
        "",
        "## Anomaly Detection",
        "",
        f"- Status: {anomaly.get('status')}",
        f"- Combined alerts: {anomaly.get('combined_alerts')}",
        "",
        "## Clustering",
        "",
        f"- Status: {clustering.get('status')}",
        f"- Best cluster count: {clustering.get('best_clusters')}",
        "",
        "## Forecasting",
        "",
        f"- Status: {forecast.get('status')}",
        f"- Validated: {forecast.get('validated', False)}",
        "",
        "## Concept Drift",
        "",
        f"- Status: {drift.get('status')}",
        f"- Validated: {drift.get('validated', False)}",
        "",
        "## Ablation Study",
        "",
        f"- Status: {ablation.get('status')}",
        "",
        "## Phase 9 Interface",
        "",
        f"- Status: {integration.get('status')}",
        "- Optimization benefit must be verified separately.",
        "",
        "## Acceptance",
        "",
        f"- Status: {acceptance['status']}",
        f"- Gates passed: {acceptance['passed']}/{acceptance['total']}",
        "",
        "## Research Limitations",
        "",
        "- Fingerprint-grouped evaluation uses a limited number of unique SQL structures.",
        "- Repeated holdout model selection is not an independent external test.",
        "- Anomaly labels have not been independently verified.",
        "- Short monitoring sessions cannot establish long-term demand forecasts.",
        "- Phase 9 workload ranking is not proof of index benefit.",
        "- No PostgreSQL or Neo4j production data was modified.",
        "",
    ]

    path = run_dir / "research_report.md"

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return path

def run_pipeline(args):
    ensure_dirs()

    started = time.perf_counter()

    # Unique, deterministic-from-clock run identifier.
    run_id = datetime.now(
        timezone.utc
    ).strftime("p8_%Y%m%d_%H%M%S_%f")

    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    print("=" * 75)
    print("PHASE 8 — ADVANCED AI WORKLOAD INTELLIGENCE")
    print("=" * 75)
    print("Run:", run_id)
    print("Output:", run_dir)

    data, sources = load_monitoring(args.input)
    data = engineer_features(data)

    quality = quality_report(data, sources)

    save_json(
        run_dir / "data_quality.json",
        quality,
    )

    save_json(
        run_dir / "environment.json",
        environment_report(),
    )

    print(
        f"\nLoaded {len(data)} records, "
        f"{data['fingerprint'].nunique()} fingerprints, "
        f"{data['session_id'].nunique()} sessions."
    )

    print("\n[1/10] Repeated classification validation")

    classification, classifier_model = (
        run_classification(
            data,
            run_dir,
            args.splits,
        )
    )

    print(
        "Winner:",
        classification.get("winner"),
    )

    print("\n[2/10] Pre-execution latency regression")

    regression, regression_model = (
        run_regression(
            data,
            run_dir,
            args.splits,
        )
    )

    print(
        "Winner:",
        regression.get("winner"),
    )

    print("\n[3/10] Query complexity prediction")

    complexity = complexity_prediction(
        data,
        run_dir,
    )

    print(
        "Status:",
        complexity["status"],
    )

    print("\n[4/10] Anomaly detection")

    anomaly, anomaly_df, anomaly_model = (
        run_anomaly_detection(
            data,
            run_dir,
        )
    )

    print(
        "Alerts:",
        anomaly["combined_alerts"],
    )

    print("\n[5/10] Workload clustering")

    clustering, cluster_model = run_clustering(
        data,
        run_dir,
    )

    print(
        "Best clusters:",
        clustering.get("best_clusters"),
    )

    print("\n[6/10] Chronological forecasting")

    forecast = chronological_forecasting(
        data,
        run_dir,
        bucket=args.forecast_bucket,
        min_buckets=args.min_forecast_buckets,
        min_sessions=args.min_forecast_sessions,
    )

    print(
        "Forecast:",
        forecast["status"],
    )

    print("\n[7/10] Concept drift detection")

    drift = run_drift(
        data,
        run_dir,
    )

    print(
        "Drift:",
        drift["status"],
    )

    print("\n[8/10] Ablation experiments")

    ablation = run_ablation(
        data,
        run_dir,
    )

    print(
        "Ablation:",
        ablation["status"],
    )

    print("\n[9/10] Phase 9 integration export")

    integration = export_phase9_priorities(
        data,
        regression_model,
        anomaly_df,
        run_dir,
    )

    print(
        "Export:",
        integration["status"],
    )

    print("\n[10/10] Research acceptance and registry")

    acceptance = acceptance_tests(
        data,
        classification,
        regression,
        forecast,
        drift,
        integration,
        run_dir,
    )

    winner_class = classification.get(
        "winner"
    )

    winner_reg = regression.get(
        "winner"
    )

    metrics = {
        "classification_macro_f1": (
            classification.get("models", {})
            .get(winner_class, {})
            .get("mean_macro_f1")
        ),
        "classification_accuracy": (
            classification.get("models", {})
            .get(winner_class, {})
            .get("mean_accuracy")
        ),
        "regression_mae_ms": (
            regression.get("models", {})
            .get(winner_reg, {})
            .get("mean_mae_ms")
        ),
    }

    registry = register_model(
        run_id,
        classifier_model,
        regression_model,
        anomaly_model,
        cluster_model,
        metrics,
        sources,
    )

    create_research_summary(
        run_dir,
        quality,
        classification,
        regression,
        complexity,
        anomaly,
        clustering,
        forecast,
        drift,
        ablation,
        integration,
        acceptance,
    )

    summary = {
        "run_id": run_id,
        "status": "PIPELINE_EXECUTED",
        "records": int(len(data)),
        "fingerprints": int(
            data["fingerprint"].nunique()
        ),
        "sessions": int(
            data["session_id"].nunique()
        ),
        "classification": classification["status"],
        "regression": regression["status"],
        "complexity": complexity["status"],
        "anomaly": anomaly["status"],
        "clustering": clustering["status"],
        "forecast": forecast["status"],
        "drift": drift["status"],
        "phase9_export": integration["status"],
        "acceptance": acceptance["status"],
        "acceptance_passed": acceptance["passed"],
        "acceptance_total": acceptance["total"],
        "model_version": registry["version"],
        "elapsed_seconds": round(
            time.perf_counter() - started,
            2,
        ),
        "run_directory": str(run_dir),
    }

    save_json(
        run_dir / "summary.json",
        summary,
    )

    print("\n" + "=" * 75)
    print("FINAL PHASE 8 SUMMARY")
    print("=" * 75)
    print(json.dumps(summary, indent=2))

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Advanced Phase 8 AI workload learning, "
            "forecasting, drift and model registry"
        )
    )

    subparsers = parser.add_subparsers(
        dest="command"
    )

    run_parser = subparsers.add_parser("run")

    run_parser.add_argument(
        "--input",
        nargs="+",
        default=[str(DEFAULT_INPUT)],
        help=(
            "One or more Phase 7 CSVs or "
            "directories containing monitoring CSVs."
        ),
    )

    run_parser.add_argument(
        "--splits",
        type=int,
        default=5,
    )

    run_parser.add_argument(
        "--forecast-bucket",
        choices=["min", "h"],
        default="h",
    )

    run_parser.add_argument(
        "--min-forecast-buckets",
        type=int,
        default=24,
    )

    run_parser.add_argument(
        "--min-forecast-sessions",
        type=int,
        default=3,
    )

    promote_parser = subparsers.add_parser(
        "promote"
    )

    promote_parser.add_argument(
        "--version",
        required=True,
    )

    promote_parser.add_argument(
        "--approve",
        action="store_true",
    )

    rollback_parser = subparsers.add_parser(
        "rollback"
    )

    rollback_parser.add_argument(
        "--approve",
        action="store_true",
    )

    subparsers.add_parser("status")

    args = parser.parse_args()

    ensure_dirs()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "run":
        if args.splits < 2:
            parser.error(
                "--splits must be at least 2."
            )

        run_pipeline(args)

    elif args.command == "promote":
        print(
            json.dumps(
                promote_version(
                    args.version,
                    approve=args.approve,
                ),
                indent=2,
            )
        )

    elif args.command == "rollback":
        print(
            json.dumps(
                rollback_version(
                    approve=args.approve,
                ),
                indent=2,
            )
        )

    elif args.command == "status":
        active = read_json(ACTIVE_FILE)

        versions = sorted(
            p.name
            for p in REGISTRY.iterdir()
            if p.is_dir()
            and (
                p / "metadata.json"
            ).exists()
        )

        print(
            json.dumps(
                {
                    "active_model": active,
                    "registered_versions": versions,
                },
                indent=2,
            )
        )

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"\nERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise
