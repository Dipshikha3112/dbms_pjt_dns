
from __future__ import annotations

import argparse
import hashlib
import json
import math
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

from catboost import CatBoostClassifier, CatBoostRegressor
from scipy.stats import spearmanr, wilcoxon
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNet,
    HuberRegressor,
    LogisticRegression,
    Ridge,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVR

warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42
EPS = 1e-6

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = (
    ROOT / "results" / "phase7_monitoring"
    / "phase7_query_monitoring.csv"
)
OUTPUT_ROOT = ROOT / "results" / "phase8_v4"

SQL_FEATURES = [
    "sql_length",
    "token_count",
    "join_count",
    "where_count",
    "group_count",
    "order_count",
    "having_count",
    "aggregate_count",
    "subquery_count",
    "predicate_count",
    "table_count",
    "distinct_count",
    "limit_count",
    "has_cte",
    "has_union",
    "has_window",
    "has_exists",
    "has_like",
    "has_between",
    "has_in",
    "has_or",
    "has_case",
    "has_offset",
    "has_left_join",
    "has_right_join",
    "has_cross_join",
    "has_null",
]

PLANNER_ALIASES = {
    "estimated_cost": [
        "estimated_cost",
        "total_cost",
        "plan_total_cost",
        "estimated_total_cost",
    ],
    "startup_cost": [
        "startup_cost",
        "plan_startup_cost",
    ],
    "estimated_rows": [
        "estimated_rows",
        "plan_rows",
        "estimated_plan_rows",
    ],
    "estimated_width": [
        "estimated_width",
        "plan_width",
    ],
    "plan_node_count": [
        "plan_node_count",
        "estimated_plan_node_count",
    ],
}

PLANNER_FEATURES = list(PLANNER_ALIASES)

FORBIDDEN = {
    "execution_time_ms",
    "planning_time_ms",
    "actual_rows",
    "actual_loops",
    "actual_total_time",
    "shared_hit_blocks",
    "shared_read_blocks",
    "shared_written_blocks",
    "shared_dirtied_blocks",
    "temp_read_blocks",
    "temp_written_blocks",
    "workload_type",
    "target_class",
    "query_family",
    "query_name",
    "fingerprint",
    "fingerprint_hash",
    "session_id",
    "timestamp",
    "source_file",
}

def native(x):
    if isinstance(x, dict):
        return {str(k): native(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [native(v) for v in x]
    if isinstance(x, np.ndarray):
        return native(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        x = float(x)
        return x if np.isfinite(x) else None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (pd.Timestamp, datetime)):
        return str(x)
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x

def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(native(data), indent=2, allow_nan=False),
        encoding="utf-8",
    )

def pick_column(df, choices):
    for name in choices:
        if name in df.columns:
            return name
    return None

def normalize_sql(value):
    s = str(value or "")
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.S)
    s = re.sub(r"--[^\n]*", " ", s)
    s = re.sub(r"'(?:''|[^'])*'", " ? ", s)
    s = re.sub(r"\b\d+(?:\.\d+)?\b", " ? ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()

def sql_hash(sql):
    return hashlib.sha256(
        normalize_sql(sql).encode("utf-8")
    ).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()

def load_dataset(paths):
    frames = []
    sources = []

    for raw in paths:
        path = Path(raw).expanduser().resolve()
        files = sorted(path.glob("*.csv")) if path.is_dir() else [path]

        for file in files:
            if not file.exists():
                raise FileNotFoundError(file)

            frame = pd.read_csv(file, low_memory=False)

            sql_col = pick_column(
                frame, ["sql", "query_sql", "sql_text", "query_text"]
            )
            latency_col = pick_column(
                frame,
                ["execution_time_ms", "latency_ms", "duration_ms"]
            )

            if sql_col is None or latency_col is None:
                if path.is_dir():
                    continue
                raise ValueError(
                    f"Required SQL/latency columns missing: {list(frame.columns)}"
                )

            frame["sql"] = frame[sql_col].fillna("").astype(str)
            frame["execution_time_ms"] = pd.to_numeric(
                frame[latency_col], errors="coerce"
            )

            role = pick_column(
                frame,
                ["workload_type", "workload_class", "user_role"]
            )
            frame["target_class"] = (
                frame[role].fillna("unknown").astype(str).str.lower()
                if role else "unknown"
            )

            fp = pick_column(
                frame,
                ["fingerprint_hash", "fingerprint", "query_fingerprint"]
            )
            frame["fingerprint"] = (
                frame[fp].fillna("").astype(str)
                if fp else ""
            )

            missing = frame["fingerprint"].str.strip().eq("")
            frame.loc[missing, "fingerprint"] = (
                frame.loc[missing, "sql"].map(sql_hash)
            )

            ts = pick_column(
                frame,
                ["monitored_at", "timestamp", "executed_at", "recorded_at"]
            )
            frame["timestamp"] = (
                pd.to_datetime(frame[ts], utc=True, errors="coerce")
                if ts else pd.NaT
            )

            if "session_id" not in frame:
                frame["session_id"] = file.stem

            if "query_family" not in frame:
                frame["query_family"] = "unknown"

            frame["source_file"] = str(file)
            frame["source_row"] = np.arange(len(frame))

            for canonical, aliases in PLANNER_ALIASES.items():
                original = pick_column(frame, aliases)
                if original:
                    frame[canonical] = pd.to_numeric(
                        frame[original], errors="coerce"
                    )

            frames.append(frame)
            sources.append({
                "path": str(file),
                "sha256": file_hash(file),
                "rows": len(frame),
            })

    if not frames:
        raise RuntimeError("No valid monitoring CSV files.")

    df = pd.concat(frames, ignore_index=True)
    df = df[
        df["sql"].str.strip().ne("")
        & df["execution_time_ms"].notna()
        & np.isfinite(df["execution_time_ms"])
        & (df["execution_time_ms"] >= 0)
    ].copy()

    if len(df) < 50:
        raise RuntimeError("Too few valid execution records.")

    return df.reset_index(drop=True), sources

def engineer_features(df):
    df = df.copy()
    df["model_text"] = df["sql"].map(normalize_sql)
    s = df["model_text"]

    df["sql_length"] = s.str.len()
    df["token_count"] = s.str.split().str.len()

    patterns = {
        "join_count": r"\bjoin\b",
        "where_count": r"\bwhere\b",
        "group_count": r"\bgroup\s+by\b",
        "order_count": r"\border\s+by\b",
        "having_count": r"\bhaving\b",
        "aggregate_count": r"\b(?:count|sum|avg|min|max)\s*\(",
        "subquery_count": r"\(\s*select\b",
        "predicate_count": r"(?<![<>!])=(?!=)|>=|<=|<>|!=",
        "distinct_count": r"\bdistinct\b",
        "limit_count": r"\blimit\b",
    }

    for col, pattern in patterns.items():
        df[col] = s.str.count(pattern)

    df["table_count"] = (
        s.str.count(r"\bfrom\b")
        + s.str.count(r"\bjoin\b")
    )

    flags = {
        "has_cte": r"^\s*with\b",
        "has_union": r"\bunion\b",
        "has_window": r"\bover\s*\(",
        "has_exists": r"\bexists\b",
        "has_like": r"\blike\b",
        "has_between": r"\bbetween\b",
        "has_in": r"\bin\s*\(",
        "has_or": r"\bor\b",
        "has_case": r"\bcase\b",
        "has_offset": r"\boffset\b",
        "has_left_join": r"\bleft\s+join\b",
        "has_right_join": r"\bright\s+join\b",
        "has_cross_join": r"\bcross\s+join\b",
        "has_null": r"\bis\s+(?:not\s+)?null\b",
    }

    for col, pattern in flags.items():
        df[col] = s.str.contains(pattern, regex=True).astype(int)

    available_planner = [
        c for c in PLANNER_FEATURES
        if c in df and pd.to_numeric(df[c], errors="coerce").notna().any()
    ]

    for col in SQL_FEATURES + available_planner:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, available_planner

def label_identifiability(df):
    usable = df[df["target_class"] != "unknown"]
    if usable.empty:
        return {"status": "NO_ROLE_LABELS"}

    table = pd.crosstab(
        usable["fingerprint"],
        usable["target_class"],
    )

    probabilities = table.div(table.sum(axis=1), axis=0)
    entropy = -(
        probabilities
        * np.log2(probabilities.clip(lower=EPS))
    ).sum(axis=1)

    ceiling = table.max(axis=1).sum() / table.to_numpy().sum()

    return {
        "status": "EVALUATED",
        "fingerprints": len(table),
        "ambiguous_fingerprints": int(
            (table.gt(0).sum(axis=1) > 1).sum()
        ),
        "mean_fingerprint_entropy_bits": float(entropy.mean()),
        "in_sample_fingerprint_lookup_accuracy_ceiling": float(ceiling),
        "interpretation": (
            "This is an in-sample ceiling for deterministic "
            "fingerprint-only lookup, not an unseen-query "
            "generalization bound or a bound for models "
            "with additional requester metadata."
        ),
    }

def variance_diagnostics(df):
    data = df[["fingerprint", "execution_time_ms"]].copy()
    data["log_latency"] = np.log1p(data["execution_time_ms"])

    grouped = data.groupby("fingerprint")["log_latency"]
    counts = grouped.size()
    means = grouped.mean()
    overall = data["log_latency"].mean()

    n = len(data)
    k = len(counts)

    if k < 2 or n <= k:
        return {"status": "INSUFFICIENT_GROUPS"}

    ss_between = float(
        (counts * (means - overall) ** 2).sum()
    )

    ss_within = float(
        grouped.apply(
            lambda x: ((x - x.mean()) ** 2).sum()
        ).sum()
    )

    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (n - k)

    n0 = (
        n - (counts.pow(2).sum() / n)
    ) / (k - 1)

    between_var = max(
        0.0,
        (ms_between - ms_within) / max(n0, EPS)
    )

    within_var = max(0.0, ms_within)

    icc = between_var / max(
        between_var + within_var, EPS
    )

    mean_cluster_size = n / k
    design_effect = 1 + (mean_cluster_size - 1) * icc

    effective_n = n / max(design_effect, EPS)

    return {
        "status": "EVALUATED",
        "fingerprints": k,
        "records": n,
        "icc_log_latency": icc,
        "between_variance": between_var,
        "within_variance": within_var,
        "design_effect_approx": design_effect,
        "effective_sample_size_approx": effective_n,
        "warning": (
            "One-way random-effects approximation. "
            "Effective sample size depends on the estimand; "
            "unseen-query evaluation still has only "
            f"{k} independent fingerprint groups."
        ),
    }

def planner_diagnostics(df):
    if "estimated_cost" not in df:
        return {"status": "NO_ESTIMATED_COST"}

    temp = df[
        ["fingerprint", "estimated_cost", "execution_time_ms"]
    ].dropna()

    temp = temp[temp["estimated_cost"] >= 0]

    if len(temp) < 10:
        return {"status": "INSUFFICIENT_COST_DATA"}

    fingerprint = temp.groupby("fingerprint").agg(
        cost=("estimated_cost", "median"),
        latency=("execution_time_ms", "median"),
    )

    row_corr = spearmanr(
        temp["estimated_cost"],
        temp["execution_time_ms"],
    ).statistic

    fp_corr = (
        spearmanr(
            fingerprint["cost"],
            fingerprint["latency"],
        ).statistic
        if len(fingerprint) >= 3 else np.nan
    )

    return {
        "status": "EVALUATED",
        "row_level_spearman": row_corr,
        "fingerprint_level_spearman": fp_corr,
        "fingerprints": len(fingerprint),
    }

def make_preprocessor(numeric, text_mode="none"):
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
                ("scale", StandardScaler()),
            ]),
            numeric,
        )
    ]

    if text_mode in ("word", "char"):
        if text_mode == "char":
            vectorizer = TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                max_features=2500,
                sublinear_tf=True,
            )
        else:
            vectorizer = TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                max_features=2000,
                sublinear_tf=True,
                token_pattern=r"(?u)\b\w+\b",
            )

        transformers.append(
            ("text", vectorizer, "model_text")
        )

    return ColumnTransformer(
        transformers,
        sparse_threshold=0.3,
    )

def model_pipeline(numeric, estimator, text_mode="none"):
    return Pipeline([
        ("features", make_preprocessor(numeric, text_mode)),
        ("model", estimator),
    ])

class CostCalibrator:
    def __init__(self, mode="ridge", alpha=1.0):
        self.mode = mode
        self.alpha = alpha

    def fit(self, frame, target):
        self.has_cost = (
            "estimated_cost" in frame
            and frame["estimated_cost"].notna().any()
        )

        if self.has_cost:
            cost = pd.to_numeric(
                frame["estimated_cost"],
                errors="coerce",
            ).fillna(0).clip(lower=0)

            x = np.log1p(cost.to_numpy()).reshape(-1, 1)
        else:
            x = np.zeros((len(frame), 1))

        self.model = (
            HuberRegressor(
                alpha=self.alpha,
                max_iter=1000,
            )
            if self.mode == "huber"
            else Ridge(alpha=self.alpha)
        )

        self.model.fit(x, target)
        return self

    def predict(self, frame):
        if self.has_cost:
            cost = pd.to_numeric(
                frame["estimated_cost"],
                errors="coerce",
            ).fillna(0).clip(lower=0)

            x = np.log1p(cost.to_numpy()).reshape(-1, 1)
        else:
            x = np.zeros((len(frame), 1))

        return self.model.predict(x)

class ResidualRegressor:
    def __init__(self, numeric, kind="catboost"):
        self.numeric = numeric
        self.kind = kind

    def fit(self, frame, target):
        self.calibrator = CostCalibrator(
            mode="ridge",
            alpha=10.0,
        ).fit(frame, target)

        residual = target - self.calibrator.predict(frame)

        if self.kind == "catboost":
            estimator = CatBoostRegressor(
                iterations=180,
                depth=2,
                learning_rate=0.04,
                l2_leaf_reg=10,
                loss_function="RMSE",
                verbose=False,
                random_seed=SEED,
                thread_count=4,
                allow_writing_files=False,
            )
        else:
            estimator = HistGradientBoostingRegressor(
                max_iter=150,
                max_leaf_nodes=7,
                min_samples_leaf=12,
                learning_rate=0.04,
                l2_regularization=5.0,
                random_state=SEED,
            )

        self.residual_model = model_pipeline(
            self.numeric,
            estimator,
        )

        self.residual_model.fit(frame, residual)
        return self

    def predict(self, frame):
        return (
            self.calibrator.predict(frame)
            + self.residual_model.predict(frame)
        )

def regression_candidates(numeric, has_cost):
    models = {
        "median_baseline": model_pipeline(
            numeric,
            DummyRegressor(strategy="median"),
        ),
        "ridge_numeric": model_pipeline(
            numeric,
            Ridge(alpha=20),
        ),
        "ridge_char_tfidf": model_pipeline(
            numeric,
            Ridge(alpha=100),
            "char",
        ),
        "elastic_net": model_pipeline(
            numeric,
            ElasticNet(
                alpha=0.1,
                l1_ratio=0.3,
                max_iter=5000,
                random_state=SEED,
            ),
        ),
        "huber": model_pipeline(
            numeric,
            HuberRegressor(
                alpha=2,
                max_iter=1000,
            ),
        ),
        "svr_rbf": model_pipeline(
            numeric,
            SVR(
                C=5,
                epsilon=0.15,
            ),
        ),
        "random_forest": model_pipeline(
            numeric,
            RandomForestRegressor(
                n_estimators=160,
                min_samples_leaf=3,
                max_features=0.8,
                random_state=SEED,
                n_jobs=-1,
            ),
        ),
        "extra_trees": model_pipeline(
            numeric,
            ExtraTreesRegressor(
                n_estimators=160,
                min_samples_leaf=3,
                random_state=SEED,
                n_jobs=-1,
            ),
        ),
        "catboost_shallow": model_pipeline(
            numeric,
            CatBoostRegressor(
                iterations=240,
                depth=2,
                learning_rate=0.04,
                l2_leaf_reg=10,
                verbose=False,
                random_seed=SEED,
                thread_count=4,
                allow_writing_files=False,
            ),
        ),
        "hist_gradient_boosting": model_pipeline(
            numeric,
            HistGradientBoostingRegressor(
                max_iter=160,
                max_leaf_nodes=7,
                min_samples_leaf=12,
                learning_rate=0.04,
                l2_regularization=5,
                random_state=SEED,
            ),
        ),
    }

    if has_cost:
        models.update({
            "cost_calibration_ridge": CostCalibrator(
                "ridge", alpha=10
            ),
            "cost_calibration_huber": CostCalibrator(
                "huber", alpha=2
            ),
            "cost_residual_catboost": ResidualRegressor(
                numeric, "catboost"
            ),
            "cost_residual_histgb": ResidualRegressor(
                numeric, "histgb"
            ),
        })

    return models

def classification_candidates(numeric):
    return {
        "majority": model_pipeline(
            numeric,
            DummyClassifier(strategy="most_frequent"),
        ),
        "logistic_char": model_pipeline(
            numeric,
            LogisticRegression(
                C=0.5,
                class_weight="balanced",
                max_iter=4000,
                random_state=SEED,
            ),
            "char",
        ),
        "logistic_word": model_pipeline(
            numeric,
            LogisticRegression(
                C=0.5,
                class_weight="balanced",
                max_iter=4000,
                random_state=SEED,
            ),
            "word",
        ),
        "linear_svm_char": model_pipeline(
            numeric,
            LinearSVC(
                C=0.1,
                max_iter=15000,
                tol=1e-4,
                dual=False,
                class_weight="balanced",
                random_state=SEED,
            ),
            "char",
        ),
        "linear_svm_word": model_pipeline(
            numeric,
            LinearSVC(
                C=0.1,
                max_iter=15000,
                tol=1e-4,
                dual=False,
                class_weight="balanced",
                random_state=SEED,
            ),
            "word",
        ),
        "random_forest": model_pipeline(
            numeric,
            RandomForestClassifier(
                n_estimators=160,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                random_state=SEED,
                n_jobs=-1,
            ),
        ),
        "catboost": model_pipeline(
            numeric,
            CatBoostClassifier(
                iterations=220,
                depth=2,
                learning_rate=0.04,
                l2_leaf_reg=10,
                verbose=False,
                random_seed=SEED,
                thread_count=4,
                allow_writing_files=False,
            ),
        ),
    }

def log_target(y):
    return np.log1p(
        np.maximum(np.asarray(y, dtype=float), 0)
    )

def predict_ms(model, frame, train_max_ms):
    predicted_log = np.asarray(
        model.predict(frame),
        dtype=float,
    )

    # Bound inverse-transform extrapolation using
    # only the training-set latency range.
    cap = max(1000, train_max_ms * 5)

    return np.expm1(
        np.clip(predicted_log, 0, np.log1p(cap))
    )

def reg_metrics(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)

    ylog = log_target(y)
    plog = log_target(pred)

    ratio = np.maximum(
        (pred + 1) / (y + 1),
        (y + 1) / (pred + 1),
    )

    corr = (
        spearmanr(y, pred).statistic
        if len(y) >= 3
        and np.std(y) > EPS
        and np.std(pred) > EPS
        else np.nan
    )

    return {
        "mae_ms": float(mean_absolute_error(y, pred)),
        "rmse_ms": float(
            np.sqrt(mean_squared_error(y, pred))
        ),
        "r2_ms": float(r2_score(y, pred))
        if len(y) > 1 else np.nan,
        "log_mae": float(
            mean_absolute_error(ylog, plog)
        ),
        "median_qerror": float(np.median(ratio)),
        "p90_qerror": float(np.quantile(ratio, 0.9)),
        "spearman": float(corr),
    }

def bootstrap_ci(values, n_boot=1000, seed=SEED):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 2:
        return [None, None]

    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)

    for i in range(n_boot):
        means[i] = rng.choice(
            x, size=len(x), replace=True
        ).mean()

    return [
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    ]

def grouped_folds(df, n_splits):
    groups = df["fingerprint"].to_numpy()
    k = len(np.unique(groups))

    if k < 4:
        raise RuntimeError(
            "Need at least four distinct fingerprints."
        )

    splitter = GroupKFold(
        n_splits=min(n_splits, k),
        shuffle=True,
        random_state=SEED,
    )

    return list(
        splitter.split(df, groups=groups)
    )

def evaluate_unseen_regression(
    df,
    numeric,
    out,
    n_splits,
    bootstrap,
):
    models = regression_candidates(
        numeric,
        "estimated_cost" in numeric,
    )

    folds = grouped_folds(df, n_splits)
    rows = []
    errors = []
    predictions = []

    print("\n" + "=" * 85)
    print("UNSEEN QUERY LATENCY — FINGERPRINT-GROUPED EVALUATION")
    print("=" * 85)

    for name, definition in models.items():
        for fold, (train_idx, test_idx) in enumerate(folds):
            train = df.iloc[train_idx]
            test = df.iloc[test_idx]

            assert set(train["fingerprint"]).isdisjoint(
                set(test["fingerprint"])
            )

            model = clone(definition) if isinstance(
                definition, Pipeline
            ) else None

            if model is None:
                import copy
                model = copy.deepcopy(definition)

            try:
                model.fit(
                    train,
                    log_target(train["execution_time_ms"]),
                )

                pred = predict_ms(
                    model,
                    test,
                    float(train["execution_time_ms"].max()),
                )

                metrics = reg_metrics(
                    test["execution_time_ms"],
                    pred,
                )

                rows.append({
                    "model": name,
                    "fold": fold,
                    "n_test": len(test),
                    "n_fingerprints": (
                        test["fingerprint"].nunique()
                    ),
                    **metrics,
                })

                predictions.append(
                    pd.DataFrame({
                        "model": name,
                        "fold": fold,
                        "fingerprint": (
                            test["fingerprint"].to_numpy()
                        ),
                        "actual_ms": (
                            test["execution_time_ms"].to_numpy()
                        ),
                        "predicted_ms": pred,
                    })
                )

            except Exception as exc:
                errors.append({
                    "model": name,
                    "fold": fold,
                    "error": repr(exc),
                })

    if not rows:
        raise RuntimeError(
            "Every unseen-query regression model failed."
        )

    scores = pd.DataFrame(rows)
    pred_df = pd.concat(
        predictions, ignore_index=True
    )

    scores.to_csv(
        out / "unseen_regression_folds.csv",
        index=False,
    )
    pred_df.to_csv(
        out / "unseen_regression_predictions.csv",
        index=False,
    )

    per_fp = (
        pred_df.groupby(["model", "fingerprint"])
        .apply(
            lambda g: pd.Series({
                "mae_ms": np.mean(
                    np.abs(
                        g["actual_ms"] - g["predicted_ms"]
                    )
                ),
                "log_mae": np.mean(
                    np.abs(
                        np.log1p(g["actual_ms"])
                        - np.log1p(g["predicted_ms"])
                    )
                ),
                "median_qerror": np.median(
                    np.maximum(
                        (g["predicted_ms"] + 1)
                        / (g["actual_ms"] + 1),
                        (g["actual_ms"] + 1)
                        / (g["predicted_ms"] + 1),
                    )
                ),
            }),
            include_groups=False,
        )
        .reset_index()
    )

    per_fp.to_csv(
        out / "unseen_per_fingerprint_errors.csv",
        index=False,
    )

    summary = (
        per_fp.groupby("model")
        .agg(
            fingerprint_mae_ms=("mae_ms", "mean"),
            fingerprint_log_mae=("log_mae", "mean"),
            fingerprint_qerror=(
                "median_qerror", "median"
            ),
            evaluated_fingerprints=(
                "fingerprint", "nunique"
            ),
        )
        .reset_index()
    )

    fold_summary = (
        scores.groupby("model")
        .agg(
            row_mae_ms=("mae_ms", "mean"),
            row_r2_ms=("r2_ms", "mean"),
            row_spearman=("spearman", "mean"),
            folds_completed=("fold", "count"),
        )
        .reset_index()
    )

    summary = summary.merge(
        fold_summary, on="model"
    )

    summary = summary[
        summary["folds_completed"] == len(folds)
    ].copy()

    ci_rows = []

    for model_name in summary["model"]:
        values = per_fp.loc[
            per_fp["model"] == model_name,
            "log_mae",
        ]

        ci = bootstrap_ci(
            values,
            n_boot=bootstrap,
        )

        ci_rows.append({
            "model": model_name,
            "log_mae_ci_low": ci[0],
            "log_mae_ci_high": ci[1],
        })

    summary = summary.merge(
        pd.DataFrame(ci_rows),
        on="model",
    )

    summary = summary.sort_values(
        [
            "fingerprint_log_mae",
            "fingerprint_mae_ms",
        ]
    ).reset_index(drop=True)

    summary["rank"] = np.arange(
        1, len(summary) + 1
    )

    summary.to_csv(
        out / "unseen_regression_leaderboard.csv",
        index=False,
    )

    if summary.empty:
        raise RuntimeError(
            "No model completed all grouped folds."
        )

    winner = summary.iloc[0]["model"]

    print(
        summary[
            [
                "rank",
                "model",
                "fingerprint_log_mae",
                "fingerprint_mae_ms",
                "fingerprint_qerror",
                "row_r2_ms",
                "log_mae_ci_low",
                "log_mae_ci_high",
            ]
        ].round(4).to_string(index=False)
    )

    print("\nBEST UNSEEN-QUERY MODEL:", winner)

    # Paired fingerprint-level comparisons.
    pivot = per_fp.pivot(
        index="fingerprint",
        columns="model",
        values="log_mae",
    )

    paired = []

    for other in summary["model"]:
        if other == winner:
            continue

        common = pivot[
            [winner, other]
        ].dropna()

        if len(common) < 5:
            continue

        # Positive difference means winner has
        # lower log-MAE than the competitor.
        improvement = (
            common[other] - common[winner]
        ).to_numpy()

        try:
            pvalue = float(
                wilcoxon(improvement).pvalue
            )
        except ValueError:
            pvalue = 1.0

        paired.append({
            "winner": winner,
            "competitor": other,
            "n_fingerprints": len(common),
            "mean_log_mae_improvement": (
                improvement.mean()
            ),
            "ci_low": bootstrap_ci(
                improvement, bootstrap
            )[0],
            "ci_high": bootstrap_ci(
                improvement, bootstrap
            )[1],
            "p_value_uncorrected": pvalue,
        })

    paired_df = pd.DataFrame(paired)

    if not paired_df.empty:
        # Holm multiple-comparison correction.
        order = np.argsort(
            paired_df["p_value_uncorrected"].to_numpy()
        )

        raw = paired_df[
            "p_value_uncorrected"
        ].to_numpy()

        adjusted = np.empty(len(raw))
        running = 0.0

        for rank, idx in enumerate(order):
            candidate = min(
                1.0,
                raw[idx] * (len(raw) - rank),
            )
            running = max(running, candidate)
            adjusted[idx] = running

        paired_df["holm_adjusted_p"] = adjusted

    paired_df.to_csv(
        out / "paired_model_comparisons.csv",
        index=False,
    )

    save_json(
        out / "unseen_regression_errors.json",
        errors,
    )

    return {
        "winner": winner,
        "leaderboard": summary,
        "per_fingerprint": per_fp,
        "predictions": pred_df,
        "models": models,
        "errors": errors,
    }

def evaluate_classification(
    df,
    numeric,
    out,
    n_splits,
):
    data = df[
        df["target_class"] != "unknown"
    ].reset_index(drop=True)

    if data["target_class"].nunique() < 2:
        return {"status": "INSUFFICIENT_LABELS"}

    models = classification_candidates(numeric)
    folds = grouped_folds(data, n_splits)

    rows = []
    predictions = []
    errors = []

    labels = sorted(data["target_class"].unique())

    print("\n" + "=" * 85)
    print("ROLE CLASSIFICATION — GROUPED EVALUATION")
    print("=" * 85)

    for name, definition in models.items():
        for fold, (tr, te) in enumerate(folds):
            train = data.iloc[tr]
            test = data.iloc[te]

            if train["target_class"].nunique() < 2:
                errors.append({
                    "model": name,
                    "fold": fold,
                    "error": "Training fold contains one class.",
                })
                continue

            model = clone(definition)

            try:
                model.fit(
                    train,
                    train["target_class"],
                )

                pred = model.predict(test)

                rows.append({
                    "model": name,
                    "fold": fold,
                    "accuracy": accuracy_score(
                        test["target_class"], pred
                    ),
                    "macro_f1": f1_score(
                        test["target_class"],
                        pred,
                        labels=labels,
                        average="macro",
                        zero_division=0,
                    ),
                    "balanced_accuracy": (
                        balanced_accuracy_score(
                            test["target_class"],
                            pred,
                        )
                    ),
                })

                predictions.append(
                    pd.DataFrame({
                        "model": name,
                        "fingerprint": (
                            test["fingerprint"].to_numpy()
                        ),
                        "actual": (
                            test["target_class"].to_numpy()
                        ),
                        "predicted": pred,
                    })
                )

            except Exception as exc:
                errors.append({
                    "model": name,
                    "fold": fold,
                    "error": repr(exc),
                })

    if not rows:
        return {
            "status": "ALL_MODELS_FAILED",
            "errors": errors,
        }

    scores = pd.DataFrame(rows)

    leaderboard = (
        scores.groupby("model")
        .agg(
            accuracy=("accuracy", "mean"),
            macro_f1=("macro_f1", "mean"),
            balanced_accuracy=(
                "balanced_accuracy", "mean"
            ),
            std_macro_f1=("macro_f1", "std"),
            folds_completed=("fold", "count"),
        )
        .reset_index()
    )

    leaderboard = leaderboard[
        leaderboard["folds_completed"] == len(folds)
    ].sort_values(
        ["macro_f1", "balanced_accuracy"],
        ascending=False,
    )

    leaderboard.to_csv(
        out / "classification_leaderboard.csv",
        index=False,
    )

    scores.to_csv(
        out / "classification_folds.csv",
        index=False,
    )

    if not predictions:
        return {"status": "NO_PREDICTIONS"}

    all_predictions = pd.concat(
        predictions,
        ignore_index=True,
    )

    all_predictions.to_csv(
        out / "classification_predictions.csv",
        index=False,
    )

    if leaderboard.empty:
        return {
            "status": "NO_MODEL_COMPLETED_ALL_FOLDS",
            "errors": errors,
        }

    winner = leaderboard.iloc[0]["model"]

    best_pred = all_predictions[
        all_predictions["model"] == winner
    ]

    report = classification_report(
        best_pred["actual"],
        best_pred["predicted"],
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    save_json(
        out / "classification_per_class.json",
        report,
    )

    cm = confusion_matrix(
        best_pred["actual"],
        best_pred["predicted"],
        labels=labels,
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30)
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
            )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(
        "Fingerprint-grouped role classification"
    )

    fig.tight_layout()
    fig.savefig(
        out / "classification_confusion.png",
        dpi=180,
    )
    plt.close(fig)

    print(
        leaderboard.round(4).to_string(index=False)
    )
    print("\nBEST CLASSIFIER:", winner)

    save_json(
        out / "classification_errors.json",
        errors,
    )

    return {
        "status": "EVALUATED",
        "winner": winner,
        "leaderboard": leaderboard,
        "models": models,
    }

def known_query_prediction(history, method):
    if not history:
        return None

    values = np.asarray(history, dtype=float)

    if method == "last":
        return float(values[-1])

    if method == "median":
        return float(np.median(values))

    if method == "rolling_median_5":
        return float(np.median(values[-5:]))

    if method == "rolling_median_10":
        return float(np.median(values[-10:]))

    if method == "ewma_03":
        result = values[0]
        for value in values[1:]:
            result = 0.3 * value + 0.7 * result
        return float(result)

    if method == "ewma_07":
        result = values[0]
        for value in values[1:]:
            result = 0.7 * value + 0.3 * result
        return float(result)

    return None

def known_query_evaluation(
    df,
    out,
    warmup=3,
):
    data = df.dropna(
        subset=["timestamp"]
    ).copy()

    if len(data) < 50:
        return {"status": "INSUFFICIENT_TIMESTAMPS"}

    data = data.sort_values(
        ["timestamp", "source_row"],
        kind="mergesort",
    ).reset_index(drop=True)

    if data["timestamp"].nunique() < 10:
        return {"status": "INSUFFICIENT_TIME_VARIATION"}

    # Only observations strictly earlier than the
    # current timestamp are allowed as history.
    history = {}
    all_prior = []

    methods = [
        "global_median",
        "last",
        "median",
        "rolling_median_5",
        "rolling_median_10",
        "ewma_03",
        "ewma_07",
        "shrinkage_median",
    ]

    rows = []

    for timestamp, batch in data.groupby(
        "timestamp", sort=True
    ):
        # Predictions for an entire timestamp batch
        # are computed before updating any history.
        pending_updates = []

        for _, record in batch.iterrows():
            fp = record["fingerprint"]
            observed = float(
                record["execution_time_ms"]
            )

            values = history.get(fp, [])

            if len(values) >= warmup and all_prior:
                global_median = float(
                    np.median(all_prior)
                )

                for method in methods:
                    if method == "global_median":
                        pred = global_median

                    elif method == "shrinkage_median":
                        local = float(np.median(values))
                        weight = len(values) / (
                            len(values) + 5
                        )
                        pred = (
                            weight * local
                            + (1 - weight) * global_median
                        )

                    else:
                        pred = known_query_prediction(
                            values, method
                        )

                    rows.append({
                        "model": method,
                        "fingerprint": fp,
                        "timestamp": timestamp,
                        "actual_ms": observed,
                        "predicted_ms": pred,
                        "history_count": len(values),
                    })

            pending_updates.append((fp, observed))

        for fp, observed in pending_updates:
            history.setdefault(fp, []).append(observed)
            all_prior.append(observed)

    if not rows:
        return {
            "status": "INSUFFICIENT_REPEATED_QUERIES"
        }

    predictions = pd.DataFrame(rows)

    predictions.to_csv(
        out / "known_query_predictions.csv",
        index=False,
    )

    summary_rows = []

    for name, group in predictions.groupby("model"):
        metrics = reg_metrics(
            group["actual_ms"],
            group["predicted_ms"],
        )

        fp_errors = group.groupby("fingerprint").apply(
            lambda g: np.mean(
                np.abs(
                    np.log1p(g["actual_ms"])
                    - np.log1p(g["predicted_ms"])
                )
            ),
            include_groups=False,
        )

        summary_rows.append({
            "model": name,
            **metrics,
            "fingerprint_log_mae": fp_errors.mean(),
            "fingerprints": len(fp_errors),
        })

    leaderboard = pd.DataFrame(
        summary_rows
    ).sort_values(
        ["fingerprint_log_mae", "mae_ms"]
    )

    leaderboard.to_csv(
        out / "known_query_leaderboard.csv",
        index=False,
    )

    print("\n" + "=" * 85)
    print("KNOWN QUERY — EXPANDING HISTORY EVALUATION")
    print("=" * 85)
    print(
        leaderboard.round(4).to_string(index=False)
    )

    winner = leaderboard.iloc[0]["model"]
    print("\nBEST KNOWN-QUERY ESTIMATOR:", winner)

    return {
        "status": "EVALUATED",
        "winner": winner,
        "leaderboard": leaderboard,
        "warning": (
            "This is prequential evaluation within "
            "the available monitoring history, not "
            "independent multi-session validation."
        ),
    }

def learning_curve(
    df,
    numeric,
    out,
    trials=5,
):
    fingerprints = np.array(
        sorted(df["fingerprint"].unique())
    )

    n = len(fingerprints)

    if n < 10:
        return {"status": "TOO_FEW_FINGERPRINTS"}

    rng = np.random.default_rng(SEED)

    test_count = max(2, int(n * 0.2))
    train_pool_size = n - test_count

    sizes = sorted(set(
        int(x) for x in [
            5, 10, 15, 20, train_pool_size
        ]
        if 3 <= x <= train_pool_size
    ))

    model = model_pipeline(
        numeric,
        Ridge(alpha=100),
        "char",
    )

    rows = []

    for trial in range(trials):
        shuffled = rng.permutation(fingerprints)

        test_fps = set(
            shuffled[:test_count]
        )
        train_fps = shuffled[test_count:]

        test = df[
            df["fingerprint"].isin(test_fps)
        ]

        for size in sizes:
            selected = set(train_fps[:size])

            train = df[
                df["fingerprint"].isin(selected)
            ]

            candidate = clone(model)

            try:
                candidate.fit(
                    train,
                    log_target(
                        train["execution_time_ms"]
                    ),
                )

                pred = predict_ms(
                    candidate,
                    test,
                    train["execution_time_ms"].max(),
                )

                metrics = reg_metrics(
                    test["execution_time_ms"],
                    pred,
                )

                rows.append({
                    "trial": trial,
                    "training_fingerprints": size,
                    "log_mae": metrics["log_mae"],
                    "mae_ms": metrics["mae_ms"],
                })

            except Exception:
                continue

    result = pd.DataFrame(rows)

    if result.empty:
        return {"status": "FAILED"}

    result.to_csv(
        out / "learning_curve.csv",
        index=False,
    )

    grouped = (
        result.groupby("training_fingerprints")
        ["log_mae"]
        .agg(["mean", "std"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(
        grouped["training_fingerprints"],
        grouped["mean"],
        yerr=grouped["std"].fillna(0),
        marker="o",
        capsize=4,
    )

    ax.set_xlabel("Training SQL fingerprints")
    ax.set_ylabel("Log latency MAE")
    ax.set_title(
        "Unseen-query learning curve"
    )
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(
        out / "learning_curve.png",
        dpi=180,
    )
    plt.close(fig)

    return {
        "status": "EVALUATED",
        "training_fingerprint_sizes": sizes,
        "trials": trials,
    }

def conformal_evaluation(
    df,
    numeric,
    definition,
    out,
    alpha=0.10,
    repeats=5,
):
    if df["fingerprint"].nunique() < 10:
        return {"status": "INSUFFICIENT_FINGERPRINTS"}

    splitter = GroupShuffleSplit(
        n_splits=repeats,
        test_size=0.25,
        random_state=SEED + 700,
    )

    rows = []
    predictions = []

    for repeat, (dev_idx, test_idx) in enumerate(
        splitter.split(
            df,
            groups=df["fingerprint"],
        )
    ):
        dev = df.iloc[dev_idx].reset_index(drop=True)
        test = df.iloc[test_idx].reset_index(drop=True)

        inner = GroupShuffleSplit(
            n_splits=1,
            test_size=0.3,
            random_state=SEED + repeat,
        )

        proper_idx, cal_idx = next(
            inner.split(
                dev,
                groups=dev["fingerprint"],
            )
        )

        proper = dev.iloc[proper_idx]
        cal = dev.iloc[cal_idx]

        if cal["fingerprint"].nunique() < 2:
            continue

        import copy
        model = copy.deepcopy(definition)

        model.fit(
            proper,
            log_target(
                proper["execution_time_ms"]
            ),
        )

        ceiling = float(
            proper["execution_time_ms"].max()
        )

        cal_pred = predict_ms(
            model, cal, ceiling
        )

        test_pred = predict_ms(
            model, test, ceiling
        )

        # One calibration score per fingerprint:
        # maximum absolute log residual in group.
        # This is conservative for row-level coverage.
        cal_scores = pd.DataFrame({
            "fingerprint": cal["fingerprint"].to_numpy(),
            "error": np.abs(
                np.log1p(
                    cal["execution_time_ms"].to_numpy()
                )
                - np.log1p(cal_pred)
            ),
        })

        cluster_scores = (
            cal_scores.groupby("fingerprint")
            ["error"].max().to_numpy()
        )

        m = len(cluster_scores)
        level = min(
            1.0,
            math.ceil((m + 1) * (1 - alpha)) / m,
        )

        radius = float(
            np.quantile(
                cluster_scores,
                level,
                method="higher",
            )
        )

        center_log = np.log1p(test_pred)

        lower = np.expm1(
            np.maximum(0, center_log - radius)
        )
        upper = np.expm1(
            center_log + radius
        )

        actual = test["execution_time_ms"].to_numpy()

        covered = (
            (actual >= lower)
            & (actual <= upper)
        )

        temp = pd.DataFrame({
            "repeat": repeat,
            "fingerprint": test["fingerprint"].to_numpy(),
            "actual_ms": actual,
            "predicted_ms": test_pred,
            "lower_ms": lower,
            "upper_ms": upper,
            "covered": covered,
        })

        predictions.append(temp)

        rows.append({
            "repeat": repeat,
            "coverage": float(covered.mean()),
            "mean_width_ms": float(
                np.mean(upper - lower)
            ),
            "calibration_fingerprints": m,
            "test_fingerprints": (
                test["fingerprint"].nunique()
            ),
        })

    if not rows:
        return {"status": "INSUFFICIENT_CALIBRATION"}

    pd.DataFrame(rows).to_csv(
        out / "conformal_repeated_results.csv",
        index=False,
    )

    all_predictions = pd.concat(
        predictions,
        ignore_index=True,
    )

    all_predictions.to_csv(
        out / "conformal_predictions.csv",
        index=False,
    )

    summary = {
        "status": "EVALUATED",
        "nominal_coverage": 1 - alpha,
        "mean_empirical_coverage": float(
            np.mean([r["coverage"] for r in rows])
        ),
        "mean_interval_width_ms": float(
            np.mean([r["mean_width_ms"] for r in rows])
        ),
        "repeats": len(rows),
        "method": (
            "Group-held-out calibration using "
            "maximum log residual per fingerprint"
        ),
        "caution": (
            "Coverage guarantees require exchangeable "
            "fingerprint groups and an appropriate "
            "cluster-level target. Only 29 fingerprints "
            "make interval calibration coarse."
        ),
    }

    save_json(
        out / "conformal_report.json",
        summary,
    )

    return summary

def phase9_ranking(df, out):
    grouped = (
        df.groupby("fingerprint")
        .agg(
            frequency=("execution_time_ms", "size"),
            median_ms=("execution_time_ms", "median"),
            mean_ms=("execution_time_ms", "mean"),
            p90_ms=(
                "execution_time_ms",
                lambda s: s.quantile(0.90),
            ),
        )
        .reset_index()
    )

    grouped["cumulative_ms"] = (
        grouped["frequency"]
        * grouped["mean_ms"]
    )

    if "estimated_cost" in df:
        cost = (
            df.groupby("fingerprint")
            ["estimated_cost"]
            .median()
            .rename("median_planner_cost")
            .reset_index()
        )
        grouped = grouped.merge(
            cost, on="fingerprint", how="left"
        )
    else:
        grouped["median_planner_cost"] = np.nan

    grouped["rank_observed_total"] = (
        grouped["cumulative_ms"]
        .rank(ascending=False, method="min")
    )

    grouped["rank_frequency"] = (
        grouped["frequency"]
        .rank(ascending=False, method="min")
    )

    grouped["rank_median_latency"] = (
        grouped["median_ms"]
        .rank(ascending=False, method="min")
    )

    grouped["rank_planner_cost"] = (
        grouped["median_planner_cost"]
        .rank(ascending=False, method="min")
    )

    grouped = grouped.sort_values(
        "rank_observed_total"
    )

    grouped.to_csv(
        out / "phase9_query_priorities.csv",
        index=False,
    )

    # These comparisons measure how well proxy
    # rankings retrieve high observed cumulative
    # time. They do NOT measure index speedup.
    oracle = grouped.sort_values(
        "cumulative_ms", ascending=False
    )

    total = max(
        oracle["cumulative_ms"].sum(),
        EPS,
    )

    rows = []

    strategies = {
        "observed_total": "cumulative_ms",
        "frequency": "frequency",
        "median_latency": "median_ms",
        "planner_cost": "median_planner_cost",
    }

    for name, column in strategies.items():
        if grouped[column].notna().sum() == 0:
            continue

        ranked = grouped.sort_values(
            column, ascending=False
        )

        for k in [3, 5, 10]:
            if k > len(grouped):
                continue

            selected = ranked.head(k)
            oracle_selected = oracle.head(k)

            captured = (
                selected["cumulative_ms"].sum()
                / total
            )

            oracle_captured = (
                oracle_selected["cumulative_ms"].sum()
                / total
            )

            rows.append({
                "strategy": name,
                "top_k": k,
                "observed_time_fraction": captured,
                "oracle_fraction": oracle_captured,
                "fraction_of_oracle": (
                    captured
                    / max(oracle_captured, EPS)
                ),
            })

    evaluation = pd.DataFrame(rows)

    evaluation.to_csv(
        out / "phase9_proxy_ranking_comparison.csv",
        index=False,
    )

    return {
        "status": "PROXY_EVALUATED",
        "queries": len(grouped),
        "ranking_comparison": evaluation.to_dict(
            orient="records"
        ),
        "critical_limitation": (
            "Observed cumulative execution time is "
            "the proxy target, not measured index "
            "benefit. The observed-total strategy "
            "is an oracle for this proxy by definition. "
            "A real Phase 9 ranking comparison requires "
            "measured per-query index improvements."
        ),
    }

def plot_leaderboard(df, metric, title, path, ascending=True):
    if df is None or df.empty:
        return

    data = df.sort_values(
        metric, ascending=ascending
    ).head(15)

    fig, ax = plt.subplots(
        figsize=(11, max(5, 0.48 * len(data)))
    )

    ax.barh(
        data["model"],
        data[metric],
    )

    ax.invert_yaxis()
    ax.set_xlabel(metric)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

def model_suitability(unseen, known, classification, out):
    rows = []

    lb = unseen["leaderboard"]

    best = lb.iloc[0]

    baseline = lb[
        lb["model"] == "median_baseline"
    ]

    baseline_score = (
        float(
            baseline.iloc[0]["fingerprint_log_mae"]
        )
        if not baseline.empty
        else None
    )

    best_score = float(
        best["fingerprint_log_mae"]
    )

    gain = (
        100 * (baseline_score - best_score)
        / max(baseline_score, EPS)
        if baseline_score is not None
        else None
    )

    rows.append({
        "task": "unseen_query_latency",
        "best_model": unseen["winner"],
        "selection_metric": "fingerprint_log_mae",
        "score": best_score,
        "baseline_score": baseline_score,
        "improvement_pct": gain,
        "phase9_recommendation": (
            "ADVISORY_CANDIDATE"
            if gain is not None and gain > 0
            else "PREFER_BASELINE"
        ),
        "note": (
            "Compare uncertainty and paired fingerprint "
            "errors before using predictions for triage."
        ),
    })

    if known.get("status") == "EVALUATED":
        klb = known["leaderboard"]

        rows.append({
            "task": "known_query_latency",
            "best_model": known["winner"],
            "selection_metric": "fingerprint_log_mae",
            "score": float(
                klb.iloc[0]["fingerprint_log_mae"]
            ),
            "baseline_score": None,
            "improvement_pct": None,
            "phase9_recommendation": (
                "ADVISORY_CANDIDATE"
            ),
            "note": (
                "Validate on independent future sessions."
            ),
        })

    if classification.get("status") == "EVALUATED":
        clb = classification["leaderboard"]

        rows.append({
            "task": "role_classification",
            "best_model": classification["winner"],
            "selection_metric": "macro_f1",
            "score": float(
                clb.iloc[0]["macro_f1"]
            ),
            "baseline_score": None,
            "improvement_pct": None,
            "phase9_recommendation": (
                "NOT_REQUIRED_FOR_INDEX_SELECTION"
            ),
            "note": (
                "Use observed requester role metadata "
                "when available instead of inferring it."
            ),
        })

    result = pd.DataFrame(rows)

    result.to_csv(
        out / "model_suitability.csv",
        index=False,
    )

    return result

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        nargs="+",
        default=[str(DEFAULT_INPUT)],
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--sql-only",
        action="store_true",
    )

    parser.add_argument(
        "--skip-learning-curve",
        action="store_true",
    )

    parser.add_argument(
        "--skip-intervals",
        action="store_true",
    )

    args = parser.parse_args()

    if args.folds < 2:
        parser.error("--folds must be at least 2")

    if args.bootstrap < 100:
        parser.error("--bootstrap must be at least 100")

    if not 0 < args.alpha < 1:
        parser.error("--alpha must be between 0 and 1")

    start = time.perf_counter()

    run_id = datetime.now(
        timezone.utc
    ).strftime("p8v4_%Y%m%d_%H%M%S_%f")

    out = OUTPUT_ROOT / "runs" / run_id
    out.mkdir(parents=True, exist_ok=False)

    print("=" * 85)
    print("PHASE 8 V4 — RESEARCH-GRADE ML EVALUATION")
    print("=" * 85)
    print("Run:", run_id)
    print("Output:", out)

    df, sources = load_dataset(args.input)
    df, planner = engineer_features(df)

    numeric = list(SQL_FEATURES)

    if not args.sql_only:
        numeric += planner

    if set(numeric) & FORBIDDEN:
        raise RuntimeError(
            "Post-execution feature leakage detected."
        )

    print("\nDATASET")
    print("Records:", len(df))
    print(
        "Unique fingerprints:",
        df["fingerprint"].nunique(),
    )
    print(
        "Sessions:",
        df["session_id"].nunique(),
    )
    print("Numeric features:", len(numeric))
    print("Planner features:", [] if args.sql_only else planner)

    save_json(
        out / "config.json",
        {
            "run_id": run_id,
            "sources": sources,
            "seed": SEED,
            "folds": args.folds,
            "bootstrap": args.bootstrap,
            "alpha": args.alpha,
            "sql_only": args.sql_only,
            "numeric_features": numeric,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
    )

    print("\n[1/9] Dataset diagnostics")

    diagnostics = {
        "label_identifiability": label_identifiability(df),
        "variance_decomposition": variance_diagnostics(df),
        "planner_correlation": planner_diagnostics(df),
        "records": len(df),
        "fingerprints": df["fingerprint"].nunique(),
        "sessions": df["session_id"].nunique(),
    }

    save_json(
        out / "dataset_diagnostics.json",
        diagnostics,
    )

    print(
        json.dumps(
            native(diagnostics),
            indent=2,
        )
    )

    print("\n[2/9] Unseen-query regression")

    unseen = evaluate_unseen_regression(
        df,
        numeric,
        out,
        args.folds,
        args.bootstrap,
    )

    print("\n[3/9] Classification")

    classification = evaluate_classification(
        df,
        numeric,
        out,
        args.folds,
    )

    print("\n[4/9] Known-query prediction")

    known = known_query_evaluation(
        df,
        out,
    )

    print("\n[5/9] Learning curve")

    learning = (
        {"status": "SKIPPED"}
        if args.skip_learning_curve
        else learning_curve(
            df, numeric, out
        )
    )

    print("\n[6/9] Prediction intervals")

    intervals = {"status": "SKIPPED"}

    if not args.skip_intervals:
        intervals = conformal_evaluation(
            df,
            numeric,
            unseen["models"][unseen["winner"]],
            out,
            args.alpha,
        )

    print("\n[7/9] Phase 9 ranking comparison")

    phase9 = phase9_ranking(
        df,
        out,
    )

    save_json(
        out / "phase9_report.json",
        phase9,
    )

    print("\n[8/9] Model suitability")

    suitability = model_suitability(
        unseen,
        known,
        classification,
        out,
    )

    print(
        suitability.to_string(index=False)
    )

    print("\n[9/9] Charts and artifacts")

    plot_leaderboard(
        unseen["leaderboard"],
        "fingerprint_log_mae",
        "Unseen-query model comparison",
        out / "unseen_model_comparison.png",
    )

    if classification.get("status") == "EVALUATED":
        plot_leaderboard(
            classification["leaderboard"],
            "macro_f1",
            "Workload classification comparison",
            out / "classification_comparison.png",
            ascending=False,
        )

    if known.get("status") == "EVALUATED":
        plot_leaderboard(
            known["leaderboard"],
            "fingerprint_log_mae",
            "Known-query prediction comparison",
            out / "known_query_comparison.png",
        )

    # Fit the chosen unseen-query model on all
    # available data for a research artifact.
    # Reported CV metrics come from held-out folds,
    # not from this full-data refit.
    import copy

    final_unseen = copy.deepcopy(
        unseen["models"][unseen["winner"]]
    )

    final_unseen.fit(
        df,
        log_target(df["execution_time_ms"]),
    )

    joblib.dump(
        {
            "model": final_unseen,
            "numeric_features": numeric,
            "train_max_ms": float(
                df["execution_time_ms"].max()
            ),
            "winner": unseen["winner"],
            "training_fingerprints": (
                df["fingerprint"].nunique()
            ),
            "training_records": len(df),
            "note": (
                "Load only trusted joblib files. "
                "Model trained on all available "
                "records after grouped evaluation."
            ),
        },
        out / "best_unseen_model.joblib",
    )

    if classification.get("status") == "EVALUATED":
        final_classifier = clone(
            classification["models"][
                classification["winner"]
            ]
        )

        final_classifier.fit(
            df,
            df["target_class"],
        )

        joblib.dump(
            final_classifier,
            out / "best_classifier.joblib",
        )

    # Historical fingerprint statistics are
    # descriptive artifacts, not unseen-query
    # evaluation features.
    history = (
        df.groupby("fingerprint")
        ["execution_time_ms"]
        .agg(
            ["count", "median", "mean", "std"]
        )
        .reset_index()
    )

    history.to_csv(
        out / "known_fingerprint_history.csv",
        index=False,
    )

    summary = {
        "run_id": run_id,
        "status": "COMPLETED",
        "records": len(df),
        "fingerprints": df["fingerprint"].nunique(),
        "sessions": df["session_id"].nunique(),
        "feature_mode": (
            "SQL_ONLY"
            if args.sql_only
            else "SQL_PLUS_AVAILABLE_PLANNER"
        ),
        "best_unseen_latency_model": (
            unseen["winner"]
        ),
        "best_unseen_fingerprint_log_mae": float(
            unseen["leaderboard"].iloc[0][
                "fingerprint_log_mae"
            ]
        ),
        "best_unseen_fingerprint_mae_ms": float(
            unseen["leaderboard"].iloc[0][
                "fingerprint_mae_ms"
            ]
        ),
        "best_classifier": classification.get(
            "winner"
        ),
        "best_classification_macro_f1": (
            float(
                classification["leaderboard"]
                .iloc[0]["macro_f1"]
            )
            if classification.get("status") == "EVALUATED"
            else None
        ),
        "best_known_query_estimator": known.get(
            "winner"
        ),
        "best_known_query_log_mae": (
            float(
                known["leaderboard"]
                .iloc[0]["fingerprint_log_mae"]
            )
            if known.get("status") == "EVALUATED"
            else None
        ),
        "learning_curve_status": learning["status"],
        "interval_status": intervals["status"],
        "interval_coverage": intervals.get(
            "mean_empirical_coverage"
        ),
        "phase9_status": phase9["status"],
        "elapsed_seconds": round(
            time.perf_counter() - start,
            2,
        ),
        "output": str(out),
        "limitations": [
            "Only 29 SQL fingerprints in current data.",
            "Only one monitoring session.",
            "Grouped CV selects and evaluates models "
            "on the same set of outer folds; reported "
            "winner score is selection-biased.",
            "A separate frozen unseen-query holdout "
            "is required for unbiased final assessment.",
            "Known-query evaluation is within-session.",
            "No actual index benefit measured in V4.",
            "No production databases modified.",
        ],
    }

    save_json(
        out / "summary.json",
        summary,
    )

    print("\n" + "=" * 85)
    print("FINAL PHASE 8 V4 RESULTS")
    print("=" * 85)

    print(
        json.dumps(
            native(summary),
            indent=2,
        )
    )

    print("\nIMPORTANT:")
    print(
        "The best model is the best observed candidate "
        "under this evaluation, not yet a validated "
        "production winner."
    )

    print("\nALL REPORTS SAVED TO:")
    print(out)

if __name__ == "__main__":
    main()
