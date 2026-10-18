
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
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

from scipy.stats import spearmanr, wilcoxon
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    LogisticRegression,
    Ridge,
    ElasticNet,
    HuberRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    StratifiedGroupKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC, SVR

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42
EPS = 1e-9

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUT = (
    ROOT
    / "results"
    / "phase7_monitoring"
    / "phase7_query_monitoring.csv"
)

OUTPUT_ROOT = ROOT / "results" / "phase8_v5"

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

LEAKAGE_COLUMNS = {
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
    "target_class",
    "expensive",
    "fingerprint",
    "session_id",
    "timestamp",
}

def to_native(value):
    if isinstance(value, dict):
        return {
            str(k): to_native(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [to_native(x) for x in value]

    if isinstance(value, np.ndarray):
        return to_native(value.tolist())

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else None

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, (datetime, pd.Timestamp, Path)):
        return str(value)

    if isinstance(value, float) and not np.isfinite(value):
        return None

    return value

def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            to_native(data),
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

def pick_column(df, choices):
    for column in choices:
        if column in df.columns:
            return column
    return None

def normalize_sql(value):
    sql = str(value or "")

    sql = re.sub(
        r"/\*.*?\*/",
        " ",
        sql,
        flags=re.S,
    )

    sql = re.sub(
        r"--[^\n]*",
        " ",
        sql,
    )

    sql = re.sub(
        r"'(?:''|[^'])*'",
        " ? ",
        sql,
    )

    sql = re.sub(
        r"\b\d+(?:\.\d+)?\b",
        " ? ",
        sql,
    )

    sql = re.sub(
        r"\s+",
        " ",
        sql,
    )

    return sql.strip().lower()

def fingerprint_sql(sql):
    return hashlib.sha256(
        normalize_sql(sql).encode("utf-8")
    ).hexdigest()

def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()

def load_data(inputs):
    frames = []
    sources = []

    for item in inputs:
        path = Path(item).expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(path)

        files = (
            sorted(path.glob("*.csv"))
            if path.is_dir()
            else [path]
        )

        for file in files:
            raw = pd.read_csv(
                file,
                low_memory=False,
            )

            sql_col = pick_column(
                raw,
                [
                    "sql",
                    "query_sql",
                    "sql_text",
                    "query_text",
                ],
            )

            latency_col = pick_column(
                raw,
                [
                    "execution_time_ms",
                    "latency_ms",
                    "duration_ms",
                ],
            )

            if sql_col is None or latency_col is None:
                if path.is_dir():
                    continue

                raise ValueError(
                    f"Required columns not found in {file}\n"
                    f"Available: {list(raw.columns)}"
                )

            df = raw.copy()

            df["sql"] = (
                df[sql_col]
                .fillna("")
                .astype(str)
            )

            df["execution_time_ms"] = pd.to_numeric(
                df[latency_col],
                errors="coerce",
            )

            fp_col = pick_column(
                df,
                [
                    "fingerprint_hash",
                    "fingerprint",
                    "query_fingerprint",
                ],
            )

            if fp_col:
                df["fingerprint"] = (
                    df[fp_col]
                    .fillna("")
                    .astype(str)
                )
            else:
                df["fingerprint"] = ""

            missing = (
                df["fingerprint"]
                .str.strip()
                .eq("")
            )

            df.loc[
                missing,
                "fingerprint",
            ] = df.loc[
                missing,
                "sql",
            ].map(fingerprint_sql)

            role_col = pick_column(
                df,
                [
                    "workload_type",
                    "workload_class",
                    "user_role",
                ],
            )

            if role_col:
                df["target_class"] = (
                    df[role_col]
                    .fillna("unknown")
                    .astype(str)
                    .str.lower()
                )
            else:
                df["target_class"] = "unknown"

            ts_col = pick_column(
                df,
                [
                    "monitored_at",
                    "timestamp",
                    "executed_at",
                    "recorded_at",
                ],
            )

            if ts_col:
                df["timestamp"] = pd.to_datetime(
                    df[ts_col],
                    utc=True,
                    errors="coerce",
                )
            else:
                df["timestamp"] = pd.NaT

            if "session_id" not in df:
                df["session_id"] = file.stem

            df["source_row"] = np.arange(
                len(df)
            )

            df["source_file"] = str(file)

            for canonical, aliases in PLANNER_ALIASES.items():
                original = pick_column(
                    df,
                    aliases,
                )

                if original:
                    df[canonical] = pd.to_numeric(
                        df[original],
                        errors="coerce",
                    )

            frames.append(df)

            sources.append({
                "path": str(file),
                "sha256": sha256_file(file),
                "rows": len(df),
            })

    if not frames:
        raise RuntimeError(
            "No valid monitoring files found."
        )

    data = pd.concat(
        frames,
        ignore_index=True,
    )

    valid = (
        data["sql"].str.strip().ne("")
        & data["execution_time_ms"].notna()
        & np.isfinite(
            data["execution_time_ms"]
        )
        & (
            data["execution_time_ms"] >= 0
        )
    )

    data = (
        data.loc[valid]
        .reset_index(drop=True)
    )

    if len(data) < 50:
        raise RuntimeError(
            "Insufficient valid monitoring records."
        )

    return data, sources

def feature_engineering(df):
    df = df.copy()

    df["model_text"] = (
        df["sql"].map(normalize_sql)
    )

    text = df["model_text"]

    df["sql_length"] = text.str.len()
    df["token_count"] = (
        text.str.split().str.len()
    )

    patterns = {
        "join_count": r"\bjoin\b",
        "where_count": r"\bwhere\b",
        "group_count": r"\bgroup\s+by\b",
        "order_count": r"\border\s+by\b",
        "having_count": r"\bhaving\b",
        "aggregate_count": (
            r"\b(?:count|sum|avg|min|max)\s*\("
        ),
        "subquery_count": r"\(\s*select\b",
        "predicate_count": (
            r"(?<![<>!])=(?!=)|>=|<=|<>|!="
        ),
        "distinct_count": r"\bdistinct\b",
        "limit_count": r"\blimit\b",
    }

    for name, pattern in patterns.items():
        df[name] = text.str.count(pattern)

    df["table_count"] = (
        text.str.count(r"\bfrom\b")
        + text.str.count(r"\bjoin\b")
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
        "has_null": (
            r"\bis\s+(?:not\s+)?null\b"
        ),
    }

    for name, pattern in flags.items():
        df[name] = (
            text.str.contains(
                pattern,
                regex=True,
            )
            .astype(int)
        )

    available_planner = []

    for name in PLANNER_ALIASES:
        if name in df:
            numeric = pd.to_numeric(
                df[name],
                errors="coerce",
            )

            if numeric.notna().any():
                df[name] = numeric
                available_planner.append(name)

    return df, available_planner

def make_features(numeric, text_mode="none"):
    transformers = [
        (
            "numeric",
            Pipeline([
                (
                    "impute",
                    SimpleImputer(
                        strategy="constant",
                        fill_value=0,
                    ),
                ),
                (
                    "scale",
                    StandardScaler(),
                ),
            ]),
            numeric,
        )
    ]

    if text_mode == "char":
        transformers.append(
            (
                "text",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(3, 5),
                    max_features=2500,
                    sublinear_tf=True,
                ),
                "model_text",
            )
        )

    elif text_mode == "word":
        transformers.append(
            (
                "text",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    max_features=2000,
                    token_pattern=r"(?u)\b\w+\b",
                    sublinear_tf=True,
                ),
                "model_text",
            )
        )

    return ColumnTransformer(
        transformers,
        sparse_threshold=0.3,
    )

def make_pipeline(numeric, model, text_mode="none"):
    return Pipeline([
        (
            "features",
            make_features(
                numeric,
                text_mode,
            ),
        ),
        ("model", model),
    ])

class CostCalibrator(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        mode="ridge",
        alpha=10.0,
    ):
        self.mode = mode
        self.alpha = alpha

    def _cost(self, X):
        if "estimated_cost" not in X:
            return np.zeros(
                (len(X), 1)
            )

        cost = pd.to_numeric(
            X["estimated_cost"],
            errors="coerce",
        ).fillna(0)

        return np.log1p(
            np.clip(
                cost.to_numpy(dtype=float),
                0,
                None,
            )
        ).reshape(-1, 1)

    def fit(self, X, y):
        self.model_ = (
            HuberRegressor(
                alpha=self.alpha,
                max_iter=2000,
            )
            if self.mode == "huber"
            else Ridge(
                alpha=self.alpha
            )
        )

        self.model_.fit(
            self._cost(X),
            y,
        )

        return self

    def predict(self, X):
        return self.model_.predict(
            self._cost(X)
        )

class CostResidualRegressor(
    BaseEstimator,
    RegressorMixin,
):
    def __init__(
        self,
        numeric=None,
        backend="catboost",
    ):
        self.numeric = numeric
        self.backend = backend

    def fit(self, X, y):
        self.calibrator_ = CostCalibrator(
            mode="ridge",
            alpha=10,
        )

        self.calibrator_.fit(X, y)

        residual = (
            np.asarray(y)
            - self.calibrator_.predict(X)
        )

        if self.backend == "catboost":
            estimator = CatBoostRegressor(
                iterations=240,
                depth=2,
                learning_rate=0.04,
                l2_leaf_reg=10,
                verbose=False,
                random_seed=SEED,
                thread_count=4,
                allow_writing_files=False,
            )

        elif self.backend == "xgboost":
            estimator = XGBRegressor(
                n_estimators=180,
                max_depth=2,
                learning_rate=0.04,
                min_child_weight=8,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=10,
                objective="reg:squarederror",
                random_state=SEED,
                n_jobs=4,
            )

        elif self.backend == "lightgbm":
            estimator = LGBMRegressor(
                n_estimators=180,
                max_depth=3,
                num_leaves=7,
                min_child_samples=12,
                learning_rate=0.04,
                reg_lambda=10,
                verbosity=-1,
                random_state=SEED,
                n_jobs=4,
            )

        else:
            estimator = HistGradientBoostingRegressor(
                max_iter=180,
                max_leaf_nodes=7,
                min_samples_leaf=12,
                learning_rate=0.04,
                l2_regularization=5,
                random_state=SEED,
            )

        self.residual_model_ = make_pipeline(
            self.numeric,
            estimator,
        )

        self.residual_model_.fit(
            X,
            residual,
        )

        return self

    def predict(self, X):
        return (
            self.calibrator_.predict(X)
            + self.residual_model_.predict(X)
        )

def regression_models(numeric):
    models = {
        "median_baseline": make_pipeline(
            numeric,
            DummyRegressor(
                strategy="median"
            ),
        ),
        "ridge_numeric": make_pipeline(
            numeric,
            Ridge(alpha=20),
        ),
        "ridge_char": make_pipeline(
            numeric,
            Ridge(alpha=100),
            "char",
        ),
        "elastic_net": make_pipeline(
            numeric,
            ElasticNet(
                alpha=0.1,
                l1_ratio=0.3,
                max_iter=5000,
                random_state=SEED,
            ),
        ),
        "huber": make_pipeline(
            numeric,
            HuberRegressor(
                alpha=2,
                max_iter=2000,
            ),
        ),
        "svr_rbf": make_pipeline(
            numeric,
            SVR(
                C=5,
                epsilon=0.15,
            ),
        ),
        "random_forest": make_pipeline(
            numeric,
            RandomForestRegressor(
                n_estimators=160,
                min_samples_leaf=3,
                max_features=0.8,
                random_state=SEED,
                n_jobs=-1,
            ),
        ),
        "extra_trees": make_pipeline(
            numeric,
            ExtraTreesRegressor(
                n_estimators=160,
                min_samples_leaf=3,
                random_state=SEED,
                n_jobs=-1,
            ),
        ),
        "hist_gradient_boosting": make_pipeline(
            numeric,
            HistGradientBoostingRegressor(
                max_iter=180,
                max_leaf_nodes=7,
                min_samples_leaf=12,
                learning_rate=0.04,
                l2_regularization=5,
                random_state=SEED,
            ),
        ),
    }

    if HAS_CATBOOST:
        models["catboost"] = make_pipeline(
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
        )

    if HAS_XGBOOST:
        models["xgboost"] = make_pipeline(
            numeric,
            XGBRegressor(
                n_estimators=180,
                max_depth=2,
                learning_rate=0.04,
                min_child_weight=8,
                reg_lambda=10,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="reg:squarederror",
                random_state=SEED,
                n_jobs=4,
            ),
        )

    if HAS_LIGHTGBM:
        models["lightgbm"] = make_pipeline(
            numeric,
            LGBMRegressor(
                n_estimators=180,
                num_leaves=7,
                max_depth=3,
                min_child_samples=12,
                learning_rate=0.04,
                reg_lambda=10,
                verbosity=-1,
                random_state=SEED,
                n_jobs=4,
            ),
        )

    if "estimated_cost" in numeric:
        models["cost_calibration_ridge"] = (
            CostCalibrator(
                "ridge",
                10,
            )
        )

        models["cost_calibration_huber"] = (
            CostCalibrator(
                "huber",
                2,
            )
        )

        models["cost_residual_histgb"] = (
            CostResidualRegressor(
                numeric=numeric,
                backend="histgb",
            )
        )

        if HAS_CATBOOST:
            models["cost_residual_catboost"] = (
                CostResidualRegressor(
                    numeric=numeric,
                    backend="catboost",
                )
            )

        if HAS_XGBOOST:
            models["cost_residual_xgboost"] = (
                CostResidualRegressor(
                    numeric=numeric,
                    backend="xgboost",
                )
            )

        if HAS_LIGHTGBM:
            models["cost_residual_lightgbm"] = (
                CostResidualRegressor(
                    numeric=numeric,
                    backend="lightgbm",
                )
            )

    return models

def binary_models(numeric):
    models = {
        "majority_baseline": make_pipeline(
            numeric,
            DummyClassifier(
                strategy="most_frequent"
            ),
        ),
        "logistic_numeric": make_pipeline(
            numeric,
            LogisticRegression(
                C=0.5,
                class_weight="balanced",
                max_iter=4000,
                random_state=SEED,
            ),
        ),
        "logistic_char": make_pipeline(
            numeric,
            LogisticRegression(
                C=0.5,
                class_weight="balanced",
                max_iter=4000,
                random_state=SEED,
            ),
            "char",
        ),
        "linear_svm_char": make_pipeline(
            numeric,
            LinearSVC(
                C=0.1,
                class_weight="balanced",
                max_iter=15000,
                dual=False,
                random_state=SEED,
            ),
            "char",
        ),
        "rbf_svm": make_pipeline(
            numeric,
            SVC(
                C=2,
                kernel="rbf",
                class_weight="balanced",
                probability=True,
                random_state=SEED,
            ),
        ),
        "random_forest": make_pipeline(
            numeric,
            RandomForestClassifier(
                n_estimators=160,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                random_state=SEED,
                n_jobs=-1,
            ),
        ),
        "extra_trees": make_pipeline(
            numeric,
            ExtraTreesClassifier(
                n_estimators=160,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=SEED,
                n_jobs=-1,
            ),
        ),
        "hist_gradient_boosting": make_pipeline(
            numeric,
            HistGradientBoostingClassifier(
                max_iter=180,
                max_leaf_nodes=7,
                min_samples_leaf=12,
                learning_rate=0.04,
                l2_regularization=5,
                random_state=SEED,
            ),
        ),
    }

    if HAS_CATBOOST:
        models["catboost"] = make_pipeline(
            numeric,
            CatBoostClassifier(
                iterations=240,
                depth=2,
                learning_rate=0.04,
                l2_leaf_reg=10,
                verbose=False,
                random_seed=SEED,
                thread_count=4,
                allow_writing_files=False,
            ),
        )

    if HAS_XGBOOST:
        models["xgboost"] = make_pipeline(
            numeric,
            XGBClassifier(
                n_estimators=180,
                max_depth=2,
                learning_rate=0.04,
                min_child_weight=8,
                reg_lambda=10,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=SEED,
                n_jobs=4,
            ),
        )

    if HAS_LIGHTGBM:
        models["lightgbm"] = make_pipeline(
            numeric,
            LGBMClassifier(
                n_estimators=180,
                num_leaves=7,
                max_depth=3,
                min_child_samples=12,
                learning_rate=0.04,
                reg_lambda=10,
                verbosity=-1,
                random_state=SEED,
                n_jobs=4,
            ),
        )

    return models

def role_models(numeric):
    models = {
        "majority": make_pipeline(
            numeric,
            DummyClassifier(
                strategy="most_frequent"
            ),
        ),
        "logistic_char": make_pipeline(
            numeric,
            LogisticRegression(
                C=0.5,
                class_weight="balanced",
                max_iter=4000,
                random_state=SEED,
            ),
            "char",
        ),
        "linear_svm_char": make_pipeline(
            numeric,
            LinearSVC(
                C=0.1,
                class_weight="balanced",
                max_iter=15000,
                dual=False,
                random_state=SEED,
            ),
            "char",
        ),
        "random_forest": make_pipeline(
            numeric,
            RandomForestClassifier(
                n_estimators=160,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                random_state=SEED,
                n_jobs=-1,
            ),
        ),
    }

    if HAS_CATBOOST:
        models["catboost"] = make_pipeline(
            numeric,
            CatBoostClassifier(
                iterations=240,
                depth=2,
                learning_rate=0.04,
                l2_leaf_reg=10,
                verbose=False,
                random_seed=SEED,
                thread_count=4,
                allow_writing_files=False,
            ),
        )

    return models

def predict_log_to_ms(
    model,
    X,
    train_max_ms,
):
    pred = np.asarray(
        model.predict(X),
        dtype=float,
    )

    upper = max(
        1000.0,
        float(train_max_ms) * 5,
    )

    return np.expm1(
        np.clip(
            pred,
            0,
            np.log1p(upper),
        )
    )

def regression_metrics(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)

    ratio = np.maximum(
        (pred + 1) / (y + 1),
        (y + 1) / (pred + 1),
    )

    correlation = np.nan

    if (
        len(y) >= 3
        and np.std(y) > EPS
        and np.std(pred) > EPS
    ):
        correlation = spearmanr(
            y,
            pred,
        ).statistic

    return {
        "mae_ms": float(
            mean_absolute_error(y, pred)
        ),
        "rmse_ms": float(
            np.sqrt(
                mean_squared_error(y, pred)
            )
        ),
        "r2": (
            float(r2_score(y, pred))
            if len(y) > 1
            else np.nan
        ),
        "log_mae": float(
            mean_absolute_error(
                np.log1p(y),
                np.log1p(pred),
            )
        ),
        "median_qerror": float(
            np.median(ratio)
        ),
        "p90_qerror": float(
            np.quantile(ratio, 0.9)
        ),
        "spearman": float(correlation),
    }

def binary_metrics(y, pred, score=None):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)

    result = {
        "accuracy": float(
            accuracy_score(y, pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, pred)
        ),
        "macro_f1": float(
            f1_score(
                y,
                pred,
                average="macro",
                labels=[0, 1],
                zero_division=0,
            )
        ),
        "precision_expensive": float(
            precision_score(
                y,
                pred,
                zero_division=0,
            )
        ),
        "recall_expensive": float(
            recall_score(
                y,
                pred,
                zero_division=0,
            )
        ),
        "f1_expensive": float(
            f1_score(
                y,
                pred,
                zero_division=0,
            )
        ),
    }

    if (
        score is not None
        and len(np.unique(y)) == 2
    ):
        try:
            result["roc_auc"] = float(
                roc_auc_score(y, score)
            )
        except ValueError:
            result["roc_auc"] = np.nan

        result["average_precision"] = float(
            average_precision_score(
                y,
                score,
            )
        )

    else:
        result["roc_auc"] = np.nan
        result["average_precision"] = np.nan

    return result

def model_scores(model, X):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)

        classes = (
            model.classes_
            if hasattr(model, "classes_")
            else model.named_steps["model"].classes_
        )

        classes = list(classes)

        if 1 in classes:
            return probs[:, classes.index(1)]

    if hasattr(model, "decision_function"):
        return np.asarray(
            model.decision_function(X),
            dtype=float,
        )

    return np.asarray(
        model.predict(X),
        dtype=float,
    )

def make_outer_splits(
    df,
    repeats,
    test_size,
    seed=SEED,
):
    splitter = GroupShuffleSplit(
        n_splits=repeats,
        test_size=test_size,
        random_state=seed,
    )

    return list(
        splitter.split(
            df,
            groups=df["fingerprint"],
        )
    )

def inner_group_splits(
    df,
    folds,
    seed,
):
    groups = df["fingerprint"].to_numpy()

    unique = np.unique(groups)

    if len(unique) < 4:
        raise RuntimeError(
            "Too few fingerprints for inner CV."
        )

    splitter = GroupKFold(
        n_splits=min(folds, len(unique)),
        shuffle=True,
        random_state=seed,
    )

    return list(
        splitter.split(
            df,
            groups=groups,
        )
    )

def expensive_threshold(
    train,
    quantile,
    fixed_threshold,
):
    if fixed_threshold is not None:
        return float(fixed_threshold)

    # Threshold derived exclusively from the
    # outer training partition.
    return float(
        train["execution_time_ms"].quantile(
            quantile
        )
    )

def expensive_label(df, threshold):
    return (
        df["execution_time_ms"]
        .to_numpy(dtype=float)
        >= threshold
    ).astype(int)

def fingerprint_binary_metrics(
    data,
    actual_col="actual",
    predicted_col="predicted",
):
    records = []

    for fp, group in data.groupby("fingerprint"):
        actual = group[actual_col].to_numpy()
        pred = group[predicted_col].to_numpy()

        records.append({
            "fingerprint": fp,
            "accuracy": accuracy_score(
                actual,
                pred,
            ),
            "balanced_accuracy": (
                balanced_accuracy_score(
                    actual,
                    pred,
                )
                if len(np.unique(actual)) == 2
                else np.nan
            ),
            "macro_f1": f1_score(
                actual,
                pred,
                average="macro",
                labels=[0, 1],
                zero_division=0,
            ),
        })

    return pd.DataFrame(records)

def evaluate_expensive_classifier(
    df,
    numeric,
    out,
    outer_repeats,
    inner_folds,
    test_size,
    expensive_quantile,
    fixed_threshold,
):
    models = binary_models(numeric)

    all_outer_rows = []
    all_predictions = []
    inner_rows = []
    failures = []

    print("\n" + "=" * 95)
    print("EXPERIMENT A — EXPENSIVE QUERY CLASSIFICATION")
    print("Nested fingerprint-grouped model selection")
    print("=" * 95)

    outer_splits = make_outer_splits(
        df,
        outer_repeats,
        test_size,
        seed=SEED + 100,
    )

    for repeat, (dev_idx, test_idx) in enumerate(
        outer_splits,
        start=1,
    ):
        dev = (
            df.iloc[dev_idx]
            .reset_index(drop=True)
        )

        test = (
            df.iloc[test_idx]
            .reset_index(drop=True)
        )

        assert set(
            dev["fingerprint"]
        ).isdisjoint(
            set(test["fingerprint"])
        )

        inner_splits = inner_group_splits(
            dev,
            inner_folds,
            SEED + repeat,
        )

        candidate_results = []

        for name, definition in models.items():
            fold_scores = []

            for fold, (tr, va) in enumerate(
                inner_splits,
                start=1,
            ):
                train = dev.iloc[tr]
                valid = dev.iloc[va]

                threshold = expensive_threshold(
                    train,
                    expensive_quantile,
                    fixed_threshold,
                )

                y_train = expensive_label(
                    train,
                    threshold,
                )

                y_valid = expensive_label(
                    valid,
                    threshold,
                )

                if len(np.unique(y_train)) < 2:
                    continue

                try:
                    model = clone(definition)

                    model.fit(
                        train,
                        y_train,
                    )

                    pred = model.predict(
                        valid
                    )

                    metrics = binary_metrics(
                        y_valid,
                        pred,
                    )

                    fold_scores.append(
                        metrics["balanced_accuracy"]
                    )

                    inner_rows.append({
                        "outer_repeat": repeat,
                        "inner_fold": fold,
                        "model": name,
                        "threshold_ms": threshold,
                        **metrics,
                    })

                except Exception as exc:
                    failures.append({
                        "task": "expensive",
                        "repeat": repeat,
                        "fold": fold,
                        "model": name,
                        "error": repr(exc),
                    })

            if len(fold_scores) == len(inner_splits):
                candidate_results.append({
                    "model": name,
                    "mean_inner_balanced_accuracy": (
                        float(
                            np.mean(fold_scores)
                        )
                    ),
                })

        if not candidate_results:
            print(
                f"Repeat {repeat}: no valid classifier."
            )
            continue

        selection = pd.DataFrame(
            candidate_results
        ).sort_values(
            [
                "mean_inner_balanced_accuracy",
                "model",
            ],
            ascending=[False, True],
        )

        winner = selection.iloc[0]["model"]

        final_threshold = expensive_threshold(
            dev,
            expensive_quantile,
            fixed_threshold,
        )

        y_dev = expensive_label(
            dev,
            final_threshold,
        )

        y_test = expensive_label(
            test,
            final_threshold,
        )

        if len(np.unique(y_dev)) < 2:
            continue

        model = clone(models[winner])

        model.fit(dev, y_dev)

        pred = model.predict(test)
        scores = model_scores(model, test)

        metrics = binary_metrics(
            y_test,
            pred,
            scores,
        )

        all_outer_rows.append({
            "repeat": repeat,
            "selected_model": winner,
            "threshold_ms": final_threshold,
            "train_fingerprints": (
                dev["fingerprint"].nunique()
            ),
            "test_fingerprints": (
                test["fingerprint"].nunique()
            ),
            "test_positive_fraction": (
                float(y_test.mean())
            ),
            **metrics,
        })

        all_predictions.append(
            pd.DataFrame({
                "repeat": repeat,
                "fingerprint": (
                    test["fingerprint"].to_numpy()
                ),
                "actual_ms": (
                    test["execution_time_ms"].to_numpy()
                ),
                "threshold_ms": final_threshold,
                "actual": y_test,
                "predicted": pred,
                "score": scores,
                "selected_model": winner,
            })
        )

        print(
            f"Outer {repeat}/{outer_repeats} | "
            f"{winner:26s} | "
            f"accuracy={metrics['accuracy']:.4f} | "
            f"balanced={metrics['balanced_accuracy']:.4f} | "
            f"macroF1={metrics['macro_f1']:.4f} | "
            f"expensive_recall={metrics['recall_expensive']:.4f}"
        )

    if not all_outer_rows:
        return {
            "status": "FAILED",
            "reason": "No valid outer evaluation.",
        }

    outer = pd.DataFrame(all_outer_rows)

    outer.to_csv(
        out / "expensive_outer_results.csv",
        index=False,
    )

    if inner_rows:
        pd.DataFrame(inner_rows).to_csv(
            out / "expensive_inner_results.csv",
            index=False,
        )

    predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    predictions.to_csv(
        out / "expensive_predictions.csv",
        index=False,
    )

    summary = {
        "status": "EVALUATED",
        "outer_repeats": len(outer),
        "mean_accuracy": float(
            outer["accuracy"].mean()
        ),
        "std_accuracy": float(
            outer["accuracy"].std()
        ),
        "mean_balanced_accuracy": float(
            outer["balanced_accuracy"].mean()
        ),
        "mean_macro_f1": float(
            outer["macro_f1"].mean()
        ),
        "mean_expensive_recall": float(
            outer["recall_expensive"].mean()
        ),
        "mean_expensive_precision": float(
            outer["precision_expensive"].mean()
        ),
        "mean_average_precision": float(
            outer["average_precision"].mean()
        ),
        "model_selection_frequency": (
            outer["selected_model"]
            .value_counts()
            .to_dict()
        ),
        "important": (
            "Outer metrics evaluate an inner-CV "
            "model-selection procedure. Repeated "
            "outer splits overlap and should not "
            "be treated as independent samples."
        ),
    }

    save_json(
        out / "expensive_summary.json",
        summary,
    )

    save_json(
        out / "expensive_failures.json",
        failures,
    )

    print("\nEXPENSIVE CLASSIFICATION SUMMARY")
    print(
        json.dumps(
            to_native(summary),
            indent=2,
        )
    )

    return {
        **summary,
        "outer": outer,
        "predictions": predictions,
        "models": models,
    }

def evaluate_unseen_regression(
    df,
    numeric,
    out,
    outer_repeats,
    inner_folds,
    test_size,
):
    models = regression_models(numeric)

    outer_rows = []
    inner_rows = []
    predictions = []
    failures = []

    print("\n" + "=" * 95)
    print("EXPERIMENT B — UNSEEN QUERY LATENCY")
    print("Nested fingerprint-grouped regression evaluation")
    print("=" * 95)

    outer_splits = make_outer_splits(
        df,
        outer_repeats,
        test_size,
        seed=SEED + 300,
    )

    for repeat, (dev_idx, test_idx) in enumerate(
        outer_splits,
        start=1,
    ):
        dev = (
            df.iloc[dev_idx]
            .reset_index(drop=True)
        )

        test = (
            df.iloc[test_idx]
            .reset_index(drop=True)
        )

        inner_splits = inner_group_splits(
            dev,
            inner_folds,
            SEED + repeat + 1000,
        )

        candidates = []

        for name, definition in models.items():
            fold_scores = []

            for fold, (tr, va) in enumerate(
                inner_splits,
                start=1,
            ):
                train = dev.iloc[tr]
                valid = dev.iloc[va]

                model = clone(definition)

                try:
                    model.fit(
                        train,
                        np.log1p(
                            train["execution_time_ms"]
                        ),
                    )

                    pred = predict_log_to_ms(
                        model,
                        valid,
                        train["execution_time_ms"].max(),
                    )

                    metrics = regression_metrics(
                        valid["execution_time_ms"],
                        pred,
                    )

                    # Equal weight to each independent
                    # SQL fingerprint, not each row.
                    tmp = pd.DataFrame({
                        "fingerprint": (
                            valid["fingerprint"].to_numpy()
                        ),
                        "error": np.abs(
                            np.log1p(
                                valid["execution_time_ms"]
                                .to_numpy()
                            )
                            - np.log1p(pred)
                        ),
                    })

                    fingerprint_log_mae = (
                        tmp.groupby("fingerprint")
                        ["error"]
                        .mean()
                        .mean()
                    )

                    fold_scores.append(
                        float(fingerprint_log_mae)
                    )

                    inner_rows.append({
                        "outer_repeat": repeat,
                        "inner_fold": fold,
                        "model": name,
                        "fingerprint_log_mae": (
                            fingerprint_log_mae
                        ),
                        **metrics,
                    })

                except Exception as exc:
                    failures.append({
                        "task": "unseen_regression",
                        "repeat": repeat,
                        "fold": fold,
                        "model": name,
                        "error": repr(exc),
                    })

            if len(fold_scores) == len(inner_splits):
                candidates.append({
                    "model": name,
                    "inner_log_mae": float(
                        np.mean(fold_scores)
                    ),
                })

        if not candidates:
            continue

        selection = pd.DataFrame(
            candidates
        ).sort_values(
            ["inner_log_mae", "model"],
            ascending=[True, True],
        )

        winner = selection.iloc[0]["model"]

        final_model = clone(
            models[winner]
        )

        final_model.fit(
            dev,
            np.log1p(
                dev["execution_time_ms"]
            ),
        )

        pred = predict_log_to_ms(
            final_model,
            test,
            dev["execution_time_ms"].max(),
        )

        metrics = regression_metrics(
            test["execution_time_ms"],
            pred,
        )

        tmp = pd.DataFrame({
            "fingerprint": (
                test["fingerprint"].to_numpy()
            ),
            "log_error": np.abs(
                np.log1p(
                    test["execution_time_ms"]
                    .to_numpy()
                )
                - np.log1p(pred)
            ),
            "abs_error_ms": np.abs(
                test["execution_time_ms"]
                .to_numpy()
                - pred
            ),
        })

        fp_log_mae = (
            tmp.groupby("fingerprint")
            ["log_error"]
            .mean()
            .mean()
        )

        fp_mae_ms = (
            tmp.groupby("fingerprint")
            ["abs_error_ms"]
            .mean()
            .mean()
        )

        outer_rows.append({
            "repeat": repeat,
            "selected_model": winner,
            "train_fingerprints": (
                dev["fingerprint"].nunique()
            ),
            "test_fingerprints": (
                test["fingerprint"].nunique()
            ),
            "fingerprint_log_mae": fp_log_mae,
            "fingerprint_mae_ms": fp_mae_ms,
            **metrics,
        })

        predictions.append(
            pd.DataFrame({
                "repeat": repeat,
                "fingerprint": (
                    test["fingerprint"].to_numpy()
                ),
                "actual_ms": (
                    test["execution_time_ms"].to_numpy()
                ),
                "predicted_ms": pred,
                "selected_model": winner,
            })
        )

        print(
            f"Outer {repeat}/{outer_repeats} | "
            f"{winner:26s} | "
            f"fp_logMAE={fp_log_mae:.4f} | "
            f"fp_MAE={fp_mae_ms:.2f}ms | "
            f"R2={metrics['r2']:.4f}"
        )

    if not outer_rows:
        return {
            "status": "FAILED",
            "reason": "No valid regression outer splits.",
        }

    outer = pd.DataFrame(outer_rows)

    outer.to_csv(
        out / "unseen_outer_results.csv",
        index=False,
    )

    if inner_rows:
        pd.DataFrame(inner_rows).to_csv(
            out / "unseen_inner_results.csv",
            index=False,
        )

    all_pred = pd.concat(
        predictions,
        ignore_index=True,
    )

    all_pred.to_csv(
        out / "unseen_predictions.csv",
        index=False,
    )

    summary = {
        "status": "EVALUATED",
        "outer_repeats": len(outer),
        "mean_fingerprint_log_mae": float(
            outer["fingerprint_log_mae"].mean()
        ),
        "mean_fingerprint_mae_ms": float(
            outer["fingerprint_mae_ms"].mean()
        ),
        "mean_row_mae_ms": float(
            outer["mae_ms"].mean()
        ),
        "median_outer_r2": float(
            outer["r2"].median()
        ),
        "mean_median_qerror": float(
            outer["median_qerror"].mean()
        ),
        "model_selection_frequency": (
            outer["selected_model"]
            .value_counts()
            .to_dict()
        ),
        "warning": (
            "Nested grouped evaluation estimates "
            "performance of model selection. "
            "Outer repeats overlap; independent "
            "new fingerprints are still required."
        ),
    }

    save_json(
        out / "unseen_summary.json",
        summary,
    )

    save_json(
        out / "unseen_failures.json",
        failures,
    )

    print("\nUNSEEN LATENCY SUMMARY")
    print(
        json.dumps(
            to_native(summary),
            indent=2,
        )
    )

    return {
        **summary,
        "outer": outer,
        "predictions": all_pred,
        "models": models,
    }

def known_query_evaluation(df, out, warmup=3):
    data = df.dropna(
        subset=["timestamp"]
    ).copy()

    if len(data) < 50:
        return {
            "status": "INSUFFICIENT_TIMESTAMPS"
        }

    data = data.sort_values(
        ["timestamp", "source_row"],
        kind="mergesort",
    ).reset_index(drop=True)

    if data["timestamp"].nunique() < 10:
        return {
            "status": "INSUFFICIENT_TIME_VARIATION"
        }

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

    history = {}
    global_history = []
    rows = []

    for timestamp, batch in data.groupby(
        "timestamp",
        sort=True,
    ):
        updates = []

        for _, record in batch.iterrows():
            fp = record["fingerprint"]
            actual = float(
                record["execution_time_ms"]
            )

            values = history.get(fp, [])

            if (
                len(values) >= warmup
                and global_history
            ):
                arr = np.asarray(
                    values,
                    dtype=float,
                )

                global_median = float(
                    np.median(global_history)
                )

                estimates = {
                    "global_median": global_median,
                    "last": float(arr[-1]),
                    "median": float(
                        np.median(arr)
                    ),
                    "rolling_median_5": float(
                        np.median(arr[-5:])
                    ),
                    "rolling_median_10": float(
                        np.median(arr[-10:])
                    ),
                }

                for alpha, name in [
                    (0.3, "ewma_03"),
                    (0.7, "ewma_07"),
                ]:
                    estimate = arr[0]

                    for value in arr[1:]:
                        estimate = (
                            alpha * value
                            + (1 - alpha) * estimate
                        )

                    estimates[name] = float(
                        estimate
                    )

                weight = (
                    len(arr)
                    / (len(arr) + 5)
                )

                estimates["shrinkage_median"] = (
                    weight * np.median(arr)
                    + (1 - weight) * global_median
                )

                for name in methods:
                    rows.append({
                        "model": name,
                        "fingerprint": fp,
                        "timestamp": timestamp,
                        "actual_ms": actual,
                        "predicted_ms": estimates[name],
                    })

            updates.append(
                (fp, actual)
            )

        # Update history only after predicting
        # all records at this timestamp.
        for fp, actual in updates:
            history.setdefault(
                fp, []
            ).append(actual)

            global_history.append(actual)

    if not rows:
        return {
            "status": "INSUFFICIENT_HISTORY"
        }

    pred = pd.DataFrame(rows)

    pred.to_csv(
        out / "known_predictions.csv",
        index=False,
    )

    results = []

    for name, group in pred.groupby("model"):
        metrics = regression_metrics(
            group["actual_ms"],
            group["predicted_ms"],
        )

        tmp = group.copy()

        tmp["log_error"] = np.abs(
            np.log1p(tmp["actual_ms"])
            - np.log1p(tmp["predicted_ms"])
        )

        fp_log_mae = (
            tmp.groupby("fingerprint")
            ["log_error"]
            .mean()
            .mean()
        )

        results.append({
            "model": name,
            "fingerprint_log_mae": fp_log_mae,
            **metrics,
        })

    leaderboard = pd.DataFrame(
        results
    ).sort_values(
        [
            "fingerprint_log_mae",
            "mae_ms",
        ]
    )

    leaderboard.to_csv(
        out / "known_leaderboard.csv",
        index=False,
    )

    winner = leaderboard.iloc[0]["model"]

    print("\n" + "=" * 95)
    print("EXPERIMENT C — KNOWN QUERY LATENCY")
    print("=" * 95)

    print(
        leaderboard.round(4).to_string(
            index=False
        )
    )

    print("\nBEST KNOWN-QUERY MODEL:", winner)

    return {
        "status": "EVALUATED",
        "winner": winner,
        "leaderboard": leaderboard,
        "note": (
            "Within-session chronological evaluation. "
            "Validate on future independent sessions."
        ),
    }

def role_classification(df, numeric, out, folds):
    data = df[
        df["target_class"] != "unknown"
    ].reset_index(drop=True)

    if data["target_class"].nunique() < 2:
        return {
            "status": "INSUFFICIENT_ROLE_LABELS"
        }

    labels = sorted(
        data["target_class"].unique()
    )

    models = role_models(numeric)

    splitter = GroupKFold(
        n_splits=min(
            folds,
            data["fingerprint"].nunique(),
        ),
        shuffle=True,
        random_state=SEED,
    )

    rows = []

    for name, definition in models.items():
        for fold, (tr, te) in enumerate(
            splitter.split(
                data,
                groups=data["fingerprint"],
            ),
            start=1,
        ):
            train = data.iloc[tr]
            test = data.iloc[te]

            if train["target_class"].nunique() < 2:
                continue

            try:
                model = clone(definition)

                model.fit(
                    train,
                    train["target_class"],
                )

                pred = model.predict(test)

                rows.append({
                    "model": name,
                    "fold": fold,
                    "accuracy": accuracy_score(
                        test["target_class"],
                        pred,
                    ),
                    "macro_f1": f1_score(
                        test["target_class"],
                        pred,
                        labels=labels,
                        average="macro",
                        zero_division=0,
                    ),
                })

            except Exception:
                continue

    if not rows:
        return {"status": "FAILED"}

    scores = pd.DataFrame(rows)

    scores.to_csv(
        out / "role_folds.csv",
        index=False,
    )

    leaderboard = (
        scores.groupby("model")
        .agg(
            accuracy=("accuracy", "mean"),
            macro_f1=("macro_f1", "mean"),
            folds=("fold", "count"),
        )
        .reset_index()
    )

    leaderboard = leaderboard[
        leaderboard["folds"]
        == min(
            folds,
            data["fingerprint"].nunique(),
        )
    ].sort_values(
        "macro_f1",
        ascending=False,
    )

    leaderboard.to_csv(
        out / "role_leaderboard.csv",
        index=False,
    )

    if leaderboard.empty:
        return {
            "status": "NO_COMPLETE_MODELS"
        }

    winner = leaderboard.iloc[0]["model"]

    print("\n" + "=" * 95)
    print("EXPERIMENT D — ROLE CLASSIFICATION DIAGNOSTIC")
    print("=" * 95)

    print(
        leaderboard.round(4).to_string(
            index=False
        )
    )

    return {
        "status": "EVALUATED",
        "winner": winner,
        "leaderboard": leaderboard,
    }

def conformal_intervals(
    df,
    numeric,
    model_definition,
    out,
    alpha=0.10,
    repeats=5,
):
    splitter = GroupShuffleSplit(
        n_splits=repeats,
        test_size=0.25,
        random_state=SEED + 900,
    )

    rows = []

    for repeat, (dev_idx, test_idx) in enumerate(
        splitter.split(
            df,
            groups=df["fingerprint"],
        ),
        start=1,
    ):
        dev = (
            df.iloc[dev_idx]
            .reset_index(drop=True)
        )

        test = (
            df.iloc[test_idx]
            .reset_index(drop=True)
        )

        inner = GroupShuffleSplit(
            n_splits=1,
            test_size=0.3,
            random_state=SEED + repeat,
        )

        tr, ca = next(
            inner.split(
                dev,
                groups=dev["fingerprint"],
            )
        )

        proper = dev.iloc[tr]
        calibration = dev.iloc[ca]

        model = clone(
            model_definition
        )

        model.fit(
            proper,
            np.log1p(
                proper["execution_time_ms"]
            ),
        )

        max_ms = proper[
            "execution_time_ms"
        ].max()

        cal_pred = predict_log_to_ms(
            model,
            calibration,
            max_ms,
        )

        test_pred = predict_log_to_ms(
            model,
            test,
            max_ms,
        )

        residual = np.abs(
            np.log1p(
                calibration["execution_time_ms"]
                .to_numpy()
            )
            - np.log1p(cal_pred)
        )

        score_df = pd.DataFrame({
            "fingerprint": (
                calibration["fingerprint"]
                .to_numpy()
            ),
            "residual": residual,
        })

        # Conservative maximum error per
        # independent fingerprint group.
        scores = (
            score_df.groupby("fingerprint")
            ["residual"]
            .max()
            .to_numpy()
        )

        m = len(scores)

        if m < 2:
            continue

        rank = math.ceil(
            (m + 1) * (1 - alpha)
        )

        if rank > m:
            # No finite distribution-free
            # conformal quantile at this level.
            radius = np.inf
        else:
            radius = np.sort(scores)[
                rank - 1
            ]

        center = np.log1p(
            test_pred
        )

        lower = np.expm1(
            np.maximum(
                0,
                center - radius,
            )
        )

        upper = np.expm1(
            center + radius
        )

        actual = test[
            "execution_time_ms"
        ].to_numpy()

        covered = (
            (actual >= lower)
            & (actual <= upper)
        )

        rows.append({
            "repeat": repeat,
            "coverage": float(
                covered.mean()
            ),
            "mean_width_ms": (
                float(
                    np.mean(
                        upper - lower
                    )
                )
                if np.isfinite(radius)
                else np.inf
            ),
            "finite_interval": (
                bool(
                    np.isfinite(radius)
                )
            ),
            "calibration_fingerprints": m,
            "test_fingerprints": (
                test["fingerprint"].nunique()
            ),
        })

    if not rows:
        return {
            "status": "INSUFFICIENT_CALIBRATION"
        }

    report = pd.DataFrame(rows)

    report.to_csv(
        out / "conformal_results.csv",
        index=False,
    )

    summary = {
        "status": "EVALUATED",
        "nominal_coverage": 1 - alpha,
        "mean_coverage": float(
            report["coverage"].mean()
        ),
        "finite_interval_fraction": float(
            report["finite_interval"].mean()
        ),
        "mean_finite_width_ms": (
            float(
                report.loc[
                    report["finite_interval"],
                    "mean_width_ms",
                ].mean()
            )
            if report["finite_interval"].any()
            else None
        ),
        "warning": (
            "Small calibration fingerprint counts "
            "can make valid conformal intervals "
            "infinite at 90% nominal coverage."
        ),
    }

    save_json(
        out / "conformal_summary.json",
        summary,
    )

    return summary

def phase9_ranking(df, out):
    grouped = (
        df.groupby("fingerprint")
        .agg(
            frequency=(
                "execution_time_ms",
                "size",
            ),
            mean_ms=(
                "execution_time_ms",
                "mean",
            ),
            median_ms=(
                "execution_time_ms",
                "median",
            ),
            p90_ms=(
                "execution_time_ms",
                lambda x: x.quantile(0.9),
            ),
        )
        .reset_index()
    )

    grouped["observed_total_ms"] = (
        grouped["frequency"]
        * grouped["mean_ms"]
    )

    if "estimated_cost" in df:
        costs = (
            df.groupby("fingerprint")
            ["estimated_cost"]
            .median()
            .rename("planner_cost")
            .reset_index()
        )

        grouped = grouped.merge(
            costs,
            on="fingerprint",
            how="left",
        )

    else:
        grouped["planner_cost"] = np.nan

    strategies = {
        "observed_total_oracle": (
            "observed_total_ms"
        ),
        "frequency": "frequency",
        "median_latency": "median_ms",
        "planner_cost": "planner_cost",
    }

    total = max(
        grouped["observed_total_ms"].sum(),
        EPS,
    )

    oracle = grouped.sort_values(
        "observed_total_ms",
        ascending=False,
    )

    rows = []

    for strategy, column in strategies.items():
        if grouped[column].notna().sum() == 0:
            continue

        ranked = grouped.sort_values(
            column,
            ascending=False,
        )

        for k in [3, 5, 10]:
            if k > len(grouped):
                continue

            captured = (
                ranked.head(k)[
                    "observed_total_ms"
                ].sum()
                / total
            )

            oracle_capture = (
                oracle.head(k)[
                    "observed_total_ms"
                ].sum()
                / total
            )

            rows.append({
                "strategy": strategy,
                "top_k": k,
                "observed_burden_capture": (
                    captured
                ),
                "fraction_of_oracle": (
                    captured
                    / max(
                        oracle_capture,
                        EPS,
                    )
                ),
            })

    grouped.sort_values(
        "observed_total_ms",
        ascending=False,
    ).to_csv(
        out / "phase9_priorities.csv",
        index=False,
    )

    comparison = pd.DataFrame(rows)

    comparison.to_csv(
        out / "phase9_ranking_comparison.csv",
        index=False,
    )

    print("\n" + "=" * 95)
    print("EXPERIMENT E — PHASE 9 RANKING")
    print("=" * 95)

    print(
        comparison.round(4).to_string(
            index=False
        )
    )

    return {
        "status": "PROXY_EVALUATED",
        "query_fingerprints": len(grouped),
        "warning": (
            "This evaluates observed workload burden, "
            "not actual index speedup. "
            "Observed-total is an oracle for this "
            "proxy and cannot be used as evidence "
            "that a predictive model is superior."
        ),
    }

def plot_bar(
    frame,
    x,
    y,
    title,
    path,
):
    if frame is None or frame.empty:
        return

    fig, ax = plt.subplots(
        figsize=(
            10,
            max(5, 0.4 * len(frame)),
        )
    )

    data = frame.sort_values(y)

    ax.barh(
        data[x].astype(str),
        data[y],
    )

    ax.set_xlabel(y)
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=180,
    )
    plt.close(fig)

def choose_final_model(
    df,
    numeric,
    models,
    task,
    folds,
    expensive_quantile=None,
    fixed_threshold=None,
):
    # Full-dataset model selection using grouped CV.
    # This selection is distinct from nested outer
    # performance reporting.
    splits = inner_group_splits(
        df,
        folds,
        SEED + 2000,
    )

    rows = []

    for name, definition in models.items():
        values = []

        for tr, va in splits:
            train = df.iloc[tr]
            valid = df.iloc[va]

            model = clone(definition)

            try:
                if task == "regression":
                    model.fit(
                        train,
                        np.log1p(
                            train["execution_time_ms"]
                        ),
                    )

                    pred = predict_log_to_ms(
                        model,
                        valid,
                        train["execution_time_ms"].max(),
                    )

                    temp = pd.DataFrame({
                        "fingerprint": (
                            valid["fingerprint"]
                            .to_numpy()
                        ),
                        "error": np.abs(
                            np.log1p(
                                valid["execution_time_ms"]
                                .to_numpy()
                            )
                            - np.log1p(pred)
                        ),
                    })

                    score = (
                        temp.groupby("fingerprint")
                        ["error"]
                        .mean()
                        .mean()
                    )

                else:
                    threshold = expensive_threshold(
                        train,
                        expensive_quantile,
                        fixed_threshold,
                    )

                    y_train = expensive_label(
                        train,
                        threshold,
                    )

                    y_valid = expensive_label(
                        valid,
                        threshold,
                    )

                    if len(np.unique(y_train)) < 2:
                        continue

                    model.fit(
                        train,
                        y_train,
                    )

                    pred = model.predict(valid)

                    score = balanced_accuracy_score(
                        y_valid,
                        pred,
                    )

                values.append(
                    float(score)
                )

            except Exception:
                continue

        if len(values) == len(splits):
            rows.append({
                "model": name,
                "score": np.mean(values),
            })

    if not rows:
        return None, None, pd.DataFrame()

    selection = pd.DataFrame(rows)

    selection = selection.sort_values(
        ["score", "model"],
        ascending=[
            task == "regression",
            True,
        ],
    )

    winner = selection.iloc[0]["model"]

    final = clone(
        models[winner]
    )

    if task == "regression":
        final.fit(
            df,
            np.log1p(
                df["execution_time_ms"]
            ),
        )

    else:
        threshold = expensive_threshold(
            df,
            expensive_quantile,
            fixed_threshold,
        )

        final.fit(
            df,
            expensive_label(
                df,
                threshold,
            ),
        )

    return winner, final, selection

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        nargs="+",
        default=[str(DEFAULT_INPUT)],
    )

    parser.add_argument(
        "--outer-repeats",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--inner-folds",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--expensive-quantile",
        type=float,
        default=0.75,
    )

    parser.add_argument(
        "--expensive-threshold-ms",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--sql-only",
        action="store_true",
    )

    parser.add_argument(
        "--skip-role",
        action="store_true",
    )

    parser.add_argument(
        "--skip-intervals",
        action="store_true",
    )

    args = parser.parse_args()

    if args.outer_repeats < 2:
        parser.error(
            "--outer-repeats must be at least 2"
        )

    if args.inner_folds < 2:
        parser.error(
            "--inner-folds must be at least 2"
        )

    if not 0 < args.test_size < 0.5:
        parser.error(
            "--test-size must be between 0 and 0.5"
        )

    if not 0 < args.expensive_quantile < 1:
        parser.error(
            "--expensive-quantile must be between 0 and 1"
        )

    if (
        args.expensive_threshold_ms is not None
        and args.expensive_threshold_ms < 0
    ):
        parser.error(
            "--expensive-threshold-ms cannot be negative"
        )

    started = time.perf_counter()

    run_id = datetime.now(
        timezone.utc
    ).strftime(
        "p8v5_%Y%m%d_%H%M%S_%f"
    )

    out = (
        OUTPUT_ROOT
        / "runs"
        / run_id
    )

    out.mkdir(
        parents=True,
        exist_ok=False,
    )

    print("=" * 95)
    print(
        "PHASE 8 V5 — ADAPTIVE HEALTHCARE "
        "DATABASE WORKLOAD INTELLIGENCE"
    )
    print("=" * 95)

    print("Run:", run_id)
    print("Output:", out)

    df, sources = load_data(
        args.input
    )

    df, planner = feature_engineering(
        df
    )

    numeric = list(SQL_FEATURES)

    if not args.sql_only:
        numeric.extend(planner)

    if set(numeric) & LEAKAGE_COLUMNS:
        raise RuntimeError(
            "Leakage-sensitive feature detected."
        )

    fingerprints = (
        df["fingerprint"].nunique()
    )

    if fingerprints < 10:
        raise RuntimeError(
            "At least 10 independent fingerprints "
            "are required for this evaluation."
        )

    print("\nDATASET")
    print("Rows:", len(df))
    print("Fingerprints:", fingerprints)
    print(
        "Sessions:",
        df["session_id"].nunique(),
    )
    print("Numeric features:", len(numeric))
    print(
        "Planner features:",
        [] if args.sql_only else planner,
    )

    save_json(
        out / "experiment_config.json",
        {
            "run_id": run_id,
            "sources": sources,
            "seed": SEED,
            "outer_repeats": args.outer_repeats,
            "inner_folds": args.inner_folds,
            "test_size": args.test_size,
            "expensive_quantile": (
                args.expensive_quantile
            ),
            "fixed_expensive_threshold_ms": (
                args.expensive_threshold_ms
            ),
            "sql_only": args.sql_only,
            "numeric_features": numeric,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "catboost_installed": HAS_CATBOOST,
            "xgboost_installed": HAS_XGBOOST,
            "lightgbm_installed": HAS_LIGHTGBM,
        },
    )

    print("\n[1/8] Expensive-query classification")

    expensive = evaluate_expensive_classifier(
        df,
        numeric,
        out,
        args.outer_repeats,
        args.inner_folds,
        args.test_size,
        args.expensive_quantile,
        args.expensive_threshold_ms,
    )

    print("\n[2/8] Unseen-query latency regression")

    unseen = evaluate_unseen_regression(
        df,
        numeric,
        out,
        args.outer_repeats,
        args.inner_folds,
        args.test_size,
    )

    print("\n[3/8] Known-query latency")

    known = known_query_evaluation(
        df,
        out,
    )

    print("\n[4/8] Role classification diagnostic")

    role = (
        {"status": "SKIPPED"}
        if args.skip_role
        else role_classification(
            df,
            numeric,
            out,
            args.inner_folds,
        )
    )

    print("\n[5/8] Final full-data model selection")

    best_reg_name = None
    best_reg_model = None

    if unseen.get("status") == "EVALUATED":
        (
            best_reg_name,
            best_reg_model,
            reg_selection,
        ) = choose_final_model(
            df,
            numeric,
            unseen["models"],
            task="regression",
            folds=args.inner_folds,
        )

        reg_selection.to_csv(
            out / "final_regression_selection.csv",
            index=False,
        )

        if best_reg_model is not None:
            joblib.dump(
                {
                    "model": best_reg_model,
                    "model_name": best_reg_name,
                    "numeric_features": numeric,
                    "train_max_ms": float(
                        df["execution_time_ms"].max()
                    ),
                    "training_records": len(df),
                    "training_fingerprints": fingerprints,
                },
                out / "best_unseen_latency.joblib",
            )

    best_cls_name = None
    best_cls_model = None

    if expensive.get("status") == "EVALUATED":
        (
            best_cls_name,
            best_cls_model,
            cls_selection,
        ) = choose_final_model(
            df,
            numeric,
            expensive["models"],
            task="classification",
            folds=args.inner_folds,
            expensive_quantile=(
                args.expensive_quantile
            ),
            fixed_threshold=(
                args.expensive_threshold_ms
            ),
        )

        cls_selection.to_csv(
            out / "final_classifier_selection.csv",
            index=False,
        )

        if best_cls_model is not None:
            threshold = expensive_threshold(
                df,
                args.expensive_quantile,
                args.expensive_threshold_ms,
            )

            joblib.dump(
                {
                    "model": best_cls_model,
                    "model_name": best_cls_name,
                    "threshold_ms": threshold,
                    "numeric_features": numeric,
                    "training_records": len(df),
                    "training_fingerprints": fingerprints,
                },
                out / "best_expensive_classifier.joblib",
            )

    print("\n[6/8] Prediction intervals")

    intervals = {"status": "SKIPPED"}

    if (
        not args.skip_intervals
        and best_reg_name is not None
    ):
        intervals = conformal_intervals(
            df,
            numeric,
            unseen["models"][best_reg_name],
            out,
            alpha=0.10,
            repeats=5,
        )

    print("\n[7/8] Phase 9 ranking")

    phase9 = phase9_ranking(
        df,
        out,
    )

    print("\n[8/8] Reports and plots")

    if expensive.get("status") == "EVALUATED":
        frequency = (
            expensive["outer"]["selected_model"]
            .value_counts()
            .rename_axis("model")
            .reset_index(name="selected_times")
        )

        frequency.to_csv(
            out / "expensive_model_frequency.csv",
            index=False,
        )

        plot_bar(
            frequency,
            "model",
            "selected_times",
            "Expensive-query model selection frequency",
            out / "expensive_model_frequency.png",
        )

    if unseen.get("status") == "EVALUATED":
        frequency = (
            unseen["outer"]["selected_model"]
            .value_counts()
            .rename_axis("model")
            .reset_index(name="selected_times")
        )

        frequency.to_csv(
            out / "unseen_model_frequency.csv",
            index=False,
        )

        plot_bar(
            frequency,
            "model",
            "selected_times",
            "Unseen-query model selection frequency",
            out / "unseen_model_frequency.png",
        )

    if known.get("status") == "EVALUATED":
        plot_bar(
            known["leaderboard"],
            "model",
            "fingerprint_log_mae",
            "Known-query latency model comparison",
            out / "known_model_comparison.png",
        )

    summary = {
        "run_id": run_id,
        "status": "COMPLETED",
        "records": len(df),
        "fingerprints": fingerprints,
        "sessions": df["session_id"].nunique(),
        "feature_mode": (
            "SQL_ONLY"
            if args.sql_only
            else "SQL_PLUS_PLANNER"
        ),
        "expensive_query_classification": {
            k: v
            for k, v in expensive.items()
            if k not in [
                "outer",
                "predictions",
                "models",
            ]
        },
        "unseen_query_latency": {
            k: v
            for k, v in unseen.items()
            if k not in [
                "outer",
                "predictions",
                "models",
            ]
        },
        "known_query_latency": {
            "status": known.get("status"),
            "winner": known.get("winner"),
        },
        "role_classification": {
            "status": role.get("status"),
            "winner": role.get("winner"),
        },
        "final_full_data_classifier": best_cls_name,
        "final_full_data_regressor": best_reg_name,
        "prediction_intervals": intervals,
        "phase9": phase9,
        "elapsed_seconds": round(
            time.perf_counter() - started,
            2,
        ),
        "limitations": [
            "Only 29 independent SQL fingerprints.",
            "Only one monitoring session.",
            "Repeated outer holdouts overlap.",
            "No independent future-session validation.",
            "Expensive-query labels depend on "
            "training-set latency threshold.",
            "Known-query evaluation is within-session.",
            "Phase 9 ranking uses workload burden "
            "rather than measured index benefit.",
            "No production databases modified.",
        ],
    }

    save_json(
        out / "summary.json",
        summary,
    )

    print("\n" + "=" * 95)
    print("FINAL V5 MODEL COMPARISON")
    print("=" * 95)

    print(
        "BEST EXPENSIVE-QUERY CLASSIFIER:",
        best_cls_name,
    )

    print(
        "BEST UNSEEN-QUERY REGRESSOR:",
        best_reg_name,
    )

    print(
        "BEST KNOWN-QUERY ESTIMATOR:",
        known.get("winner"),
    )

    print(
        "BEST ROLE CLASSIFIER:",
        role.get("winner"),
    )

    if expensive.get("status") == "EVALUATED":
        print(
            "NESTED EXPENSIVE CLASSIFICATION ACCURACY:",
            f"{expensive['mean_accuracy'] * 100:.2f}%",
        )

        print(
            "NESTED EXPENSIVE CLASSIFICATION MACRO F1:",
            f"{expensive['mean_macro_f1']:.4f}",
        )

        print(
            "NESTED EXPENSIVE CLASSIFICATION BALANCED ACCURACY:",
            f"{expensive['mean_balanced_accuracy']:.4f}",
        )

    if unseen.get("status") == "EVALUATED":
        print(
            "NESTED UNSEEN FINGERPRINT LOG MAE:",
            f"{unseen['mean_fingerprint_log_mae']:.4f}",
        )

        print(
            "NESTED UNSEEN FINGERPRINT MAE:",
            f"{unseen['mean_fingerprint_mae_ms']:.2f} ms",
        )

    print("\nRESULT DIRECTORY:")
    print(out)

    print("\nIMPORTANT:")
    print(
        "High accuracy is meaningful only when "
        "both classes are represented and performance "
        "generalizes to unseen SQL fingerprints."
    )

if __name__ == "__main__":
    main()
