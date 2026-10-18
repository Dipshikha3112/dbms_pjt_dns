
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

from scipy.sparse import csr_matrix, hstack
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_extraction.text import TfidfVectorizer
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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

PHASE6 = RESULTS / "phase6_workloads" / "combined_workload.csv"
PHASE7 = RESULTS / "phase7_monitoring" / "phase7_query_monitoring.csv"

SEED = 42
RNG = np.random.default_rng(SEED)

LOG = logging.getLogger("phase8_final")

SQL_COLUMNS = ["sql", "query", "sql_text", "query_text"]
LABEL_COLUMNS = ["workload_type", "role_label", "query_family"]
TIME_COLUMNS = ["monitored_at", "executed_at", "timestamp"]
LATENCY_COLUMNS = [
    "execution_time_ms", "latency_ms", "duration_ms",
    "elapsed_ms",
]

KEYWORDS = [
    "SELECT", "FROM", "WHERE", "JOIN", "LEFT JOIN",
    "RIGHT JOIN", "INNER JOIN", "GROUP BY",
    "ORDER BY", "HAVING", "DISTINCT", "UNION",
    "EXISTS", "BETWEEN", "LIKE", "LIMIT",
    "OFFSET", "CASE", "WHEN", "WITH", "IN",
    "COUNT", "SUM", "AVG", "MIN", "MAX",
    "OVER", "PARTITION BY", "COALESCE",
]

OUT = None


def setup_logging(path):
    path.mkdir(parents=True, exist_ok=True)

    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s"
    )

    for handler in [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            path / "execution.log", encoding="utf-8"
        ),
    ]:
        handler.setFormatter(formatter)
        LOG.addHandler(handler)


def json_clean(obj):
    if isinstance(obj, dict):
        return {
            str(k): json_clean(v)
            for k, v in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [json_clean(v) for v in obj]

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return (
            float(obj)
            if np.isfinite(obj) else None
        )

    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None

    return obj


def save_json(name, obj):
    with (OUT / name).open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            json_clean(obj),
            f,
            indent=2,
            ensure_ascii=False,
        )


def column(df, aliases, required=False):
    lookup = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for alias in aliases:
        if alias in lookup:
            return lookup[alias]

    if required:
        raise ValueError(
            f"Missing required column. Expected one of: {aliases}. "
            f"Available: {list(df.columns)}"
        )

    return None


def digest(value):
    return hashlib.sha256(
        str(value).encode(
            "utf-8", errors="replace"
        )
    ).hexdigest()[:20]


def normalize_sql(sql):
    sql = str(sql or "")

    sql = re.sub(
        r"/\*.*?\*/",
        " ",
        sql,
        flags=re.S,
    )
    sql = re.sub(r"--[^\n]*", " ", sql)

    sql = re.sub(
        r"'(?:''|[^'])*'",
        " STRING_LITERAL ",
        sql,
    )

    sql = re.sub(
        r"\b\d+(?:\.\d+)?\b",
        " NUMBER_LITERAL ",
        sql,
    )

    sql = re.sub(
        r"\s+",
        " ",
        sql.upper(),
    ).strip()

    return sql


def count_keyword(sql, word):
    return len(
        re.findall(
            r"\b" +
            re.escape(word).replace(r"\ ", r"\s+") +
            r"\b",
            sql,
        )
    )


def sql_features(sql):
    original = str(sql or "")
    s = normalize_sql(original)

    tokens = re.findall(
        r"[A-Z_][A-Z_0-9]*|[^\s]",
        s,
    )

    result = {
        "sql_length": len(original),
        "normalized_length": len(s),
        "tokens": len(tokens),
        "unique_tokens": len(set(tokens)),
        "joins": count_keyword(s, "JOIN"),
        "selects": count_keyword(s, "SELECT"),
        "subqueries": max(
            0,
            count_keyword(s, "SELECT") - 1,
        ),
        "parentheses": s.count("("),
        "commas": s.count(","),
        "and_count": count_keyword(s, "AND"),
        "or_count": count_keyword(s, "OR"),
        "comparison_count": len(
            re.findall(
                r"<=|>=|<>|!=|(?<![<>!])=(?!=)|<|>",
                s,
            )
        ),
    }

    for keyword in KEYWORDS:
        result[
            "kw_" +
            keyword.lower().replace(" ", "_")
        ] = count_keyword(s, keyword)

    return result


def feature_frame(sql_values):
    return pd.DataFrame(
        [
            sql_features(s)
            for s in sql_values
        ]
    ).fillna(0).astype(float)


def read_workload(path):
    LOG.info("Loading Phase 6 workload")

    df = pd.read_csv(path, low_memory=False)

    sql_col = column(
        df, SQL_COLUMNS, required=True
    )
    label_col = column(
        df, LABEL_COLUMNS, required=True
    )

    result = pd.DataFrame({
        "sql": df[sql_col].fillna("").astype(str),
        "label": (
            df[label_col]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        ),
    })

    result = result[
        result["sql"].str.strip().ne("") &
        result["label"].ne("")
    ].copy()

    result["normalized_sql"] = (
        result["sql"].map(normalize_sql)
    )

    # Group on exact normalized SQL, not row position.
    # Identical normalized templates cannot cross splits.
    result["group"] = (
        result["normalized_sql"].map(digest)
    )

    result = result.reset_index(drop=True)

    if len(result) < 40:
        raise ValueError(
            "Insufficient Phase 6 workload records."
        )

    return result


def read_monitoring(path):
    LOG.info("Loading Phase 7 monitoring")

    df = pd.read_csv(path, low_memory=False)

    time_col = column(
        df, TIME_COLUMNS, required=True
    )
    latency_col = column(
        df, LATENCY_COLUMNS
    )
    sql_col = column(df, SQL_COLUMNS)
    label_col = column(df, LABEL_COLUMNS)

    try:
        timestamps = pd.to_datetime(
            df[time_col],
            utc=True,
            errors="coerce",
            format="mixed",
        )
    except (TypeError, ValueError):
        timestamps = pd.to_datetime(
            df[time_col],
            utc=True,
            errors="coerce",
        )

    result = pd.DataFrame({
        "timestamp": timestamps,
        "sql": (
            df[sql_col].fillna("").astype(str)
            if sql_col else ""
        ),
        "label": (
            df[label_col].fillna("unknown")
            .astype(str).str.strip().str.lower()
            if label_col else "unknown"
        ),
        "latency_ms": (
            pd.to_numeric(
                df[latency_col],
                errors="coerce",
            )
            if latency_col else np.nan
        ),
    })

    result = result[
        result["timestamp"].notna()
    ].copy()

    result.loc[
        (result["latency_ms"] < 0) |
        (~np.isfinite(result["latency_ms"])),
        "latency_ms",
    ] = np.nan

    result = result.sort_values(
        "timestamp", kind="stable"
    ).reset_index(drop=True)

    if result.empty:
        raise ValueError(
            "No valid monitoring timestamps."
        )

    result["normalized_sql"] = (
        result["sql"].map(normalize_sql)
    )

    result["group"] = (
        result["normalized_sql"].map(digest)
    )

    return result


def metrics(y, pred):
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(
            y, pred
        ),
        "macro_f1": f1_score(
            y, pred,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y, pred,
            average="weighted",
            zero_division=0,
        ),
    }


def regression_metrics(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)

    return {
        "mae": mean_absolute_error(y, pred),
        "rmse": np.sqrt(
            mean_squared_error(y, pred)
        ),
        "r2": (
            r2_score(y, pred)
            if len(y) > 1 else None
        ),
    }


def split_groups(df, n_splits=5):
    n_splits = min(
        n_splits,
        df["group"].nunique(),
    )

    if n_splits < 2:
        raise ValueError(
            "Insufficient distinct SQL groups."
        )

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=SEED,
    )

    splits = list(
        splitter.split(
            df["normalized_sql"],
            df["label"],
            df["group"],
        )
    )

    all_classes = set(df["label"])

    def split_quality(pair):
        tr, te = pair

        train_classes = set(
            df.iloc[tr]["label"]
        )
        test_classes = set(
            df.iloc[te]["label"]
        )

        coverage = (
            len(train_classes & all_classes) +
            len(test_classes & all_classes)
        )

        size_penalty = abs(
            len(te) / len(df) - 1 / n_splits
        )

        return (
            coverage,
            -size_penalty,
        )

    best = max(splits, key=split_quality)

    return best, splits


def make_text_features(
    train_text,
    test_text,
    train_numeric,
    test_numeric,
):
    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 2),
        min_df=2,
        max_features=6000,
        sublinear_tf=True,
    )

    train_tfidf = vectorizer.fit_transform(
        train_text
    )

    test_tfidf = vectorizer.transform(
        test_text
    )

    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(
        train_numeric
    )
    test_scaled = scaler.transform(
        test_numeric
    )

    X_train = hstack([
        train_tfidf,
        csr_matrix(train_scaled),
    ]).tocsr()

    X_test = hstack([
        test_tfidf,
        csr_matrix(test_scaled),
    ]).tocsr()

    return (
        X_train,
        X_test,
        vectorizer,
        scaler,
    )


def make_logistic():
    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        C=1.0,
        random_state=SEED,
    )


def make_forest():
    return RandomForestClassifier(
        n_estimators=250,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=-1,
    )


def fit_classifiers(
    train,
    test,
    train_numeric,
    test_numeric,
):
    (
        X_train,
        X_test,
        vectorizer,
        scaler,
    ) = make_text_features(
        train["normalized_sql"],
        test["normalized_sql"],
        train_numeric,
        test_numeric,
    )

    models = {
        "tfidf_logistic": make_logistic(),
        "numeric_random_forest": make_forest(),
        "majority_baseline": DummyClassifier(
            strategy="most_frequent"
        ),
    }

    models["tfidf_logistic"].fit(
        X_train,
        train["label"],
    )

    models["numeric_random_forest"].fit(
        train_numeric,
        train["label"],
    )

    models["majority_baseline"].fit(
        train_numeric,
        train["label"],
    )

    predictions = {
        "tfidf_logistic": models[
            "tfidf_logistic"
        ].predict(X_test),

        "numeric_random_forest": models[
            "numeric_random_forest"
        ].predict(test_numeric),

        "majority_baseline": models[
            "majority_baseline"
        ].predict(test_numeric),
    }

    scores = {
        name: metrics(
            test["label"],
            prediction,
        )
        for name, prediction
        in predictions.items()
    }

    return (
        models,
        predictions,
        scores,
        vectorizer,
        scaler,
    )


def classification_stage(workload, X):
    LOG.info(
        "Classification: group-safe holdout and CV"
    )

    (train_idx, test_idx), folds = split_groups(
        workload
    )

    train = workload.iloc[train_idx]
    test = workload.iloc[test_idx]

    train_numeric = X.iloc[train_idx]
    test_numeric = X.iloc[test_idx]

    (
        models,
        predictions,
        scores,
        vectorizer,
        scaler,
    ) = fit_classifiers(
        train,
        test,
        train_numeric,
        test_numeric,
    )

    candidate_names = [
        "tfidf_logistic",
        "numeric_random_forest",
    ]

    # This selection is descriptive on the holdout;
    # do not report the selected score as unbiased
    # post-selection performance.
    selected = max(
        candidate_names,
        key=lambda name: scores[name]["macro_f1"],
    )

    labels = sorted(workload["label"].unique())

    output = pd.DataFrame({
        "actual": test["label"].to_numpy(),
        "group": test["group"].to_numpy(),
    })

    for name, pred in predictions.items():
        output[name] = pred

    output.to_csv(
        OUT / "classification_predictions.csv",
        index=False,
    )

    for name, pred in predictions.items():
        save_json(
            f"classification_report_{name}.json",
            classification_report(
                test["label"],
                pred,
                labels=labels,
                output_dict=True,
                zero_division=0,
            ),
        )

    cm = confusion_matrix(
        test["label"],
        predictions[selected],
        labels=labels,
    )

    pd.DataFrame(
        cm,
        index=labels,
        columns=labels,
    ).to_csv(
        OUT / "classification_confusion_matrix.csv"
    )

    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.colorbar()
    plt.xticks(
        range(len(labels)),
        labels,
        rotation=30,
    )
    plt.yticks(
        range(len(labels)),
        labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(
        f"Grouped holdout: {selected}"
    )
    plt.tight_layout()
    plt.savefig(
        OUT / "classification_confusion_matrix.png",
        dpi=160,
    )
    plt.close()

    cv_records = []

    for fold, (tr, te) in enumerate(
        folds, start=1
    ):
        fold_train = workload.iloc[tr]
        fold_test = workload.iloc[te]

        if fold_train["label"].nunique() < 2:
            continue

        (
            _,
            _,
            fold_scores,
            _,
            _,
        ) = fit_classifiers(
            fold_train,
            fold_test,
            X.iloc[tr],
            X.iloc[te],
        )

        for name, score in fold_scores.items():
            cv_records.append({
                "fold": fold,
                "model": name,
                "train_rows": len(tr),
                "test_rows": len(te),
                "train_classes": fold_train[
                    "label"
                ].nunique(),
                "test_classes": fold_test[
                    "label"
                ].nunique(),
                **score,
            })

    cv_df = pd.DataFrame(cv_records)

    cv_df.to_csv(
        OUT / "classification_grouped_cv.csv",
        index=False,
    )

    cv_summary = (
        cv_df.groupby("model")
        .agg(
            folds=("fold", "count"),
            mean_accuracy=("accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            std_macro_f1=("macro_f1", "std"),
        )
        .reset_index()
    )

    cv_summary.to_csv(
        OUT / "classification_cv_summary.csv",
        index=False,
    )

    # Fit final candidate models on all Phase 6 records.
    full_vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 2),
        min_df=2,
        max_features=6000,
        sublinear_tf=True,
    )

    full_text = full_vectorizer.fit_transform(
        workload["normalized_sql"]
    )

    full_scaler = StandardScaler()
    full_numeric = full_scaler.fit_transform(X)

    full_matrix = hstack([
        full_text,
        csr_matrix(full_numeric),
    ]).tocsr()

    final_logistic = make_logistic()
    final_logistic.fit(
        full_matrix,
        workload["label"],
    )

    final_forest = make_forest()
    final_forest.fit(
        X,
        workload["label"],
    )

    joblib.dump(
        {
            "tfidf_logistic": final_logistic,
            "numeric_random_forest": final_forest,
            "tfidf_vectorizer": full_vectorizer,
            "numeric_scaler": full_scaler,
            "numeric_features": list(X.columns),
            "selected_on_holdout": selected,
            "selection_warning": (
                "Model selection on holdout is exploratory; "
                "CV metrics provide additional comparison."
            ),
        },
        OUT / "classification_models.joblib",
    )

    result = {
        "status": "COMPLETED",
        "holdout": {
            "train_rows": len(train),
            "test_rows": len(test),
            "train_groups": train[
                "group"
            ].nunique(),
            "test_groups": test[
                "group"
            ].nunique(),
            "train_classes": sorted(
                train["label"].unique()
            ),
            "test_classes": sorted(
                test["label"].unique()
            ),
            "missing_test_classes": sorted(
                set(labels) -
                set(test["label"])
            ),
            "scores": scores,
        },
        "selected_candidate": selected,
        "cv": cv_summary.to_dict(
            orient="records"
        ),
        "limitations": [
            "Role classification is not hybrid-engine routing",
            "SQL-group generalization is challenging",
            "Class coverage is reported explicitly",
            "Holdout-selected model score is exploratory",
        ],
    }

    save_json(
        "classification_metrics.json",
        result,
    )

    return result


def clustering_stage(X, workload):
    LOG.info("Clustering SQL workloads")

    scaler = StandardScaler()
    Z = scaler.fit_transform(X)

    candidates = []

    for k in range(2, min(9, len(X) - 1) + 1):
        model = KMeans(
            n_clusters=k,
            random_state=SEED,
            n_init=10,
        )

        labels = model.fit_predict(Z)

        if len(set(labels)) < 2:
            continue

        score = silhouette_score(
            Z,
            labels,
            sample_size=min(950, len(Z)),
            random_state=SEED,
        )

        candidates.append(
            (k, score, model)
        )

    if not candidates:
        return {
            "status": "INSUFFICIENT_VARIATION"
        }

    k, score, model = max(
        candidates,
        key=lambda x: x[1],
    )

    assignments = pd.DataFrame({
        "group": workload["group"],
        "label": workload["label"],
        "cluster": model.labels_,
    })

    assignments.to_csv(
        OUT / "cluster_assignments.csv",
        index=False,
    )

    pd.DataFrame([
        {
            "k": item[0],
            "silhouette": item[1],
        }
        for item in candidates
    ]).to_csv(
        OUT / "cluster_selection.csv",
        index=False,
    )

    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "features": list(X.columns),
        },
        OUT / "clustering_model.joblib",
    )

    return {
        "status": "COMPLETED",
        "best_k": k,
        "silhouette": score,
    }


def latency_stage(monitoring):
    LOG.info(
        "Latency prediction with SQL-group holdout"
    )

    df = monitoring[
        monitoring["latency_ms"].notna() &
        monitoring["sql"].str.strip().ne("")
    ].copy()

    if len(df) < 60:
        return {
            "status": "INSUFFICIENT_DATA"
        }

    X = feature_frame(df["sql"])

    unique_groups = df["group"].unique()

    if len(unique_groups) < 4:
        return {
            "status": "INSUFFICIENT_GROUPS"
        }

    shuffled = np.random.default_rng(
        SEED
    ).permutation(unique_groups)

    cutoff = max(
        1, int(0.75 * len(shuffled))
    )

    train_groups = set(
        shuffled[:cutoff]
    )

    mask = df["group"].isin(
        train_groups
    ).to_numpy()

    train_idx = np.where(mask)[0]
    test_idx = np.where(~mask)[0]

    if len(train_idx) < 20 or len(test_idx) < 10:
        return {
            "status": "INSUFFICIENT_SPLIT"
        }

    y = df["latency_ms"].to_numpy()

    target = np.log1p(y)

    model = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=15,
        random_state=SEED,
    )

    model.fit(
        X.iloc[train_idx],
        target[train_idx],
    )

    pred = np.maximum(
        0,
        np.expm1(
            model.predict(
                X.iloc[test_idx]
            )
        ),
    )

    baseline = np.full(
        len(test_idx),
        np.median(y[train_idx]),
    )

    scores = {
        "model": regression_metrics(
            y[test_idx], pred
        ),
        "median_baseline": regression_metrics(
            y[test_idx], baseline
        ),
    }

    pd.DataFrame({
        "actual_ms": y[test_idx],
        "predicted_ms": pred,
        "median_baseline_ms": baseline,
        "sql_group": df.iloc[
            test_idx
        ]["group"].to_numpy(),
    }).to_csv(
        OUT / "latency_predictions.csv",
        index=False,
    )

    joblib.dump(
        {
            "model": model,
            "feature_columns": list(X.columns),
            "target": "log1p_latency_ms",
        },
        OUT / "latency_model.joblib",
    )

    result = {
        "status": "COMPLETED",
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "scores": scores,
        "beats_baseline": (
            scores["model"]["mae"] <
            scores["median_baseline"]["mae"]
        ),
    }

    save_json(
        "latency_metrics.json",
        result,
    )

    return result


def optimization_stage(monitoring):
    LOG.info("Ranking optimization candidates")

    df = monitoring.copy()

    valid = df[
        df["latency_ms"].notna()
    ].copy()

    if valid.empty:
        return {
            "status": "NO_LATENCY_DATA"
        }

    grouped = valid.groupby(
        "group"
    )["latency_ms"]

    ranking = grouped.agg(
        executions="size",
        mean_ms="mean",
        median_ms="median",
        max_ms="max",
    ).reset_index()

    ranking["p95_ms"] = (
        grouped.quantile(0.95).to_numpy()
    )

    ranking["observed_total_ms"] = (
        ranking["executions"] *
        ranking["mean_ms"]
    )

    ranking = ranking.sort_values(
        "observed_total_ms",
        ascending=False,
    )

    ranking["rank"] = np.arange(
        1, len(ranking) + 1
    )

    ranking.to_csv(
        OUT / "optimization_candidate_ranking.csv",
        index=False,
    )

    return {
        "status": "COMPLETED",
        "groups_ranked": len(ranking),
        "interpretation": (
            "Observed latency burden only; "
            "not proof of an index benefit"
        ),
    }


def build_bins(monitoring, seconds):
    start = monitoring[
        "timestamp"
    ].min().floor(f"{seconds}s")

    end = monitoring[
        "timestamp"
    ].max().floor(f"{seconds}s")

    index = pd.date_range(
        start,
        end,
        freq=f"{seconds}s",
    )

    counts = (
        monitoring.set_index("timestamp")
        .resample(f"{seconds}s")
        .size()
        .reindex(index, fill_value=0)
    )

    return pd.DataFrame({
        "timestamp": counts.index,
        "count": counts.to_numpy(),
    })


def lagged(values, lags):
    features = []
    targets = []
    indices = []

    for t in range(lags, len(values)):
        past = values[t-lags:t]

        features.append(
            list(past[::-1]) + [
                np.mean(past),
                np.std(past),
                np.min(past),
                np.max(past),
                past[-1] - past[0],
            ]
        )

        targets.append(values[t])
        indices.append(t)

    return (
        np.asarray(features, dtype=float),
        np.asarray(targets, dtype=float),
        np.asarray(indices, dtype=int),
    )


def subset_mae(actual, predicted, mask):
    if not np.any(mask):
        return None

    return mean_absolute_error(
        actual[mask],
        predicted[mask],
    )


def forecasting_stage(monitoring, seconds):
    LOG.info(
        "Forecasting with chronological and active-bin evaluation"
    )

    bins = build_bins(
        monitoring, seconds
    )

    bins.to_csv(
        OUT / "query_arrival_bins.csv",
        index=False,
    )

    values = bins["count"].to_numpy(
        dtype=float
    )

    if len(values) < 45:
        return {
            "status": "INSUFFICIENT_BINS",
            "bins": len(values),
        }

    lags = min(
        9,
        max(4, len(values) // 10),
    )

    X, y, indices = lagged(
        values, lags
    )

    split = int(0.75 * len(y))

    if split < 20 or len(y) - split < 10:
        return {
            "status": "INSUFFICIENT_SPLIT"
        }

    train_X = X[:split]
    test_X = X[split:]

    train_y = y[:split]
    test_y = y[split:]

    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=8,
        min_samples_leaf=2,
        random_state=SEED,
        n_jobs=-1,
    )

    model.fit(
        train_X, train_y
    )

    predictions = {
        "random_forest": np.maximum(
            0,
            model.predict(test_X),
        ),
        "persistence": values[
            indices[split:] - 1
        ],
        "rolling_mean": np.asarray([
            np.mean(
                values[t-lags:t]
            )
            for t in indices[split:]
        ]),
        "zero_baseline": np.zeros_like(
            test_y
        ),
        "training_mean_baseline": np.full_like(
            test_y,
            np.mean(train_y),
        ),
    }

    active = test_y > 0
    zero = test_y == 0

    scores = {}

    for name, pred in predictions.items():
        scores[name] = {
            **regression_metrics(
                test_y, pred
            ),
            "active_bin_mae": subset_mae(
                test_y,
                pred,
                active,
            ),
            "zero_bin_mae": subset_mae(
                test_y,
                pred,
                zero,
            ),
        }

    best_baseline = min(
        [
            k for k in scores
            if k != "random_forest"
        ],
        key=lambda k: scores[k]["mae"],
    )

    rf_mae = scores[
        "random_forest"
    ]["mae"]

    baseline_mae = scores[
        best_baseline
    ]["mae"]

    improvement = (
        100 * (
            baseline_mae - rf_mae
        ) / baseline_mae
        if baseline_mae > 0
        else None
    )

    output = pd.DataFrame({
        "timestamp": bins.iloc[
            indices[split:]
        ]["timestamp"].to_numpy(),
        "actual": test_y,
    })

    for name, pred in predictions.items():
        output[name] = pred

    output.to_csv(
        OUT / "forecast_predictions.csv",
        index=False,
    )

    plt.figure(figsize=(11, 5))

    plt.plot(
        test_y,
        label="Actual",
        marker=".",
    )

    plt.plot(
        predictions["random_forest"],
        label="Random Forest",
    )

    plt.plot(
        predictions[best_baseline],
        label=f"Best baseline: {best_baseline}",
    )

    plt.title(
        f"Short-term query arrivals: {seconds}s bins"
    )
    plt.xlabel("Chronological test bin")
    plt.ylabel("Query count")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUT / "forecast_comparison.png",
        dpi=160,
    )
    plt.close()

    joblib.dump(
        {
            "model": model,
            "lags": lags,
            "bin_seconds": seconds,
            "scope": "within_session",
        },
        OUT / "forecast_model.joblib",
    )

    result = {
        "status": "COMPLETED_WITHIN_SESSION",
        "total_bins": len(bins),
        "train_bins": len(train_y),
        "test_bins": len(test_y),
        "test_zero_bins": int(
            zero.sum()
        ),
        "test_active_bins": int(
            active.sum()
        ),
        "test_zero_fraction": float(
            zero.mean()
        ),
        "lag_count": lags,
        "scores": scores,
        "best_baseline": best_baseline,
        "rf_beats_best_baseline": bool(
            rf_mae < baseline_mae
        ),
        "mae_improvement_pct": improvement,
        "active_bin_caution": (
            "Active-bin MAE is descriptive on a "
            "target-conditioned subset, not a separate "
            "unbiased forecast evaluation."
        ),
        "limitations": [
            "Only one short monitoring session",
            "No independent-session validation",
            "No daily or weekly demand prediction",
            "Potential scripted benchmark-order effects",
        ],
    }

    save_json(
        "forecast_metrics.json",
        result,
    )

    return result


def js_divergence(a, b):
    ca = Counter(map(str, a))
    cb = Counter(map(str, b))

    categories = sorted(
        set(ca) | set(cb)
    )

    if not categories:
        return None

    p = np.asarray([
        ca.get(x, 0)
        for x in categories
    ], dtype=float)

    q = np.asarray([
        cb.get(x, 0)
        for x in categories
    ], dtype=float)

    p /= p.sum()
    q /= q.sum()

    m = (p + q) / 2

    def kl(x, y):
        mask = x > 0
        return np.sum(
            x[mask] *
            np.log2(
                x[mask] / y[mask]
            )
        )

    return float(
        (kl(p, m) + kl(q, m)) / 2
    )


def psi(reference, current):
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

    if min(
        len(reference), len(current)
    ) < 15:
        return None

    if len(np.unique(reference)) < 2:
        return None

    edges = np.unique(
        np.quantile(
            reference,
            np.linspace(0, 1, 9),
        )
    )

    if len(edges) < 3:
        return None

    edges[0] = -np.inf
    edges[-1] = np.inf

    a = np.histogram(
        reference,
        bins=edges,
    )[0].astype(float)

    b = np.histogram(
        current,
        bins=edges,
    )[0].astype(float)

    a = np.maximum(
        a / max(a.sum(), 1),
        1e-6,
    )
    b = np.maximum(
        b / max(b.sum(), 1),
        1e-6,
    )

    a /= a.sum()
    b /= b.sum()

    return float(
        np.sum(
            (b - a) * np.log(b / a)
        )
    )


def drift_stage(monitoring):
    LOG.info(
        "Within-session drift and composition diagnostics"
    )

    if len(monitoring) < 80:
        return {
            "status": "INSUFFICIENT_EVENTS"
        }

    midpoint = (
        monitoring["timestamp"].iloc[0] +
        (
            monitoring["timestamp"].iloc[-1] -
            monitoring["timestamp"].iloc[0]
        ) / 2
    )

    early = monitoring[
        monitoring["timestamp"] < midpoint
    ].copy()

    late = monitoring[
        monitoring["timestamp"] >= midpoint
    ].copy()

    if min(len(early), len(late)) < 30:
        return {
            "status": "INSUFFICIENT_WINDOW_EVENTS"
        }

    js = js_divergence(
        early["label"],
        late["label"],
    )

    early_prop = (
        early["label"]
        .value_counts(normalize=True)
    )

    late_prop = (
        late["label"]
        .value_counts(normalize=True)
    )

    categories = sorted(
        set(early_prop.index) |
        set(late_prop.index)
    )

    composition = pd.DataFrame({
        "label": categories,
        "early_fraction": [
            early_prop.get(x, 0)
            for x in categories
        ],
        "late_fraction": [
            late_prop.get(x, 0)
            for x in categories
        ],
    })

    composition.to_csv(
        OUT / "workload_composition_drift.csv",
        index=False,
    )

    plt.figure(figsize=(10, 5))

    positions = np.arange(
        len(categories)
    )

    plt.bar(
        positions - 0.2,
        composition["early_fraction"],
        width=0.4,
        label="Early window",
    )

    plt.bar(
        positions + 0.2,
        composition["late_fraction"],
        width=0.4,
        label="Late window",
    )

    plt.xticks(
        positions,
        categories,
        rotation=30,
    )

    plt.ylabel("Proportion")
    plt.title(
        "Within-session workload composition shift"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUT / "workload_drift.png",
        dpi=160,
    )
    plt.close()

    records = []

    early_sql = feature_frame(
        early["sql"]
    )

    late_sql = feature_frame(
        late["sql"]
    )

    for feature in early_sql.columns:
        value = psi(
            early_sql[feature],
            late_sql[feature],
        )

        records.append({
            "feature": feature,
            "metric": "PSI",
            "value": value,
            "flag_ge_0_2": (
                value >= 0.2
                if value is not None
                else False
            ),
        })

    latency_psi = psi(
        early["latency_ms"],
        late["latency_ms"],
    )

    records.append({
        "feature": "latency_ms",
        "metric": "PSI",
        "value": latency_psi,
        "flag_ge_0_2": (
            latency_psi >= 0.2
            if latency_psi is not None
            else False
        ),
    })

    psi_df = pd.DataFrame(records)

    psi_df.to_csv(
        OUT / "drift_psi_metrics.csv",
        index=False,
    )

    # Exploratory permutation reference.
    # This shuffles individual labels and therefore does
    # not preserve temporal autocorrelation.
    labels = monitoring[
        "label"
    ].to_numpy()

    n_early = len(early)

    permutation_scores = []

    permutation_rng = np.random.default_rng(
        SEED
    )

    for _ in range(500):
        shuffled = permutation_rng.permutation(
            labels
        )

        permutation_scores.append(
            js_divergence(
                shuffled[:n_early],
                shuffled[n_early:],
            )
        )

    exploratory_p = (
        (
            1 + sum(
                x >= js
                for x in permutation_scores
            )
        ) /
        (1 + len(permutation_scores))
    )

    result = {
        "status": "COMPLETED_DIAGNOSTIC_ONLY",
        "early_events": len(early),
        "late_events": len(late),
        "midpoint_utc": midpoint,
        "js_divergence_bits": js,
        "exploratory_iid_permutation_p": exploratory_p,
        "permutations": 500,
        "psi_features_evaluated": int(
            psi_df["value"].notna().sum()
        ),
        "psi_features_flagged": int(
            psi_df["flag_ge_0_2"].sum()
        ),
        "limitations": [
            "Single short benchmark session",
            "Workload ordering may explain the shift",
            "Permutation null ignores temporal dependence",
            "Permutation p-value is exploratory only",
            "PSI thresholds are descriptive heuristics",
            "No proof of long-term concept drift",
        ],
    }

    save_json(
        "drift_metrics.json",
        result,
    )

    return result


def monitoring_stage(monitoring):
    LOG.info("Creating monitoring visualizations")

    start = monitoring["timestamp"].min()
    end = monitoring["timestamp"].max()

    duration = (
        end - start
    ).total_seconds()

    plt.figure(figsize=(11, 5))

    plt.plot(
        monitoring["timestamp"],
        monitoring["latency_ms"],
        linewidth=0.8,
    )

    plt.xlabel("Time (UTC)")
    plt.ylabel("Latency (ms)")
    plt.title("Observed Phase 7 execution latency")
    plt.xticks(rotation=25)
    plt.tight_layout()

    plt.savefig(
        OUT / "latency_timeline.png",
        dpi=160,
    )
    plt.close()

    return {
        "status": "COMPLETED",
        "records": len(monitoring),
        "first_timestamp": start,
        "last_timestamp": end,
        "duration_seconds": duration,
        "duration_minutes": duration / 60,
        "valid_latency_records": int(
            monitoring["latency_ms"]
            .notna().sum()
        ),
        "sql_groups": monitoring[
            "group"
        ].nunique(),
    }


def execute(name, function, summary):
    try:
        result = function()

        summary["components"][name] = result

        LOG.info(
            "%s => %s",
            name,
            result.get("status", "UNKNOWN"),
        )

    except Exception as exc:
        LOG.exception(
            "Component failed: %s",
            name,
        )

        summary["components"][name] = {
            "status": "FAILED",
            "error": str(exc),
        }

        summary["errors"].append({
            "component": name,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })

    save_json(
        "summary_partial.json",
        summary,
    )


def report(summary):
    components = summary["components"]

    lines = [
        "# Phase 8 — Final Workload Intelligence Evaluation",
        "",
        "## Experiment",
        "",
        f"- Phase 6: {summary['phase6_source']}",
        f"- Phase 7: {summary['phase7_source']}",
        "- Database connections: none",
        "- Database modifications: none",
        "- Random seed: 42",
        "",
        "## Components",
        "",
    ]

    for name, value in components.items():
        lines.append(
            f"- {name}: "
            f"{value.get('status', 'UNKNOWN')}"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "Classification uses SQL-group-separated validation.",
        "TF-IDF and numeric structural features are compared "
        "against a majority-class baseline.",
        "",
        "Latency regression uses a SQL-group-separated holdout "
        "and a training-median baseline.",
        "",
        "Forecasting uses real recorded timestamps, "
        "a chronological holdout, multiple baselines, "
        "and active-bin error diagnostics.",
        "",
        "Drift analysis measures short-session workload "
        "composition differences and feature PSI.",
        "",
        "## Limitations",
        "",
        "- The temporal trace covers only one short session.",
        "- Daily or weekly demand prediction is unsupported.",
        "- Benchmark ordering can create artificial shifts.",
        "- No long-term concept drift is established.",
        "- Class imbalance and group concentration can "
        "limit grouped holdout coverage.",
        "- Model selection on the holdout is exploratory.",
        "- No causal optimizer speedup is demonstrated.",
        "- No SQL/Cypher/vector routing model is established.",
        "",
        "## Database safety",
        "",
        "This program reads CSV files only and creates "
        "new output files in a timestamped directory.",
        "It does not connect to PostgreSQL or Neo4j.",
        "",
    ])

    (OUT / "research_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main():
    global OUT

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phase6",
        default=str(PHASE6),
    )

    parser.add_argument(
        "--phase7",
        default=str(PHASE7),
    )

    parser.add_argument(
        "--bin-seconds",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    if args.bin_seconds < 1:
        parser.error(
            "--bin-seconds must be >= 1"
        )

    stamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S_%f")

    OUT = (
        RESULTS /
        "phase8_final" /
        stamp
    )

    setup_logging(OUT)

    summary = {
        "phase": 8,
        "status": "RUNNING",
        "phase6_source": args.phase6,
        "phase7_source": args.phase7,
        "output": str(OUT),
        "components": {},
        "errors": [],
        "database_modifications": "NONE",
    }

    LOG.info("=" * 72)
    LOG.info(
        "PHASE 8: FINAL REFINED IMPLEMENTATION"
    )
    LOG.info("=" * 72)

    LOG.info(
        "No database connections or modifications"
    )

    try:
        workload = read_workload(
            Path(args.phase6)
        )

        X = feature_frame(
            workload["sql"]
        )

        X.to_csv(
            OUT / "sql_features.csv",
            index=False,
        )

        summary["components"]["dataset"] = {
            "status": "COMPLETED",
            "rows": len(workload),
            "features": X.shape[1],
            "groups": workload[
                "group"
            ].nunique(),
            "classes": workload[
                "label"
            ].value_counts().to_dict(),
        }

        execute(
            "clustering",
            lambda: clustering_stage(
                X, workload
            ),
            summary,
        )

        execute(
            "classification",
            lambda: classification_stage(
                workload, X
            ),
            summary,
        )

    except Exception as exc:
        LOG.exception(
            "Phase 6 processing failed"
        )

        summary["errors"].append({
            "component": "phase6",
            "error": str(exc),
        })

    try:
        monitoring = read_monitoring(
            Path(args.phase7)
        )

        execute(
            "monitoring",
            lambda: monitoring_stage(
                monitoring
            ),
            summary,
        )

        execute(
            "latency_prediction",
            lambda: latency_stage(
                monitoring
            ),
            summary,
        )

        execute(
            "optimization_ranking",
            lambda: optimization_stage(
                monitoring
            ),
            summary,
        )

        execute(
            "forecasting",
            lambda: forecasting_stage(
                monitoring,
                args.bin_seconds,
            ),
            summary,
        )

        execute(
            "temporal_drift",
            lambda: drift_stage(
                monitoring
            ),
            summary,
        )

    except Exception as exc:
        LOG.exception(
            "Phase 7 processing failed"
        )

        summary["errors"].append({
            "component": "phase7",
            "error": str(exc),
        })

    summary["status"] = (
        "COMPLETED_WITH_ERRORS"
        if summary["errors"]
        else "COMPLETED_WITH_LIMITATIONS"
    )

    summary["finished_utc"] = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    save_json(
        "summary.json",
        summary,
    )

    report(summary)

    print("\n" + "=" * 72)
    print("PHASE 8 FINAL RESULTS")
    print("=" * 72)

    print(
        "Status:",
        summary["status"],
    )

    print(
        "Output:",
        OUT,
    )

    for name, value in summary[
        "components"
    ].items():
        print(
            f"{name:25s} => "
            f"{value.get('status', 'UNKNOWN')}"
        )

    print(
        "Errors:",
        len(summary["errors"]),
    )

    print(
        "Database modifications: NONE"
    )

    print("=" * 72)

    return (
        1 if summary["errors"]
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
