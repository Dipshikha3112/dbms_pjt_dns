
import argparse
import json
import re
import hashlib
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
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

V6_ROOT = ROOT / "results" / "phase8_v6_validation"

PRED_FILE = (
    V6_ROOT / "predictions" / "independent_predictions.csv"
)

RAW_FILE = (
    V6_ROOT / "monitoring" / "independent_validation_raw.csv"
)

V5_FILE = (
    ROOT / "results" / "phase7_monitoring"
    / "phase7_query_monitoring.csv"
)

OUT = V6_ROOT / "v6_improved_generalization"

MODELS = OUT / "models"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"

for folder in (OUT, MODELS, TABLES, FIGURES, REPORTS):
    folder.mkdir(parents=True, exist_ok=True)

SEED = 42
THRESHOLD_MS = 33.25075

FEATURES = [
    "log_planner_cost",
    "log_startup_cost",
    "log_estimated_rows",
    "log_plan_nodes",
    "sql_length",
    "sql_tokens",
    "join_count",
    "where_count",
    "and_count",
    "or_count",
    "group_by_count",
    "order_by_count",
    "having_count",
    "limit_count",
    "distinct_count",
    "aggregate_count",
    "subquery_count",
    "case_count",
    "union_count",
    "window_count",
    "table_count",
    "predicate_count",
    "comparison_count",
    "like_count",
    "in_count",
    "between_count",
    "is_null_count",
    "select_count",
    "literal_count",
    "nested_depth",
    "scale_log",
    "planner_cost_per_row",
    "planner_cost_per_node",
    "estimated_rows_per_node",
    "join_cost_interaction",
    "filter_cost_interaction",
    "sort_indicator",
    "hash_join_indicator",
    "nested_loop_indicator",
    "seq_scan_indicator",
    "index_scan_indicator"
]

def banner(message):
    print("\n" + "=" * 88)
    print(message)
    print("=" * 88, flush=True)

def save_json(path, data):
    def convert(x):
        if isinstance(x, dict):
            return {str(k): convert(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [convert(v) for v in x]
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            x = float(x)
        if isinstance(x, float) and not np.isfinite(x):
            return None
        if isinstance(x, Path):
            return str(x)
        return x

    path.write_text(
        json.dumps(convert(data), indent=2),
        encoding="utf-8"
    )

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def normalize_scale(value):
    text = str(value).lower().strip()
    if "50" in text:
        return "50k"
    if "10" in text:
        return "10k"
    if text in ("1k", "1", "1000"):
        return "1k"
    return "unknown"

def scale_numeric(value):
    return {
        "1k": 1000,
        "10k": 10000,
        "50k": 50000
    }.get(normalize_scale(value), np.nan)

def normalize_sql(sql):
    text = str(sql).lower()
    text = re.sub(r"'(?:''|[^'])*'", " ? ", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\b", " ? ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def sql_hash(sql):
    return hashlib.sha256(
        normalize_sql(sql).encode("utf-8")
    ).hexdigest()

def numeric_column(df, candidates, default=np.nan):
    for c in candidates:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)

def text_column(df, candidates, default=""):
    for c in candidates:
        if c in df.columns:
            return df[c].fillna(default).astype(str)
    return pd.Series(default, index=df.index, dtype=str)

def safe_log(series):
    x = pd.to_numeric(series, errors="coerce")
    return np.log1p(x.clip(lower=0))

def sql_features(sql):
    s = normalize_sql(sql)
    def count(pattern):
        return len(re.findall(pattern, s, flags=re.I))

    depth = 0
    maximum = 0
    for ch in s:
        if ch == "(":
            depth += 1
            maximum = max(maximum, depth)
        elif ch == ")":
            depth = max(0, depth - 1)

    return {
        "sql_length": len(s),
        "sql_tokens": len(s.split()),
        "join_count": count(r"\bjoin\b"),
        "where_count": count(r"\bwhere\b"),
        "and_count": count(r"\band\b"),
        "or_count": count(r"\bor\b"),
        "group_by_count": count(r"\bgroup\s+by\b"),
        "order_by_count": count(r"\border\s+by\b"),
        "having_count": count(r"\bhaving\b"),
        "limit_count": count(r"\blimit\b"),
        "distinct_count": count(r"\bdistinct\b"),
        "aggregate_count": count(
            r"\b(count|sum|avg|min|max)\s*\("
        ),
        "subquery_count": max(
            0, count(r"\bselect\b") - 1
        ),
        "case_count": count(r"\bcase\b"),
        "union_count": count(r"\bunion\b"),
        "window_count": count(r"\bover\s*\("),
        "table_count": count(r"\bfrom\b") + count(r"\bjoin\b"),
        "predicate_count": count(r"\band\b") + count(r"\bor\b") + count(r"\bwhere\b"),
        "comparison_count": count(r"(?:>=|<=|<>|!=|=|>|<)"),
        "like_count": count(r"\blike\b"),
        "in_count": count(r"\bin\s*\("),
        "between_count": count(r"\bbetween\b"),
        "is_null_count": count(r"\bis\s+(?:not\s+)?null\b"),
        "select_count": count(r"\bselect\b"),
        "literal_count": count(r"\?"),
        "nested_depth": maximum
    }

def prepare(df, source):
    d = df.copy()

    d["sql_text"] = text_column(
        d, ["sql", "sql_text", "query_sql", "query_text", "query"]
    )

    if d["sql_text"].str.strip().eq("").all():
        raise RuntimeError(
            f"{source} has no usable SQL column."
        )

    d["latency_ms"] = numeric_column(
        d,
        [
            "execution_time_ms",
            "latency_ms",
            "duration_ms",
            "elapsed_ms",
            "wall_time_ms"
        ]
    )

    if source == "V6":
        if "database_scale" in d:
            d["scale"] = d["database_scale"].map(normalize_scale)
        elif "scale" in d:
            d["scale"] = d["scale"].map(normalize_scale)
        else:
            raise RuntimeError("V6 database scale is missing.")
    else:
        d["scale"] = "1k"

    d["scale_value"] = d["scale"].map(scale_numeric)

    if source == "V6" and "fingerprint" in d:
        d["fingerprint"] = d["fingerprint"].astype(str)
    else:
        d["fingerprint"] = d["sql_text"].map(sql_hash)

    d["planner_cost"] = numeric_column(
        d,
        [
            "estimated_cost",
            "planner_total_cost",
            "plan_total_cost",
            "total_cost",
            "planner_cost"
        ]
    )

    d["startup_cost"] = numeric_column(
        d,
        [
            "startup_cost",
            "planner_startup_cost",
            "plan_startup_cost"
        ]
    )

    d["estimated_rows_value"] = numeric_column(
        d,
        [
            "estimated_rows",
            "planner_rows",
            "plan_rows",
            "root_plan_rows"
        ]
    )

    d["plan_nodes_value"] = numeric_column(
        d,
        [
            "plan_node_count",
            "planner_node_count",
            "plan_nodes"
        ]
    )

    d["plan_text"] = text_column(
        d, ["plan_node_types", "plan_types"], ""
    ).str.lower()

    d["session"] = text_column(
        d, ["session_id", "session"], "unknown"
    )

    d["source"] = source

    if source == "V6":
        d["original_predicted_ms"] = numeric_column(
            d, ["predicted_latency_ms"]
        )
        d["original_predicted_class"] = numeric_column(
            d, ["predicted_expensive"]
        )
    else:
        d["original_predicted_ms"] = np.nan
        d["original_predicted_class"] = np.nan

    d = d[
        d["latency_ms"].notna()
        & np.isfinite(d["latency_ms"])
        & (d["latency_ms"] >= 0)
    ].copy()

    d["target"] = (
        d["latency_ms"] >= THRESHOLD_MS
    ).astype(int)

    structures = [
        sql_features(x) for x in d["sql_text"]
    ]

    sf = pd.DataFrame(structures, index=d.index)

    for col in sf.columns:
        d[col] = sf[col]

    d["log_planner_cost"] = safe_log(d["planner_cost"])
    d["log_startup_cost"] = safe_log(d["startup_cost"])
    d["log_estimated_rows"] = safe_log(d["estimated_rows_value"])
    d["log_plan_nodes"] = safe_log(d["plan_nodes_value"])
    d["scale_log"] = safe_log(d["scale_value"])

    d["planner_cost_per_row"] = (
        d["planner_cost"] /
        (d["estimated_rows_value"].clip(lower=0) + 1)
    )

    d["planner_cost_per_node"] = (
        d["planner_cost"] /
        (d["plan_nodes_value"].clip(lower=0) + 1)
    )

    d["estimated_rows_per_node"] = (
        d["estimated_rows_value"] /
        (d["plan_nodes_value"].clip(lower=0) + 1)
    )

    d["join_cost_interaction"] = (
        d["join_count"] * d["log_planner_cost"]
    )

    d["filter_cost_interaction"] = (
        d["predicate_count"] * d["log_planner_cost"]
    )

    for name, token in {
        "sort_indicator": "sort",
        "hash_join_indicator": "hash join",
        "nested_loop_indicator": "nested loop",
        "seq_scan_indicator": "seq scan",
        "index_scan_indicator": "index scan"
    }.items():
        d[name] = d["plan_text"].str.contains(
            token, regex=False
        ).astype(int)

    d[FEATURES] = d[FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )

    return d.reset_index(drop=True)

def load_data():
    banner("LOADING EXISTING V5 AND V6 DATA")

    if not PRED_FILE.exists():
        raise FileNotFoundError(
            f"V6 predictions not found:\n{PRED_FILE}"
        )

    raw_v6 = pd.read_csv(PRED_FILE, low_memory=False)
    v6 = prepare(raw_v6, "V6")

    print("V6 records:", len(v6))
    print("V6 fingerprints:", v6["fingerprint"].nunique())
    print("V6 scales:", v6["scale"].value_counts().to_dict())

    v5 = None
    if V5_FILE.exists():
        try:
            v5 = prepare(
                pd.read_csv(V5_FILE, low_memory=False),
                "V5"
            )
            print("V5 original records:", len(v5))
        except Exception as e:
            print("[WARNING] Original V5 data unavailable:", e)
    else:
        print("[WARNING] Original monitoring file not found.")

    return v5, v6

def split_v6(v6):
    banner("GROUPED V6 TRAIN / VALIDATION / TEST SPLIT")

    groups = v6["fingerprint"].astype(str)

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=.20, random_state=SEED
    )

    development_idx, test_idx = next(
        splitter.split(v6, groups=groups)
    )

    development = v6.iloc[development_idx].copy()
    test = v6.iloc[test_idx].copy()

    splitter2 = GroupShuffleSplit(
        n_splits=1, test_size=.25, random_state=SEED + 1
    )

    train_idx, validation_idx = next(
        splitter2.split(
            development,
            groups=development["fingerprint"]
        )
    )

    train = development.iloc[train_idx].copy()
    validation = development.iloc[validation_idx].copy()

    assert set(train.fingerprint).isdisjoint(validation.fingerprint)
    assert set(train.fingerprint).isdisjoint(test.fingerprint)
    assert set(validation.fingerprint).isdisjoint(test.fingerprint)

    train["partition"] = "train"
    validation["partition"] = "validation"
    test["partition"] = "test"

    assignments = pd.concat([
        train[["fingerprint", "partition"]],
        validation[["fingerprint", "partition"]],
        test[["fingerprint", "partition"]]
    ]).drop_duplicates()

    assignments.to_csv(
        TABLES / "fingerprint_partition_manifest.csv",
        index=False
    )

    for name, d in [
        ("TRAIN", train),
        ("VALIDATION", validation),
        ("TEST", test)
    ]:
        print(
            name,
            "rows:", len(d),
            "fingerprints:", d.fingerprint.nunique(),
            "expensive rate:", round(d.target.mean(), 4)
        )

    return train, validation, test

def make_weights(df):
    frequency = df.groupby("fingerprint")[
        "fingerprint"
    ].transform("size")

    weights = 1.0 / frequency.to_numpy(float)
    return weights / np.mean(weights)

def classification_metrics(y, pred):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)

    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(
            y, pred, average="macro", zero_division=0
        ),
        "expensive_precision": precision_score(
            y, pred, zero_division=0
        ),
        "expensive_recall": recall_score(
            y, pred, zero_division=0
        ),
        "specificity": tn / max(tn + fp, 1),
        "false_negative_rate": fn / max(fn + tp, 1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }

def regression_metrics(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.maximum(
        np.asarray(pred, dtype=float), 0
    )

    return {
        "mae_ms": mean_absolute_error(y, pred),
        "rmse_ms": np.sqrt(
            mean_squared_error(y, pred)
        ),
        "log_mae": np.mean(
            np.abs(np.log1p(y) - np.log1p(pred))
        ),
        "r2": r2_score(y, pred),
        "median_absolute_error_ms": np.median(
            np.abs(y - pred)
        ),
        "median_q_error": np.median(
            np.maximum(
                (y + 1) / (pred + 1),
                (pred + 1) / (y + 1)
            )
        )
    }

def models_classification():
    return {
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=350,
            max_features=.85,
            min_samples_leaf=3,
            class_weight="balanced",
            n_jobs=-1,
            random_state=SEED
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=220,
            learning_rate=.045,
            max_leaf_nodes=15,
            l2_regularization=5,
            random_state=SEED
        )
    }

def models_regression():
    return {
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=350,
            min_samples_leaf=3,
            max_features=.9,
            n_jobs=-1,
            random_state=SEED
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=SEED
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=.045,
            max_leaf_nodes=15,
            l2_regularization=5,
            random_state=SEED
        )
    }

def choose_threshold(y, probability):
    candidates = np.linspace(.05, .95, 91)

    rows = []
    for threshold in candidates:
        pred = (probability >= threshold).astype(int)
        m = classification_metrics(y, pred)

        rows.append({
            "threshold": threshold,
            **m
        })

    table = pd.DataFrame(rows)

    # Prefer balanced performance and expensive-query detection.
    # Selection uses validation data only.
    table["selection_score"] = (
        .45 * table["balanced_accuracy"]
        + .35 * table["macro_f1"]
        + .20 * table["expensive_recall"]
    )

    selected = table.sort_values(
        ["selection_score", "macro_f1"],
        ascending=False
    ).iloc[0]

    return float(selected["threshold"]), table

def feature_matrices(train, validation, test):
    imputer = SimpleImputer(
        strategy="median",
        keep_empty_features=True
    )

    x_train = imputer.fit_transform(train[FEATURES])
    x_val = imputer.transform(validation[FEATURES])
    x_test = imputer.transform(test[FEATURES])

    if x_train.shape[1] != len(FEATURES):
        raise RuntimeError("Feature dimension mismatch.")

    return imputer, x_train, x_val, x_test

def train_classifier(train, validation, x_train, x_val):
    banner("TRAINING IMPROVED CLASSIFIERS")

    y_train = train["target"].to_numpy(int)
    y_val = validation["target"].to_numpy(int)

    if len(np.unique(y_train)) < 2:
        raise RuntimeError(
            "Training partition has only one class."
        )

    weights = make_weights(train)
    results = []
    fitted = {}

    for name, model in models_classification().items():
        print("\nTraining classifier:", name, flush=True)

        model.fit(
            x_train, y_train,
            sample_weight=weights
        )

        probabilities = model.predict_proba(x_val)[:, 1]

        threshold, threshold_table = choose_threshold(
            y_val, probabilities
        )

        pred = (probabilities >= threshold).astype(int)

        m = classification_metrics(y_val, pred)

        selection = (
            .45 * m["balanced_accuracy"]
            + .35 * m["macro_f1"]
            + .20 * m["expensive_recall"]
        )

        results.append({
            "model": name,
            "threshold": threshold,
            "selection_score": selection,
            **m
        })

        threshold_table.to_csv(
            TABLES / f"threshold_search_{name}.csv",
            index=False
        )

        fitted[name] = model

        print(
            "Accuracy:", round(m["accuracy"], 4),
            "Balanced:", round(m["balanced_accuracy"], 4),
            "F1:", round(m["macro_f1"], 4),
            "Recall:", round(m["expensive_recall"], 4),
            "Threshold:", round(threshold, 3)
        )

    leaderboard = pd.DataFrame(results).sort_values(
        "selection_score", ascending=False
    )

    leaderboard.to_csv(
        TABLES / "classification_validation_leaderboard.csv",
        index=False
    )

    winner = leaderboard.iloc[0]
    name = winner["model"]

    print("\nSelected classifier:", name)

    return fitted[name], float(winner["threshold"]), leaderboard

def train_regressor(train, validation, x_train, x_val):
    banner("TRAINING IMPROVED LATENCY REGRESSORS")

    y_train = np.log1p(
        train["latency_ms"].to_numpy(float)
    )
    y_val = validation["latency_ms"].to_numpy(float)

    weights = make_weights(train)

    results = []
    fitted = {}

    for name, model in models_regression().items():
        print("\nTraining regressor:", name, flush=True)

        model.fit(
            x_train, y_train,
            sample_weight=weights
        )

        prediction = np.expm1(
            np.clip(model.predict(x_val), 0, 25)
        )

        m = regression_metrics(y_val, prediction)

        results.append({
            "model": name,
            **m
        })

        fitted[name] = model

        print(
            "MAE:", round(m["mae_ms"], 3),
            "Log MAE:", round(m["log_mae"], 4),
            "R2:", round(m["r2"], 4)
        )

    leaderboard = pd.DataFrame(results).sort_values(
        ["log_mae", "mae_ms"]
    )

    leaderboard.to_csv(
        TABLES / "regression_validation_leaderboard.csv",
        index=False
    )

    name = leaderboard.iloc[0]["model"]

    print("\nSelected regressor:", name)

    return fitted[name], leaderboard

def predict_regression(model, x):
    return np.expm1(
        np.clip(model.predict(x), 0, 25)
    )

def feature_shift(v5, v6):
    banner("GENERALIZATION AUDIT: FEATURE SHIFT")

    if v5 is None or v5.empty:
        print("Original V5 workload unavailable.")
        return

    columns = [
        "latency_ms",
        "planner_cost",
        "estimated_rows_value",
        "join_count",
        "table_count",
        "sql_length",
        "predicate_count",
        "nested_depth"
    ]

    rows = []

    for scale, subset in v6.groupby("scale"):
        for feature in columns:
            a = pd.to_numeric(
                v5[feature], errors="coerce"
            ).dropna().to_numpy()

            b = pd.to_numeric(
                subset[feature], errors="coerce"
            ).dropna().to_numpy()

            if len(a) < 10 or len(b) < 10:
                continue

            result = ks_2samp(a, b)

            rows.append({
                "scale": scale,
                "feature": feature,
                "v5_median": np.median(a),
                "v6_median": np.median(b),
                "v5_p90": np.quantile(a, .9),
                "v6_p90": np.quantile(b, .9),
                "ks_statistic": result.statistic,
                "ks_p_descriptive": result.pvalue
            })

    if rows:
        pd.DataFrame(rows).to_csv(
            TABLES / "feature_shift.csv", index=False
        )
        print(pd.DataFrame(rows).to_string(index=False))
        print(
            "NOTE: Repeated execution rows violate the "
            "independence assumption of ordinary KS p-values."
        )

def failure_audit():
    banner("GENERALIZATION AUDIT: FAILED EXECUTIONS")

    if not RAW_FILE.exists():
        print("Raw execution file not found.")
        return

    raw = pd.read_csv(RAW_FILE, low_memory=False)

    if "status" not in raw:
        print("No execution status column.")
        return

    status = raw["status"].astype(str).str.lower()

    failed = raw[
        ~status.isin(["ok", "success", "successful", "true"])
    ].copy()

    if "error" in raw:
        has_error = (
            raw["error"].fillna("").astype(str).str.strip().ne("")
        )
        has_error &= ~raw["error"].astype(str).str.lower().isin(
            ["nan", "none", "null"]
        )
        failed = raw[
            (~status.isin(["ok", "success", "successful", "true"]))
            | has_error
        ].copy()

    failed["scale"] = text_column(
        failed, ["database_scale", "scale"], "unknown"
    ).map(normalize_scale)

    def classify_error(s):
        t = str(s).lower()
        if "timeout" in t or "canceling statement" in t:
            return "timeout"
        if "syntax" in t:
            return "syntax"
        if "does not exist" in t:
            return "missing_relation"
        if "permission" in t:
            return "permission"
        if "connection" in t:
            return "connection"
        return "other_or_unknown"

    failed["failure_type"] = text_column(
        failed, ["error"], ""
    ).map(classify_error)

    failed.to_csv(
        TABLES / "failed_executions.csv",
        index=False
    )

    summary = failed.groupby(
        ["scale", "failure_type"]
    ).size().reset_index(name="count")

    summary.to_csv(
        TABLES / "failure_summary.csv",
        index=False
    )

    print("Raw records:", len(raw))
    print("Failed records by status/error:", len(failed))
    print(summary.to_string(index=False))

    if len(raw) - len(failed) != 5271:
        print(
            "[WARNING] Failure count does not reconcile "
            "with the 5271 prediction rows."
        )

def original_model_audit(test):
    banner("FROZEN V5 MODEL COMPARISON")

    valid = test.dropna(
        subset=["original_predicted_class"]
    )

    if len(valid):
        m = classification_metrics(
            valid["target"],
            valid["original_predicted_class"].astype(int)
        )

        print("Original frozen classifier on held-out V6 subset:")
        print(json.dumps(m, indent=2))

        save_json(
            REPORTS / "original_v5_classifier_test.json", m
        )

    valid_reg = test.dropna(
        subset=["original_predicted_ms"]
    )

    if len(valid_reg):
        m = regression_metrics(
            valid_reg["latency_ms"],
            valid_reg["original_predicted_ms"]
        )

        print("Original frozen regressor on held-out V6 subset:")
        print(json.dumps(m, indent=2))

        save_json(
            REPORTS / "original_v5_regressor_test.json", m
        )

def bootstrap_test(test, B):
    banner("FINGERPRINT-CLUSTERED BOOTSTRAP")

    grouped = [
        g for _, g in test.groupby("fingerprint")
    ]

    rng = np.random.default_rng(SEED + 100)

    results = []

    for i in range(B):
        idx = rng.integers(
            0, len(grouped), len(grouped)
        )

        sample = pd.concat(
            [grouped[j] for j in idx],
            ignore_index=True
        )

        m = classification_metrics(
            sample["target"],
            sample["improved_class"]
        )

        r = regression_metrics(
            sample["latency_ms"],
            sample["improved_latency_ms"]
        )

        row = {
            "iteration": i,
            "accuracy": m["accuracy"],
            "balanced_accuracy": m["balanced_accuracy"],
            "macro_f1": m["macro_f1"],
            "expensive_recall": m["expensive_recall"],
            "mae_ms": r["mae_ms"],
            "log_mae": r["log_mae"]
        }

        old = sample["original_predicted_class"]
        valid = old.notna()

        if valid.all():
            original = classification_metrics(
                sample["target"],
                old.astype(int)
            )
            row["accuracy_gain_vs_v5"] = (
                m["accuracy"] - original["accuracy"]
            )
            row["recall_gain_vs_v5"] = (
                m["expensive_recall"]
                - original["expensive_recall"]
            )

        results.append(row)

    draws = pd.DataFrame(results)

    draws.to_csv(
        TABLES / "bootstrap_draws.csv",
        index=False
    )

    summary = []

    for metric in draws.columns:
        if metric == "iteration":
            continue

        values = draws[metric].dropna().to_numpy()

        if len(values) == 0:
            continue

        summary.append({
            "metric": metric,
            "bootstrap_mean": np.mean(values),
            "ci95_low": np.quantile(values, .025),
            "ci95_high": np.quantile(values, .975)
        })

    result = pd.DataFrame(summary)

    result.to_csv(
        TABLES / "bootstrap_confidence_intervals.csv",
        index=False
    )

    print(result.to_string(index=False))

def per_scale_report(test):
    banner("HELD-OUT TEST PERFORMANCE BY SCALE")

    rows = []

    for scale, g in test.groupby("scale"):
        c = classification_metrics(
            g["target"], g["improved_class"]
        )

        r = regression_metrics(
            g["latency_ms"], g["improved_latency_ms"]
        )

        row = {
            "scale": scale,
            "executions": len(g),
            "fingerprints": g.fingerprint.nunique(),
            **c,
            **r
        }

        rows.append(row)

    result = pd.DataFrame(rows)

    result.to_csv(
        TABLES / "test_per_scale.csv",
        index=False
    )

    print(result.to_string(index=False))

def baseline_report(v5, train, test):
    banner("BASELINE COMPARISON")

    reference = v5 if v5 is not None and len(v5) else train

    majority = int(
        reference["target"].mean() >= .5
    )

    median_latency = float(
        reference["latency_ms"].median()
    )

    y = test["target"].to_numpy(int)
    actual_ms = test["latency_ms"].to_numpy(float)

    baselines = []

    for name, pred in [
        ("Improved classifier", test["improved_class"]),
        ("Original V5 classifier", test["original_predicted_class"]),
        ("Frozen majority", np.full(len(test), majority)),
        ("Always inexpensive", np.zeros(len(test), dtype=int)),
        ("Always expensive", np.ones(len(test), dtype=int))
    ]:
        pred = np.asarray(pred, dtype=float)
        valid = np.isfinite(pred)

        if valid.sum() == 0:
            continue

        m = classification_metrics(
            y[valid], pred[valid].astype(int)
        )

        baselines.append({
            "task": "classification",
            "model": name,
            "reference": (
                "V5 original"
                if v5 is not None and len(v5)
                else "V6 train only"
            ),
            **m
        })

    pd.DataFrame(baselines).to_csv(
        TABLES / "classification_baselines.csv",
        index=False
    )

    reg_rows = []

    for name, pred in [
        ("Improved regressor", test["improved_latency_ms"]),
        ("Original V5 regressor", test["original_predicted_ms"]),
        ("Frozen median", np.full(len(test), median_latency))
    ]:
        pred = np.asarray(pred, dtype=float)
        valid = np.isfinite(pred)

        if valid.sum() == 0:
            continue

        m = regression_metrics(
            actual_ms[valid], pred[valid]
        )

        reg_rows.append({
            "model": name,
            "reference": (
                "V5 original"
                if v5 is not None and len(v5)
                else "V6 train only"
            ),
            **m
        })

    pd.DataFrame(reg_rows).to_csv(
        TABLES / "regression_baselines.csv",
        index=False
    )

    print("\nClassification baselines:")
    print(pd.DataFrame(baselines).to_string(index=False))

    print("\nRegression baselines:")
    print(pd.DataFrame(reg_rows).to_string(index=False))

def make_figures(test):
    banner("GENERATING PUBLICATION FIGURES")

    cm = confusion_matrix(
        test["target"],
        test["improved_class"],
        labels=[0, 1]
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm, cmap="Blues")

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center", fontsize=14
            )

    ax.set_xticks([0, 1], ["Inexpensive", "Expensive"])
    ax.set_yticks([0, 1], ["Inexpensive", "Expensive"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("V6 improved classifier — held-out fingerprints")

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(
        FIGURES / "confusion_matrix.png", dpi=250
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))

    for scale, g in test.groupby("scale"):
        ax.scatter(
            np.log1p(g["latency_ms"]),
            np.log1p(g["improved_latency_ms"]),
            s=15, alpha=.45, label=scale
        )

    upper = max(
        np.log1p(test["latency_ms"]).max(),
        np.log1p(test["improved_latency_ms"]).max()
    )

    ax.plot(
        [0, upper], [0, upper],
        "--", color="black", label="Perfect prediction"
    )

    ax.set_xlabel("log1p(actual latency, ms)")
    ax.set_ylabel("log1p(predicted latency, ms)")
    ax.set_title("Improved latency regression")
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        FIGURES / "latency_predictions.png", dpi=250
    )
    plt.close(fig)

    per_scale = pd.read_csv(
        TABLES / "test_per_scale.csv"
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        per_scale["scale"],
        per_scale["accuracy"] * 100,
        label="Accuracy"
    )
    ax.plot(
        per_scale["scale"],
        per_scale["balanced_accuracy"] * 100,
        marker="o",
        color="black",
        label="Balanced accuracy"
    )
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage")
    ax.set_title("Held-out classification performance by scale")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        FIGURES / "classification_by_scale.png", dpi=250
    )
    plt.close(fig)

def write_report(train, validation, test, classifier_name,
                 regressor_name, threshold, c, r):

    lines = [
        "# Phase 8 V6 — Improved Generalization Experiment",
        "",
        "## Experimental design",
        "",
        "This is a post-V6 development experiment.",
        "It must not replace the original frozen V5 independent-validation result.",
        "",
        f"- V6 training executions: {len(train)}",
        f"- V6 validation executions: {len(validation)}",
        f"- V6 held-out test executions: {len(test)}",
        f"- Held-out fingerprints: {test.fingerprint.nunique()}",
        "- Split unit: SQL fingerprint",
        "- V6 split fractions: approximately 60/20/20 by fingerprint",
        "- Hyperparameter/model selection: validation partition only",
        "- Test partition: excluded from model fitting and threshold tuning",
        "",
        "## Selected models",
        "",
        f"- Classifier: {classifier_name}",
        f"- Regressor: {regressor_name}",
        f"- Probability threshold: {threshold:.4f}",
        "",
        "## Held-out classification",
        ""
    ]

    for k, v in c.items():
        lines.append(f"- {k}: {v}")

    lines.extend([
        "",
        "## Held-out regression",
        ""
    ])

    for k, v in r.items():
        lines.append(f"- {k}: {v}")

    lines.extend([
        "",
        "## Limitations",
        "",
        "- The V6 corpus was previously inspected, so this is "
        "a post-hoc development split rather than a pristine prospective test.",
        "- All three scales share SQL fingerprints; grouped splitting "
        "prevents exact fingerprint overlap between partitions.",
        "- A random fingerprint split is not a leave-one-scale-out test.",
        "- Scale-aware training may improve within-scale interpolation "
        "without proving generalization to entirely unseen database sizes.",
        "- Timeouts are excluded from ordinary latency regression; "
        "results are conditional on successful execution.",
        "- The original V5 and new V6 classifiers use different "
        "training protocols and feature sets.",
        "- The original V5 model remains frozen and is evaluated "
        "only as a reference.",
        "- This experiment is not a final independent external validation.",
        "- Any subsequent tuning based on the held-out results "
        "requires another untouched evaluation set.",
        ""
    ])

    (REPORTS / "RESEARCH_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

def main():
    global THRESHOLD_MS

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bootstrap", type=int, default=1000
    )

    parser.add_argument(
        "--threshold-ms", type=float, default=33.25075
    )

    args = parser.parse_args()

    THRESHOLD_MS = args.threshold_ms

    banner("PHASE 8 V6 — IMPROVED GENERALIZATION")
    print("Project:", ROOT)
    print("Output:", OUT)
    print("Database access: NO")
    print("Production modifications: NO")
    print("Original V5 model modifications: NO")
    print("Classification cutoff:", THRESHOLD_MS, "ms")

    v5, v6 = load_data()

    if len(v6) < 100:
        raise RuntimeError("Insufficient V6 records.")

    train, validation, test = split_v6(v6)

    # V5 original monitoring is used for audit and baseline
    # reference, not automatically mixed into V6 training.
    # This avoids treating potentially incompatible feature
    # definitions as equivalent.

    imputer, x_train, x_val, x_test = feature_matrices(
        train, validation, test
    )

    classifier, probability_threshold, class_board = (
        train_classifier(train, validation, x_train, x_val)
    )

    regressor, reg_board = train_regressor(
        train, validation, x_train, x_val
    )

    banner("HELD-OUT TEST EVALUATION")

    probabilities = classifier.predict_proba(x_test)[:, 1]

    test["improved_probability"] = probabilities

    test["improved_class"] = (
        probabilities >= probability_threshold
    ).astype(int)

    test["improved_latency_ms"] = predict_regression(
        regressor, x_test
    )

    class_result = classification_metrics(
        test["target"],
        test["improved_class"]
    )

    reg_result = regression_metrics(
        test["latency_ms"],
        test["improved_latency_ms"]
    )

    print("\nIMPROVED CLASSIFIER:")
    print(json.dumps(class_result, indent=2))

    print("\nIMPROVED REGRESSOR:")
    print(json.dumps(reg_result, indent=2))

    test.to_csv(
        TABLES / "heldout_test_predictions.csv",
        index=False
    )

    joblib.dump(
        {
            "imputer": imputer,
            "classifier": classifier,
            "regressor": regressor,
            "features": FEATURES,
            "probability_threshold": probability_threshold,
            "expensive_threshold_ms": THRESHOLD_MS,
            "scale_aware": True
        },
        MODELS / "v6_improved_models.joblib"
    )

    save_json(
        REPORTS / "experiment_manifest.json",
        {
            "seed": SEED,
            "classification_cutoff_ms": THRESHOLD_MS,
            "train_fingerprints": train.fingerprint.nunique(),
            "validation_fingerprints": validation.fingerprint.nunique(),
            "test_fingerprints": test.fingerprint.nunique(),
            "v6_sha256": sha256_file(PRED_FILE),
            "v5_sha256": (
                sha256_file(V5_FILE)
                if V5_FILE.exists() else None
            ),
            "features": FEATURES,
            "selected_classifier": class_board.iloc[0]["model"],
            "selected_regressor": reg_board.iloc[0]["model"],
            "probability_threshold": probability_threshold
        }
    )

    original_model_audit(test)

    per_scale_report(test)

    baseline_report(v5, train, test)

    feature_shift(v5, v6)

    failure_audit()

    bootstrap_test(test, args.bootstrap)

    make_figures(test)

    write_report(
        train,
        validation,
        test,
        class_board.iloc[0]["model"],
        reg_board.iloc[0]["model"],
        probability_threshold,
        class_result,
        reg_result
    )

    banner("PHASE 8 V6 COMPLETE")

    print("Classifier:", class_board.iloc[0]["model"])
    print("Regressor:", reg_board.iloc[0]["model"])
    print("Accuracy:", round(class_result["accuracy"] * 100, 2), "%")
    print("Balanced accuracy:",
          round(class_result["balanced_accuracy"] * 100, 2), "%")
    print("Macro F1:", round(class_result["macro_f1"], 4))
    print("Expensive-query recall:",
          round(class_result["expensive_recall"] * 100, 2), "%")
    print("Regression MAE:",
          round(reg_result["mae_ms"], 3), "ms")
    print("Regression log MAE:",
          round(reg_result["log_mae"], 4))
    print("Results:", OUT)

if __name__ == "__main__":
    main()
