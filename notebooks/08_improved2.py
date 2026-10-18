
from pathlib import Path
from datetime import datetime
import json
import re
import warnings

import joblib
import numpy as np
import pandas as pd
import sqlglot

from sqlglot import exp

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent

DATA = (
    ROOT / "results"
    / "phase8_2_expanded"
    / "phase82_training.csv"
)

OUT = (
    ROOT / "results"
    / "phase8_2_expanded"
    / "model_results_final"
)

OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
REPEATS = 20
TEST_SIZE = 0.25
CV_FOLDS = 3

LABELS = ["admin", "doctor", "research"]
FEATURE_SETS = ["sql_only", "sql_ast"]

NUMERIC = [
    "sql_length",
    "sql_words",
    "join_count",
    "where_count",
    "group_count",
    "order_count",
    "having_count",
    "subquery_count",
    "table_count",
    "column_count",
    "ast_depth"
]


def stage(message):
    print(
        "\n" + "=" * 76
        + "\n" + message
        + "\n" + "=" * 76,
        flush=True
    )


def normalize_sql(sql):
    sql = str(sql)

    sql = re.sub(
        r"/\*.*?\*/",
        " ",
        sql,
        flags=re.S
    )

    sql = re.sub(
        r"--[^\n]*",
        " ",
        sql
    )

    sql = re.sub(
        r"'(?:''|[^'])*'",
        " STR_VALUE ",
        sql
    )

    sql = re.sub(
        r"\b\d+(?:\.\d+)?\b",
        " NUM_VALUE ",
        sql
    )

    return re.sub(
        r"\s+",
        " ",
        sql
    ).strip().lower()


def ast_features(sql):
    try:
        tree = sqlglot.parse_one(
            str(sql),
            read="postgres"
        )

        nodes = list(tree.walk())

        tables = [
            n.name.lower()
            for n in tree.find_all(exp.Table)
            if n.name
        ]

        columns = [
            n.name.lower()
            for n in tree.find_all(exp.Column)
            if n.name
        ]

        tokens = (
            ["table_" + t for t in tables]
            + ["column_" + c for c in columns]
            + [
                "node_" + type(n).__name__.lower()
                for n in nodes[:150]
            ]
        )

        return {
            "ast_text": " ".join(tokens) or "empty",
            "table_count": len(tables),
            "column_count": len(columns),
            "ast_depth": max(
                (
                    getattr(n, "depth", 0)
                    for n in nodes
                ),
                default=0
            )
        }

    except Exception:
        return {
            "ast_text": "parse_failed",
            "table_count": 0,
            "column_count": 0,
            "ast_depth": 0
        }


def prepare_data():
    stage("1. LOADING EXPANDED PHASE 8.2 DATA")

    if not DATA.exists():
        raise FileNotFoundError(
            f"Missing dataset: {DATA}"
        )

    df = pd.read_csv(
        DATA,
        low_memory=False
    )

    required = [
        "sql",
        "role_label",
        "query_hash",
        "fingerprint_hash",
        "source"
    ]

    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    df = df.dropna(
        subset=required
    ).copy()

    df["role_label"] = (
        df["role_label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df = df[
        df["role_label"].isin(LABELS)
    ].copy()

    for col in [
        "query_hash",
        "fingerprint_hash",
        "source"
    ]:
        df[col] = df[col].astype(str)

    conflicts = (
        df.groupby("query_hash")["role_label"]
        .nunique()
    )

    if (conflicts > 1).any():
        raise RuntimeError(
            "Conflicting role labels remain."
        )

    df["sql_text"] = (
        df["sql"].map(normalize_sql)
    )

    unique_sql = df[
        ["query_hash", "sql"]
    ].drop_duplicates("query_hash")

    cache = {
        row.query_hash: ast_features(row.sql)
        for row in unique_sql.itertuples(
            index=False
        )
    }

    df["ast_text"] = df["query_hash"].map(
        lambda h: cache[h]["ast_text"]
    )

    for col in [
        "table_count",
        "column_count",
        "ast_depth"
    ]:
        df[col] = df["query_hash"].map(
            lambda h, c=col: cache[h][c]
        )

    s = df["sql_text"]

    df["sql_length"] = s.str.len()
    df["sql_words"] = s.str.split().str.len()

    patterns = {
        "join_count": r"\bjoin\b",
        "where_count": r"\bwhere\b",
        "group_count": r"\bgroup\s+by\b",
        "order_count": r"\border\s+by\b",
        "having_count": r"\bhaving\b",
        "subquery_count": r"\bselect\b"
    }

    for col, pattern in patterns.items():
        df[col] = s.str.count(pattern)

    df["subquery_count"] = (
        df["subquery_count"] - 1
    ).clip(lower=0)

    df[NUMERIC] = (
        df[NUMERIC]
        .fillna(0)
        .astype(float)
    )

    df = df.reset_index(drop=True)

    print("Records:", len(df))
    print("SQL hashes:", df.query_hash.nunique())
    print(
        "Fingerprints:",
        df.fingerprint_hash.nunique()
    )
    print(
        "Classes:",
        df.role_label.value_counts().to_dict()
    )
    print(
        "Sources:",
        df.source.value_counts().to_dict()
    )

    return df


def candidate_models():
    return {
        "linear_svm": LinearSVC(
            C=1.0,
            class_weight="balanced",
            max_iter=10000,
            random_state=SEED
        ),

        "logistic": LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            random_state=SEED
        ),

        "extra_trees": ExtraTreesClassifier(
            n_estimators=150,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=SEED
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=150,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=SEED
        )
    }


def build_pipeline(estimator, feature_set):
    transformers = [
        (
            "char",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                max_features=10000,
                sublinear_tf=True
            ),
            "sql_text"
        ),

        (
            "word",
            TfidfVectorizer(
                token_pattern=r"(?u)\b\w+\b",
                ngram_range=(1, 2),
                max_features=6000,
                sublinear_tf=True
            ),
            "sql_text"
        )
    ]

    if feature_set == "sql_ast":
        transformers.extend([
            (
                "ast",
                TfidfVectorizer(
                    token_pattern=r"(?u)\b\w+\b",
                    ngram_range=(1, 2),
                    max_features=6000
                ),
                "ast_text"
            ),

            (
                "numeric",
                StandardScaler(),
                NUMERIC
            )
        ])

    return Pipeline([
        (
            "features",
            ColumnTransformer(
                transformers,
                sparse_threshold=0.3
            )
        ),
        (
            "classifier",
            clone(estimator)
        )
    ])


def fingerprint_weights(df):
    counts = (
        df["fingerprint_hash"]
        .value_counts()
    )

    weights = (
        df["fingerprint_hash"]
        .map(
            lambda fp: 1.0 / counts[fp]
        )
        .to_numpy(dtype=float)
    )

    return weights / weights.mean()


def fit_model(model, train, weighted):
    kwargs = {}

    if weighted:
        kwargs["classifier__sample_weight"] = (
            fingerprint_weights(train)
        )

    model.fit(
        train,
        train["role_label"],
        **kwargs
    )

    return model


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),

        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=LABELS,
                average="macro",
                zero_division=0
            )
        ),

        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=LABELS,
                average="weighted",
                zero_division=0
            )
        )
    }


def select_model(train, feature_set, seed):
    cv = StratifiedGroupKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=seed
    )

    splits = list(
        cv.split(
            train,
            train["role_label"],
            groups=train["fingerprint_hash"]
        )
    )

    results = []

    for name, estimator in candidate_models().items():

        for weighted in [False, True]:
            fold_scores = []

            for train_idx, val_idx in splits:
                tr = train.iloc[train_idx]
                val = train.iloc[val_idx]

                if tr.role_label.nunique() < 2:
                    fold_scores.append(0.0)
                    continue

                try:
                    model = build_pipeline(
                        estimator,
                        feature_set
                    )

                    fit_model(
                        model,
                        tr,
                        weighted
                    )

                    predictions = model.predict(val)

                    metrics = calculate_metrics(
                        val.role_label,
                        predictions
                    )

                    fold_scores.append(
                        metrics["macro_f1"]
                    )

                except (ValueError, RuntimeError) as exc:
                    print(
                        "CV warning:",
                        name,
                        "weighted:",
                        weighted,
                        str(exc)[:150],
                        flush=True
                    )

                    fold_scores.append(0.0)

            results.append({
                "model": name,
                "weighted": weighted,
                "cv_macro_f1": float(
                    np.mean(fold_scores)
                ),
                "cv_std": float(
                    np.std(fold_scores)
                ),
                "fold_scores": json.dumps(
                    fold_scores
                )
            })

    best = max(
        results,
        key=lambda row: row["cv_macro_f1"]
    )

    return best, results


def source_metrics(test, predictions, source):
    mask = (
        test["source"].to_numpy() == source
    )

    if not mask.any():
        return {
            "accuracy": None,
            "macro_f1": None,
            "n": 0
        }

    metrics = calculate_metrics(
        test["role_label"].to_numpy()[mask],
        predictions[mask]
    )

    metrics["n"] = int(mask.sum())

    return metrics


def evaluate_feature_set(df, feature_set):
    stage(
        "2. FINGERPRINT-GROUPED EVALUATION: "
        + feature_set
    )

    splitter = GroupShuffleSplit(
        n_splits=REPEATS,
        test_size=TEST_SIZE,
        random_state=SEED
    )

    result_rows = []
    cv_rows = []
    prediction_rows = []

    for run, (train_idx, test_idx) in enumerate(
        splitter.split(
            df,
            df.role_label,
            groups=df.fingerprint_hash
        ),
        start=1
    ):
        train = df.iloc[train_idx].copy()
        test = df.iloc[test_idx].copy()

        overlap = (
            set(train.fingerprint_hash)
            & set(test.fingerprint_hash)
        )

        if overlap:
            raise RuntimeError(
                "Fingerprint leakage detected."
            )

        if train.role_label.nunique() < 2:
            print(
                "Skipping split with insufficient classes:",
                run
            )
            continue

        best, selection_results = select_model(
            train,
            feature_set,
            SEED + run
        )

        for item in selection_results:
            cv_rows.append({
                "run": run,
                "feature_set": feature_set,
                **item
            })

        model = build_pipeline(
            candidate_models()[best["model"]],
            feature_set
        )

        fit_model(
            model,
            train,
            best["weighted"]
        )

        predictions = model.predict(test)

        metrics = calculate_metrics(
            test.role_label,
            predictions
        )

        original = source_metrics(
            test,
            predictions,
            "phase6_original"
        )

        expanded = source_metrics(
            test,
            predictions,
            "phase82_expansion"
        )

        majority = (
            train.role_label
            .value_counts()
            .idxmax()
        )

        baseline = float(
            (test.role_label == majority).mean()
        )

        result_rows.append({
            "run": run,
            "feature_set": feature_set,
            "model": best["model"],
            "weighted": best["weighted"],
            "cv_macro_f1": best["cv_macro_f1"],
            "train_records": len(train),
            "test_records": len(test),
            "train_fingerprints": (
                train.fingerprint_hash.nunique()
            ),
            "test_fingerprints": (
                test.fingerprint_hash.nunique()
            ),
            "test_classes": (
                test.role_label.nunique()
            ),
            "baseline_accuracy": baseline,
            **metrics,
            "original_n": original["n"],
            "original_accuracy": original["accuracy"],
            "original_macro_f1": original["macro_f1"],
            "expanded_n": expanded["n"],
            "expanded_accuracy": expanded["accuracy"],
            "expanded_macro_f1": expanded["macro_f1"]
        })

        prediction_df = test[
            [
                "query_hash",
                "fingerprint_hash",
                "role_label",
                "source"
            ]
        ].copy()

        prediction_df["predicted_role"] = predictions

        prediction_df["correct"] = (
            prediction_df.role_label.to_numpy()
            == predictions
        )

        prediction_df["run"] = run
        prediction_df["feature_set"] = feature_set

        prediction_rows.append(
            prediction_df
        )

        print(
            f"Run {run:02d}/{REPEATS} | "
            f"model={best['model']} | "
            f"weighted={best['weighted']} | "
            f"accuracy={metrics['accuracy']:.4f} | "
            f"macroF1={metrics['macro_f1']:.4f} | "
            f"CV={best['cv_macro_f1']:.4f}",
            flush=True
        )

    results_df = pd.DataFrame(
        result_rows
    )

    predictions_df = pd.concat(
        prediction_rows,
        ignore_index=True
    )

    results_df.to_csv(
        OUT / f"{feature_set}_repeated_results.csv",
        index=False
    )

    pd.DataFrame(cv_rows).to_csv(
        OUT / f"{feature_set}_cv_results.csv",
        index=False
    )

    predictions_df.to_csv(
        OUT / f"{feature_set}_predictions.csv",
        index=False
    )

    error_analysis = (
        predictions_df
        .groupby(
            [
                "fingerprint_hash",
                "role_label",
                "source"
            ],
            dropna=False
        )
        .agg(
            test_predictions=("correct", "size"),
            correct=("correct", "sum"),
            accuracy=("correct", "mean"),
            distinct_runs=("run", "nunique")
        )
        .reset_index()
        .sort_values(
            ["accuracy", "test_predictions"],
            ascending=[True, False]
        )
    )

    error_analysis.to_csv(
        OUT / (
            f"{feature_set}_"
            "fingerprint_error_analysis.csv"
        ),
        index=False
    )

    class_report = classification_report(
        predictions_df.role_label,
        predictions_df.predicted_role,
        labels=LABELS,
        zero_division=0,
        output_dict=True
    )

    (
        OUT / f"{feature_set}_classification_report.json"
    ).write_text(
        json.dumps(class_report, indent=2),
        encoding="utf-8"
    )

    matrix = confusion_matrix(
        predictions_df.role_label,
        predictions_df.predicted_role,
        labels=LABELS
    )

    pd.DataFrame(
        matrix,
        index=LABELS,
        columns=LABELS
    ).to_csv(
        OUT / f"{feature_set}_confusion_matrix.csv"
    )

    def average(column):
        series = results_df[column]

        if series.notna().any():
            return float(series.mean())

        return None

    summary = {
        "feature_set": feature_set,
        "evaluated_splits": len(results_df),
        "mean_accuracy": average("accuracy"),
        "std_accuracy": float(
            results_df.accuracy.std(ddof=1)
        ),
        "min_accuracy": float(
            results_df.accuracy.min()
        ),
        "max_accuracy": float(
            results_df.accuracy.max()
        ),
        "mean_macro_f1": average("macro_f1"),
        "std_macro_f1": float(
            results_df.macro_f1.std(ddof=1)
        ),
        "mean_original_accuracy": average(
            "original_accuracy"
        ),
        "mean_expanded_accuracy": average(
            "expanded_accuracy"
        ),
        "weighted_selected_splits": int(
            results_df.weighted.sum()
        ),
        "splits_missing_test_class": int(
            (results_df.test_classes < 3).sum()
        )
    }

    return summary


def train_final_model(df, feature_set):
    stage(
        "3. TRAINING FINAL MODEL: "
        + feature_set
    )

    best, selection_results = select_model(
        df,
        feature_set,
        SEED + 1000
    )

    pd.DataFrame(
        selection_results
    ).to_csv(
        OUT / f"{feature_set}_final_cv_selection.csv",
        index=False
    )

    model = build_pipeline(
        candidate_models()[best["model"]],
        feature_set
    )

    fit_model(
        model,
        df,
        best["weighted"]
    )

    model_path = (
        OUT / f"phase82_final_{feature_set}.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    print("Selected model:", best["model"])
    print("Fingerprint weighting:", best["weighted"])
    print(
        "Training CV macro F1:",
        round(best["cv_macro_f1"], 4)
    )
    print("Saved:", model_path)

    return {
        "feature_set": feature_set,
        "model": best["model"],
        "weighted": bool(best["weighted"]),
        "training_cv_macro_f1": best["cv_macro_f1"],
        "model_path": str(model_path)
    }


def main():
    stage(
        "PHASE 8.2 - FINAL ROBUST SQL WORKLOAD CLASSIFIER"
    )

    df = prepare_data()

    summaries = []

    for feature_set in FEATURE_SETS:
        summary = evaluate_feature_set(
            df,
            feature_set
        )

        summaries.append(summary)

    comparison = pd.DataFrame(
        summaries
    )

    comparison.to_csv(
        OUT / "phase82_comparison.csv",
        index=False
    )

    final_models = []

    for feature_set in FEATURE_SETS:
        final_models.append(
            train_final_model(
                df,
                feature_set
            )
        )

    report = {
        "phase": "8.2",
        "created_at": datetime.now().isoformat(),
        "dataset": str(DATA),
        "records": len(df),
        "fingerprints": int(
            df.fingerprint_hash.nunique()
        ),
        "evaluation": (
            "20 fingerprint-grouped holdouts with "
            "training-only grouped CV for model "
            "and weighting selection"
        ),
        "results": summaries,
        "final_models": final_models,
        "limitations": [
            "Synthetic queries have not been executed.",
            "Repeated holdouts overlap and are not "
            "independent external test datasets.",
            "GroupShuffleSplit does not guarantee "
            "all three classes in every test split.",
            "Final models use all available records.",
            "Final models have no separate independent "
            "test score.",
            "This evaluates SQL role prediction, "
            "not database optimization effectiveness."
        ]
    }

    (
        OUT / "phase82_report.json"
    ).write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    stage("PHASE 8.2 COMPLETE")

    print(
        comparison.to_string(
            index=False
        )
    )

    print("\nFinal models:")

    for item in final_models:
        print(
            item["feature_set"],
            "->",
            item["model"],
            "| weighted:",
            item["weighted"]
        )

    print("\nAll outputs saved to:")
    print(OUT)


if __name__ == "__main__":
    main()
