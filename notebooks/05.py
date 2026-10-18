from pathlib import Path
import json
import math
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
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
from sklearn.model_selection import (
    GroupShuffleSplit,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from xgboost import (
    XGBClassifier,
    XGBRegressor,
)

warnings.filterwarnings("ignore")


# ============================================================
# PHASE 8
# AI WORKLOAD LEARNER
#
# Adaptive Healthcare Knowledge Database
#
# INPUT:
#   results/phase7_monitoring/phase7_query_monitoring.csv
#
# OUTPUT:
#   results/phase8_ai_workload/
#   models/phase8/
#
# IMPORTANT:
#   - Does NOT modify PostgreSQL
#   - Does NOT create indexes
#   - Does NOT optimize database yet
#   - Phase 9 owns physical-design recommendations
# ============================================================


print("=" * 100)
print("PHASE 8 - AI WORKLOAD LEARNER")
print("Adaptive Healthcare Knowledge Database")
print("=" * 100)


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PHASE7_DIR = (
    PROJECT_ROOT
    / "results"
    / "phase7_monitoring"
)

INPUT_FILE = (
    PHASE7_DIR
    / "phase7_query_monitoring.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "phase8_ai_workload"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "phase8"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. OUTPUT FILES
# ============================================================

FEATURE_DATASET_FILE = (
    RESULTS_DIR
    / "phase8_feature_dataset.csv"
)

FINGERPRINT_PROFILE_FILE = (
    RESULTS_DIR
    / "phase8_fingerprint_profiles.csv"
)

CLUSTER_FILE = (
    RESULTS_DIR
    / "phase8_clusters.csv"
)

CLUSTER_SUMMARY_FILE = (
    RESULTS_DIR
    / "phase8_cluster_summary.csv"
)

CLASSIFICATION_RESULTS_FILE = (
    RESULTS_DIR
    / "phase8_workload_classification_results.csv"
)

RISK_RESULTS_FILE = (
    RESULTS_DIR
    / "phase8_risk_model_results.csv"
)

DEMAND_RESULTS_FILE = (
    RESULTS_DIR
    / "phase8_demand_model_results.csv"
)

OPTIMIZATION_PRIORITY_FILE = (
    RESULTS_DIR
    / "phase8_optimization_priority.csv"
)

FEATURE_IMPORTANCE_FILE = (
    RESULTS_DIR
    / "phase8_feature_importance.csv"
)

METADATA_FILE = (
    RESULTS_DIR
    / "phase8_metadata.json"
)

REPORT_FILE = (
    RESULTS_DIR
    / "phase8_report.txt"
)


# ============================================================
# 3. RANDOM SEED
# ============================================================

RANDOM_STATE = 42

np.random.seed(
    RANDOM_STATE
)


# ============================================================
# 4. DISPLAY HELPER
# ============================================================

def stage(title):

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


# ============================================================
# 5. SAFE NUMERIC HELPERS
# ============================================================

def safe_numeric(series):

    return pd.to_numeric(
        series,
        errors="coerce"
    )


def safe_log1p(series):

    values = safe_numeric(
        series
    ).fillna(0)

    values = values.clip(
        lower=0
    )

    return np.log1p(
        values
    )


# ============================================================
# 6. LIST / TEXT HELPERS
# ============================================================

def count_csv_items(value):

    if pd.isna(value):
        return 0

    text = str(value).strip()

    if not text:
        return 0

    return len(
        [
            x
            for x in text.split(",")
            if x.strip()
        ]
    )


def text_length(value):

    if pd.isna(value):
        return 0

    return len(
        str(value)
    )


# ============================================================
# 7. RMSE COMPATIBILITY HELPER
# ============================================================

def calculate_rmse(
    y_true,
    y_pred
):

    return math.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )


# ============================================================
# 8. CHECK INPUT
# ============================================================

stage("CHECKING PHASE 7 INPUT")


if not INPUT_FILE.exists():

    raise FileNotFoundError(
        "Phase 7 monitoring file was not found:\n"
        + str(INPUT_FILE)
    )


df = pd.read_csv(
    INPUT_FILE
)


print(
    "Phase 7 monitoring file:",
    INPUT_FILE
)

print(
    "Rows:",
    f"{len(df):,}"
)

print(
    "Columns:",
    len(df.columns)
)


# ============================================================
# 9. VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = {

    "workload_type",
    "query_family",
    "query_name",
    "complexity",

    "query_hash",
    "fingerprint_hash",
    "query_fingerprint",

    "tables_accessed",
    "columns_accessed",

    "join_count",
    "filter_count",

    "execution_time_ms",
    "planning_time_ms",
    "estimated_cost",

    "estimated_rows",
    "actual_rows",

    "index_used",
    "index_scan_count",
    "seq_scan_count",
    "bitmap_scan_count",

    "shared_hit_blocks",
    "shared_read_blocks",

    "temp_read_blocks",
    "temp_written_blocks",

    "io_read_blocks",
    "io_write_blocks",

    "plan_node_count",
    "sql"
}


missing_columns = (
    required_columns
    - set(df.columns)
)


if missing_columns:

    raise RuntimeError(
        "Phase 7 dataset is missing required columns:\n"
        + str(
            sorted(
                missing_columns
            )
        )
    )


print(
    "Phase 7 schema validation: PASS"
)


# ============================================================
# 10. KEEP SUCCESSFUL ROWS
# ============================================================

if "status" in df.columns:

    df = df[
        df["status"] == "SUCCESS"
    ].copy()


df.reset_index(
    drop=True,
    inplace=True
)


print(
    "Successful monitoring rows:",
    f"{len(df):,}"
)


# ============================================================
# 11. BASIC CLEANING
# ============================================================

stage("CLEANING PHASE 7 DATA")


numeric_columns = [

    "join_count",
    "filter_count",

    "execution_time_ms",
    "planning_time_ms",
    "wall_time_ms",

    "estimated_cost",
    "startup_cost",

    "estimated_rows",
    "actual_rows",

    "actual_startup_time_ms",
    "actual_total_time_ms",

    "index_scan_count",
    "seq_scan_count",
    "bitmap_scan_count",

    "shared_hit_blocks",
    "shared_read_blocks",
    "shared_dirtied_blocks",
    "shared_written_blocks",

    "local_hit_blocks",
    "local_read_blocks",

    "temp_read_blocks",
    "temp_written_blocks",

    "io_read_blocks",
    "io_write_blocks",

    "plan_node_count"
]


for column in numeric_columns:

    if column in df.columns:

        df[column] = safe_numeric(
            df[column]
        )


if df["index_used"].dtype == object:

    df["index_used"] = (
        df["index_used"]
        .astype(str)
        .str.lower()
        .map(
            {
                "true": 1,
                "false": 0,
                "1": 1,
                "0": 0
            }
        )
        .fillna(0)
        .astype(int)
    )

else:

    df["index_used"] = (
        df["index_used"]
        .fillna(0)
        .astype(int)
    )


print(
    "Cleaning: COMPLETE"
)


# ============================================================
# 12. FEATURE ENGINEERING
# ============================================================

stage("FEATURE ENGINEERING")


# ------------------------------------------------------------
# Structural features
# ------------------------------------------------------------

df["table_count"] = (
    df["tables_accessed"]
    .apply(
        count_csv_items
    )
)

df["column_count"] = (
    df["columns_accessed"]
    .apply(
        count_csv_items
    )
)

df["sql_length"] = (
    df["sql"]
    .apply(
        text_length
    )
)

df["fingerprint_length"] = (
    df["query_fingerprint"]
    .apply(
        text_length
    )
)


# ------------------------------------------------------------
# Cost / row / latency log transforms
# ------------------------------------------------------------

df["log_estimated_cost"] = (
    safe_log1p(
        df["estimated_cost"]
    )
)

df["log_estimated_rows"] = (
    safe_log1p(
        df["estimated_rows"]
    )
)

df["log_actual_rows"] = (
    safe_log1p(
        df["actual_rows"]
    )
)

df["log_execution_time"] = (
    safe_log1p(
        df["execution_time_ms"]
    )
)

df["log_io_read_blocks"] = (
    safe_log1p(
        df["io_read_blocks"]
    )
)

df["log_shared_hits"] = (
    safe_log1p(
        df["shared_hit_blocks"]
    )
)


# ------------------------------------------------------------
# Cardinality estimation error
# ------------------------------------------------------------

df["row_estimation_ratio"] = (

    (
        df["actual_rows"].fillna(0)
        + 1
    )

    /

    (
        df["estimated_rows"].fillna(0)
        + 1
    )
)


df["log_row_estimation_ratio"] = (
    np.log1p(
        df[
            "row_estimation_ratio"
        ].clip(
            lower=0
        )
    )
)


# ------------------------------------------------------------
# Scan features
# ------------------------------------------------------------

df["total_scan_count"] = (

    df["index_scan_count"].fillna(0)

    +

    df["seq_scan_count"].fillna(0)

    +

    df["bitmap_scan_count"].fillna(0)
)


df["seq_scan_ratio"] = (

    df["seq_scan_count"].fillna(0)

    /

    (
        df["total_scan_count"]
        + 1
    )
)


df["index_scan_ratio"] = (

    df["index_scan_count"].fillna(0)

    /

    (
        df["total_scan_count"]
        + 1
    )
)


# ------------------------------------------------------------
# I/O features
# ------------------------------------------------------------

df["total_buffer_activity"] = (

    df["shared_hit_blocks"].fillna(0)

    +

    df["shared_read_blocks"].fillna(0)

    +

    df["temp_read_blocks"].fillna(0)

    +

    df["temp_written_blocks"].fillna(0)
)


df["physical_read_ratio"] = (

    df["io_read_blocks"].fillna(0)

    /

    (
        df["total_buffer_activity"]
        + 1
    )
)


# ------------------------------------------------------------
# Query complexity score
# ------------------------------------------------------------

complexity_map = {

    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3
}


df["complexity_score"] = (

    df["complexity"]
    .astype(str)
    .str.upper()
    .map(
        complexity_map
    )
    .fillna(0)
)


# ------------------------------------------------------------
# Fingerprint frequency
#
# IMPORTANT:
# Use the normalized database fingerprint.
# ------------------------------------------------------------

fingerprint_frequency = (

    df
    .groupby(
        "fingerprint_hash"
    )
    .size()
    .to_dict()
)


df["fingerprint_frequency"] = (

    df["fingerprint_hash"]
    .map(
        fingerprint_frequency
    )
    .fillna(1)
    .astype(int)
)


df["log_fingerprint_frequency"] = (
    np.log1p(
        df[
            "fingerprint_frequency"
        ]
    )
)


# ------------------------------------------------------------
# Query family frequency
# ------------------------------------------------------------

family_frequency = (

    df
    .groupby(
        "query_family"
    )
    .size()
    .to_dict()
)


df["family_frequency"] = (

    df["query_family"]
    .map(
        family_frequency
    )
    .fillna(1)
    .astype(int)
)


# ============================================================
# 13. CREATE PERFORMANCE RISK LABEL
# ============================================================

stage("CREATING PERFORMANCE-RISK LABEL")


# Research-oriented rule:
#
# HIGH:
#   >= 1000 ms
#   OR extreme cost
#   OR extreme I/O
#
# MEDIUM:
#   >= 50 ms
#   OR moderately high cost
#   OR significant sequential scan/I/O
#
# LOW:
#   everything else
#
# The thresholds are explicit and saved in metadata.


HIGH_LATENCY_MS = 1000.0
MEDIUM_LATENCY_MS = 50.0

HIGH_COST = 100000.0
MEDIUM_COST = 10000.0

HIGH_IO_READS = 50000
MEDIUM_IO_READS = 1000


def assign_risk(row):

    execution_time = (
        row["execution_time_ms"]
        if pd.notna(
            row["execution_time_ms"]
        )
        else 0
    )

    cost = (
        row["estimated_cost"]
        if pd.notna(
            row["estimated_cost"]
        )
        else 0
    )

    io_reads = (
        row["io_read_blocks"]
        if pd.notna(
            row["io_read_blocks"]
        )
        else 0
    )

    seq_scans = (
        row["seq_scan_count"]
        if pd.notna(
            row["seq_scan_count"]
        )
        else 0
    )


    if (
        execution_time
        >= HIGH_LATENCY_MS

        or

        cost
        >= HIGH_COST

        or

        io_reads
        >= HIGH_IO_READS
    ):

        return "HIGH"


    if (
        execution_time
        >= MEDIUM_LATENCY_MS

        or

        cost
        >= MEDIUM_COST

        or

        io_reads
        >= MEDIUM_IO_READS

        or

        seq_scans
        >= 2
    ):

        return "MEDIUM"


    return "LOW"


df["performance_risk"] = (
    df.apply(
        assign_risk,
        axis=1
    )
)


risk_map = {

    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2
}


df["performance_risk_code"] = (

    df["performance_risk"]
    .map(
        risk_map
    )
)


print(
    df[
        "performance_risk"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# 14. OPTIMIZATION PRESSURE SCORE
# ============================================================

stage("CALCULATING OPTIMIZATION PRESSURE")


# This is NOT the Phase 9 recommendation.
#
# It is a prioritization signal:
#
# frequent + expensive + slow + I/O heavy + sequential scans
#
# Phase 9 will inspect actual SQL and generate physical-design
# candidates.


def percentile_rank(series):

    return (
        series
        .rank(
            method="average",
            pct=True
        )
        .fillna(0)
    )


latency_rank = percentile_rank(
    df["execution_time_ms"]
)

cost_rank = percentile_rank(
    df["estimated_cost"]
)

io_rank = percentile_rank(
    df["io_read_blocks"]
)

frequency_rank = percentile_rank(
    df["fingerprint_frequency"]
)

seq_rank = percentile_rank(
    df["seq_scan_count"]
)


df["optimization_pressure_score"] = (

    100

    * (

        0.30
        * latency_rank

        +

        0.25
        * cost_rank

        +

        0.20
        * io_rank

        +

        0.15
        * frequency_rank

        +

        0.10
        * seq_rank
    )
)


# ============================================================
# 15. SAVE FEATURE DATASET
# ============================================================

df.to_csv(
    FEATURE_DATASET_FILE,
    index=False
)


print(
    "Feature dataset saved:"
)

print(
    FEATURE_DATASET_FILE
)


# ============================================================
# 16. FINGERPRINT PROFILES
# ============================================================

stage("BUILDING NORMALIZED QUERY PROFILES")


fingerprint_profiles = (

    df

    .groupby(
        "fingerprint_hash",
        dropna=False
    )

    .agg(

        query_fingerprint=(
            "query_fingerprint",
            "first"
        ),

        query_name=(
            "query_name",
            "first"
        ),

        query_family=(
            "query_family",
            "first"
        ),

        dominant_workload_type=(
            "workload_type",
            lambda x:
                x.mode().iloc[0]
                if not x.mode().empty
                else x.iloc[0]
        ),

        dominant_complexity=(
            "complexity",
            lambda x:
                x.mode().iloc[0]
                if not x.mode().empty
                else x.iloc[0]
        ),

        frequency=(
            "fingerprint_hash",
            "size"
        ),

        avg_execution_time_ms=(
            "execution_time_ms",
            "mean"
        ),

        median_execution_time_ms=(
            "execution_time_ms",
            "median"
        ),

        p95_execution_time_ms=(
            "execution_time_ms",
            lambda x:
                x.quantile(0.95)
        ),

        max_execution_time_ms=(
            "execution_time_ms",
            "max"
        ),

        avg_planning_time_ms=(
            "planning_time_ms",
            "mean"
        ),

        avg_estimated_cost=(
            "estimated_cost",
            "mean"
        ),

        avg_estimated_rows=(
            "estimated_rows",
            "mean"
        ),

        avg_actual_rows=(
            "actual_rows",
            "mean"
        ),

        avg_join_count=(
            "join_count",
            "mean"
        ),

        avg_filter_count=(
            "filter_count",
            "mean"
        ),

        avg_table_count=(
            "table_count",
            "mean"
        ),

        avg_column_count=(
            "column_count",
            "mean"
        ),

        index_usage_rate=(
            "index_used",
            "mean"
        ),

        avg_index_scans=(
            "index_scan_count",
            "mean"
        ),

        avg_seq_scans=(
            "seq_scan_count",
            "mean"
        ),

        avg_bitmap_scans=(
            "bitmap_scan_count",
            "mean"
        ),

        avg_io_read_blocks=(
            "io_read_blocks",
            "mean"
        ),

        avg_io_write_blocks=(
            "io_write_blocks",
            "mean"
        ),

        avg_shared_hits=(
            "shared_hit_blocks",
            "mean"
        ),

        avg_temp_reads=(
            "temp_read_blocks",
            "mean"
        ),

        avg_temp_writes=(
            "temp_written_blocks",
            "mean"
        ),

        avg_plan_nodes=(
            "plan_node_count",
            "mean"
        ),

        avg_optimization_pressure=(
            "optimization_pressure_score",
            "mean"
        ),

        high_risk_count=(
            "performance_risk",
            lambda x:
                int(
                    (
                        x == "HIGH"
                    ).sum()
                )
        ),

        medium_risk_count=(
            "performance_risk",
            lambda x:
                int(
                    (
                        x == "MEDIUM"
                    ).sum()
                )
        ),

        low_risk_count=(
            "performance_risk",
            lambda x:
                int(
                    (
                        x == "LOW"
                    ).sum()
                )
        )
    )

    .reset_index()
)


fingerprint_profiles[
    "high_risk_rate"
] = (

    fingerprint_profiles[
        "high_risk_count"
    ]

    /

    fingerprint_profiles[
        "frequency"
    ]
)


fingerprint_profiles[
    "medium_risk_rate"
] = (

    fingerprint_profiles[
        "medium_risk_count"
    ]

    /

    fingerprint_profiles[
        "frequency"
    ]
)


fingerprint_profiles = (

    fingerprint_profiles

    .sort_values(
        "avg_optimization_pressure",
        ascending=False
    )

    .reset_index(
        drop=True
    )
)


fingerprint_profiles.to_csv(
    FINGERPRINT_PROFILE_FILE,
    index=False
)


print(
    "Normalized fingerprints:",
    len(
        fingerprint_profiles
    )
)

print()

print(
    fingerprint_profiles[
        [
            "query_name",
            "query_family",
            "frequency",
            "avg_execution_time_ms",
            "avg_estimated_cost",
            "avg_io_read_blocks",
            "avg_optimization_pressure"
        ]
    ]
    .head(15)
    .to_string(
        index=False
    )
)


# ============================================================
# 17. UNSUPERVISED QUERY CLUSTERING
# ============================================================

stage("UNSUPERVISED QUERY-PATTERN CLUSTERING")


cluster_features = [

    "log_execution_time",
    "log_estimated_cost",
    "log_estimated_rows",
    "log_actual_rows",

    "join_count",
    "filter_count",

    "table_count",
    "column_count",

    "index_used",

    "index_scan_count",
    "seq_scan_count",
    "bitmap_scan_count",

    "log_io_read_blocks",
    "log_shared_hits",

    "plan_node_count",
    "complexity_score",

    "log_fingerprint_frequency"
]


cluster_df = (
    df[
        cluster_features
    ]
    .copy()
)


cluster_df = (
    cluster_df
    .replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )
    .fillna(0)
)


scaler = StandardScaler()


X_cluster = scaler.fit_transform(
    cluster_df
)


# ------------------------------------------------------------
# Find a reasonable K using silhouette score
# ------------------------------------------------------------

silhouette_records = []

best_k = None
best_score = -1


max_k = min(
    8,
    len(df) - 1
)


for k in range(
    2,
    max_k + 1
):

    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=20
    )

    labels = model.fit_predict(
        X_cluster
    )

    score = silhouette_score(
        X_cluster,
        labels
    )

    silhouette_records.append(
        {
            "k": k,
            "silhouette_score": score
        }
    )

    if score > best_score:

        best_score = score
        best_k = k


silhouette_df = pd.DataFrame(
    silhouette_records
)


silhouette_df.to_csv(
    RESULTS_DIR
    / "phase8_silhouette_scores.csv",
    index=False
)


print(
    "Selected cluster count:",
    best_k
)

print(
    "Best silhouette score:",
    f"{best_score:.4f}"
)


# ------------------------------------------------------------
# Final clustering
# ------------------------------------------------------------

kmeans = KMeans(
    n_clusters=best_k,
    random_state=RANDOM_STATE,
    n_init=30
)


df["cluster_id"] = (
    kmeans.fit_predict(
        X_cluster
    )
)


joblib.dump(
    scaler,
    MODEL_DIR
    / "cluster_scaler.joblib"
)


joblib.dump(
    kmeans,
    MODEL_DIR
    / "kmeans_query_clusters.joblib"
)


cluster_output_columns = [

    "workload_type",
    "query_family",
    "query_name",
    "complexity",

    "fingerprint_hash",

    "execution_time_ms",
    "estimated_cost",
    "actual_rows",

    "index_used",
    "seq_scan_count",
    "index_scan_count",

    "io_read_blocks",

    "fingerprint_frequency",
    "performance_risk",

    "optimization_pressure_score",

    "cluster_id"
]


df[
    cluster_output_columns
].to_csv(
    CLUSTER_FILE,
    index=False
)


cluster_summary = (

    df

    .groupby(
        "cluster_id"
    )

    .agg(

        executions=(
            "cluster_id",
            "size"
        ),

        unique_fingerprints=(
            "fingerprint_hash",
            "nunique"
        ),

        avg_execution_time_ms=(
            "execution_time_ms",
            "mean"
        ),

        median_execution_time_ms=(
            "execution_time_ms",
            "median"
        ),

        max_execution_time_ms=(
            "execution_time_ms",
            "max"
        ),

        avg_cost=(
            "estimated_cost",
            "mean"
        ),

        avg_rows=(
            "actual_rows",
            "mean"
        ),

        avg_joins=(
            "join_count",
            "mean"
        ),

        avg_seq_scans=(
            "seq_scan_count",
            "mean"
        ),

        index_usage_rate=(
            "index_used",
            "mean"
        ),

        avg_io_reads=(
            "io_read_blocks",
            "mean"
        ),

        avg_pressure=(
            "optimization_pressure_score",
            "mean"
        )
    )

    .reset_index()
)


cluster_summary.to_csv(
    CLUSTER_SUMMARY_FILE,
    index=False
)


print()

print(
    cluster_summary.to_string(
        index=False
    )
)


# ============================================================
# 18. ML FEATURE DEFINITIONS
# ============================================================

stage("PREPARING SUPERVISED ML FEATURES")


# ------------------------------------------------------------
# IMPORTANT:
#
# We intentionally do NOT use execution_time_ms as an input
# when predicting performance risk, because risk itself uses
# execution time.
#
# This prevents trivial target leakage.
# ------------------------------------------------------------


numeric_predictor_features = [

    "estimated_cost",
    "startup_cost",

    "estimated_rows",

    "join_count",
    "filter_count",

    "table_count",
    "column_count",

    "index_used",

    "index_scan_count",
    "seq_scan_count",
    "bitmap_scan_count",

    "plan_node_count",

    "complexity_score",

    "fingerprint_frequency",
    "family_frequency"
]


categorical_predictor_features = [

    "query_family",
    "complexity"
]


available_numeric = [

    x
    for x in numeric_predictor_features
    if x in df.columns
]


available_categorical = [

    x
    for x in categorical_predictor_features
    if x in df.columns
]


model_features = (

    available_numeric
    +
    available_categorical
)


print(
    "Numeric predictors:",
    available_numeric
)

print()

print(
    "Categorical predictors:",
    available_categorical
)


# ============================================================
# 19. PREPROCESSOR FACTORY
# ============================================================

def make_preprocessor():

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )


    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )


    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                available_numeric
            ),
            (
                "categorical",
                categorical_pipeline,
                available_categorical
            )
        ]
    )


# ============================================================
# 20. WORKLOAD CLASSIFICATION
# ============================================================

stage("MODEL 1 - WORKLOAD CLASSIFICATION")


# ------------------------------------------------------------
# Goal:
# Learn structural differences between:
# doctor / admin / research.
#
# "mixed" is not a true semantic class.
# It is a deployment mixture of those workloads.
#
# Therefore:
# train only on doctor/admin/research.
# ------------------------------------------------------------


classification_df = df[
    df["workload_type"].isin(
        [
            "doctor",
            "admin",
            "research"
        ]
    )
].copy()


class_mapping = {

    "doctor": 0,
    "admin": 1,
    "research": 2
}


inverse_class_mapping = {

    value: key
    for key, value
    in class_mapping.items()
}


classification_df[
    "workload_target"
] = (

    classification_df[
        "workload_type"
    ]
    .map(
        class_mapping
    )
)


X = classification_df[
    model_features
].copy()

y = classification_df[
    "workload_target"
].copy()


# ------------------------------------------------------------
# Group split by fingerprint.
#
# Prevents the exact same normalized query pattern from
# appearing in both train and test when possible.
# ------------------------------------------------------------

groups = classification_df[
    "fingerprint_hash"
]


group_splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=RANDOM_STATE
)


train_idx, test_idx = next(
    group_splitter.split(
        X,
        y,
        groups=groups
    )
)


X_train = X.iloc[
    train_idx
]

X_test = X.iloc[
    test_idx
]

y_train = y.iloc[
    train_idx
]

y_test = y.iloc[
    test_idx
]


print(
    "Training rows:",
    len(X_train)
)

print(
    "Testing rows:",
    len(X_test)
)


# ------------------------------------------------------------
# Random Forest baseline
# ------------------------------------------------------------

rf_classifier = Pipeline(
    steps=[
        (
            "preprocessor",
            make_preprocessor()
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=400,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]
)


rf_classifier.fit(
    X_train,
    y_train
)


rf_pred = rf_classifier.predict(
    X_test
)


rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)


rf_f1 = f1_score(
    y_test,
    rf_pred,
    average="weighted"
)


# ------------------------------------------------------------
# XGBoost
# ------------------------------------------------------------

xgb_classifier = Pipeline(
    steps=[
        (
            "preprocessor",
            make_preprocessor()
        ),
        (
            "model",
            XGBClassifier(
                n_estimators=350,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]
)


xgb_classifier.fit(
    X_train,
    y_train
)


xgb_pred = xgb_classifier.predict(
    X_test
)


xgb_accuracy = accuracy_score(
    y_test,
    xgb_pred
)


xgb_f1 = f1_score(
    y_test,
    xgb_pred,
    average="weighted"
)


classification_results = pd.DataFrame(
    [
        {
            "model":
                "RandomForest",

            "accuracy":
                rf_accuracy,

            "weighted_f1":
                rf_f1
        },

        {
            "model":
                "XGBoost",

            "accuracy":
                xgb_accuracy,

            "weighted_f1":
                xgb_f1
        }
    ]
)


classification_results.to_csv(
    CLASSIFICATION_RESULTS_FILE,
    index=False
)


print()

print(
    classification_results.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# Select best classifier
# ------------------------------------------------------------

if xgb_f1 >= rf_f1:

    best_workload_classifier = (
        xgb_classifier
    )

    best_workload_name = "XGBoost"

    best_workload_pred = (
        xgb_pred
    )

else:

    best_workload_classifier = (
        rf_classifier
    )

    best_workload_name = (
        "RandomForest"
    )

    best_workload_pred = (
        rf_pred
    )


joblib.dump(
    best_workload_classifier,
    MODEL_DIR
    / "workload_classifier.joblib"
)


# ------------------------------------------------------------
# Save detailed classification report
# ------------------------------------------------------------

classification_report_dict = (
    classification_report(
        y_test,
        best_workload_pred,
        labels=[
            0,
            1,
            2
        ],
        target_names=[
            "doctor",
            "admin",
            "research"
        ],
        output_dict=True,
        zero_division=0
    )
)


with open(
    RESULTS_DIR
    / "phase8_workload_classification_report.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        classification_report_dict,
        f,
        indent=4
    )


cm = confusion_matrix(
    y_test,
    best_workload_pred,
    labels=[
        0,
        1,
        2
    ]
)


cm_df = pd.DataFrame(
    cm,
    index=[
        "actual_doctor",
        "actual_admin",
        "actual_research"
    ],
    columns=[
        "pred_doctor",
        "pred_admin",
        "pred_research"
    ]
)


cm_df.to_csv(
    RESULTS_DIR
    / "phase8_workload_confusion_matrix.csv"
)


print()

print(
    "Best workload classifier:",
    best_workload_name
)

print()

print(
    cm_df.to_string()
)


# ============================================================
# 21. PERFORMANCE RISK CLASSIFICATION
# ============================================================

stage("MODEL 2 - PERFORMANCE RISK CLASSIFICATION")


risk_df = df.copy()


X_risk = risk_df[
    model_features
].copy()


y_risk = risk_df[
    "performance_risk_code"
].copy()


# ------------------------------------------------------------
# Use stratified split when possible.
# ------------------------------------------------------------

X_train_risk, X_test_risk, y_train_risk, y_test_risk = (
    train_test_split(
        X_risk,
        y_risk,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_risk
    )
)


# ------------------------------------------------------------
# Random Forest Risk Model
# ------------------------------------------------------------

rf_risk = Pipeline(
    steps=[
        (
            "preprocessor",
            make_preprocessor()
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]
)


rf_risk.fit(
    X_train_risk,
    y_train_risk
)


rf_risk_pred = rf_risk.predict(
    X_test_risk
)


rf_risk_accuracy = accuracy_score(
    y_test_risk,
    rf_risk_pred
)


rf_risk_f1 = f1_score(
    y_test_risk,
    rf_risk_pred,
    average="weighted"
)


# ------------------------------------------------------------
# XGBoost Risk Model
# ------------------------------------------------------------

xgb_risk = Pipeline(
    steps=[
        (
            "preprocessor",
            make_preprocessor()
        ),
        (
            "model",
            XGBClassifier(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]
)


xgb_risk.fit(
    X_train_risk,
    y_train_risk
)


xgb_risk_pred = xgb_risk.predict(
    X_test_risk
)


xgb_risk_accuracy = accuracy_score(
    y_test_risk,
    xgb_risk_pred
)


xgb_risk_f1 = f1_score(
    y_test_risk,
    xgb_risk_pred,
    average="weighted"
)


risk_results = pd.DataFrame(
    [
        {
            "model":
                "RandomForest",

            "accuracy":
                rf_risk_accuracy,

            "weighted_f1":
                rf_risk_f1
        },

        {
            "model":
                "XGBoost",

            "accuracy":
                xgb_risk_accuracy,

            "weighted_f1":
                xgb_risk_f1
        }
    ]
)


risk_results.to_csv(
    RISK_RESULTS_FILE,
    index=False
)


print(
    risk_results.to_string(
        index=False
    )
)


if xgb_risk_f1 >= rf_risk_f1:

    best_risk_model = (
        xgb_risk
    )

    best_risk_name = "XGBoost"

    best_risk_pred = (
        xgb_risk_pred
    )

else:

    best_risk_model = (
        rf_risk
    )

    best_risk_name = (
        "RandomForest"
    )

    best_risk_pred = (
        rf_risk_pred
    )


joblib.dump(
    best_risk_model,
    MODEL_DIR
    / "performance_risk_classifier.joblib"
)


risk_report = classification_report(
    y_test_risk,
    best_risk_pred,
    labels=[
        0,
        1,
        2
    ],
    target_names=[
        "LOW",
        "MEDIUM",
        "HIGH"
    ],
    output_dict=True,
    zero_division=0
)


with open(
    RESULTS_DIR
    / "phase8_risk_classification_report.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        risk_report,
        f,
        indent=4
    )


print()

print(
    "Best risk classifier:",
    best_risk_name
)


# ============================================================
# 22. DEMAND / FREQUENCY LEARNING
# ============================================================

stage("MODEL 3 - QUERY DEMAND / FREQUENCY LEARNING")


# ------------------------------------------------------------
# Unit of analysis:
# normalized fingerprint.
#
# Target:
# observed execution frequency.
#
# This learns which structural query patterns tend to receive
# more demand in the observed workload.
#
# This is NOT a time-series forecast because Phase 7 currently
# contains one monitoring window, not many historical windows.
#
# Once multiple monitoring windows exist, this model can be
# replaced/extended with true temporal forecasting.
# ------------------------------------------------------------


demand_df = fingerprint_profiles.copy()


demand_df[
    "log_avg_execution"
] = np.log1p(
    demand_df[
        "avg_execution_time_ms"
    ].clip(
        lower=0
    )
)


demand_df[
    "log_avg_cost"
] = np.log1p(
    demand_df[
        "avg_estimated_cost"
    ].clip(
        lower=0
    )
)


demand_df[
    "log_avg_rows"
] = np.log1p(
    demand_df[
        "avg_actual_rows"
    ].clip(
        lower=0
    )
)


demand_df[
    "log_avg_io"
] = np.log1p(
    demand_df[
        "avg_io_read_blocks"
    ].clip(
        lower=0
    )
)


demand_numeric_features = [

    "log_avg_execution",
    "log_avg_cost",
    "log_avg_rows",
    "log_avg_io",

    "avg_join_count",
    "avg_filter_count",

    "avg_table_count",
    "avg_column_count",

    "index_usage_rate",

    "avg_index_scans",
    "avg_seq_scans",
    "avg_bitmap_scans",

    "avg_plan_nodes"
]


demand_categorical_features = [

    "query_family",
    "dominant_workload_type",
    "dominant_complexity"
]


demand_features = (

    demand_numeric_features
    +
    demand_categorical_features
)


def make_demand_preprocessor():

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )


    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )


    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                demand_numeric_features
            ),
            (
                "categorical",
                categorical_pipeline,
                demand_categorical_features
            )
        ]
    )


X_demand = demand_df[
    demand_features
].copy()


y_demand = demand_df[
    "frequency"
].astype(float)


# Small fingerprint-level dataset.
# Use deterministic holdout.
X_train_demand, X_test_demand, y_train_demand, y_test_demand = (
    train_test_split(
        X_demand,
        y_demand,
        test_size=0.25,
        random_state=RANDOM_STATE
    )
)


# ------------------------------------------------------------
# Random Forest Demand Regressor
# ------------------------------------------------------------

rf_demand = Pipeline(
    steps=[
        (
            "preprocessor",
            make_demand_preprocessor()
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=500,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]
)


rf_demand.fit(
    X_train_demand,
    y_train_demand
)


rf_demand_pred = rf_demand.predict(
    X_test_demand
)


rf_demand_mae = mean_absolute_error(
    y_test_demand,
    rf_demand_pred
)


rf_demand_rmse = calculate_rmse(
    y_test_demand,
    rf_demand_pred
)


rf_demand_r2 = r2_score(
    y_test_demand,
    rf_demand_pred
)


# ------------------------------------------------------------
# XGBoost Demand Regressor
# ------------------------------------------------------------

xgb_demand = Pipeline(
    steps=[
        (
            "preprocessor",
            make_demand_preprocessor()
        ),
        (
            "model",
            XGBRegressor(
                n_estimators=400,
                max_depth=4,
                learning_rate=0.04,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]
)


xgb_demand.fit(
    X_train_demand,
    y_train_demand
)


xgb_demand_pred = xgb_demand.predict(
    X_test_demand
)


xgb_demand_mae = mean_absolute_error(
    y_test_demand,
    xgb_demand_pred
)


xgb_demand_rmse = calculate_rmse(
    y_test_demand,
    xgb_demand_pred
)


xgb_demand_r2 = r2_score(
    y_test_demand,
    xgb_demand_pred
)


demand_results = pd.DataFrame(
    [
        {
            "model":
                "RandomForest",

            "MAE":
                rf_demand_mae,

            "RMSE":
                rf_demand_rmse,

            "R2":
                rf_demand_r2
        },

        {
            "model":
                "XGBoost",

            "MAE":
                xgb_demand_mae,

            "RMSE":
                xgb_demand_rmse,

            "R2":
                xgb_demand_r2
        }
    ]
)


demand_results.to_csv(
    DEMAND_RESULTS_FILE,
    index=False
)


print(
    demand_results.to_string(
        index=False
    )
)


if xgb_demand_mae <= rf_demand_mae:

    best_demand_model = (
        xgb_demand
    )

    best_demand_name = "XGBoost"

else:

    best_demand_model = (
        rf_demand
    )

    best_demand_name = (
        "RandomForest"
    )


joblib.dump(
    best_demand_model,
    MODEL_DIR
    / "query_demand_regressor.joblib"
)


print()

print(
    "Best demand model:",
    best_demand_name
)


# ============================================================
# 23. PREDICT DEMAND FOR ALL FINGERPRINTS
# ============================================================

demand_df[
    "predicted_frequency"
] = (

    best_demand_model
    .predict(
        demand_df[
            demand_features
        ]
    )
)


demand_df[
    "predicted_frequency"
] = (

    demand_df[
        "predicted_frequency"
    ]
    .clip(
        lower=0
    )
)


# ============================================================
# 24. PHASE 9 OPTIMIZATION PRIORITY
# ============================================================

stage("CREATING PHASE 9 OPTIMIZATION PRIORITY")


# ------------------------------------------------------------
# Combine:
#
# measured performance
# observed frequency
# learned demand
# sequential scan behavior
# I/O
#
# This does NOT create an index.
# It ranks patterns for Phase 9 investigation.
# ------------------------------------------------------------


priority_df = fingerprint_profiles.copy()


predicted_frequency_map = dict(
    zip(
        demand_df[
            "fingerprint_hash"
        ],
        demand_df[
            "predicted_frequency"
        ]
    )
)


priority_df[
    "predicted_frequency"
] = (

    priority_df[
        "fingerprint_hash"
    ]
    .map(
        predicted_frequency_map
    )
)


priority_df[
    "latency_rank"
] = percentile_rank(
    priority_df[
        "avg_execution_time_ms"
    ]
)


priority_df[
    "cost_rank"
] = percentile_rank(
    priority_df[
        "avg_estimated_cost"
    ]
)


priority_df[
    "io_rank"
] = percentile_rank(
    priority_df[
        "avg_io_read_blocks"
    ]
)


priority_df[
    "observed_frequency_rank"
] = percentile_rank(
    priority_df[
        "frequency"
    ]
)


priority_df[
    "predicted_frequency_rank"
] = percentile_rank(
    priority_df[
        "predicted_frequency"
    ]
)


priority_df[
    "seq_scan_rank"
] = percentile_rank(
    priority_df[
        "avg_seq_scans"
    ]
)


priority_df[
    "optimization_priority_score"
] = (

    100

    * (

        0.30
        * priority_df[
            "latency_rank"
        ]

        +

        0.20
        * priority_df[
            "cost_rank"
        ]

        +

        0.15
        * priority_df[
            "io_rank"
        ]

        +

        0.15
        * priority_df[
            "observed_frequency_rank"
        ]

        +

        0.10
        * priority_df[
            "predicted_frequency_rank"
        ]

        +

        0.10
        * priority_df[
            "seq_scan_rank"
        ]
    )
)


def priority_label(score):

    if score >= 75:
        return "CRITICAL"

    if score >= 55:
        return "HIGH"

    if score >= 35:
        return "MEDIUM"

    return "LOW"


priority_df[
    "optimization_priority"
] = (

    priority_df[
        "optimization_priority_score"
    ]
    .apply(
        priority_label
    )
)


priority_df = (

    priority_df

    .sort_values(
        "optimization_priority_score",
        ascending=False
    )

    .reset_index(
        drop=True
    )
)


priority_df[
    "priority_rank"
] = (
    np.arange(
        1,
        len(priority_df) + 1
    )
)


priority_df.to_csv(
    OPTIMIZATION_PRIORITY_FILE,
    index=False
)


print(
    priority_df[
        [
            "priority_rank",
            "query_name",
            "query_family",
            "frequency",
            "predicted_frequency",
            "avg_execution_time_ms",
            "avg_estimated_cost",
            "avg_io_read_blocks",
            "avg_seq_scans",
            "optimization_priority_score",
            "optimization_priority"
        ]
    ]
    .head(15)
    .to_string(
        index=False
    )
)


# ============================================================
# 25. FEATURE IMPORTANCE
# ============================================================

stage("EXTRACTING FEATURE IMPORTANCE")


# ------------------------------------------------------------
# Feature importance from best risk model.
# ------------------------------------------------------------

try:

    preprocessor = (
        best_risk_model
        .named_steps[
            "preprocessor"
        ]
    )

    fitted_model = (
        best_risk_model
        .named_steps[
            "model"
        ]
    )

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )


    if hasattr(
        fitted_model,
        "feature_importances_"
    ):

        importances = (
            fitted_model
            .feature_importances_
        )


        feature_importance_df = pd.DataFrame(
            {
                "feature":
                    feature_names,

                "importance":
                    importances
            }
        )


        feature_importance_df = (

            feature_importance_df

            .sort_values(
                "importance",
                ascending=False
            )

            .reset_index(
                drop=True
            )
        )


        feature_importance_df.to_csv(
            FEATURE_IMPORTANCE_FILE,
            index=False
        )


        print(
            feature_importance_df
            .head(20)
            .to_string(
                index=False
            )
        )

    else:

        feature_importance_df = (
            pd.DataFrame()
        )


except Exception as exc:

    print(
        "Feature importance extraction skipped:"
    )

    print(
        exc
    )

    feature_importance_df = (
        pd.DataFrame()
    )


# ============================================================
# 26. VISUALIZATION 1
# RISK DISTRIBUTION
# ============================================================

stage("CREATING PHASE 8 VISUALIZATIONS")


risk_counts = (

    df[
        "performance_risk"
    ]
    .value_counts()
    .reindex(
        [
            "LOW",
            "MEDIUM",
            "HIGH"
        ],
        fill_value=0
    )
)


plt.figure(
    figsize=(8, 5)
)

risk_counts.plot(
    kind="bar"
)

plt.title(
    "Phase 8 Performance Risk Distribution"
)

plt.xlabel(
    "Risk Class"
)

plt.ylabel(
    "Number of Query Executions"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR
    / "phase8_risk_distribution.png",
    dpi=160
)

plt.close()


# ============================================================
# 27. VISUALIZATION 2
# CLUSTER SIZE
# ============================================================

cluster_counts = (

    df[
        "cluster_id"
    ]
    .value_counts()
    .sort_index()
)


plt.figure(
    figsize=(8, 5)
)

cluster_counts.plot(
    kind="bar"
)

plt.title(
    "Phase 8 Learned Query Clusters"
)

plt.xlabel(
    "Cluster ID"
)

plt.ylabel(
    "Number of Query Executions"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR
    / "phase8_cluster_distribution.png",
    dpi=160
)

plt.close()


# ============================================================
# 28. VISUALIZATION 3
# TOP OPTIMIZATION PRIORITIES
# ============================================================

top_priority = (
    priority_df
    .head(15)
    .copy()
)


plt.figure(
    figsize=(11, 7)
)

plt.barh(
    top_priority[
        "query_name"
    ],
    top_priority[
        "optimization_priority_score"
    ]
)

plt.xlabel(
    "Optimization Priority Score"
)

plt.ylabel(
    "Query Pattern"
)

plt.title(
    "Top Phase 9 Optimization Candidates"
)

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR
    / "phase8_top_optimization_priorities.png",
    dpi=160
)

plt.close()


# ============================================================
# 29. VISUALIZATION 4
# LATENCY VS COST
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.scatter(
    np.log1p(
        df[
            "estimated_cost"
        ].clip(
            lower=0
        )
    ),
    np.log1p(
        df[
            "execution_time_ms"
        ].clip(
            lower=0
        )
    ),
    alpha=0.6
)

plt.xlabel(
    "log(1 + Estimated Cost)"
)

plt.ylabel(
    "log(1 + Execution Time ms)"
)

plt.title(
    "PostgreSQL Estimated Cost vs Measured Execution Time"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR
    / "phase8_cost_vs_latency.png",
    dpi=160
)

plt.close()


# ============================================================
# 30. SAVE MODEL CONFIGURATION
# ============================================================

model_config = {

    "random_state":
        RANDOM_STATE,

    "workload_class_mapping":
        class_mapping,

    "risk_mapping":
        risk_map,

    "model_features":
        model_features,

    "numeric_features":
        available_numeric,

    "categorical_features":
        available_categorical,

    "demand_features":
        demand_features,

    "selected_cluster_count":
        int(best_k),

    "silhouette_score":
        float(best_score),

    "risk_thresholds": {

        "high_latency_ms":
            HIGH_LATENCY_MS,

        "medium_latency_ms":
            MEDIUM_LATENCY_MS,

        "high_cost":
            HIGH_COST,

        "medium_cost":
            MEDIUM_COST,

        "high_io_reads":
            HIGH_IO_READS,

        "medium_io_reads":
            MEDIUM_IO_READS
    }
}


with open(
    MODEL_DIR
    / "phase8_model_config.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        model_config,
        f,
        indent=4
    )


# ============================================================
# 31. METADATA
# ============================================================

stage("SAVING PHASE 8 METADATA")


metadata = {

    "phase":
        8,

    "phase_name":
        "AI Workload Learner",

    "generated_at":
        datetime.now().isoformat(),

    "source_file":
        str(INPUT_FILE),

    "source_rows":
        int(
            len(df)
        ),

    "unique_raw_queries":
        int(
            df[
                "query_hash"
            ].nunique()
        ),

    "unique_normalized_fingerprints":
        int(
            df[
                "fingerprint_hash"
            ].nunique()
        ),

    "query_families":
        int(
            df[
                "query_family"
            ].nunique()
        ),

    "workload_types":
        sorted(
            df[
                "workload_type"
            ]
            .dropna()
            .unique()
            .tolist()
        ),

    "selected_clusters":
        int(
            best_k
        ),

    "silhouette_score":
        float(
            best_score
        ),

    "best_workload_classifier":
        best_workload_name,

    "best_workload_classifier_f1":
        float(
            max(
                rf_f1,
                xgb_f1
            )
        ),

    "best_risk_classifier":
        best_risk_name,

    "best_risk_classifier_f1":
        float(
            max(
                rf_risk_f1,
                xgb_risk_f1
            )
        ),

    "best_demand_model":
        best_demand_name,

    "demand_model_note":
        (
            "Frequency learning is based on the "
            "single Phase 7 monitoring window. "
            "It is not claimed as a true temporal "
            "forecast. True future-demand forecasting "
            "requires multiple time-ordered monitoring "
            "windows."
        ),

    "phase9_note":
        (
            "Optimization priority scores identify "
            "query patterns for physical-design "
            "investigation. No indexes, materialized "
            "views, partitions, or schema changes "
            "are applied in Phase 8."
        )
}


with open(
    METADATA_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )


# ============================================================
# 32. FINAL REPORT
# ============================================================

stage("GENERATING PHASE 8 REPORT")


top_candidate = (
    priority_df.iloc[0]
    if not priority_df.empty
    else None
)


if top_candidate is not None:

    top_candidate_text = f"""
Top optimization candidate:
{top_candidate['query_name']}

Query family:
{top_candidate['query_family']}

Observed frequency:
{int(top_candidate['frequency'])}

Average execution time:
{top_candidate['avg_execution_time_ms']:.3f} ms

Average estimated cost:
{top_candidate['avg_estimated_cost']:.3f}

Average I/O reads:
{top_candidate['avg_io_read_blocks']:.3f}

Optimization priority score:
{top_candidate['optimization_priority_score']:.2f}

Priority:
{top_candidate['optimization_priority']}
"""

else:

    top_candidate_text = (
        "No optimization candidate available."
    )


report = f"""
PHASE 8 - AI WORKLOAD LEARNER
======================================================================

Project:
Adaptive Healthcare Knowledge Database


SOURCE DATA
----------------------------------------------------------------------

Source:
{INPUT_FILE}

Monitored executions:
{len(df):,}

Unique concrete queries:
{df['query_hash'].nunique():,}

Unique normalized fingerprints:
{df['fingerprint_hash'].nunique():,}

Query families:
{df['query_family'].nunique():,}


FEATURE ENGINEERING
----------------------------------------------------------------------

Features include:

- PostgreSQL estimated cost
- estimated rows
- actual rows
- join count
- filter count
- table count
- column count
- index usage
- index scans
- sequential scans
- bitmap scans
- plan-node count
- buffer activity
- physical I/O
- query complexity
- normalized fingerprint frequency
- query-family frequency
- cardinality estimation behavior


UNSUPERVISED WORKLOAD LEARNING
----------------------------------------------------------------------

Algorithm:
KMeans

Selected number of clusters:
{best_k}

Silhouette score:
{best_score:.4f}


WORKLOAD CLASSIFICATION
----------------------------------------------------------------------

Classes:
Doctor
Admin
Research

Mixed workload is excluded as a semantic training class because
it is a deployment mixture of doctor/admin/research queries.

Random Forest weighted F1:
{rf_f1:.4f}

XGBoost weighted F1:
{xgb_f1:.4f}

Selected model:
{best_workload_name}


PERFORMANCE-RISK CLASSIFICATION
----------------------------------------------------------------------

Classes:
LOW
MEDIUM
HIGH

Random Forest weighted F1:
{rf_risk_f1:.4f}

XGBoost weighted F1:
{xgb_risk_f1:.4f}

Selected model:
{best_risk_name}


DEMAND LEARNING
----------------------------------------------------------------------

Random Forest MAE:
{rf_demand_mae:.4f}

Random Forest RMSE:
{rf_demand_rmse:.4f}

Random Forest R2:
{rf_demand_r2:.4f}

XGBoost MAE:
{xgb_demand_mae:.4f}

XGBoost RMSE:
{xgb_demand_rmse:.4f}

XGBoost R2:
{xgb_demand_r2:.4f}

Selected demand model:
{best_demand_name}

IMPORTANT:
The current dataset contains one monitoring window.

Therefore this phase learns observed workload demand/frequency
patterns. It does NOT claim a validated time-series forecast.

True future-demand forecasting will require multiple monitoring
windows collected over time.


OPTIMIZATION PRIORITY
----------------------------------------------------------------------

{top_candidate_text}


RESEARCH SAFETY
----------------------------------------------------------------------

Phase 8 performs analysis and machine learning only.

It does NOT:

- create indexes
- drop indexes
- modify tables
- create partitions
- create materialized views
- change PostgreSQL configuration

All physical database optimization belongs to Phase 9.


PHASE 9 INPUT
----------------------------------------------------------------------

Phase 9 receives:

1. Normalized query profiles
2. Learned workload clusters
3. Performance-risk predictions
4. Observed query frequency
5. Learned demand estimates
6. Optimization-priority ranking
7. Query cost and I/O behavior
8. Sequential/index scan behavior

Phase 9 will use these signals together with actual SQL structure
to generate candidate physical designs and evaluate them before
anything is applied.


STATUS
----------------------------------------------------------------------

PHASE 8 AI WORKLOAD LEARNING COMPLETED
"""


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        report.strip()
    )


# ============================================================
# 33. FINAL OUTPUT
# ============================================================

stage("PHASE 8 FINAL RESULT")


print(
    "Monitoring rows              :",
    f"{len(df):,}"
)

print(
    "Unique concrete queries      :",
    f"{df['query_hash'].nunique():,}"
)

print(
    "Unique normalized patterns   :",
    f"{df['fingerprint_hash'].nunique():,}"
)

print(
    "Query families               :",
    f"{df['query_family'].nunique():,}"
)

print(
    "Selected query clusters      :",
    best_k
)

print(
    "Cluster silhouette score     :",
    f"{best_score:.4f}"
)

print(
    "Best workload classifier     :",
    best_workload_name
)

print(
    "Workload classifier F1       :",
    f"{max(rf_f1, xgb_f1):.4f}"
)

print(
    "Best risk classifier         :",
    best_risk_name
)

print(
    "Risk classifier F1           :",
    f"{max(rf_risk_f1, xgb_risk_f1):.4f}"
)

print(
    "Best demand model            :",
    best_demand_name
)


if top_candidate is not None:

    print(
        "Top optimization candidate  :",
        top_candidate[
            "query_name"
        ]
    )

    print(
        "Top candidate family        :",
        top_candidate[
            "query_family"
        ]
    )

    print(
        "Top candidate score         :",
        f"{top_candidate['optimization_priority_score']:.2f}"
    )


print()

print(
    "Results directory:"
)

print(
    RESULTS_DIR
)


print()

print(
    "Models directory:"
)

print(
    MODEL_DIR
)


print()

print(
    "Generated result files:"
)


for path in sorted(
    RESULTS_DIR.iterdir()
):

    print(
        " -",
        path.name
    )


print()

print(
    "Generated model files:"
)


for path in sorted(
    MODEL_DIR.iterdir()
):

    print(
        " -",
        path.name
    )


print()

print("=" * 100)
print("PHASE 8 STATUS: COMPLETE")
print("=" * 100)

print(
    "Phase 7 dataset loading          PASS"
)

print(
    "Feature engineering              COMPLETE"
)

print(
    "Query fingerprint profiling      COMPLETE"
)

print(
    "Query clustering                 COMPLETE"
)

print(
    "Workload classification          COMPLETE"
)

print(
    "Performance-risk learning        COMPLETE"
)

print(
    "Demand/frequency learning        COMPLETE"
)

print(
    "Optimization prioritization      COMPLETE"
)

print(
    "Model persistence                COMPLETE"
)

print(
    "Research outputs                 COMPLETE"
)

print()

print(
    "NEXT:"
)

print(
    "PHASE 9 -> SELF-OPTIMIZING DATABASE"
)