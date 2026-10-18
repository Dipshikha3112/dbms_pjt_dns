
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg
from psycopg import sql

from scipy.sparse import hstack, csr_matrix
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    RandomForestRegressor,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    silhouette_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = (
    ROOT / "results"
    / "phase7_monitoring"
    / "phase7_query_monitoring.csv"
)
RESULTS = ROOT / "results" / "phase8_9_v2"
RESULTS.mkdir(parents=True, exist_ok=True)

SEED = 42
BENCHMARK_DB = "adaptive_healthcare_phase9_benchmark"

NUMERIC_FEATURES = [
    "sql_length",
    "word_count",
    "join_count",
    "filter_count",
    "where_count",
    "group_count",
    "order_count",
    "subquery_count",
    "aggregate_count",
    "distinct_count",
    "table_count",
    "predicate_count",
    "limit_count",
    "estimated_cost",
    "startup_cost",
    "estimated_rows",
    "actual_rows",
    "planning_time_ms",
    "execution_time_ms",
    "index_scan_count",
    "seq_scan_count",
    "bitmap_scan_count",
    "shared_hit_blocks",
    "shared_read_blocks",
    "shared_dirtied_blocks",
    "shared_written_blocks",
    "temp_read_blocks",
    "temp_written_blocks",
    "plan_node_count",
]

# These features are useful for explaining monitored workloads.
# Runtime measurements may not exist for an incoming unexecuted query.
STRUCTURAL_FEATURES = [
    "sql_length",
    "word_count",
    "join_count",
    "filter_count",
    "where_count",
    "group_count",
    "order_count",
    "subquery_count",
    "aggregate_count",
    "distinct_count",
    "table_count",
    "predicate_count",
    "limit_count",
]

def now():
    return datetime.now(timezone.utc).isoformat()

def write_json(filename, data):
    path = RESULTS / filename
    path.write_text(
        json.dumps(data, indent=2, default=str),
        encoding="utf-8",
    )
    return path

def clean_sql(text):
    text = str(text or "")
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"--[^\n]*", " ", text)
    text = re.sub(r"'(?:''|[^'])*'", "?", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\b", "?", text)
    return re.sub(r"\s+", " ", text).strip().lower()

def sql_hash(text):
    return hashlib.sha256(
        clean_sql(text).encode("utf-8")
    ).hexdigest()[:24]

def load_data(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, low_memory=False)

    if "sql" not in df.columns:
        raise ValueError(
            "Expected Phase 7 column 'sql'. "
            f"Available columns: {list(df.columns)}"
        )

    if "workload_type" not in df.columns:
        raise ValueError("Missing workload_type column.")

    df = df.copy()
    df["sql"] = df["sql"].fillna("").astype(str)
    df = df[df["sql"].str.strip().ne("")].copy()

    if "fingerprint_hash" in df.columns:
        df["fingerprint"] = (
            df["fingerprint_hash"]
            .fillna("")
            .astype(str)
        )
        missing = df["fingerprint"].eq("")
        df.loc[missing, "fingerprint"] = (
            df.loc[missing, "sql"].map(sql_hash)
        )
    else:
        df["fingerprint"] = df["sql"].map(sql_hash)

    if "monitored_at" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["monitored_at"],
            errors="coerce",
            utc=True,
        )
    else:
        df["timestamp"] = pd.NaT

    df["workload_type"] = (
        df["workload_type"]
        .fillna("unknown")
        .astype(str)
        .str.lower()
    )

    if "query_family" not in df.columns:
        df["query_family"] = "unknown"

    if "tables_accessed" not in df.columns:
        df["tables_accessed"] = ""

    if "plan_node_types" not in df.columns:
        df["plan_node_types"] = ""

    for col in [
        "join_count",
        "filter_count",
        "estimated_cost",
        "startup_cost",
        "estimated_rows",
        "actual_rows",
        "planning_time_ms",
        "execution_time_ms",
        "index_scan_count",
        "seq_scan_count",
        "bitmap_scan_count",
        "shared_hit_blocks",
        "shared_read_blocks",
        "shared_dirtied_blocks",
        "shared_written_blocks",
        "temp_read_blocks",
        "temp_written_blocks",
        "plan_node_count",
    ]:
        if col not in df.columns:
            df[col] = np.nan

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    return df.reset_index(drop=True)

def add_features(df):
    df = df.copy()
    text = df["sql"].str.lower()

    df["sql_length"] = text.str.len()
    df["word_count"] = text.str.split().str.len()

    df["where_count"] = text.str.count(r"\bwhere\b")
    df["group_count"] = text.str.count(
        r"\bgroup\s+by\b"
    )
    df["order_count"] = text.str.count(
        r"\border\s+by\b"
    )
    df["subquery_count"] = text.str.count(
        r"\(\s*select\b"
    )
    df["aggregate_count"] = text.str.count(
        r"\b(?:count|sum|avg|min|max)\s*\("
    )
    df["distinct_count"] = text.str.count(
        r"\bdistinct\b"
    )
    df["table_count"] = (
        text.str.count(r"\bfrom\b")
        + text.str.count(r"\bjoin\b")
    )
    df["predicate_count"] = text.str.count(
        r"(?<![<>!])=(?!=)|>=|<=|<>|!=|\blike\b|\bin\b"
    )
    df["limit_count"] = text.str.count(r"\blimit\b")

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df["normalized_sql"] = df["sql"].map(clean_sql)

    # Table names and plan operators are explanatory features.
    # No query_family, query_name or expected_use labels
    # are inserted into the model's text input.
    df["text_features"] = (
        df["normalized_sql"].fillna("")
        + " "
        + df["tables_accessed"].fillna("").astype(str)
        .str.replace(",", " ", regex=False)
    )

    return df

def train_phase8(df):
    valid = df[
        df["workload_type"].isin(
            ["doctor", "admin", "research", "mixed"]
        )
    ].copy()

    groups = valid["fingerprint"].astype(str)

    if groups.nunique() < 8:
        return {
            "status": "INSUFFICIENT_FINGERPRINTS",
            "unique_fingerprints": int(groups.nunique()),
        }

    encoder = LabelEncoder()
    y = encoder.fit_transform(valid["workload_type"])

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=SEED,
    )

    train_idx, test_idx = next(
        splitter.split(valid, y, groups=groups)
    )

    train = valid.iloc[train_idx]
    test = valid.iloc[test_idx]

    vectorizer = TfidfVectorizer(
        max_features=4000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        token_pattern=r"(?u)\b[\w]+\b",
    )

    X_train_text = vectorizer.fit_transform(
        train["text_features"]
    )
    X_test_text = vectorizer.transform(
        test["text_features"]
    )

    imputer = SimpleImputer(
        strategy="constant",
        fill_value=0,
    )
    scaler = StandardScaler()

    X_train_numeric = imputer.fit_transform(
        train[STRUCTURAL_FEATURES]
    )
    X_test_numeric = imputer.transform(
        test[STRUCTURAL_FEATURES]
    )

    X_train_numeric = scaler.fit_transform(
        X_train_numeric
    )
    X_test_numeric = scaler.transform(
        X_test_numeric
    )

    X_train = hstack(
        [
            X_train_text,
            csr_matrix(X_train_numeric),
        ],
        format="csr",
    )

    X_test = hstack(
        [
            X_test_text,
            csr_matrix(X_test_numeric),
        ],
        format="csr",
    )

    y_train = y[train_idx]
    y_test = y[test_idx]

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=350,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        ),
        "hist_gradient_boosting":
            HistGradientBoostingClassifier(
                max_iter=200,
                max_leaf_nodes=15,
                learning_rate=0.05,
                random_state=SEED,
            ),
    }

    results = {}
    trained = {}

    majority = Counter(y_train).most_common(1)[0][0]
    majority_predictions = np.full(
        len(y_test),
        majority,
    )

    results["majority_baseline"] = {
        "accuracy": float(
            accuracy_score(
                y_test,
                majority_predictions,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_test,
                majority_predictions,
                average="macro",
                zero_division=0,
            )
        ),
    }

    for name, model in models.items():
        if name == "hist_gradient_boosting":
            model.fit(
                X_train.toarray(),
                y_train,
            )
            pred = model.predict(
                X_test.toarray()
            )
        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

        results[name] = {
            "accuracy": float(
                accuracy_score(y_test, pred)
            ),
            "macro_f1": float(
                f1_score(
                    y_test,
                    pred,
                    average="macro",
                    zero_division=0,
                )
            ),
            "classification_report":
                classification_report(
                    y_test,
                    pred,
                    labels=np.arange(
                        len(encoder.classes_)
                    ),
                    target_names=list(
                        encoder.classes_
                    ),
                    output_dict=True,
                    zero_division=0,
                ),
        }

        trained[name] = model

    winner = max(
        trained,
        key=lambda name: (
            results[name]["macro_f1"],
            results[name]["accuracy"],
        ),
    )

    # Refit the winning model on the complete dataset.
    final_vectorizer = TfidfVectorizer(
        max_features=4000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        token_pattern=r"(?u)\b[\w]+\b",
    )

    all_text = final_vectorizer.fit_transform(
        valid["text_features"]
    )

    final_imputer = SimpleImputer(
        strategy="constant",
        fill_value=0,
    )
    final_scaler = StandardScaler()

    all_numeric = final_imputer.fit_transform(
        valid[STRUCTURAL_FEATURES]
    )
    all_numeric = final_scaler.fit_transform(
        all_numeric
    )

    all_features = hstack(
        [
            all_text,
            csr_matrix(all_numeric),
        ],
        format="csr",
    )

    if winner == "random_forest":
        final_model = RandomForestClassifier(
            n_estimators=350,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        )
        final_model.fit(all_features, y)
    else:
        final_model = HistGradientBoostingClassifier(
            max_iter=200,
            max_leaf_nodes=15,
            learning_rate=0.05,
            random_state=SEED,
        )
        final_model.fit(
            all_features.toarray(),
            y,
        )

    joblib.dump(
        {
            "model": final_model,
            "winner": winner,
            "vectorizer": final_vectorizer,
            "imputer": final_imputer,
            "scaler": final_scaler,
            "encoder": encoder,
            "numeric_features": STRUCTURAL_FEATURES,
            "trained_at": now(),
        },
        RESULTS / "phase8_workload_model.joblib",
    )

    report = {
        "status": "TRAINED",
        "winner": winner,
        "training_rows": int(len(train)),
        "testing_rows": int(len(test)),
        "unique_fingerprints": int(
            groups.nunique()
        ),
        "holdout_strategy":
            "Fingerprint-grouped train/test split",
        "models": results,
        "important_note": (
            "Mixed workloads reuse many query families. "
            "The same SQL structure can have different "
            "workload labels. SQL alone may therefore "
            "be insufficient to infer the requesting actor."
        ),
    }

    write_json(
        "phase8_classifier_report.json",
        report,
    )

    return report

def cluster_phase8(df):
    numeric = df[NUMERIC_FEATURES].copy()

    imputer = SimpleImputer(
        strategy="constant",
        fill_value=0,
    )
    scaler = StandardScaler()

    X = imputer.fit_transform(numeric)
    X = scaler.fit_transform(X)

    k = min(
        6,
        max(2, df["query_family"].nunique() // 5),
        len(df) - 1,
    )

    model = MiniBatchKMeans(
        n_clusters=k,
        random_state=SEED,
        batch_size=min(256, len(df)),
        n_init=10,
    )

    labels = model.fit_predict(X)

    df = df.copy()
    df["cluster"] = labels

    try:
        score = float(
            silhouette_score(
                X,
                labels,
                sample_size=min(950, len(df)),
                random_state=SEED,
            )
        )
    except Exception:
        score = None

    joblib.dump(
        {
            "model": model,
            "imputer": imputer,
            "scaler": scaler,
            "features": NUMERIC_FEATURES,
        },
        RESULTS / "phase8_cluster_model.joblib",
    )

    df.to_csv(
        RESULTS / "phase8_clustered_workload.csv",
        index=False,
    )

    report = {
        "status": "CLUSTERED",
        "clusters": int(k),
        "silhouette_score": score,
        "cluster_distribution": {
            str(key): int(value)
            for key, value in
            df["cluster"].value_counts().items()
        },
    }

    write_json(
        "phase8_clusters.json",
        report,
    )

    return report, df

def forecast_phase8(df):
    valid = df.dropna(
        subset=["timestamp"]
    ).copy()

    if valid.empty:
        result = {
            "status": "NO_TIMESTAMPS",
        }
        write_json("phase8_forecast.json", result)
        return result

    start = valid["timestamp"].min()
    end = valid["timestamp"].max()

    duration_seconds = (
        end - start
    ).total_seconds()

    hourly_buckets = (
        valid["timestamp"]
        .dt.floor("h")
        .nunique()
    )

    counts = (
        valid["workload_type"]
        .value_counts()
        .to_dict()
    )

    result = {
        "status": "SHORT_OBSERVATION_WINDOW",
        "observed_records": int(len(valid)),
        "start": str(start),
        "end": str(end),
        "duration_seconds": round(
            duration_seconds,
            3,
        ),
        "distinct_hourly_buckets": int(
            hourly_buckets
        ),
        "observed_workload_counts": {
            str(k): int(v)
            for k, v in counts.items()
        },
        "forecast_validated": False,
        "reason": (
            "The monitoring window is too short to "
            "validate hourly or daily future demand. "
            "No artificial timestamps are generated."
        ),
    }

    # Short-window forecasting is allowed only as
    # exploratory next-minute estimation.
    if duration_seconds >= 300:
        valid["minute"] = (
            valid["timestamp"].dt.floor("min")
        )

        minute_counts = (
            valid.groupby(
                ["minute", "workload_type"]
            )
            .size()
            .unstack(fill_value=0)
            .sort_index()
        )

        full_index = pd.date_range(
            minute_counts.index.min(),
            minute_counts.index.max(),
            freq="min",
            tz="UTC",
        )

        minute_counts = minute_counts.reindex(
            full_index,
            fill_value=0,
        )

        predictions = {}

        for column in minute_counts.columns:
            series = minute_counts[column].astype(float)

            if len(series) >= 5:
                prediction = float(
                    series.tail(3).mean()
                )

                predictions[str(column)] = round(
                    prediction,
                    2,
                )

        result["exploratory_next_minute"] = predictions
        result["exploratory_method"] = (
            "Three-minute moving average"
        )
        result["exploratory_warning"] = (
            "Not a validated forecast of future "
            "clinical or administrative demand."
        )

        minute_counts.to_csv(
            RESULTS / "phase8_minute_counts.csv"
        )

    write_json(
        "phase8_forecast.json",
        result,
    )

    return result

def safe_select(query):
    text = str(query or "").strip()

    if not re.match(
        r"(?is)^select\b",
        text,
    ):
        return False

    if ";" in text.rstrip(";"):
        return False

    forbidden = (
        r"\b(insert|update|delete|merge|drop|alter|"
        r"create|truncate|copy|vacuum|refresh|grant|"
        r"revoke|call|execute|pg_sleep|dblink|"
        r"pg_read_file|pg_write_file)\b"
    )

    if re.search(forbidden, text, flags=re.I):
        return False

    if re.search(
        r"\bfor\s+(update|share)\b",
        text,
        flags=re.I,
    ):
        return False

    return True

def connect_benchmark(dsn):
    conn = psycopg.connect(
        dsn,
        connect_timeout=10,
        options=(
            "-c statement_timeout=120000 "
            "-c lock_timeout=3000"
        ),
    )

    with conn.cursor() as cur:
        cur.execute(
            "SELECT current_database()"
        )
        database = cur.fetchone()[0]

    conn.rollback()

    if database != BENCHMARK_DB:
        conn.close()
        raise RuntimeError(
            "Safety stop: connected to "
            f"{database}, not {BENCHMARK_DB}."
        )

    return conn

def get_metadata(conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                n.nspname,
                c.relname,
                GREATEST(c.reltuples, 0)::bigint
            FROM pg_class c
            JOIN pg_namespace n
              ON n.oid = c.relnamespace
            WHERE c.relkind IN ('r','p')
              AND n.nspname = 'public'
        """)

        tables = {
            name: {
                "schema": schema,
                "rows": int(rows),
            }
            for schema, name, rows in cur.fetchall()
        }

        cur.execute("""
            SELECT
                table_name,
                column_name,
                data_type
            FROM information_schema.columns
            WHERE table_schema = 'public'
        """)

        columns = defaultdict(dict)

        for table, column, dtype in cur.fetchall():
            columns[table][column] = dtype

        cur.execute("""
            SELECT
                tablename,
                indexname,
                indexdef
            FROM pg_indexes
            WHERE schemaname = 'public'
        """)

        indexes = defaultdict(list)

        for table, name, definition in cur.fetchall():
            indexes[table].append({
                "name": name,
                "definition": definition,
            })

    conn.rollback()
    return tables, columns, indexes

def extract_aliases(query):
    pattern = (
        r'\b(from|join)\s+'
        r'([a-z_][a-z0-9_]*)'
        r'(?:\s+(?:as\s+)?'
        r'([a-z_][a-z0-9_]*))?'
    )

    reserved = {
        "where", "join", "left", "right",
        "inner", "outer", "full", "on",
        "group", "order", "limit", "having",
    }

    mapping = {}

    for _, table, alias in re.findall(
        pattern,
        query.lower(),
    ):
        mapping[table] = table

        if alias and alias not in reserved:
            mapping[alias] = table

    return mapping

def extract_index_columns(query, aliases, columns):
    text = query.lower()
    found = defaultdict(list)

    # Equality filters and join predicates.
    pattern = (
        r'\b([a-z_][a-z0-9_]*)\.'
        r'([a-z_][a-z0-9_]*)\s*'
        r'(=|>=|<=|>|<|\blike\b|\bin\b)'
    )

    for alias, column, operator in re.findall(
        pattern,
        text,
    ):
        table = aliases.get(alias)

        if (
            table
            and table in columns
            and column in columns[table]
            and column not in found[table]
        ):
            found[table].append(column)

    # Include ORDER BY columns for potential
    # composite indexes.
    order_match = re.search(
        r"\border\s+by\s+(.+?)(?:\blimit\b|$)",
        text,
        flags=re.S,
    )

    if order_match:
        order_text = order_match.group(1)

        for alias, column in re.findall(
            r"\b([a-z_][a-z0-9_]*)\."
            r"([a-z_][a-z0-9_]*)",
            order_text,
        ):
            table = aliases.get(alias)

            if (
                table
                and table in columns
                and column in columns[table]
                and column not in found[table]
            ):
                found[table].append(column)

    return found

def existing_index_covers(
    table,
    candidate_columns,
    indexes,
):
    target = [
        c.lower()
        for c in candidate_columns
    ]

    for index in indexes.get(table, []):
        definition = index["definition"].lower()

        match = re.search(
            r"\(([^)]+)\)",
            definition,
        )

        if not match:
            continue

        existing = [
            c.strip().strip('"').split()[0]
            for c in match.group(1).split(",")
        ]

        if existing[:len(target)] == target:
            return True

    return False

def generate_candidates(
    df,
    conn,
    max_candidates,
):
    tables, columns, indexes = get_metadata(conn)

    # Use one representative query per fingerprint,
    # ranked by monitored execution time.
    ranked = (
        df.sort_values(
            "execution_time_ms",
            ascending=False,
        )
        .drop_duplicates("fingerprint")
    )

    candidates = []
    seen = set()

    for _, row in ranked.iterrows():
        query = str(row["sql"])

        if not safe_select(query):
            continue

        aliases = extract_aliases(query)

        matches = extract_index_columns(
            query,
            aliases,
            columns,
        )

        for table, available in matches.items():
            if table not in tables:
                continue

            available = available[:3]

            for width in range(
                1,
                min(3, len(available)) + 1,
            ):
                selected = available[:width]

                key = (
                    table,
                    tuple(selected),
                )

                if key in seen:
                    continue

                seen.add(key)

                if existing_index_covers(
                    table,
                    selected,
                    indexes,
                ):
                    continue

                identifier = (
                    "p89v2_"
                    + hashlib.sha1(
                        repr(key).encode()
                    ).hexdigest()[:12]
                )

                create_sql = sql.SQL(
                    "CREATE INDEX {} ON {}.{} ({})"
                ).format(
                    sql.Identifier(identifier),
                    sql.Identifier("public"),
                    sql.Identifier(table),
                    sql.SQL(", ").join(
                        sql.Identifier(col)
                        for col in selected
                    ),
                ).as_string(conn)

                candidates.append({
                    "id": identifier,
                    "kind": (
                        "COMPOSITE_BTREE"
                        if width > 1
                        else "BTREE"
                    ),
                    "table": table,
                    "columns": selected,
                    "create_sql": create_sql,
                    "source_fingerprint": row["fingerprint"],
                    "source_query": query,
                    "historical_latency_ms": float(
                        row["execution_time_ms"]
                    ),
                    "table_rows_estimate": (
                        tables[table]["rows"]
                    ),
                })

                if len(candidates) >= max_candidates:
                    return candidates

    return candidates

def explain(conn, query, analyze=False):
    if not safe_select(query):
        raise ValueError(
            "Unsafe or unsupported query."
        )

    prefix = (
        "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) "
        if analyze
        else "EXPLAIN (FORMAT JSON) "
    )

    with conn.cursor() as cur:
        cur.execute(prefix + query)
        value = cur.fetchone()[0]

    if isinstance(value, str):
        value = json.loads(value)

    root = value[0]
    plan = root["Plan"]

    return {
        "cost": float(
            plan.get("Total Cost", 0)
        ),
        "latency_ms": (
            float(root.get("Execution Time", 0))
            if analyze else None
        ),
        "planning_ms": (
            float(root.get("Planning Time", 0))
            if analyze else None
        ),
        "shared_reads": (
            int(plan.get("Shared Read Blocks", 0))
            if analyze else None
        ),
        "shared_hits": (
            int(plan.get("Shared Hit Blocks", 0))
            if analyze else None
        ),
        "plan": plan,
    }

class RollbackExperiment(Exception):
    def __init__(self, result):
        self.result = result

def relevant_queries(df, table, limit=8):
    rows = df[
        df["tables_accessed"]
        .fillna("")
        .astype(str)
        .str.contains(
            rf"(?<![a-z0-9_]){re.escape(table)}"
            rf"(?![a-z0-9_])",
            case=False,
            regex=True,
        )
    ]

    rows = (
        rows.sort_values(
            "execution_time_ms",
            ascending=False,
        )
        .drop_duplicates("fingerprint")
    )

    return [
        str(query)
        for query in rows["sql"]
        if safe_select(query)
    ][:limit]

def evaluate_candidate(
    conn,
    candidate,
    queries,
    actual_benchmark,
    repeats,
):
    if not queries:
        return {
            "candidate_id": candidate["id"],
            "decision": "NO_RELEVANT_QUERIES",
        }

    before_costs = []
    after_costs = []
    before_times = []
    after_times = []
    errors = []

    try:
        with conn.transaction():
            for query in queries:
                try:
                    before_costs.append(
                        explain(
                            conn,
                            query,
                            analyze=False,
                        )["cost"]
                    )
                except Exception as exc:
                    errors.append(
                        f"Baseline: {str(exc)[:150]}"
                    )

            if not before_costs:
                raise RollbackExperiment({
                    "candidate_id": candidate["id"],
                    "decision": "BASELINE_FAILED",
                    "errors": errors,
                })

            with conn.cursor() as cur:
                cur.execute(
                    candidate["create_sql"]
                )

            # Re-evaluate the same queries after
            # creating the index.
            for query in queries:
                try:
                    after_costs.append(
                        explain(
                            conn,
                            query,
                            analyze=False,
                        )["cost"]
                    )
                except Exception as exc:
                    errors.append(
                        f"Candidate: {str(exc)[:150]}"
                    )

            if len(before_costs) != len(after_costs):
                raise RollbackExperiment({
                    "candidate_id": candidate["id"],
                    "decision": "INCOMPLETE_COST_COMPARISON",
                    "errors": errors,
                })

            base_cost = float(
                sum(before_costs)
            )
            new_cost = float(
                sum(after_costs)
            )

            cost_improvement = (
                (base_cost - new_cost)
                / max(base_cost, 1e-9)
                * 100
            )

            if actual_benchmark:
                # Actual benchmarks are deliberately
                # restricted to a small number of queries.
                for query in queries[:3]:
                    # The before-index latency cannot be
                    # measured here because the index
                    # already exists in this transaction.
                    # Therefore actual measurements are
                    # handled separately.
                    pass

            result = {
                "candidate_id": candidate["id"],
                "table": candidate["table"],
                "columns": candidate["columns"],
                "queries_evaluated": len(before_costs),
                "baseline_total_cost": base_cost,
                "optimized_total_cost": new_cost,
                "estimated_improvement_pct": round(
                    cost_improvement,
                    4,
                ),
                "decision": (
                    "PROVISIONAL_ACCEPT"
                    if cost_improvement >= 5
                    else "REJECT"
                    if cost_improvement <= 0
                    else "DEFER"
                ),
                "errors": errors,
            }

            raise RollbackExperiment(result)

    except RollbackExperiment as exc:
        return exc.result

def benchmark_candidate(
    conn,
    candidate,
    queries,
    repeats,
):
    # Measure baseline first, without the index.
    baseline = []

    for query in queries[:3]:
        samples = []

        for _ in range(repeats):
            with conn.transaction():
                samples.append(
                    explain(
                        conn,
                        query,
                        analyze=True,
                    )["latency_ms"]
                )

        baseline.append(
            float(np.median(samples))
        )

    # Create candidate index inside a transaction
    # and force rollback at the end.
    try:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute(
                    candidate["create_sql"]
                )

            optimized = []

            for query in queries[:3]:
                samples = []

                for _ in range(repeats):
                    samples.append(
                        explain(
                            conn,
                            query,
                            analyze=True,
                        )["latency_ms"]
                    )

                optimized.append(
                    float(np.median(samples))
                )

            baseline_total = sum(baseline)
            optimized_total = sum(optimized)

            improvement = (
                (baseline_total - optimized_total)
                / max(baseline_total, 1e-9)
                * 100
            )

            result = {
                "candidate_id": candidate["id"],
                "queries_benchmarked": len(baseline),
                "baseline_latency_ms": baseline_total,
                "optimized_latency_ms": optimized_total,
                "actual_improvement_pct": round(
                    improvement,
                    4,
                ),
                "decision": (
                    "MEASURED_IMPROVEMENT"
                    if improvement > 5
                    else "NO_CONFIRMED_IMPROVEMENT"
                ),
                "limitation": (
                    "Sequential local measurements can "
                    "be influenced by PostgreSQL cache "
                    "warming and execution order."
                ),
            }

            raise RollbackExperiment(result)

    except RollbackExperiment as exc:
        return exc.result

def recommendations(df, conn):
    tables, columns, indexes = get_metadata(conn)

    suggestions = []

    for table, metadata in tables.items():
        rows = metadata["rows"]

        if rows >= 100000:
            for column, dtype in columns[table].items():
                if dtype in (
                    "date",
                    "timestamp without time zone",
                    "timestamp with time zone",
                ):
                    suggestions.append({
                        "kind": "BRIN_REVIEW",
                        "table": table,
                        "column": column,
                        "automatically_applied": False,
                    })
                    break

        if rows >= 500000:
            suggestions.append({
                "kind": "PARTITION_REVIEW",
                "table": table,
                "automatically_applied": False,
            })

        for column, dtype in columns[table].items():
            if dtype in (
                "jsonb",
                "ARRAY",
            ):
                suggestions.append({
                    "kind": "GIN_REVIEW",
                    "table": table,
                    "column": column,
                    "automatically_applied": False,
                })
                break

    group_queries = df[
        df["sql"].str.contains(
            r"\bgroup\s+by\b",
            case=False,
            regex=True,
        )
    ]

    if not group_queries.empty:
        suggestions.append({
            "kind": "MATERIALIZED_VIEW_REVIEW",
            "query_count": int(
                len(group_queries)
            ),
            "automatically_applied": False,
        })

    return suggestions

def run_phase9(
    df,
    dsn,
    max_candidates,
    actual_benchmark,
    repeats,
):
    if not dsn:
        raise ValueError(
            "Missing PHASE89_BENCHMARK_DSN."
        )

    conn = connect_benchmark(dsn)

    try:
        candidates = generate_candidates(
            df,
            conn,
            max_candidates,
        )

        conn.rollback()

        write_json(
            "phase9_candidates.json",
            candidates,
        )

        results = []
        benchmarks = []

        for index, candidate in enumerate(
            candidates,
            start=1,
        ):
            print(
                f"Candidate {index}/{len(candidates)}: "
                f"{candidate['table']} "
                f"{candidate['columns']}",
                flush=True,
            )

            queries = relevant_queries(
                df,
                candidate["table"],
                limit=8,
            )

            try:
                result = evaluate_candidate(
                    conn,
                    candidate,
                    queries,
                    actual_benchmark,
                    repeats,
                )
                results.append(result)
            except Exception as exc:
                conn.rollback()
                results.append({
                    "candidate_id": candidate["id"],
                    "decision": "ERROR",
                    "error": str(exc)[:300],
                })

            conn.rollback()

            # Benchmark provisionally accepted indexes,
            # plus a limited number of rejected candidates
            # to test whether the cost model missed benefits.
            should_benchmark = (
                actual_benchmark
                and bool(queries)
                and (
                    result.get("decision")
                    == "PROVISIONAL_ACCEPT"
                    or index <= 3
                )
            )

            if should_benchmark:
                try:
                    measured = benchmark_candidate(
                        conn,
                        candidate,
                        queries,
                        repeats,
                    )
                    benchmarks.append(measured)
                except Exception as exc:
                    conn.rollback()
                    benchmarks.append({
                        "candidate_id": candidate["id"],
                        "decision": "ERROR",
                        "error": str(exc)[:300],
                    })

            conn.rollback()

        feedback = pd.DataFrame(benchmarks)

        if not feedback.empty:
            feedback.to_csv(
                RESULTS / "phase9_feedback.csv",
                index=False,
            )

        report = {
            "status": "EVALUATED",
            "benchmark_database": BENCHMARK_DB,
            "candidate_count": len(candidates),
            "provisional_accepts": sum(
                r.get("decision")
                == "PROVISIONAL_ACCEPT"
                for r in results
            ),
            "rejected": sum(
                r.get("decision") == "REJECT"
                for r in results
            ),
            "deferred": sum(
                r.get("decision") == "DEFER"
                for r in results
            ),
            "actual_benchmarks": len(benchmarks),
            "candidate_results": results,
            "benchmark_results": benchmarks,
            "additional_recommendations":
                recommendations(df, conn),
            "production_changes": 0,
            "persistent_benchmark_changes": 0,
            "limitations": [
                "No production indexes are modified.",
                "All candidate indexes are rolled back.",
                "Benchmark results are sensitive to cache effects.",
                "Partitioning and materialized views are recommendations only.",
                "Long-term write overhead is not measured.",
                "No autonomous production deployment is performed.",
            ],
        }

        write_json(
            "phase9_summary.json",
            report,
        )

        return report

    finally:
        conn.rollback()
        conn.close()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phase",
        choices=["8", "9", "both"],
        default="both",
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
    )
    parser.add_argument(
        "--actual-benchmark",
        action="store_true",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--benchmark-dsn",
        default=None,
    )

    args = parser.parse_args()

    if args.max_candidates < 1 or args.repeats < 1:
        parser.error(
            "Candidates and repeats must be positive."
        )

    start = time.perf_counter()

    df = load_data(args.input)
    df = add_features(df)

    print(
        f"Loaded {len(df)} monitoring records "
        f"from {args.input}",
        flush=True,
    )
    print(
        f"Unique fingerprints: "
        f"{df['fingerprint'].nunique()}",
        flush=True,
    )

    summary = {
        "started_at": now(),
        "records": len(df),
        "unique_fingerprints":
            int(df["fingerprint"].nunique()),
        "input": str(args.input),
    }

    if args.phase in ("8", "both"):
        print("\nRunning Phase 8...", flush=True)

        classifier = train_phase8(df)
        clustering, clustered = cluster_phase8(df)
        forecast = forecast_phase8(clustered)

        summary["phase8"] = {
            "classifier": classifier,
            "clustering": clustering,
            "forecast": forecast,
        }

        print(
            "Phase 8 completed.",
            flush=True,
        )

    if args.phase in ("9", "both"):
        print("\nRunning Phase 9...", flush=True)

        dsn = (
            args.benchmark_dsn
            or os.getenv(
                "PHASE89_BENCHMARK_DSN"
            )
        )

        summary["phase9"] = run_phase9(
            df,
            dsn,
            args.max_candidates,
            args.actual_benchmark,
            args.repeats,
        )

        print(
            "Phase 9 completed.",
            flush=True,
        )

    summary["finished_at"] = now()
    summary["elapsed_seconds"] = round(
        time.perf_counter() - start,
        2,
    )

    write_json(
        "combined_summary.json",
        summary,
    )

    print(
        "\nFINAL SUMMARY",
        flush=True,
    )
    print(
        json.dumps(
            {
                "records": summary["records"],
                "fingerprints":
                    summary["unique_fingerprints"],
                "elapsed_seconds":
                    summary["elapsed_seconds"],
                "output_directory": str(RESULTS),
            },
            indent=2,
        ),
        flush=True,
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)
