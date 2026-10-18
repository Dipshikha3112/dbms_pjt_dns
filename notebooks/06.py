from pathlib import Path
import os
import re
import json
import time
import hashlib
import getpass
import statistics
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg
from psycopg import sql


# =====================================================================
# PHASE 9
# SELF-OPTIMIZING DATABASE
#
# Adaptive Healthcare Knowledge Database
#
# 9A - Workload / Plan Analysis
# 9B - Physical Design Advisor
# 9C - Experimental Evaluation
# 9D - Recommendation + Feedback
#
# IMPORTANT:
# Experimental indexes are removed after evaluation by default.
# The original Phase 4/5 physical design is therefore preserved.
# =====================================================================


print("=" * 115)
print("PHASE 9 - SELF-OPTIMIZING DATABASE")
print("Adaptive Healthcare Knowledge Database")
print("=" * 115)


# =====================================================================
# 1. PATHS
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PHASE7_DIR = PROJECT_ROOT / "results" / "phase7_monitoring"
PHASE8_DIR = PROJECT_ROOT / "results" / "phase8_ai_workload"

PHASE7_FILE = PHASE7_DIR / "phase7_query_monitoring.csv"
PHASE8_PRIORITY_FILE = PHASE8_DIR / "phase8_optimization_priority.csv"

RESULTS_DIR = PROJECT_ROOT / "results" / "phase9_self_optimization"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# 2. OUTPUTS
# =====================================================================

WORKLOAD_ANALYSIS_FILE = RESULTS_DIR / "phase9_workload_analysis.csv"
CANDIDATES_FILE = RESULTS_DIR / "phase9_all_candidates.csv"
INDEX_RESULTS_FILE = RESULTS_DIR / "phase9_index_results.csv"
MV_RESULTS_FILE = RESULTS_DIR / "phase9_materialized_view_recommendations.csv"
PARTITION_FILE = RESULTS_DIR / "phase9_partition_recommendations.csv"
GIN_FILE = RESULTS_DIR / "phase9_gin_recommendations.csv"
BEST_FILE = RESULTS_DIR / "phase9_best_designs.csv"
APPROVED_FILE = RESULTS_DIR / "phase9_approved_recommendations.csv"
REJECTED_FILE = RESULTS_DIR / "phase9_rejected_recommendations.csv"
FEEDBACK_FILE = RESULTS_DIR / "phase9_feedback_dataset.csv"
PLAN_FILE = RESULTS_DIR / "phase9_plan_comparisons.json"
METADATA_FILE = RESULTS_DIR / "phase9_metadata.json"
REPORT_FILE = RESULTS_DIR / "phase9_report.txt"


# =====================================================================
# 3. DATABASE
# =====================================================================

DB_CONFIG = {
    "dbname": "adaptive_healthcare_db",
    "host": "localhost",
    "port": 5432,
    "user": "postgres",
}

password = os.getenv("POSTGRES_PASSWORD")

if not password:
    password = getpass.getpass("PostgreSQL password: ")

DB_CONFIG["password"] = password


# =====================================================================
# 4. PHASE 9 CONFIGURATION
# =====================================================================

TOP_PATTERNS = 15

BASELINE_RUNS = 3
CANDIDATE_RUNS = 3

STATEMENT_TIMEOUT_MS = 90000

MAX_INDEX_COLUMNS = 3
MAX_INDEX_CANDIDATES_PER_PATTERN = 8

MIN_TABLE_ROWS = 500

# Primary measured-performance threshold
MIN_LATENCY_IMPROVEMENT = 10.0

# Strong measured improvements may be approved regardless
# of planner cost estimate.
STRONG_LATENCY_IMPROVEMENT = 25.0

# Supporting threshold, NOT mandatory veto.
MIN_COST_IMPROVEMENT = 2.0

# Candidate must not create severe regression.
MAX_REGRESSION = 5.0

# Utility threshold
MIN_UTILITY_SCORE = 45.0

# BRIN normally makes sense for larger tables.
# Our research DB is smaller than production, therefore this is
# deliberately permissive enough to allow experimentation.
BRIN_MIN_ROWS = 100000

# Candidate materialized view:
# only recommend automatically when repeated analytical aggregation
# is detected.
MV_MIN_FREQUENCY = 10

# Keep experimental indexes?
KEEP_APPROVED_INDEXES = False

# Actually create materialized views?
#
# False is safer because generic query rewriting is not reliable.
# We recommend MVs and benchmark indexes automatically.
CREATE_MATERIALIZED_VIEWS = False

# Partitioning remains advisory.
AUTO_PARTITION = False


# =====================================================================
# 5. HELPERS
# =====================================================================

def stage(title):
    print()
    print("=" * 115)
    print(title)
    print("=" * 115)


def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def clean_sql(query):
    if pd.isna(query):
        return ""

    query = str(query).strip()

    while query.endswith(";"):
        query = query[:-1].strip()

    return query


def sha256_text(value):
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def median(values):
    values = [
        float(x)
        for x in values
        if x is not None
        and not pd.isna(x)
    ]

    if not values:
        return 0.0

    return statistics.median(values)


def mean(values):
    values = [
        float(x)
        for x in values
        if x is not None
        and not pd.isna(x)
    ]

    if not values:
        return 0.0

    return statistics.mean(values)


def pct_improvement(before, after):
    if before is None or before <= 0:
        return 0.0

    return ((before - after) / before) * 100.0


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def unique_order(values):
    seen = set()
    result = []

    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)

    return result


# =====================================================================
# 6. QUERY SAFETY
# =====================================================================

FORBIDDEN = [
    " INSERT ",
    " UPDATE ",
    " DELETE ",
    " DROP ",
    " ALTER ",
    " TRUNCATE ",
    " GRANT ",
    " REVOKE ",
    " COPY ",
    " VACUUM ",
]


def is_read_only(query):
    q = " " + query.upper() + " "

    stripped = query.strip().upper()

    if not (
        stripped.startswith("SELECT")
        or stripped.startswith("WITH")
    ):
        return False

    for keyword in FORBIDDEN:
        if keyword in q:
            return False

    return True


# =====================================================================
# 7. CONNECT
# =====================================================================

stage("CONNECTING TO POSTGRESQL")

conn = psycopg.connect(**DB_CONFIG)
conn.autocommit = True

with conn.cursor() as cur:
    cur.execute(
        "SELECT current_database(), version();"
    )

    database_name, pg_version = cur.fetchone()

print("Database:", database_name)
print("Connection: PASS")


# =====================================================================
# 8. LOAD INPUT
# =====================================================================

stage("LOADING PHASE 7 + PHASE 8")

if not PHASE7_FILE.exists():
    raise FileNotFoundError(PHASE7_FILE)

if not PHASE8_PRIORITY_FILE.exists():
    raise FileNotFoundError(PHASE8_PRIORITY_FILE)

monitor_df = pd.read_csv(PHASE7_FILE)
priority_df = pd.read_csv(PHASE8_PRIORITY_FILE)

print("Phase 7 executions:", f"{len(monitor_df):,}")
print("Phase 8 patterns:", f"{len(priority_df):,}")


required_phase7 = {
    "fingerprint_hash",
    "query_name",
    "query_family",
    "sql",
    "execution_time_ms",
    "estimated_cost",
}

missing = required_phase7 - set(monitor_df.columns)

if missing:
    raise RuntimeError(
        "Missing Phase 7 columns: "
        + str(sorted(missing))
    )


required_phase8 = {
    "fingerprint_hash",
    "priority_rank",
    "optimization_priority_score",
    "optimization_priority",
}

missing = required_phase8 - set(priority_df.columns)

if missing:
    raise RuntimeError(
        "Missing Phase 8 columns: "
        + str(sorted(missing))
    )

print("Input validation: PASS")


# =====================================================================
# 9. SCHEMA METADATA
# =====================================================================

stage("READING DATABASE METADATA")


def load_columns():
    result = {}

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                table_name,
                column_name,
                data_type
            FROM information_schema.columns
            WHERE table_schema='public'
            ORDER BY table_name, ordinal_position;
            """
        )

        for table, column, dtype in cur.fetchall():
            result.setdefault(table, {})
            result[table][column] = dtype

    return result


def load_primary_keys():
    result = {}

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                tc.table_name,
                kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema = kcu.table_schema
            WHERE tc.table_schema='public'
              AND tc.constraint_type='PRIMARY KEY'
            ORDER BY tc.table_name, kcu.ordinal_position;
            """
        )

        for table, column in cur.fetchall():
            result.setdefault(table, [])
            result[table].append(column)

    return result


def load_indexes():
    result = {}

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                tablename,
                indexname,
                indexdef
            FROM pg_indexes
            WHERE schemaname='public';
            """
        )

        for table, name, definition in cur.fetchall():
            result.setdefault(table, [])

            result[table].append(
                {
                    "name": name,
                    "definition": definition
                }
            )

    return result


def load_table_stats():
    result = {}

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                relname,
                n_live_tup,
                pg_total_relation_size(relid)
            FROM pg_stat_user_tables;
            """
        )

        for table, rows, size in cur.fetchall():
            result[table] = {
                "rows": int(rows or 0),
                "size_bytes": int(size or 0)
            }

    return result


schema = load_columns()
primary_keys = load_primary_keys()
existing_indexes = load_indexes()
table_stats = load_table_stats()

print("Tables:", len(schema))


# =====================================================================
# 10. PLAN FUNCTIONS
# =====================================================================

def walk_plan(plan):
    yield plan

    for child in plan.get("Plans", []):
        yield from walk_plan(child)


def explain_analyze(query):
    if not is_read_only(query):
        raise RuntimeError("Unsafe SQL rejected")

    with conn.cursor() as cur:

        cur.execute(
            f"SET statement_timeout={STATEMENT_TIMEOUT_MS};"
        )

        start = time.perf_counter()

        cur.execute(
            "EXPLAIN "
            "(ANALYZE TRUE, BUFFERS TRUE, FORMAT JSON) "
            + query
        )

        payload = cur.fetchone()[0]

        wall_ms = (
            time.perf_counter() - start
        ) * 1000.0

    if isinstance(payload, str):
        payload = json.loads(payload)

    root = payload[0]
    plan = root["Plan"]

    return {
        "execution_time_ms":
            safe_float(root.get("Execution Time")),

        "planning_time_ms":
            safe_float(root.get("Planning Time")),

        "wall_time_ms":
            wall_ms,

        "cost":
            safe_float(plan.get("Total Cost")),

        "actual_rows":
            safe_float(plan.get("Actual Rows")),

        "plan":
            plan,

        "raw":
            payload,
    }


def analyze_plan(plan):
    result = {
        "tables": set(),
        "seq_scan_tables": set(),
        "index_scan_tables": set(),
        "indexes_used": set(),
        "shared_hit_blocks": 0,
        "shared_read_blocks": 0,
        "temp_read_blocks": 0,
        "temp_written_blocks": 0,
        "sorts": 0,
        "aggregates": 0,
        "joins": 0,
    }

    for node in walk_plan(plan):

        table = node.get("Relation Name")
        node_type = node.get("Node Type", "")

        if table:
            result["tables"].add(table)

        if node_type == "Seq Scan" and table:
            result["seq_scan_tables"].add(table)

        if node_type in {
            "Index Scan",
            "Index Only Scan",
            "Bitmap Index Scan",
        }:
            if table:
                result["index_scan_tables"].add(table)

        if node.get("Index Name"):
            result["indexes_used"].add(
                node["Index Name"]
            )

        if node_type in {
            "Hash Join",
            "Merge Join",
            "Nested Loop",
        }:
            result["joins"] += 1

        if "Aggregate" in node_type:
            result["aggregates"] += 1

        if node_type == "Sort":
            result["sorts"] += 1

    # Root buffer statistics already represent the complete query.
    result["shared_hit_blocks"] = safe_int(
        plan.get("Shared Hit Blocks")
    )

    result["shared_read_blocks"] = safe_int(
        plan.get("Shared Read Blocks")
    )

    result["temp_read_blocks"] = safe_int(
        plan.get("Temp Read Blocks")
    )

    result["temp_written_blocks"] = safe_int(
        plan.get("Temp Written Blocks")
    )

    for key in [
        "tables",
        "seq_scan_tables",
        "index_scan_tables",
        "indexes_used",
    ]:
        result[key] = sorted(result[key])

    return result


# =====================================================================
# 11. BENCHMARK
# =====================================================================

def benchmark(query, runs):
    results = []

    for i in range(runs):
        result = explain_analyze(query)

        results.append(result)

    return {
        "runs": results,

        "latency_ms":
            median([
                r["execution_time_ms"]
                for r in results
            ]),

        "cost":
            median([
                r["cost"]
                for r in results
            ]),

        "io_reads":
            median([
                analyze_plan(r["plan"])[
                    "shared_read_blocks"
                ]
                for r in results
            ]),

        "io_hits":
            median([
                analyze_plan(r["plan"])[
                    "shared_hit_blocks"
                ]
                for r in results
            ]),

        "plan":
            results[-1]["plan"],

        "raw":
            results[-1]["raw"],
    }


# =====================================================================
# 12. REPRESENTATIVE QUERY
# =====================================================================

def representative_query(fingerprint):

    subset = monitor_df[
        monitor_df["fingerprint_hash"]
        == fingerprint
    ].copy()

    if subset.empty:
        return None

    subset["execution_time_ms"] = pd.to_numeric(
        subset["execution_time_ms"],
        errors="coerce"
    )

    med = subset["execution_time_ms"].median()

    subset["_distance"] = (
        subset["execution_time_ms"] - med
    ).abs()

    return subset.sort_values(
        "_distance"
    ).iloc[0]


# =====================================================================
# 13. QUERY FREQUENCY
# =====================================================================

frequency_map = (
    monitor_df
    .groupby("fingerprint_hash")
    .size()
    .to_dict()
)


# =====================================================================
# 14. SQL PARSING
# =====================================================================

SQL_RESERVED = {
    "WHERE",
    "JOIN",
    "LEFT",
    "RIGHT",
    "FULL",
    "INNER",
    "OUTER",
    "ON",
    "GROUP",
    "ORDER",
    "LIMIT",
    "HAVING",
    "OFFSET",
    "UNION",
    "CROSS",
}


def alias_map(query):

    result = {}

    pattern = re.compile(
        r"""
        \b(?:FROM|JOIN)\s+
        "?([A-Za-z_][A-Za-z0-9_]*)"?
        (?:\s+(?:AS\s+)?
        ([A-Za-z_][A-Za-z0-9_]*))?
        """,
        re.I | re.X
    )

    for match in pattern.finditer(query):

        table = match.group(1)
        alias = match.group(2)

        if table not in schema:
            continue

        result[table] = table

        if alias and alias.upper() not in SQL_RESERVED:
            result[alias] = table

    return result


def qualified_references(query):

    amap = alias_map(query)

    pattern = re.compile(
        r'\b([A-Za-z_][A-Za-z0-9_]*)\.'
        r'"?([A-Za-z_][A-Za-z0-9_]*)"?'
    )

    result = []

    for alias, column in pattern.findall(query):

        table = amap.get(alias)

        if not table:
            continue

        if column not in schema.get(table, {}):
            continue

        result.append(
            (table, column)
        )

    return unique_order(result)


def extract_clause(query, keyword, stops):

    upper = query.upper()

    start = upper.find(keyword)

    if start < 0:
        return ""

    start += len(keyword)

    positions = []

    for stop in stops:

        pos = upper.find(stop, start)

        if pos >= 0:
            positions.append(pos)

    end = min(positions) if positions else len(query)

    return query[start:end]


# =====================================================================
# 15. COLUMN ROLE ANALYSIS
# =====================================================================

def analyze_column_roles(query):

    refs = qualified_references(query)

    where = extract_clause(
        query,
        " WHERE ",
        [
            " GROUP BY ",
            " ORDER BY ",
            " HAVING ",
            " LIMIT ",
        ]
    )

    join_section = extract_clause(
        query,
        " FROM ",
        [
            " WHERE ",
            " GROUP BY ",
            " ORDER BY ",
            " HAVING ",
            " LIMIT ",
        ]
    )

    group = extract_clause(
        query,
        " GROUP BY ",
        [
            " ORDER BY ",
            " HAVING ",
            " LIMIT ",
        ]
    )

    order = extract_clause(
        query,
        " ORDER BY ",
        [
            " LIMIT ",
            " OFFSET ",
        ]
    )

    roles = {}

    for table, column in refs:

        roles.setdefault(
            table,
            {
                "equality": [],
                "range": [],
                "join": [],
                "group": [],
                "order": [],
                "referenced": [],
            }
        )

        roles[table]["referenced"].append(column)

        c = re.escape(column)

        if re.search(
            rf'\b{c}\b\s*=\s*(?![A-Za-z_]+\.)',
            where,
            re.I
        ):
            roles[table]["equality"].append(column)

        if re.search(
            rf'\b{c}\b\s+IN\s*\(',
            where,
            re.I
        ):
            roles[table]["equality"].append(column)

        if re.search(
            rf'\b{c}\b\s*(>=|<=|>|<|BETWEEN)',
            where,
            re.I
        ):
            roles[table]["range"].append(column)

        if re.search(
            rf'\b{c}\b',
            join_section,
            re.I
        ):
            roles[table]["join"].append(column)

        if re.search(
            rf'\b{c}\b',
            group,
            re.I
        ):
            roles[table]["group"].append(column)

        if re.search(
            rf'\b{c}\b',
            order,
            re.I
        ):
            roles[table]["order"].append(column)

    for table in roles:

        for role in roles[table]:

            roles[table][role] = unique_order(
                roles[table][role]
            )

    return roles


# =====================================================================
# 16. EXISTING INDEX COVERAGE
# =====================================================================

def index_already_exists(table, columns):

    if not columns:
        return True

    definitions = existing_indexes.get(
        table,
        []
    )

    target = [
        c.lower()
        for c in columns
    ]

    for item in definitions:

        definition = item["definition"].lower()

        match = re.search(
            r'\((.*?)\)',
            definition
        )

        if not match:
            continue

        indexed_part = match.group(1)

        indexed_columns = []

        for piece in indexed_part.split(","):

            piece = piece.strip()

            col_match = re.match(
                r'"?([a-zA-Z_][a-zA-Z0-9_]*)"?',
                piece
            )

            if col_match:
                indexed_columns.append(
                    col_match.group(1).lower()
                )

        # PostgreSQL B-tree prefix rule.
        if (
            len(indexed_columns) >= len(target)
            and
            indexed_columns[:len(target)] == target
        ):
            return True

    return False


# =====================================================================
# 17. CANDIDATE NAME
# =====================================================================

def make_index_name(method, table, columns):

    base = (
        f"ai_p9_{method}_{table}_"
        + "_".join(columns)
    )

    digest = sha256_text(base)[:7]

    base = re.sub(
        r"[^A-Za-z0-9_]",
        "_",
        base
    )

    return base[:50] + "_" + digest


# =====================================================================
# 18. B-TREE / COMPOSITE / COVERING CANDIDATES
# =====================================================================

def generate_btree_candidates(
    query,
    plan_info
):

    roles = analyze_column_roles(query)

    candidates = []

    target_tables = (
        plan_info["seq_scan_tables"]
        if plan_info["seq_scan_tables"]
        else plan_info["tables"]
    )

    for table in target_tables:

        if table not in roles:
            continue

        rows = table_stats.get(
            table,
            {}
        ).get("rows", 0)

        if rows < MIN_TABLE_ROWS:
            continue

        role = roles[table]

        equality = role["equality"]
        joins = role["join"]
        ranges = role["range"]
        order = role["order"]
        groups = role["group"]

        pk = set(
            primary_keys.get(table, [])
        )

        # Do not lead a proposed composite index with an
        # already-unique PK unless the query actually filters on it.
        useful_joins = [
            c for c in joins
            if c not in pk
            or c in equality
        ]

        proposals = []

        # Selective equality first.
        proposals.append(
            equality
            + useful_joins
            + ranges
        )

        # Join-heavy.
        proposals.append(
            useful_joins
            + equality
            + ranges
        )

        # Equality + temporal ordering.
        proposals.append(
            equality
            + ranges
            + order
        )

        # Equality + grouping.
        proposals.append(
            equality
            + groups
        )

        # Join + grouping.
        proposals.append(
            useful_joins
            + groups
        )

        # Individual useful columns.
        for column in (
            equality
            + ranges
            + useful_joins
        ):
            proposals.append([column])

        seen = set()

        for columns in proposals:

            columns = unique_order(columns)

            columns = [
                c for c in columns
                if c in schema.get(table, {})
            ]

            columns = columns[:MAX_INDEX_COLUMNS]

            if not columns:
                continue

            key = (
                table,
                tuple(columns)
            )

            if key in seen:
                continue

            seen.add(key)

            if index_already_exists(
                table,
                columns
            ):
                continue

            method = (
                "composite"
                if len(columns) > 1
                else "btree"
            )

            candidates.append(
                {
                    "design_type":
                        method.upper(),

                    "table":
                        table,

                    "columns":
                        columns,

                    "include_columns":
                        [],

                    "index_name":
                        make_index_name(
                            method,
                            table,
                            columns
                        ),

                    "reason":
                        "Derived from equality, join, "
                        "range, grouping and ordering "
                        "roles in the monitored SQL."
                }
            )

        # -------------------------------------------------------------
        # Covering index candidate
        # -------------------------------------------------------------

        key_columns = unique_order(
            equality
            + useful_joins
            + ranges
        )[:2]

        include_columns = [
            c for c in role["referenced"]
            if c not in key_columns
            and c not in pk
        ][:3]

        if (
            key_columns
            and include_columns
            and not index_already_exists(
                table,
                key_columns
            )
        ):

            candidates.append(
                {
                    "design_type":
                        "COVERING_INDEX",

                    "table":
                        table,

                    "columns":
                        key_columns,

                    "include_columns":
                        include_columns,

                    "index_name":
                        make_index_name(
                            "cover",
                            table,
                            key_columns
                        ),

                    "reason":
                        "Key columns support filtering/"
                        "joins while INCLUDE columns may "
                        "reduce heap access."
                }
            )

    return candidates[
        :MAX_INDEX_CANDIDATES_PER_PATTERN
    ]


# =====================================================================
# 19. PARTIAL INDEX DETECTION
# =====================================================================

def generate_partial_candidates(query):

    candidates = []

    roles = analyze_column_roles(query)

    where = extract_clause(
        query,
        " WHERE ",
        [
            " GROUP BY ",
            " ORDER BY ",
            " HAVING ",
            " LIMIT ",
        ]
    )

    if not where.strip():
        return candidates

    # Only generate a partial index when a stable literal predicate
    # can be isolated. UUID/patient-specific predicates are excluded
    # because they would overfit one concrete execution.

    stable_pattern = re.compile(
        r"""
        (?:
            ([A-Za-z_][A-Za-z0-9_]*)\.
        )?
        ([A-Za-z_][A-Za-z0-9_]*)
        \s*=\s*
        ('[^']*'|TRUE|FALSE)
        """,
        re.I | re.X
    )

    amap = alias_map(query)

    for match in stable_pattern.finditer(where):

        alias = match.group(1)
        column = match.group(2)
        literal = match.group(3)

        if alias:
            table = amap.get(alias)
        else:
            table = None

            matching_tables = [
                t for t in roles
                if column in schema.get(t, {})
            ]

            if len(matching_tables) == 1:
                table = matching_tables[0]

        if not table:
            continue

        if column not in schema.get(table, {}):
            continue

        # Avoid patient-specific descriptions/names as partial
        # predicates where possible.
        if column.lower() in {
            "patient_id",
            "encounter_id",
            "provider_id",
            "prescription_id",
            "claim_id",
        }:
            continue

        role = roles.get(table, {})

        key_columns = unique_order(
            role.get("join", [])
            + role.get("range", [])
            + role.get("order", [])
        )

        key_columns = [
            c for c in key_columns
            if c != column
        ][:MAX_INDEX_COLUMNS]

        if not key_columns:
            key_columns = [column]

        name = make_index_name(
            "partial",
            table,
            key_columns
        )

        predicate = (
            f"{column} = {literal}"
        )

        candidates.append(
            {
                "design_type":
                    "PARTIAL_INDEX",

                "table":
                    table,

                "columns":
                    key_columns,

                "include_columns":
                    [],

                "index_name":
                    name,

                "predicate":
                    predicate,

                "reason":
                    "Stable literal predicate detected "
                    "in repeated monitored SQL."
            }
        )

    return candidates


# =====================================================================
# 20. BRIN DETECTION
# =====================================================================

TEMPORAL_TYPES = {
    "date",
    "timestamp without time zone",
    "timestamp with time zone",
}


def generate_brin_candidates(query):

    candidates = []

    roles = analyze_column_roles(query)

    for table, role in roles.items():

        rows = table_stats.get(
            table,
            {}
        ).get("rows", 0)

        if rows < BRIN_MIN_ROWS:
            continue

        possible = unique_order(
            role["range"]
            + role["order"]
            + role["group"]
        )

        for column in possible:

            dtype = schema.get(
                table,
                {}
            ).get(column, "")

            if dtype not in TEMPORAL_TYPES:
                continue

            name = make_index_name(
                "brin",
                table,
                [column]
            )

            candidates.append(
                {
                    "design_type":
                        "BRIN",

                    "table":
                        table,

                    "columns":
                        [column],

                    "include_columns":
                        [],

                    "index_name":
                        name,

                    "reason":
                        f"Large table ({rows:,} rows) "
                        f"with temporal/range access on "
                        f"{column}."
                }
            )

    return candidates


# =====================================================================
# 21. GIN ELIGIBILITY
# =====================================================================

def detect_gin_candidates(query):

    candidates = []

    amap = alias_map(query)

    # Detect common PostgreSQL operators where GIN may be useful.
    patterns = [
        (
            r'([A-Za-z_][A-Za-z0-9_]*)\.'
            r'([A-Za-z_][A-Za-z0-9_]*)\s*@>',
            "containment"
        ),
        (
            r'([A-Za-z_][A-Za-z0-9_]*)\.'
            r'([A-Za-z_][A-Za-z0-9_]*)\s*\?',
            "json/key"
        ),
        (
            r'to_tsvector\s*\([^,]+,\s*'
            r'([A-Za-z_][A-Za-z0-9_]*)\.'
            r'([A-Za-z_][A-Za-z0-9_]*)',
            "full_text"
        ),
    ]

    for pattern, reason_type in patterns:

        for alias, column in re.findall(
            pattern,
            query,
            re.I
        ):

            table = amap.get(alias)

            if not table:
                continue

            candidates.append(
                {
                    "design_type":
                        "GIN",

                    "table":
                        table,

                    "columns":
                        [column],

                    "reason":
                        f"GIN-eligible {reason_type} "
                        "predicate detected."
                }
            )

    return candidates


# =====================================================================
# 22. MATERIALIZED VIEW ELIGIBILITY
# =====================================================================

AGGREGATE_PATTERN = re.compile(
    r'\b(COUNT|SUM|AVG|MIN|MAX)\s*\(',
    re.I
)


def detect_materialized_view_candidate(
    query,
    frequency,
    baseline_latency
):

    upper = query.upper()

    has_group = " GROUP BY " in upper

    has_aggregate = bool(
        AGGREGATE_PATTERN.search(query)
    )

    if not has_aggregate:
        return None

    if frequency < MV_MIN_FREQUENCY:
        return None

    score = 0

    if has_group:
        score += 30

    if frequency >= 20:
        score += 25
    elif frequency >= 10:
        score += 15

    if baseline_latency >= 100:
        score += 30
    elif baseline_latency >= 20:
        score += 20
    else:
        score += 10

    if " JOIN " in upper:
        score += 15

    return {
        "design_type":
            "MATERIALIZED_VIEW",

        "eligibility_score":
            min(score, 100),

        "frequency":
            frequency,

        "has_group_by":
            has_group,

        "has_aggregate":
            has_aggregate,

        "reason":
            "Repeated analytical aggregation detected. "
            "A materialized aggregate may avoid repeated "
            "joins/scans/grouping."
    }


# =====================================================================
# 23. PARTITION ELIGIBILITY
# =====================================================================

def detect_partition_candidates(query):

    roles = analyze_column_roles(query)

    results = []

    for table, role in roles.items():

        rows = table_stats.get(
            table,
            {}
        ).get("rows", 0)

        temporal = []

        for column in unique_order(
            role["range"]
            + role["group"]
            + role["order"]
        ):

            dtype = schema.get(
                table,
                {}
            ).get(column, "")

            if dtype in TEMPORAL_TYPES:
                temporal.append(column)

        if not temporal:
            continue

        # Advisory only. At this project scale partitioning may
        # not improve performance, so we explicitly record scale.
        for column in temporal:

            if rows >= 500000:
                priority = "HIGH"
            elif rows >= 100000:
                priority = "MEDIUM"
            else:
                priority = "LOW"

            results.append(
                {
                    "design_type":
                        "PARTITION_RECOMMENDATION",

                    "table":
                        table,

                    "column":
                        column,

                    "table_rows":
                        rows,

                    "priority":
                        priority,

                    "reason":
                        "Temporal workload detected. "
                        "Range partitioning can be evaluated "
                        "at larger scale; no in-place "
                        "repartitioning is performed."
                }
            )

    return results


# =====================================================================
# 24. CREATE INDEX SQL
# =====================================================================

def create_index_statement(candidate):

    table = candidate["table"]
    name = candidate["index_name"]
    columns = candidate["columns"]

    col_sql = sql.SQL(", ").join(
        sql.Identifier(c)
        for c in columns
    )

    design_type = candidate["design_type"]

    if design_type == "BRIN":

        return sql.SQL(
            "CREATE INDEX {} ON {} USING BRIN ({})"
        ).format(
            sql.Identifier(name),
            sql.Identifier(table),
            col_sql
        )

    if design_type == "PARTIAL_INDEX":

        predicate = candidate["predicate"]

        return sql.SQL(
            "CREATE INDEX {} ON {} ({}) WHERE "
        ).format(
            sql.Identifier(name),
            sql.Identifier(table),
            col_sql
        ) + sql.SQL(predicate)

    if design_type == "COVERING_INDEX":

        include = sql.SQL(", ").join(
            sql.Identifier(c)
            for c in candidate[
                "include_columns"
            ]
        )

        return sql.SQL(
            "CREATE INDEX {} ON {} ({}) INCLUDE ({})"
        ).format(
            sql.Identifier(name),
            sql.Identifier(table),
            col_sql,
            include
        )

    return sql.SQL(
        "CREATE INDEX {} ON {} ({})"
    ).format(
        sql.Identifier(name),
        sql.Identifier(table),
        col_sql
    )


# =====================================================================
# 25. CREATE / DROP EXPERIMENTAL INDEX
# =====================================================================

def create_index(candidate):

    statement = create_index_statement(
        candidate
    )

    with conn.cursor() as cur:

        cur.execute(statement)

        cur.execute(
            sql.SQL(
                "ANALYZE {}"
            ).format(
                sql.Identifier(
                    candidate["table"]
                )
            )
        )


def drop_index(name):

    with conn.cursor() as cur:

        cur.execute(
            sql.SQL(
                "DROP INDEX IF EXISTS {}"
            ).format(
                sql.Identifier(name)
            )
        )


def index_size(name):

    with conn.cursor() as cur:

        cur.execute(
            """
            SELECT pg_relation_size(%s::regclass);
            """,
            (name,)
        )

        row = cur.fetchone()

        return int(
            row[0] or 0
        )


# =====================================================================
# 26. UTILITY SCORE
# =====================================================================

def calculate_utility(
    latency_improvement,
    io_improvement,
    cost_improvement,
    priority_score,
    frequency,
    index_size_bytes,
    table_size_bytes,
):

    # ---------------------------------------------------------
    # Measured latency = strongest signal
    # ---------------------------------------------------------

    latency_component = clamp(
        latency_improvement,
        -100,
        100
    )

    latency_component = (
        latency_component / 100.0
    ) * 45.0

    # ---------------------------------------------------------
    # Measured I/O
    # ---------------------------------------------------------

    io_component = clamp(
        io_improvement,
        -100,
        100
    )

    io_component = (
        io_component / 100.0
    ) * 15.0

    # ---------------------------------------------------------
    # PostgreSQL estimated cost = supporting evidence
    # ---------------------------------------------------------

    cost_component = clamp(
        cost_improvement,
        -100,
        100
    )

    cost_component = (
        cost_component / 100.0
    ) * 10.0

    # ---------------------------------------------------------
    # AI priority
    # ---------------------------------------------------------

    priority_component = (
        clamp(
            priority_score,
            0,
            100
        )
        / 100.0
    ) * 15.0

    # ---------------------------------------------------------
    # Frequency
    # ---------------------------------------------------------

    frequency_component = (
        min(
            frequency,
            60
        )
        / 60.0
    ) * 10.0

    # ---------------------------------------------------------
    # Storage penalty
    # ---------------------------------------------------------

    storage_ratio = 0.0

    if table_size_bytes > 0:

        storage_ratio = (
            index_size_bytes
            / table_size_bytes
        )

    storage_penalty = (
        min(
            storage_ratio,
            1.0
        )
        * 5.0
    )

    raw = (
        latency_component
        + io_component
        + cost_component
        + priority_component
        + frequency_component
        - storage_penalty
    )

    # Shift into easier 0-100 research scale.
    score = clamp(
        raw + 25.0,
        0,
        100
    )

    return score


# =====================================================================
# 27. ACCEPTANCE POLICY
# =====================================================================

def decision(
    latency_improvement,
    cost_improvement,
    utility_score,
    candidate_used
):

    if latency_improvement < -MAX_REGRESSION:
        return False, "Measured performance regression"

    # Strong real-world improvement.
    if (
        latency_improvement
        >= STRONG_LATENCY_IMPROVEMENT
        and candidate_used
    ):
        return (
            True,
            "Strong measured latency improvement"
        )

    # Normal acceptance.
    if (
        latency_improvement
        >= MIN_LATENCY_IMPROVEMENT
        and candidate_used
        and utility_score
        >= MIN_UTILITY_SCORE
    ):
        return (
            True,
            "Measured latency and utility thresholds passed"
        )

    # Moderate latency improvement with planner support.
    if (
        latency_improvement
        >= MIN_LATENCY_IMPROVEMENT
        and cost_improvement
        >= MIN_COST_IMPROVEMENT
        and utility_score
        >= MIN_UTILITY_SCORE
    ):
        return (
            True,
            "Measured benefit plus planner-cost support"
        )

    return (
        False,
        "Insufficient measured benefit"
    )


# =====================================================================
# 28. RECOMMENDATION TABLE
# =====================================================================

stage("VALIDATING RECOMMENDATION TABLE")

with conn.cursor() as cur:

    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema='public'
          AND table_name='optimization_recommendation';
        """
    )

    recommendation_columns = {
        row[0]
        for row in cur.fetchall()
    }

required = {
    "recommendation_type",
    "target_table",
    "target_columns",
    "recommendation_sql",
    "reason",
    "estimated_before_cost",
    "estimated_after_cost",
    "expected_improvement_percent",
    "actual_before_ms",
    "actual_after_ms",
    "actual_improvement_percent",
    "status",
}

missing = required - recommendation_columns

if missing:
    raise RuntimeError(
        "optimization_recommendation mismatch: "
        + str(sorted(missing))
    )

print("Schema validation: PASS")


# =====================================================================
# 29. INSERT RECOMMENDATION
# =====================================================================

def save_recommendation(record):

    with conn.cursor() as cur:

        cur.execute(
            """
            INSERT INTO optimization_recommendation
            (
                recommendation_type,
                target_table,
                target_columns,
                recommendation_sql,
                reason,
                estimated_before_cost,
                estimated_after_cost,
                expected_improvement_percent,
                actual_before_ms,
                actual_after_ms,
                actual_improvement_percent,
                status,
                created_at,
                evaluated_at
            )
            VALUES
            (
                %s,%s,%s,%s,%s,
                %s,%s,%s,
                %s,%s,%s,
                %s,
                NOW(),
                NOW()
            );
            """,
            (
                record["design_type"],
                record["table"],
                record["columns_text"],
                record["recommendation_sql"],
                record["reason"],
                record["baseline_cost"],
                record["candidate_cost"],
                record["cost_improvement_percent"],
                record["baseline_latency_ms"],
                record["candidate_latency_ms"],
                record["latency_improvement_percent"],
                record["status"],
            )
        )


# =====================================================================
# 30. SELECT PHASE 8 PATTERNS
# =====================================================================

stage("SELECTING PHASE 8 AI PRIORITIES")

selected = (
    priority_df
    .sort_values("priority_rank")
    .head(TOP_PATTERNS)
    .copy()
)

print(
    selected[
        [
            "priority_rank",
            "query_name",
            "query_family",
            "optimization_priority_score",
            "optimization_priority",
        ]
    ].to_string(index=False)
)


# =====================================================================
# 31. STORAGE
# =====================================================================

workload_analysis_records = []
all_candidate_records = []
index_result_records = []
mv_records = []
partition_records = []
gin_records = []
plan_records = []


# =====================================================================
# 32. PHASE 9A-9C MAIN LOOP
# =====================================================================

stage("PHASE 9A-9C - ANALYSIS, DESIGN AND EVALUATION")


for number, (_, priority_row) in enumerate(
    selected.iterrows(),
    start=1
):

    fingerprint = priority_row[
        "fingerprint_hash"
    ]

    query_name = priority_row[
        "query_name"
    ]

    family = priority_row[
        "query_family"
    ]

    priority_score = safe_float(
        priority_row[
            "optimization_priority_score"
        ]
    )

    priority_label = priority_row[
        "optimization_priority"
    ]

    frequency = int(
        frequency_map.get(
            fingerprint,
            1
        )
    )

    print()
    print("-" * 115)

    print(
        f"[{number}/{len(selected)}] "
        f"{query_name}"
    )

    print("Family:", family)
    print("Frequency:", frequency)

    print(
        "Phase 8 priority:",
        f"{priority_score:.2f}",
        priority_label
    )

    representative = representative_query(
        fingerprint
    )

    if representative is None:

        print("Representative SQL missing")
        continue

    query = clean_sql(
        representative["sql"]
    )

    if not is_read_only(query):

        print("Unsafe SQL skipped")
        continue


    # ================================================================
    # BASELINE
    # ================================================================

    print("Benchmarking baseline...")

    try:

        baseline = benchmark(
            query,
            BASELINE_RUNS
        )

    except Exception as exc:

        print("Baseline failed:", exc)
        continue


    baseline_latency = baseline[
        "latency_ms"
    ]

    baseline_cost = baseline[
        "cost"
    ]

    baseline_io = baseline[
        "io_reads"
    ]

    plan_info = analyze_plan(
        baseline["plan"]
    )


    print(
        "Baseline latency:",
        f"{baseline_latency:.3f} ms"
    )

    print(
        "Baseline cost:",
        f"{baseline_cost:.2f}"
    )

    print(
        "Sequential scans:",
        plan_info["seq_scan_tables"]
    )


    # ================================================================
    # WORKLOAD ANALYSIS
    # ================================================================

    upper_query = query.upper()

    has_aggregate = bool(
        AGGREGATE_PATTERN.search(query)
    )

    has_group = (
        " GROUP BY " in upper_query
    )

    has_order = (
        " ORDER BY " in upper_query
    )

    has_join = (
        " JOIN " in upper_query
    )

    workload_analysis_records.append(
        {
            "fingerprint_hash":
                fingerprint,

            "query_name":
                query_name,

            "query_family":
                family,

            "frequency":
                frequency,

            "priority_score":
                priority_score,

            "priority_label":
                priority_label,

            "baseline_latency_ms":
                baseline_latency,

            "baseline_cost":
                baseline_cost,

            "baseline_io_reads":
                baseline_io,

            "tables":
                ",".join(
                    plan_info["tables"]
                ),

            "seq_scan_tables":
                ",".join(
                    plan_info[
                        "seq_scan_tables"
                    ]
                ),

            "has_join":
                has_join,

            "has_aggregate":
                has_aggregate,

            "has_group_by":
                has_group,

            "has_order_by":
                has_order,
        }
    )


    # ================================================================
    # PHASE 9B - CANDIDATE GENERATION
    # ================================================================

    candidates = []

    candidates.extend(
        generate_btree_candidates(
            query,
            plan_info
        )
    )

    candidates.extend(
        generate_partial_candidates(
            query
        )
    )

    candidates.extend(
        generate_brin_candidates(
            query
        )
    )


    # Remove duplicates
    deduped = []

    seen = set()

    for candidate in candidates:

        key = (
            candidate["design_type"],
            candidate["table"],
            tuple(candidate["columns"]),
            candidate.get(
                "predicate",
                ""
            )
        )

        if key in seen:
            continue

        seen.add(key)

        deduped.append(candidate)

    candidates = deduped[
        :MAX_INDEX_CANDIDATES_PER_PATTERN
    ]


    # ================================================================
    # GIN ADVISOR
    # ================================================================

    gin_candidates = detect_gin_candidates(
        query
    )

    for item in gin_candidates:

        item.update(
            {
                "fingerprint_hash":
                    fingerprint,

                "query_name":
                    query_name,

                "query_family":
                    family,
            }
        )

        gin_records.append(item)


    # ================================================================
    # MATERIALIZED VIEW ADVISOR
    # ================================================================

    mv = detect_materialized_view_candidate(
        query,
        frequency,
        baseline_latency
    )

    if mv:

        mv_name = (
            "mv_ai_"
            + re.sub(
                r"[^A-Za-z0-9_]",
                "_",
                family
            )[:35]
            + "_"
            + fingerprint[:7]
        )

        mv_record = {
            "fingerprint_hash":
                fingerprint,

            "query_name":
                query_name,

            "query_family":
                family,

            "materialized_view_name":
                mv_name,

            "frequency":
                frequency,

            "baseline_latency_ms":
                baseline_latency,

            "baseline_cost":
                baseline_cost,

            "eligibility_score":
                mv[
                    "eligibility_score"
                ],

            "has_group_by":
                mv[
                    "has_group_by"
                ],

            "reason":
                mv[
                    "reason"
                ],

            "status":
                "RECOMMENDED_FOR_PHASE16_EVALUATION",

            "note":
                "The system intentionally does not "
                "rewrite arbitrary analytical SQL into "
                "a materialized-view query automatically. "
                "The candidate is recorded for controlled "
                "evaluation."
        }

        mv_records.append(
            mv_record
        )

        print(
            "Materialized-view candidate:",
            mv_name
        )


    # ================================================================
    # PARTITION ADVISOR
    # ================================================================

    partitions = detect_partition_candidates(
        query
    )

    for item in partitions:

        item.update(
            {
                "fingerprint_hash":
                    fingerprint,

                "query_name":
                    query_name,

                "query_family":
                    family,
            }
        )

        partition_records.append(item)


    # ================================================================
    # PHASE 9C - INDEX EXPERIMENTS
    # ================================================================

    print(
        "Experimental index candidates:",
        len(candidates)
    )


    for candidate_number, candidate in enumerate(
        candidates,
        start=1
    ):

        table = candidate["table"]
        columns = candidate["columns"]
        index_name = candidate["index_name"]

        design_type = candidate[
            "design_type"
        ]

        table_size = table_stats.get(
            table,
            {}
        ).get(
            "size_bytes",
            0
        )

        print()

        print(
            f"  Candidate "
            f"{candidate_number}/"
            f"{len(candidates)}"
        )

        print(
            "  Type:",
            design_type
        )

        print(
            "  Table:",
            table
        )

        print(
            "  Columns:",
            columns
        )

        all_candidate_records.append(
            {
                "fingerprint_hash":
                    fingerprint,

                "query_name":
                    query_name,

                "query_family":
                    family,

                "design_type":
                    design_type,

                "table":
                    table,

                "columns":
                    ",".join(columns),

                "index_name":
                    index_name,

                "reason":
                    candidate[
                        "reason"
                    ],
            }
        )


        created = False

        try:

            create_index(candidate)

            created = True

            size_bytes = index_size(
                index_name
            )

            optimized = benchmark(
                query,
                CANDIDATE_RUNS
            )

            optimized_latency = optimized[
                "latency_ms"
            ]

            optimized_cost = optimized[
                "cost"
            ]

            optimized_io = optimized[
                "io_reads"
            ]


            latency_improvement = pct_improvement(
                baseline_latency,
                optimized_latency
            )

            cost_improvement = pct_improvement(
                baseline_cost,
                optimized_cost
            )

            io_improvement = pct_improvement(
                baseline_io,
                optimized_io
            )


            optimized_plan_info = analyze_plan(
                optimized["plan"]
            )


            used = (
                index_name
                in optimized_plan_info[
                    "indexes_used"
                ]
            )


            utility = calculate_utility(
                latency_improvement,
                io_improvement,
                cost_improvement,
                priority_score,
                frequency,
                size_bytes,
                table_size,
            )


            approved, decision_reason = decision(
                latency_improvement,
                cost_improvement,
                utility,
                used
            )


            status = (
                "APPROVED"
                if approved
                else "REJECTED"
            )


            columns_text = ",".join(
                columns
            )


            if design_type == "BRIN":

                recommendation_sql = (
                    f"CREATE INDEX {index_name} "
                    f"ON {table} USING BRIN "
                    f"({columns_text});"
                )

            elif design_type == "PARTIAL_INDEX":

                recommendation_sql = (
                    f"CREATE INDEX {index_name} "
                    f"ON {table} ({columns_text}) "
                    f"WHERE "
                    f"{candidate['predicate']};"
                )

            elif design_type == "COVERING_INDEX":

                include_text = ",".join(
                    candidate[
                        "include_columns"
                    ]
                )

                recommendation_sql = (
                    f"CREATE INDEX {index_name} "
                    f"ON {table} ({columns_text}) "
                    f"INCLUDE ({include_text});"
                )

            else:

                recommendation_sql = (
                    f"CREATE INDEX {index_name} "
                    f"ON {table} "
                    f"({columns_text});"
                )


            reason = (
                candidate["reason"]
                + " "
                + decision_reason
                + f". Phase8Priority={priority_score:.2f}; "
                + f"Frequency={frequency}; "
                + f"LatencyImprovement="
                + f"{latency_improvement:.2f}%; "
                + f"IOImprovement="
                + f"{io_improvement:.2f}%; "
                + f"CostImprovement="
                + f"{cost_improvement:.2f}%; "
                + f"Utility={utility:.2f}; "
                + f"PlannerUsedCandidate={used}."
            )


            record = {
                "fingerprint_hash":
                    fingerprint,

                "query_name":
                    query_name,

                "query_family":
                    family,

                "priority_score":
                    priority_score,

                "priority_label":
                    priority_label,

                "frequency":
                    frequency,

                "design_type":
                    design_type,

                "table":
                    table,

                "columns_text":
                    columns_text,

                "index_name":
                    index_name,

                "baseline_latency_ms":
                    baseline_latency,

                "candidate_latency_ms":
                    optimized_latency,

                "latency_improvement_percent":
                    latency_improvement,

                "baseline_cost":
                    baseline_cost,

                "candidate_cost":
                    optimized_cost,

                "cost_improvement_percent":
                    cost_improvement,

                "baseline_io_reads":
                    baseline_io,

                "candidate_io_reads":
                    optimized_io,

                "io_improvement_percent":
                    io_improvement,

                "index_size_bytes":
                    size_bytes,

                "candidate_used":
                    used,

                "utility_score":
                    utility,

                "status":
                    status,

                "decision_reason":
                    decision_reason,

                "recommendation_sql":
                    recommendation_sql,

                "reason":
                    reason,
            }


            index_result_records.append(
                record
            )


            plan_records.append(
                {
                    "fingerprint_hash":
                        fingerprint,

                    "query_name":
                        query_name,

                    "query_family":
                        family,

                    "design_type":
                        design_type,

                    "index_name":
                        index_name,

                    "baseline":
                        baseline["raw"],

                    "candidate":
                        optimized["raw"],

                    "latency_improvement_percent":
                        latency_improvement,

                    "cost_improvement_percent":
                        cost_improvement,

                    "io_improvement_percent":
                        io_improvement,

                    "utility_score":
                        utility,

                    "status":
                        status,
                }
            )


            print(
                "  Baseline:",
                f"{baseline_latency:.3f} ms"
            )

            print(
                "  Candidate:",
                f"{optimized_latency:.3f} ms"
            )

            print(
                "  Latency improvement:",
                f"{latency_improvement:.2f}%"
            )

            print(
                "  I/O improvement:",
                f"{io_improvement:.2f}%"
            )

            print(
                "  Cost improvement:",
                f"{cost_improvement:.2f}%"
            )

            print(
                "  PostgreSQL used candidate:",
                used
            )

            print(
                "  Utility score:",
                f"{utility:.2f}"
            )

            print(
                "  RESULT:",
                status
            )


        except Exception as exc:

            print(
                "  Candidate failed:",
                exc
            )

            index_result_records.append(
                {
                    "fingerprint_hash":
                        fingerprint,

                    "query_name":
                        query_name,

                    "query_family":
                        family,

                    "priority_score":
                        priority_score,

                    "priority_label":
                        priority_label,

                    "frequency":
                        frequency,

                    "design_type":
                        design_type,

                    "table":
                        table,

                    "columns_text":
                        ",".join(columns),

                    "index_name":
                        index_name,

                    "baseline_latency_ms":
                        baseline_latency,

                    "candidate_latency_ms":
                        np.nan,

                    "latency_improvement_percent":
                        np.nan,

                    "baseline_cost":
                        baseline_cost,

                    "candidate_cost":
                        np.nan,

                    "cost_improvement_percent":
                        np.nan,

                    "baseline_io_reads":
                        baseline_io,

                    "candidate_io_reads":
                        np.nan,

                    "io_improvement_percent":
                        np.nan,

                    "index_size_bytes":
                        0,

                    "candidate_used":
                        False,

                    "utility_score":
                        0,

                    "status":
                        "FAILED",

                    "decision_reason":
                        str(exc),

                    "recommendation_sql":
                        "",

                    "reason":
                        str(exc),
                }
            )


        finally:

            if created and not KEEP_APPROVED_INDEXES:

                try:
                    drop_index(index_name)

                except Exception as exc:
                    print(
                        "  WARNING cleanup failed:",
                        exc
                    )


# =====================================================================
# 33. RESULT DATAFRAMES
# =====================================================================

stage("PHASE 9D - DECISION AND FEEDBACK")

workload_df = pd.DataFrame(
    workload_analysis_records
)

candidate_df = pd.DataFrame(
    all_candidate_records
)

index_df = pd.DataFrame(
    index_result_records
)

mv_df = pd.DataFrame(
    mv_records
)

partition_df = pd.DataFrame(
    partition_records
)

gin_df = pd.DataFrame(
    gin_records
)


# =====================================================================
# 34. BEST DESIGN PER WORKLOAD
# =====================================================================

if not index_df.empty:

    successful_df = index_df[
        index_df["status"]
        != "FAILED"
    ].copy()

    if not successful_df.empty:

        best_df = (
            successful_df
            .sort_values(
                [
                    "fingerprint_hash",
                    "utility_score",
                    "latency_improvement_percent",
                ],
                ascending=[
                    True,
                    False,
                    False,
                ]
            )
            .groupby(
                "fingerprint_hash",
                as_index=False
            )
            .first()
        )

    else:
        best_df = pd.DataFrame()

else:
    best_df = pd.DataFrame()


# =====================================================================
# 35. APPROVED / REJECTED
# =====================================================================

if not index_df.empty:

    approved_df = index_df[
        index_df["status"]
        == "APPROVED"
    ].copy()

    rejected_df = index_df[
        index_df["status"]
        != "APPROVED"
    ].copy()

else:

    approved_df = pd.DataFrame()
    rejected_df = pd.DataFrame()


# =====================================================================
# 36. PERSIST ONLY BEST APPROVED DESIGN PER PATTERN
# =====================================================================

persisted = 0

if not approved_df.empty:

    best_approved = (
        approved_df
        .sort_values(
            [
                "fingerprint_hash",
                "utility_score",
                "latency_improvement_percent",
            ],
            ascending=[
                True,
                False,
                False,
            ]
        )
        .groupby(
            "fingerprint_hash",
            as_index=False
        )
        .first()
    )

    for _, record in best_approved.iterrows():

        save_recommendation(
            record.to_dict()
        )

        persisted += 1

else:

    best_approved = pd.DataFrame()


# =====================================================================
# 37. FEEDBACK DATASET
# =====================================================================

feedback_columns = [
    "fingerprint_hash",
    "query_name",
    "query_family",
    "priority_score",
    "frequency",
    "design_type",
    "table",
    "columns_text",
    "baseline_latency_ms",
    "candidate_latency_ms",
    "latency_improvement_percent",
    "baseline_cost",
    "candidate_cost",
    "cost_improvement_percent",
    "baseline_io_reads",
    "candidate_io_reads",
    "io_improvement_percent",
    "index_size_bytes",
    "candidate_used",
    "utility_score",
    "status",
]


if not index_df.empty:

    feedback_df = index_df[
        [
            c for c in feedback_columns
            if c in index_df.columns
        ]
    ].copy()

    feedback_df[
        "beneficial"
    ] = (
        feedback_df["status"]
        == "APPROVED"
    ).astype(int)

else:

    feedback_df = pd.DataFrame()


# =====================================================================
# 38. SAVE CSV
# =====================================================================

workload_df.to_csv(
    WORKLOAD_ANALYSIS_FILE,
    index=False
)

candidate_df.to_csv(
    CANDIDATES_FILE,
    index=False
)

index_df.to_csv(
    INDEX_RESULTS_FILE,
    index=False
)

mv_df.to_csv(
    MV_RESULTS_FILE,
    index=False
)

partition_df.to_csv(
    PARTITION_FILE,
    index=False
)

gin_df.to_csv(
    GIN_FILE,
    index=False
)

best_df.to_csv(
    BEST_FILE,
    index=False
)

approved_df.to_csv(
    APPROVED_FILE,
    index=False
)

rejected_df.to_csv(
    REJECTED_FILE,
    index=False
)

feedback_df.to_csv(
    FEEDBACK_FILE,
    index=False
)


with open(
    PLAN_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        plan_records,
        f,
        indent=2,
        default=str
    )


# =====================================================================
# 39. VERIFY CLEANUP
# =====================================================================

stage("VERIFYING PHYSICAL-DESIGN CLEANUP")


with conn.cursor() as cur:

    cur.execute(
        """
        SELECT indexname
        FROM pg_indexes
        WHERE schemaname='public'
          AND indexname LIKE 'ai_p9_%'
        ORDER BY indexname;
        """
    )

    remaining = [
        row[0]
        for row in cur.fetchall()
    ]


if KEEP_APPROVED_INDEXES:

    print(
        "KEEP_APPROVED_INDEXES=True"
    )

    print(
        "Remaining AI indexes:",
        remaining
    )

else:

    if remaining:

        print(
            "WARNING - experimental indexes remain:"
        )

        for name in remaining:
            print(" -", name)

    else:

        print(
            "Experimental index cleanup: PASS"
        )

        print(
            "Original baseline physical design preserved."
        )


# =====================================================================
# 40. SUMMARY
# =====================================================================

stage("PHASE 9 RESULTS SUMMARY")


print(
    "AI patterns evaluated             :",
    len(selected)
)

print(
    "Workload analyses                 :",
    len(workload_df)
)

print(
    "Index candidates generated        :",
    len(candidate_df)
)

print(
    "Index experiments completed       :",
    len(index_df)
)

print(
    "Approved index experiments        :",
    len(approved_df)
)

print(
    "Rejected/failed experiments       :",
    len(rejected_df)
)

print(
    "Materialized-view recommendations :",
    len(mv_df)
)

print(
    "BRIN experiments                  :",
    (
        int(
            (
                index_df["design_type"]
                == "BRIN"
            ).sum()
        )
        if not index_df.empty
        else 0
    )
)

print(
    "GIN recommendations               :",
    len(gin_df)
)

print(
    "Partition recommendations         :",
    len(partition_df)
)

print(
    "Recommendations persisted         :",
    persisted
)


# =====================================================================
# 41. DISPLAY BEST APPROVED RESULTS
# =====================================================================

if not best_approved.empty:

    print()
    print("BEST APPROVED PHYSICAL DESIGNS")
    print("-" * 115)

    columns = [
        "query_name",
        "query_family",
        "design_type",
        "table",
        "columns_text",
        "baseline_latency_ms",
        "candidate_latency_ms",
        "latency_improvement_percent",
        "io_improvement_percent",
        "utility_score",
    ]

    print(
        best_approved[
            columns
        ].sort_values(
            "latency_improvement_percent",
            ascending=False
        ).to_string(
            index=False
        )
    )

else:

    print()
    print(
        "No index candidate passed "
        "the final measured-benefit policy."
    )


# =====================================================================
# 42. TOP MATERIALIZED VIEW OPPORTUNITIES
# =====================================================================

if not mv_df.empty:

    print()
    print("MATERIALIZED VIEW OPPORTUNITIES")
    print("-" * 115)

    print(
        mv_df[
            [
                "query_name",
                "query_family",
                "frequency",
                "baseline_latency_ms",
                "eligibility_score",
            ]
        ]
        .sort_values(
            "eligibility_score",
            ascending=False
        )
        .to_string(
            index=False
        )
    )


# =====================================================================
# 43. METADATA
# =====================================================================

metadata = {

    "phase":
        9,

    "phase_name":
        "Self-Optimizing Database",

    "generated_at":
        datetime.now().isoformat(),

    "phase9_components": {
        "9A":
            "Workload and execution-plan analysis",

        "9B":
            "Workload-aware physical-design advisor",

        "9C":
            "Controlled experimental evaluation",

        "9D":
            "Decision, persistence and feedback",
    },

    "patterns_evaluated":
        len(selected),

    "index_candidates":
        len(candidate_df),

    "index_experiments":
        len(index_df),

    "approved_index_experiments":
        len(approved_df),

    "materialized_view_recommendations":
        len(mv_df),

    "gin_recommendations":
        len(gin_df),

    "partition_recommendations":
        len(partition_df),

    "recommendations_persisted":
        persisted,

    "baseline_runs":
        BASELINE_RUNS,

    "candidate_runs":
        CANDIDATE_RUNS,

    "acceptance_policy": {
        "minimum_latency_improvement":
            MIN_LATENCY_IMPROVEMENT,

        "strong_latency_improvement":
            STRONG_LATENCY_IMPROVEMENT,

        "minimum_cost_improvement_support":
            MIN_COST_IMPROVEMENT,

        "minimum_utility_score":
            MIN_UTILITY_SCORE,

        "maximum_regression":
            MAX_REGRESSION,
    },

    "baseline_preserved":
        (
            not KEEP_APPROVED_INDEXES
            and len(remaining) == 0
        ),

    "important_methodological_note":
        (
            "Measured execution-time improvement is "
            "the primary optimization signal. "
            "PostgreSQL estimated cost is supporting "
            "evidence and is not used as a mandatory "
            "veto."
        ),

    "materialized_view_policy":
        (
            "Repeated analytical aggregations are "
            "identified as materialized-view "
            "opportunities. Generic automatic SQL "
            "rewriting is intentionally avoided."
        ),

    "partition_policy":
        (
            "Partitioning is advisory only because "
            "in-place repartitioning would alter the "
            "frozen experimental baseline."
        ),
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


# =====================================================================
# 44. REPORT
# =====================================================================

best_improvement = 0.0
best_query = "None"
best_design = "None"

if not approved_df.empty:

    temp = approved_df.sort_values(
        "latency_improvement_percent",
        ascending=False
    ).iloc[0]

    best_improvement = safe_float(
        temp[
            "latency_improvement_percent"
        ]
    )

    best_query = temp[
        "query_name"
    ]

    best_design = (
        temp["design_type"]
        + " on "
        + temp["table"]
        + "("
        + temp["columns_text"]
        + ")"
    )


report = f"""
PHASE 9 - SELF-OPTIMIZING DATABASE
=======================================================================

Project
-------
Adaptive Healthcare Knowledge Database


PHASE 9 ARCHITECTURE
-----------------------------------------------------------------------

9A - Workload and Execution-Plan Analysis

Phase 7 monitored SQL and Phase 8 AI priorities are combined with
PostgreSQL EXPLAIN ANALYZE information.

The system identifies:
- expensive queries
- frequent workload patterns
- joins
- sequential scans
- aggregations
- temporal access
- filtering columns
- grouping columns
- ordering columns


9B - PHYSICAL DESIGN ADVISOR
-----------------------------------------------------------------------

The advisor evaluates or recommends:

1. B-tree indexes
2. Composite indexes
3. Covering indexes
4. Partial indexes
5. BRIN indexes
6. GIN indexes where eligible
7. Materialized views for repeated analytical aggregation
8. Partitioning opportunities for temporal large-table workloads


9C - EXPERIMENTAL EVALUATION
-----------------------------------------------------------------------

Each executable index candidate is:

1. Created temporarily.
2. Followed by ANALYZE.
3. Tested using repeated EXPLAIN ANALYZE.
4. Compared with the baseline.
5. Evaluated for:
   - execution latency
   - PostgreSQL estimated cost
   - I/O reads
   - physical storage overhead
   - workload frequency
   - Phase 8 AI priority
6. Assigned a utility score.
7. Approved or rejected.
8. Removed after the experiment unless permanent application is
   explicitly enabled.


9D - FEEDBACK
-----------------------------------------------------------------------

Successful recommendations are persisted to
optimization_recommendation.

All tested candidates are written to the Phase 9 feedback dataset.

The feedback dataset contains the observed outcome of each physical
design and can later be used to improve the workload learner.


RESULTS
-----------------------------------------------------------------------

AI patterns evaluated:
{len(selected)}

Index candidates generated:
{len(candidate_df)}

Index experiments:
{len(index_df)}

Approved index experiments:
{len(approved_df)}

Rejected / failed experiments:
{len(rejected_df)}

Materialized-view opportunities:
{len(mv_df)}

GIN opportunities:
{len(gin_df)}

Partition recommendations:
{len(partition_df)}

Recommendations persisted:
{persisted}


BEST MEASURED OPTIMIZATION
-----------------------------------------------------------------------

Query:
{best_query}

Design:
{best_design}

Measured latency improvement:
{best_improvement:.2f}%


DECISION POLICY
-----------------------------------------------------------------------

The final decision policy does NOT reject a strongly beneficial
physical design merely because PostgreSQL's estimated planner cost
fails an arbitrary percentage threshold.

Measured latency is the primary experimental signal.

Planner cost, I/O improvement, workload frequency, Phase 8 AI
priority and storage overhead provide supporting evidence.

This addresses the limitation observed during the Phase 9A pilot,
where an index reducing the condition-medication query latency by
approximately 51% was rejected because estimated cost improved by
only approximately 3.8%.


MATERIALIZED VIEW POLICY
-----------------------------------------------------------------------

Repeated aggregation workloads are detected automatically.

The system intentionally does not perform arbitrary automatic SQL
rewriting into materialized views because such rewriting must preserve
query semantics.

Instead, qualifying workloads are recorded as materialized-view
recommendations for controlled evaluation.


PARTITIONING POLICY
-----------------------------------------------------------------------

Partition recommendations are generated for sufficiently large
temporal workloads.

The existing PostgreSQL tables are not automatically repartitioned.

This protects the frozen research baseline and avoids destructive
schema migration during optimization experiments.


RESEARCH INTERPRETATION
-----------------------------------------------------------------------

Phase 9 implements the closed optimization loop:

Historical workload
        ->
Phase 8 AI workload learning
        ->
Optimization priority
        ->
Query/plan analysis
        ->
Physical-design candidates
        ->
Real PostgreSQL experiment
        ->
Measured benefit
        ->
Utility/safety decision
        ->
Approve/reject
        ->
Optimization feedback

This provides a stronger self-optimization methodology than simply
creating indexes for every sequential scan.


STATUS
-----------------------------------------------------------------------

PHASE 9A - Workload analysis              COMPLETE
PHASE 9B - Physical design advisor        COMPLETE
PHASE 9C - Experimental evaluation        COMPLETE
PHASE 9D - Feedback loop                  COMPLETE

PHASE 9 STATUS: COMPLETE
"""


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        report.strip()
    )


# =====================================================================
# 45. FINAL STATUS
# =====================================================================

stage("PHASE 9 FINAL RESULT")


print(
    "Patterns evaluated               :",
    len(selected)
)

print(
    "Index candidates generated       :",
    len(candidate_df)
)

print(
    "Index experiments                :",
    len(index_df)
)

print(
    "Approved index experiments       :",
    len(approved_df)
)

print(
    "Materialized-view opportunities  :",
    len(mv_df)
)

print(
    "GIN opportunities                :",
    len(gin_df)
)

print(
    "Partition recommendations        :",
    len(partition_df)
)

print(
    "Feedback samples generated       :",
    len(feedback_df)
)

print(
    "Recommendations persisted        :",
    persisted
)

print(
    "Experimental cleanup             :",
    (
        "PASS"
        if (
            KEEP_APPROVED_INDEXES
            or len(remaining) == 0
        )
        else "REVIEW"
    )
)


if not approved_df.empty:

    print()

    print(
        "Best measured query             :",
        best_query
    )

    print(
        "Best physical design            :",
        best_design
    )

    print(
        "Best latency improvement        :",
        f"{best_improvement:.2f}%"
    )


print()
print("Results directory:")
print(RESULTS_DIR)

print()
print("Generated files:")

for path in sorted(
    RESULTS_DIR.iterdir()
):
    print(" -", path.name)


print()
print("=" * 115)
print("PHASE 9 STATUS: COMPLETE")
print("=" * 115)

print("9A Workload/plan analysis          COMPLETE")
print("9B Physical-design advisor         COMPLETE")
print("9C Experimental evaluation         COMPLETE")
print("9D Recommendation/feedback         COMPLETE")

print()
print("NEXT:")
print("PHASE 10 -> BIOMEDICAL KNOWLEDGE GRAPH")


# =====================================================================
# 46. CLOSE
# =====================================================================

conn.close()