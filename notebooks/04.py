from pathlib import Path
import os
import re
import json
import time
import hashlib
import getpass
from datetime import datetime

import pandas as pd
import psycopg


# ============================================================
# PHASE 7 - QUERY MONITORING
# Adaptive Healthcare Knowledge Database
# ============================================================

print("=" * 100)
print("PHASE 7 - QUERY MONITORING")
print("Adaptive Healthcare Knowledge Database")
print("=" * 100)


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PHASE6_DIR = (
    PROJECT_ROOT /
    "results" /
    "phase6_workloads"
)

RESULTS_DIR = (
    PROJECT_ROOT /
    "results" /
    "phase7_monitoring"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

WORKLOAD_FILE = (
    PHASE6_DIR /
    "combined_workload.csv"
)

DB_NAME = "adaptive_healthcare_db"
DB_HOST = os.getenv("PGHOST", "localhost")
DB_PORT = int(os.getenv("PGPORT", "5432"))
DB_USER = os.getenv("PGUSER", "postgres")
DB_PASSWORD = os.getenv("PGPASSWORD")


# Maximum time allowed for ONE query.
# 60 seconds = 60000 milliseconds.
STATEMENT_TIMEOUT_MS = 60000


# Commit query_log rows in batches.
COMMIT_EVERY = 25


# None means execute all 950 workload instances.
MAX_QUERIES = None


# Do NOT automatically delete old query_log rows.
# Existing data is preserved.
RESET_QUERY_LOG = False


# Output files
MONITORING_CSV = (
    RESULTS_DIR /
    "phase7_query_monitoring.csv"
)

ERROR_CSV = (
    RESULTS_DIR /
    "phase7_query_errors.csv"
)

FREQUENCY_CSV = (
    RESULTS_DIR /
    "phase7_query_frequency.csv"
)

FAMILY_SUMMARY_CSV = (
    RESULTS_DIR /
    "phase7_family_summary.csv"
)

WORKLOAD_SUMMARY_CSV = (
    RESULTS_DIR /
    "phase7_workload_summary.csv"
)

INDEX_SUMMARY_CSV = (
    RESULTS_DIR /
    "phase7_index_usage_summary.csv"
)

TABLE_SUMMARY_CSV = (
    RESULTS_DIR /
    "phase7_table_access_summary.csv"
)

METADATA_JSON = (
    RESULTS_DIR /
    "phase7_metadata.json"
)

REPORT_TXT = (
    RESULTS_DIR /
    "phase7_report.txt"
)


# ============================================================
# 2. BASIC HELPERS
# ============================================================

def stage(title):

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def clean_sql(sql):

    return re.sub(
        r"\s+",
        " ",
        str(sql)
    ).strip()


def sha256_text(text):

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def safe_int(value, default=0):

    try:
        return int(value)

    except Exception:
        return default


def safe_float(value, default=0.0):

    try:
        return float(value)

    except Exception:
        return default


# ============================================================
# 3. QUERY FINGERPRINTING
# ============================================================

def create_fingerprint(sql):

    sql = clean_sql(sql)

    # UUID values
    sql = re.sub(
        r"'[0-9a-fA-F]{8}-"
        r"[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{12}'",
        "?",
        sql
    )

    # Other quoted string literals
    sql = re.sub(
        r"'(?:''|[^'])*'",
        "?",
        sql
    )

    # Numeric literals
    sql = re.sub(
        r"\b\d+(?:\.\d+)?\b",
        "?",
        sql
    )

    # Normalize whitespace again
    sql = clean_sql(sql)

    # Normalize case
    sql = sql.lower()

    return sql


# ============================================================
# 4. QUERY TYPE
# ============================================================

def get_query_type(sql):

    sql = clean_sql(sql).upper()

    if sql.startswith("SELECT"):
        return "SELECT"

    if sql.startswith("WITH"):
        return "SELECT"

    if sql.startswith("INSERT"):
        return "INSERT"

    if sql.startswith("UPDATE"):
        return "UPDATE"

    if sql.startswith("DELETE"):
        return "DELETE"

    return "OTHER"


# ============================================================
# 5. READ-ONLY SAFETY
# ============================================================

def validate_read_only_query(sql):

    cleaned = clean_sql(sql)

    upper = cleaned.upper()

    if not (
        upper.startswith("SELECT")
        or
        upper.startswith("WITH")
    ):
        return False

    forbidden = [
        " INSERT ",
        " UPDATE ",
        " DELETE ",
        " DROP ",
        " ALTER ",
        " CREATE ",
        " TRUNCATE ",
        " GRANT ",
        " REVOKE ",
        " VACUUM ",
        " ANALYZE ",
        " CALL ",
        " COPY "
    ]

    padded = " " + upper + " "

    for keyword in forbidden:

        if keyword in padded:
            return False

    return True


# ============================================================
# 6. DATABASE SCHEMA INFORMATION
# ============================================================

def load_schema_columns(conn):

    schema = {}

    with conn.cursor() as cur:

        cur.execute(
            """
            SELECT
                table_name,
                column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
            ORDER BY
                table_name,
                ordinal_position
            """
        )

        for table_name, column_name in cur.fetchall():

            schema.setdefault(
                table_name,
                set()
            ).add(
                column_name
            )

    return schema


# ============================================================
# 7. SQL TABLE/COLUMN EXTRACTION
# ============================================================

def extract_aliases(sql):

    aliases = {}

    pattern = re.compile(
        r"\b(?:FROM|JOIN)\s+"
        r"([a-zA-Z_][a-zA-Z0-9_]*)"
        r"(?:\s+(?:AS\s+)?"
        r"([a-zA-Z_][a-zA-Z0-9_]*))?",
        re.IGNORECASE
    )

    reserved = {
        "where",
        "join",
        "left",
        "right",
        "inner",
        "outer",
        "full",
        "cross",
        "group",
        "order",
        "limit",
        "having",
        "on",
        "union"
    }

    for match in pattern.finditer(sql):

        table = match.group(1)

        alias = match.group(2)

        aliases[table] = table

        if (
            alias
            and
            alias.lower() not in reserved
        ):
            aliases[alias] = table

    return aliases


def extract_columns(sql, schema):

    aliases = extract_aliases(sql)

    columns = set()

    # alias.column references
    pattern = re.compile(
        r"\b([a-zA-Z_][a-zA-Z0-9_]*)"
        r"\."
        r"([a-zA-Z_][a-zA-Z0-9_]*)\b"
    )

    for alias, column in pattern.findall(sql):

        table = aliases.get(alias)

        if (
            table
            and
            table in schema
            and
            column in schema[table]
        ):

            columns.add(
                f"{table}.{column}"
            )

    return sorted(columns)


# ============================================================
# 8. PLAN TREE WALKER
# ============================================================

def walk_plan(node):

    yield node

    for child in node.get(
        "Plans",
        []
    ):

        yield from walk_plan(child)


# ============================================================
# 9. PLAN TABLE EXTRACTION
# ============================================================

def get_plan_tables(root):

    tables = set()

    for node in walk_plan(root):

        relation = node.get(
            "Relation Name"
        )

        if relation:
            tables.add(relation)

    return sorted(tables)


# ============================================================
# 10. JOIN COUNT
# ============================================================

def count_joins(root):

    join_types = {
        "Nested Loop",
        "Hash Join",
        "Merge Join"
    }

    count = 0

    for node in walk_plan(root):

        node_type = node.get(
            "Node Type",
            ""
        )

        if node_type in join_types:
            count += 1

    return count


# ============================================================
# 11. FILTER / CONDITION COUNT
# ============================================================

def count_filters(root):

    condition_keys = {
        "Filter",
        "Index Cond",
        "Recheck Cond",
        "Hash Cond",
        "Merge Cond",
        "Join Filter",
        "TID Cond"
    }

    count = 0

    for node in walk_plan(root):

        for key in condition_keys:

            if node.get(key):
                count += 1

    return count


# ============================================================
# 12. INDEX USAGE
# ============================================================

def get_index_information(root):

    indexes = set()

    index_used = False

    index_scan_count = 0

    seq_scan_count = 0

    bitmap_scan_count = 0


    for node in walk_plan(root):

        node_type = node.get(
            "Node Type",
            ""
        )

        index_name = node.get(
            "Index Name"
        )

        if index_name:

            indexes.add(
                index_name
            )


        if node_type in {
            "Index Scan",
            "Index Only Scan",
            "Bitmap Index Scan"
        }:

            index_used = True
            index_scan_count += 1


        if node_type == "Seq Scan":

            seq_scan_count += 1


        if node_type in {
            "Bitmap Heap Scan",
            "Bitmap Index Scan"
        }:

            bitmap_scan_count += 1


    return {
        "index_used":
            index_used,

        "indexes":
            sorted(indexes),

        "index_scan_count":
            index_scan_count,

        "seq_scan_count":
            seq_scan_count,

        "bitmap_scan_count":
            bitmap_scan_count
    }


# ============================================================
# 13. BUFFER / I-O METRICS
# ============================================================

def get_io_metrics(root):

    shared_hit = safe_int(
        root.get(
            "Shared Hit Blocks",
            0
        )
    )

    shared_read = safe_int(
        root.get(
            "Shared Read Blocks",
            0
        )
    )

    shared_dirtied = safe_int(
        root.get(
            "Shared Dirtied Blocks",
            0
        )
    )

    shared_written = safe_int(
        root.get(
            "Shared Written Blocks",
            0
        )
    )

    local_hit = safe_int(
        root.get(
            "Local Hit Blocks",
            0
        )
    )

    local_read = safe_int(
        root.get(
            "Local Read Blocks",
            0
        )
    )

    local_dirtied = safe_int(
        root.get(
            "Local Dirtied Blocks",
            0
        )
    )

    local_written = safe_int(
        root.get(
            "Local Written Blocks",
            0
        )
    )

    temp_read = safe_int(
        root.get(
            "Temp Read Blocks",
            0
        )
    )

    temp_written = safe_int(
        root.get(
            "Temp Written Blocks",
            0
        )
    )

    io_read_blocks = (
        shared_read +
        local_read +
        temp_read
    )

    io_write_blocks = (
        shared_written +
        local_written +
        temp_written
    )

    return {
        "shared_hit_blocks":
            shared_hit,

        "shared_read_blocks":
            shared_read,

        "shared_dirtied_blocks":
            shared_dirtied,

        "shared_written_blocks":
            shared_written,

        "local_hit_blocks":
            local_hit,

        "local_read_blocks":
            local_read,

        "local_dirtied_blocks":
            local_dirtied,

        "local_written_blocks":
            local_written,

        "temp_read_blocks":
            temp_read,

        "temp_written_blocks":
            temp_written,

        "io_read_blocks":
            io_read_blocks,

        "io_write_blocks":
            io_write_blocks
    }


# ============================================================
# 14. PLAN NODE STATISTICS
# ============================================================

def get_plan_node_stats(root):

    nodes = list(
        walk_plan(root)
    )

    node_types = {}

    for node in nodes:

        node_type = node.get(
            "Node Type",
            "Unknown"
        )

        node_types[node_type] = (
            node_types.get(
                node_type,
                0
            )
            + 1
        )

    return {
        "plan_node_count":
            len(nodes),

        "plan_node_types":
            node_types
    }


# ============================================================
# 15. QUERY_LOG INSERT
# ============================================================

def insert_query_log(
    cur,
    query_hash_value,
    fingerprint,
    sql,
    query_type,
    tables,
    columns,
    join_count,
    filter_count,
    execution_time_ms,
    estimated_cost,
    actual_rows,
    index_used,
    io_read_blocks,
    io_write_blocks
):

    cur.execute(
        """
        INSERT INTO query_log
        (
            query_hash,
            query_fingerprint,
            query_text,
            query_type,
            tables_accessed,
            columns_accessed,
            join_count,
            filter_count,
            execution_time_ms,
            estimated_cost,
            actual_rows,
            index_used,
            cpu_usage,
            io_read_blocks,
            io_write_blocks,
            executed_at
        )
        VALUES
        (
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            NULL,
            %s,
            %s,
            CURRENT_TIMESTAMP
        )
        RETURNING query_log_id
        """,
        (
            query_hash_value,
            fingerprint,
            sql,
            query_type,
            ", ".join(tables),
            ", ".join(columns),
            join_count,
            filter_count,
            execution_time_ms,
            estimated_cost,
            actual_rows,
            index_used,
            io_read_blocks,
            io_write_blocks
        )
    )

    return cur.fetchone()[0]


# ============================================================
# 16. RUN ONE MONITORED QUERY
# ============================================================

def monitor_query(
    conn,
    sql,
    schema
):

    sql = clean_sql(sql)

    if not validate_read_only_query(sql):

        raise ValueError(
            "Query rejected by Phase 7 "
            "read-only safety check."
        )


    fingerprint = create_fingerprint(
        sql
    )

    raw_hash = sha256_text(
        sql
    )

    fingerprint_hash = sha256_text(
        fingerprint
    )

    query_type = get_query_type(
        sql
    )

    columns = extract_columns(
        sql,
        schema
    )


    with conn.cursor() as cur:

        # Local timeout applies only to this transaction.
        cur.execute(
            f"SET LOCAL statement_timeout = "
            f"{STATEMENT_TIMEOUT_MS}"
        )

        start_wall = time.perf_counter()

        cur.execute(
            """
            EXPLAIN
            (
                ANALYZE TRUE,
                BUFFERS TRUE,
                FORMAT JSON
            )
            """
            + sql
        )

        row = cur.fetchone()

        wall_time_ms = (
            time.perf_counter()
            - start_wall
        ) * 1000.0


    payload = row[0]

    if isinstance(
        payload,
        str
    ):
        payload = json.loads(
            payload
        )


    if not isinstance(
        payload,
        list
    ):

        raise RuntimeError(
            "Unexpected PostgreSQL "
            "EXPLAIN JSON format."
        )


    explain = payload[0]

    root = explain["Plan"]


    execution_time_ms = safe_float(
        explain.get(
            "Execution Time",
            wall_time_ms
        )
    )

    planning_time_ms = safe_float(
        explain.get(
            "Planning Time",
            0.0
        )
    )

    estimated_cost = safe_float(
        root.get(
            "Total Cost",
            0.0
        )
    )

    startup_cost = safe_float(
        root.get(
            "Startup Cost",
            0.0
        )
    )

    actual_rows = safe_int(
        root.get(
            "Actual Rows",
            0
        )
    )

    plan_rows = safe_int(
        root.get(
            "Plan Rows",
            0
        )
    )

    actual_startup_time = safe_float(
        root.get(
            "Actual Startup Time",
            0.0
        )
    )

    actual_total_time = safe_float(
        root.get(
            "Actual Total Time",
            execution_time_ms
        )
    )


    tables = get_plan_tables(
        root
    )

    join_count = count_joins(
        root
    )

    filter_count = count_filters(
        root
    )

    index_info = get_index_information(
        root
    )

    io_info = get_io_metrics(
        root
    )

    node_info = get_plan_node_stats(
        root
    )


    return {
        "query_hash":
            raw_hash,

        "fingerprint_hash":
            fingerprint_hash,

        "query_fingerprint":
            fingerprint,

        "query_type":
            query_type,

        "tables_accessed":
            tables,

        "columns_accessed":
            columns,

        "join_count":
            join_count,

        "filter_count":
            filter_count,

        "execution_time_ms":
            execution_time_ms,

        "planning_time_ms":
            planning_time_ms,

        "wall_time_ms":
            wall_time_ms,

        "estimated_cost":
            estimated_cost,

        "startup_cost":
            startup_cost,

        "actual_rows":
            actual_rows,

        "estimated_rows":
            plan_rows,

        "actual_startup_time_ms":
            actual_startup_time,

        "actual_total_time_ms":
            actual_total_time,

        "index_used":
            index_info[
                "index_used"
            ],

        "indexes_used":
            index_info[
                "indexes"
            ],

        "index_scan_count":
            index_info[
                "index_scan_count"
            ],

        "seq_scan_count":
            index_info[
                "seq_scan_count"
            ],

        "bitmap_scan_count":
            index_info[
                "bitmap_scan_count"
            ],

        "shared_hit_blocks":
            io_info[
                "shared_hit_blocks"
            ],

        "shared_read_blocks":
            io_info[
                "shared_read_blocks"
            ],

        "shared_dirtied_blocks":
            io_info[
                "shared_dirtied_blocks"
            ],

        "shared_written_blocks":
            io_info[
                "shared_written_blocks"
            ],

        "local_hit_blocks":
            io_info[
                "local_hit_blocks"
            ],

        "local_read_blocks":
            io_info[
                "local_read_blocks"
            ],

        "temp_read_blocks":
            io_info[
                "temp_read_blocks"
            ],

        "temp_written_blocks":
            io_info[
                "temp_written_blocks"
            ],

        "io_read_blocks":
            io_info[
                "io_read_blocks"
            ],

        "io_write_blocks":
            io_info[
                "io_write_blocks"
            ],

        "plan_node_count":
            node_info[
                "plan_node_count"
            ],

        "plan_node_types":
            node_info[
                "plan_node_types"
            ]
    }


# ============================================================
# 17. CHECK PHASE 6 INPUT
# ============================================================

stage("CHECKING PHASE 6 WORKLOAD")


if not WORKLOAD_FILE.exists():

    raise FileNotFoundError(
        "Phase 6 workload file not found:\n"
        + str(WORKLOAD_FILE)
    )


workload_df = pd.read_csv(
    WORKLOAD_FILE
)


required_columns = {
    "workload_type",
    "query_family",
    "query_name",
    "complexity",
    "expected_use",
    "query_hash",
    "sql"
}


missing_columns = (
    required_columns -
    set(workload_df.columns)
)


if missing_columns:

    raise RuntimeError(
        "Phase 6 workload is missing columns: "
        + str(
            sorted(
                missing_columns
            )
        )
    )


print(
    "Phase 6 workload found."
)

print(
    "Workload rows:",
    f"{len(workload_df):,}"
)

print(
    "Workload types:"
)

print(
    workload_df[
        "workload_type"
    ]
    .value_counts()
    .to_string()
)


if MAX_QUERIES is not None:

    workload_df = (
        workload_df
        .head(
            MAX_QUERIES
        )
        .copy()
    )


print()
print(
    "Queries selected for monitoring:",
    f"{len(workload_df):,}"
)


# ============================================================
# 18. DATABASE CONNECTION
# ============================================================

stage("DATABASE CONNECTION")


if not DB_PASSWORD:

    DB_PASSWORD = getpass.getpass(
        f"PostgreSQL password for '{DB_USER}': "
    )


conn = psycopg.connect(
    host=DB_HOST,
    port=DB_PORT,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD
)


print(
    "Connected to PostgreSQL."
)

print(
    "Database:",
    DB_NAME
)


# ============================================================
# 19. VERIFY QUERY_LOG TABLE
# ============================================================

stage("VERIFYING QUERY_LOG TABLE")


expected_query_log_columns = {
    "query_log_id",
    "query_hash",
    "query_fingerprint",
    "query_text",
    "query_type",
    "tables_accessed",
    "columns_accessed",
    "join_count",
    "filter_count",
    "execution_time_ms",
    "estimated_cost",
    "actual_rows",
    "index_used",
    "cpu_usage",
    "io_read_blocks",
    "io_write_blocks",
    "executed_at"
}


actual_query_log_columns = (
    load_schema_columns(
        conn
    ).get(
        "query_log",
        set()
    )
)


missing_log_columns = (
    expected_query_log_columns -
    actual_query_log_columns
)


if missing_log_columns:

    conn.close()

    raise RuntimeError(
        "query_log schema mismatch. "
        "Missing columns: "
        + str(
            sorted(
                missing_log_columns
            )
        )
    )


print(
    "query_log schema validation PASS"
)

print(
    "Columns:",
    len(actual_query_log_columns)
)


# ============================================================
# 20. CURRENT QUERY_LOG STATE
# ============================================================

with conn.cursor() as cur:

    cur.execute(
        """
        SELECT COUNT(*)
        FROM query_log
        """
    )

    existing_log_count = (
        cur.fetchone()[0]
    )


print(
    "Existing query_log rows:",
    f"{existing_log_count:,}"
)


if RESET_QUERY_LOG:

    stage("RESETTING QUERY_LOG")

    print(
        "RESET_QUERY_LOG=True"
    )

    print(
        "Deleting existing Phase 7 "
        "query monitoring rows..."
    )

    with conn.cursor() as cur:

        cur.execute(
            """
            DELETE FROM query_log
            """
        )

    conn.commit()

    print(
        "query_log cleared."
    )


# ============================================================
# 21. LOAD SCHEMA
# ============================================================

stage("LOADING DATABASE SCHEMA")


schema = load_schema_columns(
    conn
)


print(
    "Tables discovered:",
    len(schema)
)


# ============================================================
# 22. EXECUTE AND MONITOR WORKLOAD
# ============================================================

stage("EXECUTING PHASE 6 WORKLOAD")


print(
    "This phase uses:"
)

print(
    "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)"
)

print()

print(
    "Statement timeout:",
    f"{STATEMENT_TIMEOUT_MS / 1000:.0f} seconds/query"
)

print(
    "Total queries:",
    f"{len(workload_df):,}"
)

print()


monitoring_records = []
error_records = []

success_count = 0
failure_count = 0
timeout_count = 0

phase_start = time.perf_counter()


for position, (
    dataframe_index,
    workload_row
) in enumerate(
    workload_df.iterrows(),
    start=1
):

    sql = clean_sql(
        workload_row["sql"]
    )

    workload_type = str(
        workload_row[
            "workload_type"
        ]
    )

    query_family = str(
        workload_row[
            "query_family"
        ]
    )

    query_name = str(
        workload_row[
            "query_name"
        ]
    )

    complexity = str(
        workload_row[
            "complexity"
        ]
    )

    expected_use = str(
        workload_row[
            "expected_use"
        ]
    )


    print(
        f"[{position:>4}/{len(workload_df)}] "
        f"{workload_type:<10} | "
        f"{complexity:<6} | "
        f"{query_name[:45]}"
    )


    query_start = time.perf_counter()


    try:

        metrics = monitor_query(
            conn,
            sql,
            schema
        )


        with conn.cursor() as cur:

            query_log_id = insert_query_log(
                cur=cur,

                query_hash_value=
                    metrics[
                        "query_hash"
                    ],

                fingerprint=
                    metrics[
                        "query_fingerprint"
                    ],

                sql=sql,

                query_type=
                    metrics[
                        "query_type"
                    ],

                tables=
                    metrics[
                        "tables_accessed"
                    ],

                columns=
                    metrics[
                        "columns_accessed"
                    ],

                join_count=
                    metrics[
                        "join_count"
                    ],

                filter_count=
                    metrics[
                        "filter_count"
                    ],

                execution_time_ms=
                    metrics[
                        "execution_time_ms"
                    ],

                estimated_cost=
                    metrics[
                        "estimated_cost"
                    ],

                actual_rows=
                    metrics[
                        "actual_rows"
                    ],

                index_used=
                    metrics[
                        "index_used"
                    ],

                io_read_blocks=
                    metrics[
                        "io_read_blocks"
                    ],

                io_write_blocks=
                    metrics[
                        "io_write_blocks"
                    ]
            )


        record = {

            "workload_position":
                position,

            "phase6_row":
                int(dataframe_index),

            "query_log_id":
                query_log_id,

            "workload_type":
                workload_type,

            "query_family":
                query_family,

            "query_name":
                query_name,

            "complexity":
                complexity,

            "expected_use":
                expected_use,

            "query_hash":
                metrics[
                    "query_hash"
                ],

            "fingerprint_hash":
                metrics[
                    "fingerprint_hash"
                ],

            "query_fingerprint":
                metrics[
                    "query_fingerprint"
                ],

            "query_type":
                metrics[
                    "query_type"
                ],

            "tables_accessed":
                ", ".join(
                    metrics[
                        "tables_accessed"
                    ]
                ),

            "columns_accessed":
                ", ".join(
                    metrics[
                        "columns_accessed"
                    ]
                ),

            "join_count":
                metrics[
                    "join_count"
                ],

            "filter_count":
                metrics[
                    "filter_count"
                ],

            "execution_time_ms":
                metrics[
                    "execution_time_ms"
                ],

            "planning_time_ms":
                metrics[
                    "planning_time_ms"
                ],

            "wall_time_ms":
                metrics[
                    "wall_time_ms"
                ],

            "estimated_cost":
                metrics[
                    "estimated_cost"
                ],

            "startup_cost":
                metrics[
                    "startup_cost"
                ],

            "estimated_rows":
                metrics[
                    "estimated_rows"
                ],

            "actual_rows":
                metrics[
                    "actual_rows"
                ],

            "actual_startup_time_ms":
                metrics[
                    "actual_startup_time_ms"
                ],

            "actual_total_time_ms":
                metrics[
                    "actual_total_time_ms"
                ],

            "index_used":
                metrics[
                    "index_used"
                ],

            "indexes_used":
                ", ".join(
                    metrics[
                        "indexes_used"
                    ]
                ),

            "index_scan_count":
                metrics[
                    "index_scan_count"
                ],

            "seq_scan_count":
                metrics[
                    "seq_scan_count"
                ],

            "bitmap_scan_count":
                metrics[
                    "bitmap_scan_count"
                ],

            "shared_hit_blocks":
                metrics[
                    "shared_hit_blocks"
                ],

            "shared_read_blocks":
                metrics[
                    "shared_read_blocks"
                ],

            "shared_dirtied_blocks":
                metrics[
                    "shared_dirtied_blocks"
                ],

            "shared_written_blocks":
                metrics[
                    "shared_written_blocks"
                ],

            "local_hit_blocks":
                metrics[
                    "local_hit_blocks"
                ],

            "local_read_blocks":
                metrics[
                    "local_read_blocks"
                ],

            "temp_read_blocks":
                metrics[
                    "temp_read_blocks"
                ],

            "temp_written_blocks":
                metrics[
                    "temp_written_blocks"
                ],

            "io_read_blocks":
                metrics[
                    "io_read_blocks"
                ],

            "io_write_blocks":
                metrics[
                    "io_write_blocks"
                ],

            "plan_node_count":
                metrics[
                    "plan_node_count"
                ],

            "plan_node_types":
                json.dumps(
                    metrics[
                        "plan_node_types"
                    ],
                    sort_keys=True
                ),

            "cpu_usage":
                None,

            "status":
                "SUCCESS",

            "monitored_at":
                datetime.now().isoformat(),

            "sql":
                sql
        }


        monitoring_records.append(
            record
        )

        success_count += 1


        print(
            "      PASS | "
            f"{metrics['execution_time_ms']:.3f} ms | "
            f"cost={metrics['estimated_cost']:.2f} | "
            f"rows={metrics['actual_rows']:,} | "
            f"index={metrics['index_used']}"
        )


    except Exception as exc:

        conn.rollback()

        failure_count += 1

        error_text = str(exc)

        is_timeout = (
            "statement timeout"
            in error_text.lower()
            or
            "query canceled"
            in error_text.lower()
        )


        if is_timeout:
            timeout_count += 1


        elapsed_ms = (
            time.perf_counter()
            - query_start
        ) * 1000.0


        error_record = {

            "workload_position":
                position,

            "phase6_row":
                int(dataframe_index),

            "workload_type":
                workload_type,

            "query_family":
                query_family,

            "query_name":
                query_name,

            "complexity":
                complexity,

            "elapsed_ms":
                elapsed_ms,

            "timeout":
                is_timeout,

            "error":
                error_text,

            "sql":
                sql
        }


        error_records.append(
            error_record
        )


        print(
            "      FAIL |",
            (
                "TIMEOUT"
                if is_timeout
                else "ERROR"
            )
        )

        print(
            "      ",
            error_text[:300]
        )


    # --------------------------------------------------------
    # Periodic commit and checkpoint
    # --------------------------------------------------------

    if (
        position % COMMIT_EVERY == 0
        or
        position == len(workload_df)
    ):

        conn.commit()


        if monitoring_records:

            pd.DataFrame(
                monitoring_records
            ).to_csv(
                MONITORING_CSV,
                index=False
            )


        if error_records:

            pd.DataFrame(
                error_records
            ).to_csv(
                ERROR_CSV,
                index=False
            )


        elapsed = (
            time.perf_counter()
            - phase_start
        )


        print()
        print(
            "      CHECKPOINT"
        )

        print(
            f"      Success : "
            f"{success_count:,}"
        )

        print(
            f"      Failed  : "
            f"{failure_count:,}"
        )

        print(
            f"      Elapsed : "
            f"{elapsed / 60:.2f} min"
        )

        print()


# Final commit
conn.commit()


# ============================================================
# 23. CREATE MONITORING DATAFRAME
# ============================================================

stage("CREATING MONITORING DATASET")


if monitoring_records:

    monitoring_df = pd.DataFrame(
        monitoring_records
    )

else:

    monitoring_df = pd.DataFrame()


if error_records:

    error_df = pd.DataFrame(
        error_records
    )

else:

    error_df = pd.DataFrame()


monitoring_df.to_csv(
    MONITORING_CSV,
    index=False
)


if not error_df.empty:

    error_df.to_csv(
        ERROR_CSV,
        index=False
    )


print(
    "Successful monitored queries:",
    f"{len(monitoring_df):,}"
)

print(
    "Failed queries:",
    f"{len(error_df):,}"
)


# ============================================================
# 24. QUERY FREQUENCY ANALYSIS
# ============================================================

stage("QUERY FREQUENCY ANALYSIS")


if not monitoring_df.empty:

    frequency_df = (

        monitoring_df

        .groupby(
            [
                "fingerprint_hash",
                "query_fingerprint",
                "workload_type",
                "query_family",
                "query_name"
            ],
            dropna=False
        )

        .agg(

            frequency=(
                "query_log_id",
                "count"
            ),

            avg_execution_time_ms=(
                "execution_time_ms",
                "mean"
            ),

            min_execution_time_ms=(
                "execution_time_ms",
                "min"
            ),

            max_execution_time_ms=(
                "execution_time_ms",
                "max"
            ),

            avg_estimated_cost=(
                "estimated_cost",
                "mean"
            ),

            avg_actual_rows=(
                "actual_rows",
                "mean"
            ),

            index_usage_rate=(
                "index_used",
                "mean"
            ),

            avg_io_read_blocks=(
                "io_read_blocks",
                "mean"
            ),

            avg_io_write_blocks=(
                "io_write_blocks",
                "mean"
            )
        )

        .reset_index()
    )


    frequency_df = (
        frequency_df
        .sort_values(
            [
                "frequency",
                "avg_execution_time_ms"
            ],
            ascending=[
                False,
                False
            ]
        )
    )


    frequency_df.to_csv(
        FREQUENCY_CSV,
        index=False
    )


    print(
        "Unique query fingerprints:",
        f"{len(frequency_df):,}"
    )

    print()

    print(
        frequency_df[
            [
                "query_name",
                "frequency",
                "avg_execution_time_ms",
                "index_usage_rate"
            ]
        ]
        .head(20)
        .to_string(
            index=False
        )
    )


# ============================================================
# 25. WORKLOAD SUMMARY
# ============================================================

stage("WORKLOAD PERFORMANCE SUMMARY")


if not monitoring_df.empty:

    workload_summary = (

        monitoring_df

        .groupby(
            "workload_type"
        )

        .agg(

            executions=(
                "query_log_id",
                "count"
            ),

            avg_execution_time_ms=(
                "execution_time_ms",
                "mean"
            ),

            median_execution_time_ms=(
                "execution_time_ms",
                "median"
            ),

            min_execution_time_ms=(
                "execution_time_ms",
                "min"
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

            index_usage_rate=(
                "index_used",
                "mean"
            ),

            avg_seq_scans=(
                "seq_scan_count",
                "mean"
            ),

            avg_index_scans=(
                "index_scan_count",
                "mean"
            ),

            avg_io_read_blocks=(
                "io_read_blocks",
                "mean"
            ),

            avg_shared_hits=(
                "shared_hit_blocks",
                "mean"
            )
        )

        .reset_index()
    )


    workload_summary[
        "index_usage_percent"
    ] = (
        workload_summary[
            "index_usage_rate"
        ]
        * 100
    )


    workload_summary.to_csv(
        WORKLOAD_SUMMARY_CSV,
        index=False
    )


    print(
        workload_summary.to_string(
            index=False
        )
    )


# ============================================================
# 26. QUERY FAMILY SUMMARY
# ============================================================

stage("QUERY FAMILY PERFORMANCE")


if not monitoring_df.empty:

    family_summary = (

        monitoring_df

        .groupby(
            [
                "workload_type",
                "query_family"
            ]
        )

        .agg(

            executions=(
                "query_log_id",
                "count"
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

            index_usage_rate=(
                "index_used",
                "mean"
            ),

            avg_seq_scans=(
                "seq_scan_count",
                "mean"
            ),

            avg_io_reads=(
                "io_read_blocks",
                "mean"
            )
        )

        .reset_index()
    )


    family_summary = (
        family_summary
        .sort_values(
            "avg_execution_time_ms",
            ascending=False
        )
    )


    family_summary.to_csv(
        FAMILY_SUMMARY_CSV,
        index=False
    )


    print(
        family_summary.to_string(
            index=False
        )
    )


# ============================================================
# 27. INDEX USAGE SUMMARY
# ============================================================

stage("INDEX USAGE SUMMARY")


if not monitoring_df.empty:

    total_success = len(
        monitoring_df
    )

    index_queries = int(
        monitoring_df[
            "index_used"
        ].sum()
    )

    no_index_queries = (
        total_success
        -
        index_queries
    )

    index_percentage = (
        index_queries
        /
        total_success
        *
        100
    )


    index_summary = pd.DataFrame(
        [
            {
                "metric":
                    "successful_queries",

                "value":
                    total_success
            },

            {
                "metric":
                    "queries_using_index",

                "value":
                    index_queries
            },

            {
                "metric":
                    "queries_without_index",

                "value":
                    no_index_queries
            },

            {
                "metric":
                    "index_usage_percent",

                "value":
                    index_percentage
            },

            {
                "metric":
                    "total_seq_scans",

                "value":
                    int(
                        monitoring_df[
                            "seq_scan_count"
                        ].sum()
                    )
            },

            {
                "metric":
                    "total_index_scans",

                "value":
                    int(
                        monitoring_df[
                            "index_scan_count"
                        ].sum()
                    )
            },

            {
                "metric":
                    "total_bitmap_scans",

                "value":
                    int(
                        monitoring_df[
                            "bitmap_scan_count"
                        ].sum()
                    )
            }
        ]
    )


    index_summary.to_csv(
        INDEX_SUMMARY_CSV,
        index=False
    )


    print(
        index_summary.to_string(
            index=False
        )
    )


# ============================================================
# 28. TABLE ACCESS SUMMARY
# ============================================================

stage("TABLE ACCESS SUMMARY")


table_counter = {}


if not monitoring_df.empty:

    for value in monitoring_df[
        "tables_accessed"
    ].fillna(""):

        tables = [
            x.strip()
            for x in str(value).split(",")
            if x.strip()
        ]

        for table in tables:

            table_counter[
                table
            ] = (
                table_counter.get(
                    table,
                    0
                )
                + 1
            )


table_summary = pd.DataFrame(
    [
        {
            "table":
                table,

            "access_frequency":
                count
        }

        for table, count
        in table_counter.items()
    ]
)


if not table_summary.empty:

    table_summary = (
        table_summary
        .sort_values(
            "access_frequency",
            ascending=False
        )
    )


table_summary.to_csv(
    TABLE_SUMMARY_CSV,
    index=False
)


if not table_summary.empty:

    print(
        table_summary.to_string(
            index=False
        )
    )


# ============================================================
# 29. SLOWEST QUERIES
# ============================================================

stage("TOP 20 SLOWEST QUERIES")


if not monitoring_df.empty:

    slowest = (

        monitoring_df

        .sort_values(
            "execution_time_ms",
            ascending=False
        )

        .head(20)
    )


    print(
        slowest[
            [
                "workload_type",
                "query_family",
                "query_name",
                "complexity",
                "execution_time_ms",
                "estimated_cost",
                "actual_rows",
                "index_used",
                "seq_scan_count",
                "io_read_blocks"
            ]
        ]
        .to_string(
            index=False
        )
    )


    slowest.to_csv(
        RESULTS_DIR /
        "phase7_slowest_queries.csv",
        index=False
    )


# ============================================================
# 30. HIGHEST COST QUERIES
# ============================================================

stage("TOP 20 HIGHEST COST QUERIES")


if not monitoring_df.empty:

    highest_cost = (

        monitoring_df

        .sort_values(
            "estimated_cost",
            ascending=False
        )

        .head(20)
    )


    print(
        highest_cost[
            [
                "workload_type",
                "query_family",
                "query_name",
                "execution_time_ms",
                "estimated_cost",
                "actual_rows",
                "index_used"
            ]
        ]
        .to_string(
            index=False
        )
    )


    highest_cost.to_csv(
        RESULTS_DIR /
        "phase7_highest_cost_queries.csv",
        index=False
    )


# ============================================================
# 31. VERIFY DATABASE QUERY_LOG
# ============================================================

stage("VERIFYING QUERY_LOG DATABASE RECORDS")


with conn.cursor() as cur:

    cur.execute(
        """
        SELECT
            COUNT(*),
            COUNT(DISTINCT query_hash),
            COUNT(DISTINCT query_fingerprint),
            AVG(execution_time_ms),
            MAX(execution_time_ms),
            SUM(
                CASE
                    WHEN index_used
                    THEN 1
                    ELSE 0
                END
            )
        FROM query_log
        """
    )

    (
        db_log_count,
        db_unique_hashes,
        db_unique_fingerprints,
        db_avg_execution,
        db_max_execution,
        db_index_count
    ) = cur.fetchone()


print(
    "query_log rows:",
    f"{db_log_count:,}"
)

print(
    "Unique raw query hashes:",
    f"{db_unique_hashes:,}"
)

print(
    "Unique fingerprints:",
    f"{db_unique_fingerprints:,}"
)

print(
    "Average execution time:",
    (
        f"{db_avg_execution:.3f} ms"
        if db_avg_execution is not None
        else "N/A"
    )
)

print(
    "Maximum execution time:",
    (
        f"{db_max_execution:.3f} ms"
        if db_max_execution is not None
        else "N/A"
    )
)

print(
    "Queries using indexes:",
    f"{db_index_count or 0:,}"
)


# ============================================================
# 32. PHASE 7 METADATA
# ============================================================

stage("SAVING PHASE 7 METADATA")


phase_elapsed_seconds = (
    time.perf_counter()
    -
    phase_start
)


metadata = {

    "phase": 7,

    "phase_name":
        "Query Monitoring",

    "generated_at":
        datetime.now().isoformat(),

    "database":
        DB_NAME,

    "source_workload":
        str(WORKLOAD_FILE),

    "workload_rows_selected":
        len(workload_df),

    "successful_queries":
        success_count,

    "failed_queries":
        failure_count,

    "timeout_queries":
        timeout_count,

    "statement_timeout_ms":
        STATEMENT_TIMEOUT_MS,

    "monitoring_method":
        (
            "EXPLAIN "
            "(ANALYZE TRUE, "
            "BUFFERS TRUE, "
            "FORMAT JSON)"
        ),

    "phase_runtime_seconds":
        phase_elapsed_seconds,

    "query_log_rows_after_run":
        db_log_count,

    "cpu_usage_note":
        (
            "cpu_usage intentionally stored "
            "as NULL because PostgreSQL "
            "EXPLAIN ANALYZE does not provide "
            "direct process CPU utilization."
        ),

    "frequency_note":
        (
            "Query frequency is derived from "
            "repeated normalized "
            "query_fingerprint values."
        )
}


with open(
    METADATA_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )


# ============================================================
# 33. PHASE 7 REPORT
# ============================================================

if not monitoring_df.empty:

    average_execution = (
        monitoring_df[
            "execution_time_ms"
        ].mean()
    )

    median_execution = (
        monitoring_df[
            "execution_time_ms"
        ].median()
    )

    maximum_execution = (
        monitoring_df[
            "execution_time_ms"
        ].max()
    )

    index_usage_percent = (
        monitoring_df[
            "index_used"
        ].mean()
        * 100
    )

    total_seq_scans = int(
        monitoring_df[
            "seq_scan_count"
        ].sum()
    )

    total_index_scans = int(
        monitoring_df[
            "index_scan_count"
        ].sum()
    )

    total_io_reads = int(
        monitoring_df[
            "io_read_blocks"
        ].sum()
    )

else:

    average_execution = 0
    median_execution = 0
    maximum_execution = 0
    index_usage_percent = 0
    total_seq_scans = 0
    total_index_scans = 0
    total_io_reads = 0


report = f"""
PHASE 7 - QUERY MONITORING
======================================================================

Project:
Adaptive Healthcare Knowledge Database

Database:
{DB_NAME}

Source:
{WORKLOAD_FILE}

MONITORING METHOD
----------------------------------------------------------------------

EXPLAIN
(
    ANALYZE TRUE,
    BUFFERS TRUE,
    FORMAT JSON
)

WORKLOAD
----------------------------------------------------------------------

Selected workload instances:
{len(workload_df):,}

Successful executions:
{success_count:,}

Failed executions:
{failure_count:,}

Timeouts:
{timeout_count:,}

Statement timeout:
{STATEMENT_TIMEOUT_MS:,} ms


PERFORMANCE
----------------------------------------------------------------------

Average execution time:
{average_execution:.3f} ms

Median execution time:
{median_execution:.3f} ms

Maximum execution time:
{maximum_execution:.3f} ms

Index usage:
{index_usage_percent:.2f} %

Total sequential scans:
{total_seq_scans:,}

Total index scans:
{total_index_scans:,}

Total physical/temp read blocks:
{total_io_reads:,}


QUERY LOG
----------------------------------------------------------------------

Rows in query_log:
{db_log_count:,}

Unique raw query hashes:
{db_unique_hashes:,}

Unique query fingerprints:
{db_unique_fingerprints:,}


CPU USAGE
----------------------------------------------------------------------

CPU usage is intentionally NULL.

PostgreSQL EXPLAIN ANALYZE does not directly expose
per-query process CPU utilization.

No synthetic CPU value was generated.


QUERY FREQUENCY
----------------------------------------------------------------------

Frequency is derived from repeated query_fingerprint values.

This allows Phase 8 to learn which normalized query patterns
occur most frequently.


PHASE 8 INPUT
----------------------------------------------------------------------

Phase 7 provides real measured features including:

- workload type
- query family
- query fingerprint
- tables accessed
- columns accessed
- join count
- filter count
- execution time
- planning time
- estimated cost
- estimated rows
- actual rows
- index usage
- index names
- sequential scans
- index scans
- bitmap scans
- shared buffer hits
- physical reads
- temp reads/writes
- query frequency


STATUS
----------------------------------------------------------------------

PHASE 7 MONITORING COMPLETED
"""


with open(
    REPORT_TXT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        report.strip()
    )


# ============================================================
# 34. FINAL RESULT
# ============================================================

stage("PHASE 7 FINAL RESULT")


print(
    "Queries selected       :",
    f"{len(workload_df):,}"
)

print(
    "Successful executions  :",
    f"{success_count:,}"
)

print(
    "Failed executions      :",
    f"{failure_count:,}"
)

print(
    "Timeouts               :",
    f"{timeout_count:,}"
)

print(
    "query_log rows         :",
    f"{db_log_count:,}"
)

print(
    "Unique fingerprints    :",
    f"{db_unique_fingerprints:,}"
)

print(
    "Average execution time :",
    f"{average_execution:.3f} ms"
)

print(
    "Median execution time  :",
    f"{median_execution:.3f} ms"
)

print(
    "Maximum execution time :",
    f"{maximum_execution:.3f} ms"
)

print(
    "Index usage            :",
    f"{index_usage_percent:.2f}%"
)

print(
    "Total sequential scans :",
    f"{total_seq_scans:,}"
)

print(
    "Total index scans      :",
    f"{total_index_scans:,}"
)


print()
print(
    "Output directory:"
)

print(
    RESULTS_DIR
)


print()
print(
    "Generated files:"
)


for path in sorted(
    RESULTS_DIR.iterdir()
):

    print(
        " -",
        path.name
    )


print()
print("=" * 100)


if failure_count == 0:

    print(
        "PHASE 7 STATUS: COMPLETE"
    )

else:

    print(
        "PHASE 7 STATUS: COMPLETE WITH "
        "FAILED/TIMEOUT QUERIES RECORDED"
    )


print("=" * 100)

print(
    "Phase 6 workload loading         PASS"
)

print(
    "Read-only safety                 PASS"
)

print(
    "EXPLAIN ANALYZE monitoring       COMPLETE"
)

print(
    "Query fingerprinting             COMPLETE"
)

print(
    "Execution-time collection        COMPLETE"
)

print(
    "Query-plan analysis              COMPLETE"
)

print(
    "Index-usage detection            COMPLETE"
)

print(
    "I/O monitoring                   COMPLETE"
)

print(
    "Query frequency analysis         COMPLETE"
)

print(
    "PostgreSQL query_log population  COMPLETE"
)

print(
    "Phase 7 reports                  COMPLETE"
)

print()
print(
    "NEXT:"
)

print(
    "PHASE 8 -> AI WORKLOAD LEARNER"
)


conn.close()