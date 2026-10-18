from pathlib import Path
import os
import json
import time
import getpass
import statistics
from datetime import datetime

import pandas as pd
import psycopg
from psycopg import sql


# =====================================================================
# PHASE 9E
# MATERIALIZED VIEW VALIDATION FOR HEAVY ANALYTICAL WORKLOADS
#
# Adaptive Healthcare Knowledge Database
#
# This script:
#   1. Loads frozen Phase 7 workload data
#   2. Selects heavy analytical query families
#   3. Benchmarks original analytical SQL
#   4. Builds optimized materialized views
#   5. Adds useful indexes to the materialized views
#   6. Measures build and refresh cost
#   7. Benchmarks MV retrieval
#   8. Calculates speedup and break-even
#   9. Approves/rejects MV designs
#  10. Saves research results
#  11. Safely removes all experimental objects
#  12. Verifies cleanup
#
# Existing Phase 6/7/8/9 results are NOT modified.
# =====================================================================


print("=" * 118)
print("PHASE 9E - MATERIALIZED VIEW VALIDATION")
print("Heavy Analytical Query Performance Optimization")
print("=" * 118)


# =====================================================================
# 1. PATHS
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Try the known Phase 7 result locations.
PHASE7_CANDIDATES = [
    PROJECT_ROOT
    / "results"
    / "phase7_monitoring"
    / "phase7_query_monitoring.csv",

    PROJECT_ROOT
    / "results"
    / "phase7_query_monitoring"
    / "phase7_query_monitoring.csv",

    PROJECT_ROOT
    / "phase7_query_monitoring.csv",
]

PHASE7_FILE = None

for candidate in PHASE7_CANDIDATES:
    if candidate.exists():
        PHASE7_FILE = candidate
        break


RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "phase9e_materialized_views"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_FILE = RESULTS_DIR / "phase9e_mv_results.csv"
APPROVED_FILE = RESULTS_DIR / "phase9e_approved_mvs.csv"
REJECTED_FILE = RESULTS_DIR / "phase9e_rejected_mvs.csv"
PLAN_FILE = RESULTS_DIR / "phase9e_plan_comparison.json"
METADATA_FILE = RESULTS_DIR / "phase9e_metadata.json"
REPORT_FILE = RESULTS_DIR / "phase9e_report.txt"


# =====================================================================
# 2. CONFIGURATION
# =====================================================================

DB_CONFIG = {
    "dbname": "adaptive_healthcare_db",
    "host": "localhost",
    "port": 5432,
    "user": "postgres",
}

BENCHMARK_RUNS = 5
WARMUP_RUNS = 1

STATEMENT_TIMEOUT_MS = 120000

MIN_LATENCY_IMPROVEMENT_PERCENT = 25.0
STRONG_SPEEDUP = 2.0

# False = experimental MVs are removed after testing.
KEEP_MATERIALIZED_VIEWS = False


# =====================================================================
# 3. TARGET ANALYTICAL WORKLOADS
# =====================================================================

TARGET_FAMILIES = [
    "condition_medication",
    "drug_utilization",
    "claim_transactions",
    "procedure_utilization",
    "financial_temporal",
    "provider_ranking",
    "condition_prevalence",
    "temporal_activity",
]


# =====================================================================
# 4. MATERIALIZED VIEW DESIGNS
# =====================================================================

MV_DESIGNS = {

    "condition_medication": {

        "mv_name":
            "mv_p9e_condition_medication",

        "create_query":
        """
        SELECT
            c.description AS condition,

            COALESCE(
                dc.name,
                pi.description
            ) AS medication,

            COUNT(
                DISTINCT d.patient_id
            ) AS patient_count

        FROM diagnosis d

        JOIN condition c
          ON d.condition_id = c.condition_id

        JOIN prescription pr
          ON d.patient_id = pr.patient_id

        JOIN prescription_item pi
          ON pr.prescription_id = pi.prescription_id

        LEFT JOIN drug_concept dc
          ON pi.rxcui = dc.rxcui

        GROUP BY
            c.description,
            COALESCE(
                dc.name,
                pi.description
            )
        """,

        "query":
        """
        SELECT
            condition,
            medication,
            patient_count

        FROM mv_p9e_condition_medication

        ORDER BY patient_count DESC

        LIMIT 100
        """,

        "indexes": [

            """
            CREATE INDEX
            idx_p9e_condition_medication_count
            ON mv_p9e_condition_medication
            (patient_count DESC)
            """,

            """
            CREATE INDEX
            idx_p9e_condition_medication_condition
            ON mv_p9e_condition_medication
            (condition)
            """,
        ],

        "reason":
            "Precomputes the expensive many-to-many "
            "condition/patient/prescription/drug aggregation."
    },


    "drug_utilization": {

        "mv_name":
            "mv_p9e_drug_utilization",

        "create_query":
        """
        SELECT
            dc.rxcui,
            dc.name,

            COUNT(
                DISTINCT pr.patient_id
            ) AS patient_count,

            COUNT(
                pr.prescription_id
            ) AS prescription_count

        FROM prescription pr

        JOIN prescription_item pi
          ON pr.prescription_id = pi.prescription_id

        LEFT JOIN drug_concept dc
          ON pi.rxcui = dc.rxcui

        WHERE pi.rxcui IS NOT NULL

        GROUP BY
            dc.rxcui,
            dc.name
        """,

        "query":
        """
        SELECT
            rxcui,
            name,
            patient_count,
            prescription_count

        FROM mv_p9e_drug_utilization

        ORDER BY prescription_count DESC

        LIMIT 50
        """,

        "indexes": [

            """
            CREATE INDEX
            idx_p9e_drug_utilization_count
            ON mv_p9e_drug_utilization
            (prescription_count DESC)
            """,

            """
            CREATE INDEX
            idx_p9e_drug_utilization_rxcui
            ON mv_p9e_drug_utilization
            (rxcui)
            """
        ],

        "reason":
            "Precomputes repeated drug-utilization aggregation."
    },


    "claim_transactions": {

        "mv_name":
            "mv_p9e_claim_transactions",

        "create_query":
        """
        SELECT
            transaction_type,

            COUNT(*) AS transaction_count,

            SUM(amount) AS total_amount,

            AVG(amount) AS average_amount,

            SUM(
                patient_payment
            ) AS patient_payments

        FROM claim_transaction

        GROUP BY transaction_type
        """,

        "query":
        """
        SELECT
            transaction_type,
            transaction_count,
            total_amount,
            average_amount,
            patient_payments

        FROM mv_p9e_claim_transactions

        ORDER BY transaction_count DESC
        """,

        "indexes": [

            """
            CREATE UNIQUE INDEX
            idx_p9e_claim_transactions_type
            ON mv_p9e_claim_transactions
            (transaction_type)
            """
        ],

        "reason":
            "Avoids repeated full-table claim transaction aggregation."
    },


    "procedure_utilization": {

        "mv_name":
            "mv_p9e_procedure_utilization",

        "create_query":
        """
        SELECT
            pd.procedure_id,
            pd.source_code,
            pd.description,

            COUNT(
                pp.patient_procedure_id
            ) AS procedure_count,

            COUNT(
                DISTINCT pp.patient_id
            ) AS patient_count,

            AVG(
                pp.base_cost
            ) AS average_cost

        FROM patient_procedure pp

        JOIN procedure_definition pd
          ON pp.procedure_id = pd.procedure_id

        GROUP BY
            pd.procedure_id,
            pd.source_code,
            pd.description
        """,

        "query":
        """
        SELECT
            procedure_id,
            source_code,
            description,
            procedure_count,
            patient_count,
            average_cost

        FROM mv_p9e_procedure_utilization

        ORDER BY procedure_count DESC

        LIMIT 50
        """,

        "indexes": [

            """
            CREATE UNIQUE INDEX
            idx_p9e_procedure_utilization_id
            ON mv_p9e_procedure_utilization
            (procedure_id)
            """,

            """
            CREATE INDEX
            idx_p9e_procedure_utilization_count
            ON mv_p9e_procedure_utilization
            (procedure_count DESC)
            """
        ],

        "reason":
            "Precomputes procedure frequency and cost statistics."
    },


    "financial_temporal": {

        "mv_name":
            "mv_p9e_financial_temporal",

        "create_query":
        """
        SELECT
            DATE_TRUNC(
                'month',
                transaction_date
            ) AS month,

            COUNT(*) AS transaction_count,

            SUM(amount) AS total_amount,

            SUM(
                patient_payment
            ) AS patient_payment

        FROM claim_transaction

        WHERE transaction_date IS NOT NULL

        GROUP BY
            DATE_TRUNC(
                'month',
                transaction_date
            )
        """,

        "query":
        """
        SELECT
            month,
            transaction_count,
            total_amount,
            patient_payment

        FROM mv_p9e_financial_temporal

        ORDER BY month
        """,

        "indexes": [

            """
            CREATE UNIQUE INDEX
            idx_p9e_financial_temporal_month
            ON mv_p9e_financial_temporal
            (month)
            """
        ],

        "reason":
            "Precomputes monthly financial transaction aggregation."
    },


    "provider_ranking": {

        "mv_name":
            "mv_p9e_provider_ranking",

        "create_query":
        """
        SELECT
            p.provider_id,
            p.name,
            p.specialty,

            COUNT(
                e.encounter_id
            ) AS encounter_count,

            COUNT(
                DISTINCT e.patient_id
            ) AS patient_count

        FROM provider p

        JOIN encounter e
          ON p.provider_id = e.provider_id

        GROUP BY
            p.provider_id,
            p.name,
            p.specialty
        """,

        "query":
        """
        SELECT
            provider_id,
            name,
            specialty,
            encounter_count,
            patient_count

        FROM mv_p9e_provider_ranking

        ORDER BY encounter_count DESC

        LIMIT 25
        """,

        "indexes": [

            """
            CREATE UNIQUE INDEX
            idx_p9e_provider_ranking_id
            ON mv_p9e_provider_ranking
            (provider_id)
            """,

            """
            CREATE INDEX
            idx_p9e_provider_ranking_count
            ON mv_p9e_provider_ranking
            (encounter_count DESC)
            """
        ],

        "reason":
            "Precomputes provider activity statistics."
    },


    "condition_prevalence": {

        "mv_name":
            "mv_p9e_condition_prevalence",

        "create_query":
        """
        SELECT
            c.source_code,
            c.description,

            COUNT(
                DISTINCT d.patient_id
            ) AS patient_count,

            COUNT(
                d.diagnosis_id
            ) AS diagnosis_count

        FROM diagnosis d

        JOIN condition c
          ON d.condition_id = c.condition_id

        GROUP BY
            c.source_code,
            c.description
        """,

        "query":
        """
        SELECT
            source_code,
            description,
            patient_count,
            diagnosis_count

        FROM mv_p9e_condition_prevalence

        ORDER BY patient_count DESC

        LIMIT 50
        """,

        "indexes": [

            """
            CREATE INDEX
            idx_p9e_condition_prevalence_count
            ON mv_p9e_condition_prevalence
            (patient_count DESC)
            """,

            """
            CREATE INDEX
            idx_p9e_condition_prevalence_code
            ON mv_p9e_condition_prevalence
            (source_code)
            """
        ],

        "reason":
            "Precomputes population-level condition prevalence."
    },


    "temporal_activity": {

        "mv_name":
            "mv_p9e_temporal_activity",

        "create_query":
        """
        SELECT
            DATE_TRUNC(
                'month',
                start_time
            ) AS month,

            COUNT(*) AS encounter_count,

            COUNT(
                DISTINCT patient_id
            ) AS patient_count

        FROM encounter

        WHERE start_time IS NOT NULL

        GROUP BY
            DATE_TRUNC(
                'month',
                start_time
            )
        """,

        "query":
        """
        SELECT
            month,
            encounter_count,
            patient_count

        FROM mv_p9e_temporal_activity

        ORDER BY month
        """,

        "indexes": [

            """
            CREATE UNIQUE INDEX
            idx_p9e_temporal_activity_month
            ON mv_p9e_temporal_activity
            (month)
            """
        ],

        "reason":
            "Precomputes monthly encounter activity."
    },
}


# =====================================================================
# 5. HELPERS
# =====================================================================

def stage(title):
    print()
    print("=" * 118)
    print(title)
    print("=" * 118)


def clean_sql(query):
    query = str(query).strip()

    while query.endswith(";"):
        query = query[:-1].strip()

    return query


def median(values):
    values = [
        float(x)
        for x in values
        if x is not None
    ]

    if not values:
        return 0.0

    return statistics.median(values)


def mean(values):
    values = [
        float(x)
        for x in values
        if x is not None
    ]

    if not values:
        return 0.0

    return statistics.mean(values)


def pct_improvement(before, after):

    if before is None or before <= 0:
        return 0.0

    return (
        (before - after)
        / before
    ) * 100.0


def safe_divide(a, b):

    if b is None or b == 0:
        return 0.0

    return a / b


# =====================================================================
# 6. DATABASE PASSWORD
# =====================================================================

password = (
    os.getenv("POSTGRES_PASSWORD")
    or
    os.getenv("DB_PASSWORD")
)

if not password:
    password = getpass.getpass(
        "PostgreSQL password: "
    )

DB_CONFIG["password"] = password


# =====================================================================
# 7. CONNECT
# =====================================================================

stage("CONNECTING TO POSTGRESQL")

conn = None

try:

    conn = psycopg.connect(
        **DB_CONFIG
    )

    conn.autocommit = True

    with conn.cursor() as cur:

        cur.execute(
            """
            SELECT
                current_database(),
                version();
            """
        )

        db_name, pg_version = cur.fetchone()

    print(
        "Database:",
        db_name
    )

    print(
        "Connection: PASS"
    )

except Exception as exc:

    print(
        "Connection: FAILED"
    )

    print(exc)

    raise


# =====================================================================
# 8. LOAD PHASE 7
# =====================================================================

stage("LOADING FROZEN PHASE 7 WORKLOAD")

if PHASE7_FILE is None:

    raise FileNotFoundError(
        "Could not locate phase7_query_monitoring.csv."
    )


print(
    "Phase 7 file:",
    PHASE7_FILE
)

phase7 = pd.read_csv(
    PHASE7_FILE
)

required_columns = {
    "query_family",
    "query_name",
    "fingerprint_hash",
    "execution_time_ms",
    "sql",
}

missing = (
    required_columns
    - set(phase7.columns)
)

if missing:

    raise RuntimeError(
        "Missing Phase 7 columns: "
        + str(
            sorted(missing)
        )
    )


print(
    "Phase 7 executions:",
    f"{len(phase7):,}"
)

print(
    "Unique families:",
    phase7["query_family"].nunique()
)

print(
    "Input validation: PASS"
)


# =====================================================================
# 9. SAFE DROP FUNCTION
# =====================================================================

def drop_materialized_view(mv_name):

    with conn.cursor() as cur:

        cur.execute(
            sql.SQL(
                """
                DROP MATERIALIZED VIEW
                IF EXISTS {} CASCADE
                """
            ).format(
                sql.Identifier(mv_name)
            )
        )


# =====================================================================
# 10. CLEAN PREVIOUS EXPERIMENTAL OBJECTS
# =====================================================================

stage(
    "CLEANING PREVIOUS PHASE 9E EXPERIMENTAL OBJECTS"
)

cleanup_failures = []

for family, design in MV_DESIGNS.items():

    mv_name = design["mv_name"]

    try:

        drop_materialized_view(
            mv_name
        )

    except Exception as exc:

        cleanup_failures.append(
            {
                "mv_name": mv_name,
                "error": str(exc)
            }
        )


if cleanup_failures:

    print(
        "WARNING: Some old objects could not be removed."
    )

    for failure in cleanup_failures:

        print(
            failure["mv_name"],
            "->",
            failure["error"]
        )

else:

    print(
        "Old Phase 9E objects removed."
    )


# =====================================================================
# 11. EXPLAIN ANALYZE
# =====================================================================

def explain_analyze(query):

    query = clean_sql(query)

    with conn.cursor() as cur:

        cur.execute(
            f"""
            SET statement_timeout =
            {STATEMENT_TIMEOUT_MS};
            """
        )

        cur.execute(
            """
            EXPLAIN
            (
                ANALYZE TRUE,
                BUFFERS TRUE,
                FORMAT JSON
            )
            """
            + query
        )

        payload = cur.fetchone()[0]


    if isinstance(payload, str):

        payload = json.loads(
            payload
        )


    root = payload[0]

    plan = root["Plan"]


    return {

        "execution_time_ms":
            float(
                root.get(
                    "Execution Time",
                    0
                )
            ),

        "planning_time_ms":
            float(
                root.get(
                    "Planning Time",
                    0
                )
            ),

        "cost":
            float(
                plan.get(
                    "Total Cost",
                    0
                )
            ),

        "shared_hit_blocks":
            int(
                plan.get(
                    "Shared Hit Blocks",
                    0
                ) or 0
            ),

        "shared_read_blocks":
            int(
                plan.get(
                    "Shared Read Blocks",
                    0
                ) or 0
            ),

        "temp_read_blocks":
            int(
                plan.get(
                    "Temp Read Blocks",
                    0
                ) or 0
            ),

        "temp_written_blocks":
            int(
                plan.get(
                    "Temp Written Blocks",
                    0
                ) or 0
            ),

        "raw":
            payload,
    }


# =====================================================================
# 12. BENCHMARK
# =====================================================================

def benchmark_query(
    query,
    warmups=WARMUP_RUNS,
    runs=BENCHMARK_RUNS
):

    for _ in range(warmups):

        explain_analyze(
            query
        )


    measured = []

    for _ in range(runs):

        result = explain_analyze(
            query
        )

        measured.append(
            result
        )


    execution_times = [
        x["execution_time_ms"]
        for x in measured
    ]

    planning_times = [
        x["planning_time_ms"]
        for x in measured
    ]

    costs = [
        x["cost"]
        for x in measured
    ]

    hits = [
        x["shared_hit_blocks"]
        for x in measured
    ]

    reads = [
        x["shared_read_blocks"]
        for x in measured
    ]

    temp_reads = [
        x["temp_read_blocks"]
        for x in measured
    ]

    temp_writes = [
        x["temp_written_blocks"]
        for x in measured
    ]


    return {

        "median_execution_ms":
            median(execution_times),

        "mean_execution_ms":
            mean(execution_times),

        "min_execution_ms":
            min(execution_times),

        "max_execution_ms":
            max(execution_times),

        "median_planning_ms":
            median(planning_times),

        "median_cost":
            median(costs),

        "median_hit_blocks":
            median(hits),

        "median_read_blocks":
            median(reads),

        "median_temp_read_blocks":
            median(temp_reads),

        "median_temp_written_blocks":
            median(temp_writes),

        "runs":
            measured,
    }


# =====================================================================
# 13. REPRESENTATIVE QUERY
# =====================================================================

def representative_query(family):

    subset = phase7[
        phase7["query_family"]
        == family
    ].copy()


    if subset.empty:
        return None


    subset["execution_time_ms"] = (
        pd.to_numeric(
            subset["execution_time_ms"],
            errors="coerce"
        )
    )


    subset = subset.dropna(
        subset=[
            "execution_time_ms",
            "sql"
        ]
    )


    if subset.empty:
        return None


    med = subset[
        "execution_time_ms"
    ].median()


    subset["_distance"] = (
        subset["execution_time_ms"]
        - med
    ).abs()


    row = (
        subset
        .sort_values("_distance")
        .iloc[0]
    )


    return {

        "query_name":
            row["query_name"],

        "fingerprint_hash":
            row["fingerprint_hash"],

        "sql":
            clean_sql(
                row["sql"]
            ),

        "frequency":
            len(subset),

        "phase7_median_ms":
            float(med),
    }


# =====================================================================
# 14. MATERIALIZED VIEW SIZE
# =====================================================================

def materialized_view_size(
    mv_name
):

    with conn.cursor() as cur:

        cur.execute(
            """
            SELECT
                pg_total_relation_size(
                    %s::regclass
                );
            """,
            (mv_name,)
        )

        value = cur.fetchone()[0]


    return int(
        value or 0
    )


# =====================================================================
# 15. MATERIALIZED VIEW ROW COUNT
# =====================================================================

def materialized_view_rows(
    mv_name
):

    with conn.cursor() as cur:

        cur.execute(
            sql.SQL(
                """
                SELECT COUNT(*)
                FROM {}
                """
            ).format(
                sql.Identifier(
                    mv_name
                )
            )
        )

        value = cur.fetchone()[0]


    return int(value)


# =====================================================================
# 16. CREATE MATERIALIZED VIEW
# =====================================================================

def create_materialized_view(
    design
):

    mv_name = design[
        "mv_name"
    ]

    create_query = clean_sql(
        design[
            "create_query"
        ]
    )


    drop_materialized_view(
        mv_name
    )


    start = time.perf_counter()


    with conn.cursor() as cur:

        statement = (
            sql.SQL(
                """
                CREATE MATERIALIZED VIEW {} AS
                """
            ).format(
                sql.Identifier(
                    mv_name
                )
            )
            +
            sql.SQL(
                create_query
            )
        )

        cur.execute(
            statement
        )


    build_ms = (
        time.perf_counter()
        - start
    ) * 1000.0


    for index_statement in design[
        "indexes"
    ]:

        with conn.cursor() as cur:

            cur.execute(
                clean_sql(
                    index_statement
                )
            )


    with conn.cursor() as cur:

        cur.execute(
            sql.SQL(
                """
                ANALYZE {}
                """
            ).format(
                sql.Identifier(
                    mv_name
                )
            )
        )


    return build_ms


# =====================================================================
# 17. REFRESH MATERIALIZED VIEW
# =====================================================================

def refresh_materialized_view(
    mv_name
):

    start = time.perf_counter()


    with conn.cursor() as cur:

        cur.execute(
            sql.SQL(
                """
                REFRESH MATERIALIZED VIEW {}
                """
            ).format(
                sql.Identifier(
                    mv_name
                )
            )
        )


    refresh_ms = (
        time.perf_counter()
        - start
    ) * 1000.0


    with conn.cursor() as cur:

        cur.execute(
            sql.SQL(
                """
                ANALYZE {}
                """
            ).format(
                sql.Identifier(
                    mv_name
                )
            )
        )


    return refresh_ms


# =====================================================================
# 18. BREAK-EVEN
# =====================================================================

def calculate_break_even(
    overhead_ms,
    original_ms,
    optimized_ms
):

    saving_ms = (
        original_ms
        - optimized_ms
    )

    if saving_ms <= 0:
        return None

    return (
        overhead_ms
        / saving_ms
    )


# =====================================================================
# 19. DECISION POLICY
# =====================================================================

def decide_mv(
    improvement,
    speedup,
    frequency,
    break_even
):

    if improvement < 0:

        return (
            "REJECTED",
            "Materialized view made the query slower."
        )


    if (
        improvement
        >= MIN_LATENCY_IMPROVEMENT_PERCENT
        and
        speedup
        >= STRONG_SPEEDUP
    ):

        if (
            break_even is None
            or
            frequency >= break_even
        ):

            return (
                "APPROVED",
                "Strong measured acceleration and "
                "materialization cost is amortized by "
                "observed workload frequency."
            )


        return (
            "CONDITIONAL",
            "Strong retrieval acceleration, but "
            "observed frequency may not amortize "
            "materialization cost in one workload window."
        )


    if (
        improvement
        >= MIN_LATENCY_IMPROVEMENT_PERCENT
    ):

        return (
            "CONDITIONAL",
            "Meaningful measured improvement, but "
            "speedup is below the strong threshold."
        )


    return (
        "REJECTED",
        "Measured improvement is too small to justify "
        "materialization overhead."
    )


# =====================================================================
# 20. RUN EXPERIMENTS
# =====================================================================

stage(
    "RUNNING HEAVY ANALYTICAL MV EXPERIMENTS"
)

results = []

plan_comparisons = []


for position, family in enumerate(
    TARGET_FAMILIES,
    start=1
):

    print()
    print("-" * 118)

    print(
        f"[{position}/"
        f"{len(TARGET_FAMILIES)}] "
        f"{family}"
    )


    design = MV_DESIGNS.get(
        family
    )


    if design is None:

        print(
            "No MV design defined."
        )

        continue


    representative = representative_query(
        family
    )


    if representative is None:

        print(
            "Family not found in Phase 7."
        )

        continue


    mv_name = design[
        "mv_name"
    ]

    original_sql = representative[
        "sql"
    ]

    mv_sql = clean_sql(
        design[
            "query"
        ]
    )

    frequency = representative[
        "frequency"
    ]

    query_name = representative[
        "query_name"
    ]

    fingerprint = representative[
        "fingerprint_hash"
    ]


    print(
        "Query:",
        query_name
    )

    print(
        "Observed frequency:",
        frequency
    )

    print(
        "Phase 7 median:",
        f"{representative['phase7_median_ms']:.3f} ms"
    )


    try:

        # ---------------------------------------------------------
        # ORIGINAL QUERY
        # ---------------------------------------------------------

        print()
        print(
            "Benchmarking original query..."
        )

        baseline = benchmark_query(
            original_sql
        )

        baseline_ms = baseline[
            "median_execution_ms"
        ]

        print(
            "Original median:",
            f"{baseline_ms:.3f} ms"
        )


        # ---------------------------------------------------------
        # BUILD MV
        # ---------------------------------------------------------

        print(
            "Building materialized view..."
        )

        build_ms = (
            create_materialized_view(
                design
            )
        )


        rows = (
            materialized_view_rows(
                mv_name
            )
        )


        size_bytes = (
            materialized_view_size(
                mv_name
            )
        )


        print(
            "MV build time:",
            f"{build_ms:.3f} ms"
        )

        print(
            "MV rows:",
            f"{rows:,}"
        )

        print(
            "MV size:",
            f"{size_bytes / (1024 * 1024):.3f} MB"
        )


        # ---------------------------------------------------------
        # REFRESH COST
        # ---------------------------------------------------------

        print(
            "Testing refresh cost..."
        )

        refresh_ms = (
            refresh_materialized_view(
                mv_name
            )
        )

        print(
            "Refresh time:",
            f"{refresh_ms:.3f} ms"
        )


        # ---------------------------------------------------------
        # OPTIMIZED QUERY
        # ---------------------------------------------------------

        print(
            "Benchmarking MV query..."
        )

        optimized = benchmark_query(
            mv_sql
        )

        mv_ms = optimized[
            "median_execution_ms"
        ]


        # ---------------------------------------------------------
        # PERFORMANCE CALCULATIONS
        # ---------------------------------------------------------

        improvement = pct_improvement(
            baseline_ms,
            mv_ms
        )

        speedup = safe_divide(
            baseline_ms,
            mv_ms
        )

        saved_ms = (
            baseline_ms
            - mv_ms
        )


        break_even_build = (
            calculate_break_even(
                build_ms,
                baseline_ms,
                mv_ms
            )
        )


        break_even_refresh = (
            calculate_break_even(
                refresh_ms,
                baseline_ms,
                mv_ms
            )
        )


        cost_improvement = (
            pct_improvement(
                baseline[
                    "median_cost"
                ],
                optimized[
                    "median_cost"
                ]
            )
        )


        read_improvement = (
            pct_improvement(
                baseline[
                    "median_read_blocks"
                ],
                optimized[
                    "median_read_blocks"
                ]
            )
        )


        temp_read_improvement = (
            pct_improvement(
                baseline[
                    "median_temp_read_blocks"
                ],
                optimized[
                    "median_temp_read_blocks"
                ]
            )
        )


        estimated_window_saving_ms = (
            max(
                saved_ms,
                0
            )
            * frequency
        )


        net_window_after_build_ms = (
            estimated_window_saving_ms
            - build_ms
        )


        net_window_after_refresh_ms = (
            estimated_window_saving_ms
            - refresh_ms
        )


        status, decision_reason = (
            decide_mv(
                improvement,
                speedup,
                frequency,
                break_even_build
            )
        )


        result = {

            "query_family":
                family,

            "query_name":
                query_name,

            "fingerprint_hash":
                fingerprint,

            "observed_frequency":
                frequency,

            "phase7_median_ms":
                representative[
                    "phase7_median_ms"
                ],

            "mv_name":
                mv_name,

            "mv_rows":
                rows,

            "mv_size_bytes":
                size_bytes,

            "mv_size_mb":
                size_bytes
                / (1024 * 1024),

            "build_time_ms":
                build_ms,

            "refresh_time_ms":
                refresh_ms,

            "original_median_ms":
                baseline_ms,

            "mv_median_ms":
                mv_ms,

            "absolute_saving_ms":
                saved_ms,

            "latency_improvement_percent":
                improvement,

            "speedup_x":
                speedup,

            "original_cost":
                baseline[
                    "median_cost"
                ],

            "mv_cost":
                optimized[
                    "median_cost"
                ],

            "cost_improvement_percent":
                cost_improvement,

            "original_shared_hits":
                baseline[
                    "median_hit_blocks"
                ],

            "mv_shared_hits":
                optimized[
                    "median_hit_blocks"
                ],

            "original_shared_reads":
                baseline[
                    "median_read_blocks"
                ],

            "mv_shared_reads":
                optimized[
                    "median_read_blocks"
                ],

            "read_block_reduction_percent":
                read_improvement,

            "original_temp_reads":
                baseline[
                    "median_temp_read_blocks"
                ],

            "mv_temp_reads":
                optimized[
                    "median_temp_read_blocks"
                ],

            "temp_read_reduction_percent":
                temp_read_improvement,

            "break_even_executions_build":
                break_even_build,

            "break_even_executions_refresh":
                break_even_refresh,

            "estimated_window_saving_ms":
                estimated_window_saving_ms,

            "net_window_saving_after_build_ms":
                net_window_after_build_ms,

            "net_window_saving_after_refresh_ms":
                net_window_after_refresh_ms,

            "status":
                status,

            "decision_reason":
                decision_reason,

            "design_reason":
                design[
                    "reason"
                ],
        }


        results.append(
            result
        )


        plan_comparisons.append(
            {

                "query_family":
                    family,

                "query_name":
                    query_name,

                "mv_name":
                    mv_name,

                "original_sql":
                    original_sql,

                "mv_query":
                    mv_sql,

                "original_plan":
                    baseline[
                        "runs"
                    ][-1][
                        "raw"
                    ],

                "mv_plan":
                    optimized[
                        "runs"
                    ][-1][
                        "raw"
                    ],

                "result":
                    result,
            }
        )


        print()
        print(
            "Original:",
            f"{baseline_ms:.3f} ms"
        )

        print(
            "MV:",
            f"{mv_ms:.3f} ms"
        )

        print(
            "Improvement:",
            f"{improvement:.2f}%"
        )

        print(
            "Speedup:",
            f"{speedup:.2f}x"
        )

        print(
            "Saved per execution:",
            f"{saved_ms:.3f} ms"
        )


        if break_even_build is not None:

            print(
                "Build break-even:",
                f"{break_even_build:.2f} executions"
            )


        if break_even_refresh is not None:

            print(
                "Refresh break-even:",
                f"{break_even_refresh:.2f} executions"
            )


        print(
            "Result:",
            status
        )


    except Exception as exc:

        print(
            "MV experiment failed:",
            exc
        )

        results.append(
            {

                "query_family":
                    family,

                "query_name":
                    query_name,

                "fingerprint_hash":
                    fingerprint,

                "observed_frequency":
                    frequency,

                "mv_name":
                    mv_name,

                "status":
                    "FAILED",

                "decision_reason":
                    str(exc),

                "design_reason":
                    design[
                        "reason"
                    ],
            }
        )


    finally:

        if not KEEP_MATERIALIZED_VIEWS:

            try:

                drop_materialized_view(
                    mv_name
                )

                print(
                    "Experimental MV removed."
                )

            except Exception as cleanup_exc:

                print(
                    "WARNING: cleanup failed:",
                    cleanup_exc
                )


# =====================================================================
# 21. RESULTS DATAFRAME
# =====================================================================

stage(
    "ANALYZING PHASE 9E RESULTS"
)

results_df = pd.DataFrame(
    results
)


if results_df.empty:

    raise RuntimeError(
        "No Phase 9E experiments completed."
    )


if "status" not in results_df.columns:

    raise RuntimeError(
        "Phase 9E results contain no status column."
    )


approved_df = results_df[
    results_df["status"]
    == "APPROVED"
].copy()


other_df = results_df[
    results_df["status"]
    != "APPROVED"
].copy()


if (
    "latency_improvement_percent"
    in results_df.columns
):

    results_df = (
        results_df
        .sort_values(
            "latency_improvement_percent",
            ascending=False,
            na_position="last"
        )
    )


if (
    not approved_df.empty
    and
    "latency_improvement_percent"
    in approved_df.columns
):

    approved_df = (
        approved_df
        .sort_values(
            "latency_improvement_percent",
            ascending=False,
            na_position="last"
        )
    )


# =====================================================================
# 22. SAVE CSV RESULTS
# =====================================================================

results_df.to_csv(
    RESULT_FILE,
    index=False
)

approved_df.to_csv(
    APPROVED_FILE,
    index=False
)

other_df.to_csv(
    REJECTED_FILE,
    index=False
)


# =====================================================================
# 23. SAVE PLAN COMPARISON
# =====================================================================

with open(
    PLAN_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        plan_comparisons,
        f,
        indent=2,
        default=str
    )


# =====================================================================
# 24. DISPLAY PERFORMANCE
# =====================================================================

print()
print(
    "MATERIALIZED VIEW PERFORMANCE"
)

print("-" * 118)


display_columns = [
    "query_name",
    "original_median_ms",
    "mv_median_ms",
    "latency_improvement_percent",
    "speedup_x",
    "build_time_ms",
    "refresh_time_ms",
    "break_even_executions_build",
    "status",
]


existing_display_columns = [
    column
    for column in display_columns
    if column in results_df.columns
]


print(
    results_df[
        existing_display_columns
    ].to_string(
        index=False
    )
)


# =====================================================================
# 25. FIND BEST RESULT
# =====================================================================

successful = results_df[
    results_df["status"].isin(
        [
            "APPROVED",
            "CONDITIONAL",
        ]
    )
].copy()


if (
    not successful.empty
    and
    "latency_improvement_percent"
    in successful.columns
):

    best = (
        successful
        .sort_values(
            "latency_improvement_percent",
            ascending=False
        )
        .iloc[0]
    )

    best_query = best[
        "query_name"
    ]

    best_improvement = float(
        best[
            "latency_improvement_percent"
        ]
    )

    best_speedup = float(
        best[
            "speedup_x"
        ]
    )

else:

    best_query = "None"
    best_improvement = 0.0
    best_speedup = 0.0


# =====================================================================
# 26. CONDITION-MEDICATION RESULT
# =====================================================================

condition_result = results_df[
    results_df[
        "query_family"
    ]
    == "condition_medication"
].copy()


if not condition_result.empty:

    cm = condition_result.iloc[0]

    cm_before = float(
        cm.get(
            "original_median_ms",
            0
        )
        or 0
    )

    cm_after = float(
        cm.get(
            "mv_median_ms",
            0
        )
        or 0
    )

    cm_improvement = float(
        cm.get(
            "latency_improvement_percent",
            0
        )
        or 0
    )

    cm_speedup = float(
        cm.get(
            "speedup_x",
            0
        )
        or 0
    )

else:

    cm_before = 0.0
    cm_after = 0.0
    cm_improvement = 0.0
    cm_speedup = 0.0


# =====================================================================
# 27. METADATA
# =====================================================================

metadata = {

    "phase":
        "9E",

    "phase_name":
        "Materialized View Validation",

    "generated_at":
        datetime.now().isoformat(),

    "phase7_file":
        str(PHASE7_FILE),

    "target_families":
        TARGET_FAMILIES,

    "benchmark_runs":
        BENCHMARK_RUNS,

    "warmup_runs":
        WARMUP_RUNS,

    "experiments":
        len(results_df),

    "approved":
        len(approved_df),

    "other_outcomes":
        len(other_df),

    "best_query":
        best_query,

    "best_improvement_percent":
        best_improvement,

    "best_speedup":
        best_speedup,

    "condition_medication": {

        "before_ms":
            cm_before,

        "after_ms":
            cm_after,

        "improvement_percent":
            cm_improvement,

        "speedup":
            cm_speedup,
    },

    "keep_materialized_views":
        KEEP_MATERIALIZED_VIEWS,

    "methodology":
        (
            "Repeated warm-cache EXPLAIN ANALYZE "
            "comparison of original analytical SQL "
            "against semantically equivalent "
            "materialized-view retrieval. Build time, "
            "refresh time, storage and break-even "
            "execution count are measured."
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
        indent=4,
        default=str
    )


# =====================================================================
# 28. REPORT
# =====================================================================

report = f"""
PHASE 9E - MATERIALIZED VIEW VALIDATION
======================================================================

Project:
Adaptive Healthcare Knowledge Database


OBJECTIVE
----------------------------------------------------------------------

Evaluate whether repeated heavy analytical healthcare database queries
benefit from materialized precomputation after workload-aware indexing.


METHODOLOGY
----------------------------------------------------------------------

Analytical families evaluated:
{len(results_df)}

Benchmark runs:
{BENCHMARK_RUNS}

Warm-up runs:
{WARMUP_RUNS}

The experiment measured:

- Original query latency
- Materialized-view retrieval latency
- MV build time
- MV refresh time
- MV storage
- Planner cost
- PostgreSQL buffer statistics
- Absolute latency saving
- Percentage latency reduction
- Speedup
- Build break-even executions
- Refresh break-even executions
- Estimated workload-window benefit


RESULT SUMMARY
----------------------------------------------------------------------

Experiments:
{len(results_df)}

Approved:
{len(approved_df)}

Other outcomes:
{len(other_df)}


BEST RETRIEVAL RESULT
----------------------------------------------------------------------

Query:
{best_query}

Latency improvement:
{best_improvement:.4f}%

Retrieval speedup:
{best_speedup:.2f}x


CONDITION-MEDICATION ANALYTICAL WORKLOAD
----------------------------------------------------------------------

Original analytical execution:
{cm_before:.3f} ms

Materialized retrieval:
{cm_after:.3f} ms

Retrieval latency improvement:
{cm_improvement:.4f}%

Retrieval speedup:
{cm_speedup:.2f}x


INTERPRETATION
----------------------------------------------------------------------

Materialized views move expensive joins and aggregation from repeated
query execution into materialization/refresh time.

Therefore, the MV retrieval speedup must not be interpreted as the
speedup of recomputing the underlying analytical result.

The appropriate interpretation is:

"Materialized precomputation reduced repeated retrieval latency while
incurring a measurable build/refresh maintenance cost."

Break-even analysis determines whether the observed workload frequency
is sufficient to justify this maintenance overhead.


SAFETY
----------------------------------------------------------------------

KEEP_MATERIALIZED_VIEWS:
{KEEP_MATERIALIZED_VIEWS}

When False, all experimental Phase 9E materialized views are removed
after benchmarking and the baseline database physical design is
preserved.
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
# 29. FINAL CLEANUP VERIFICATION
#
# IMPORTANT FIX:
# fetchall() occurs INSIDE the cursor context.
# =====================================================================

stage(
    "VERIFYING CLEANUP"
)


# First make one more cleanup pass when experimental objects should
# not remain.

if not KEEP_MATERIALIZED_VIEWS:

    for family, design in MV_DESIGNS.items():

        mv_name = design[
            "mv_name"
        ]

        try:

            drop_materialized_view(
                mv_name
            )

        except Exception as exc:

            print(
                "WARNING: final cleanup problem for",
                mv_name,
                ":",
                exc
            )


# -------------------------------------------------------------
# Verification query.
#
# The previous version failed because fetchall() was called
# after leaving the "with conn.cursor()" block.
#
# Here BOTH execute() and fetchall() occur while cur is open.
# -------------------------------------------------------------

with conn.cursor() as cur:

    cur.execute(
        """
        SELECT
            matviewname

        FROM pg_matviews

        WHERE schemaname = 'public'
          AND matviewname LIKE 'mv_p9e_%'

        ORDER BY matviewname;
        """
    )

    remaining_rows = cur.fetchall()


remaining_mvs = [
    row[0]
    for row in remaining_rows
]


if KEEP_MATERIALIZED_VIEWS:

    print(
        "Materialized views retained intentionally."
    )

    if remaining_mvs:

        for mv in remaining_mvs:

            print(
                " -",
                mv
            )

    cleanup_status = "RETAINED"


else:

    if remaining_mvs:

        print(
            "WARNING: Experimental materialized views remain:"
        )

        for mv in remaining_mvs:

            print(
                " -",
                mv
            )

        cleanup_status = "REVIEW"

    else:

        print(
            "Experimental MV cleanup: PASS"
        )

        print(
            "Remaining Phase 9E MVs: 0"
        )

        print(
            "Original database baseline preserved."
        )

        cleanup_status = "PASS"


# =====================================================================
# 30. UPDATE METADATA WITH CLEANUP RESULT
# =====================================================================

metadata[
    "cleanup_status"
] = cleanup_status

metadata[
    "remaining_experimental_mvs"
] = remaining_mvs


with open(
    METADATA_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=4,
        default=str
    )


# =====================================================================
# 31. APPEND CLEANUP STATUS TO REPORT
# =====================================================================

with open(
    REPORT_FILE,
    "a",
    encoding="utf-8"
) as f:

    f.write(
        "\n\n\nCLEANUP VERIFICATION\n"
        "----------------------------------------------------------------------\n\n"
    )

    f.write(
        f"Cleanup status: {cleanup_status}\n"
    )

    f.write(
        f"Remaining experimental MVs: {len(remaining_mvs)}\n"
    )


# =====================================================================
# 32. FINAL SUMMARY
# =====================================================================

stage(
    "PHASE 9E FINAL RESULT"
)


print(
    "Analytical families evaluated :",
    len(results_df)
)

print(
    "Approved MVs                  :",
    len(approved_df)
)

print(
    "Other outcomes                :",
    len(other_df)
)


print()

print(
    "Best query                    :",
    best_query
)

print(
    "Best improvement              :",
    f"{best_improvement:.4f}%"
)

print(
    "Best retrieval speedup        :",
    f"{best_speedup:.2f}x"
)


print()

print(
    "Condition-medication before   :",
    f"{cm_before:.3f} ms"
)

print(
    "Condition-medication MV       :",
    f"{cm_after:.3f} ms"
)

print(
    "Condition-medication gain     :",
    f"{cm_improvement:.4f}%"
)

print(
    "Condition-medication speedup  :",
    f"{cm_speedup:.2f}x"
)


print()

print(
    "Cleanup                       :",
    cleanup_status
)

print(
    "Remaining experimental MVs    :",
    len(remaining_mvs)
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
    "Generated files:"
)

for path in sorted(
    RESULTS_DIR.iterdir()
):

    if path.is_file():

        print(
            " -",
            path.name
        )


# =====================================================================
# 33. FINAL STATUS
# =====================================================================

print()
print("=" * 118)


if cleanup_status == "PASS":

    print(
        "PHASE 9E STATUS: COMPLETE"
    )

else:

    print(
        "PHASE 9E STATUS: COMPLETE WITH CLEANUP REVIEW"
    )


print("=" * 118)

print(
    "Heavy analytical MV validation    COMPLETE"
)

print(
    "Build overhead analysis            COMPLETE"
)

print(
    "Refresh overhead analysis          COMPLETE"
)

print(
    "Break-even analysis                COMPLETE"
)

print(
    "Performance comparison             COMPLETE"
)

print(
    "Research outputs                   COMPLETE"
)

print(
    "Baseline cleanup                   ",
    cleanup_status
)


if cleanup_status == "PASS":

    print()
    print(
        "PHASE 9A-9E CAN NOW BE FROZEN."
    )

    print(
        "NEXT -> PHASE 10: Biomedical Knowledge Graph"
    )


# =====================================================================
# 34. CLOSE DATABASE CONNECTION
# =====================================================================

if conn is not None:

    conn.close()


print()
print(
    "Database connection closed."
)