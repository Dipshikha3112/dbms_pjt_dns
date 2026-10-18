from pathlib import Path
import os
import getpass
import json
from datetime import datetime

import pandas as pd
import psycopg


# ============================================================
# PHASE 3 INTEGRITY AUDIT
# Adaptive Healthcare Knowledge Database
# ============================================================

print("=" * 100)
print("PHASE 3 - POSTGRESQL INTEGRITY AUDIT")
print("Adaptive Healthcare Knowledge Database")
print("=" * 100)


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA = PROJECT_ROOT / "data"
SYNTHEA = DATA / "synthea"
RXNORM = DATA / "rxnorm"
RESULTS = PROJECT_ROOT / "results"

RESULTS.mkdir(parents=True, exist_ok=True)

DB_NAME = "adaptive_healthcare_db"
DB_HOST = os.getenv("PGHOST", "localhost")
DB_PORT = int(os.getenv("PGPORT", "5432"))
DB_USER = os.getenv("PGUSER", "postgres")
DB_PASSWORD = os.getenv("PGPASSWORD")

if not DB_PASSWORD:
    DB_PASSWORD = getpass.getpass(
        f"PostgreSQL password for '{DB_USER}': "
    )


# ============================================================
# 2. HELPERS
# ============================================================

def stage(name):
    print("\n" + "=" * 100)
    print(name)
    print("=" * 100)


def scalar(conn, query, params=None):
    with conn.cursor() as cur:
        cur.execute(query, params)
        row = cur.fetchone()
        return row[0] if row else None


def get_columns(conn, table_name):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = %s
            ORDER BY ordinal_position
            """,
            (table_name,)
        )

        return [
            row[0]
            for row in cur.fetchall()
        ]


def check(name, passed, detail):
    status = "PASS" if passed else "CHECK"

    audit_results.append({
        "check": name,
        "status": status,
        "detail": detail
    })

    print(
        f"{status:<8} "
        f"{name:<55} "
        f"{detail}"
    )


# ============================================================
# 3. CONNECT
# ============================================================

stage("CONNECTING TO DATABASE")

conn = psycopg.connect(
    host=DB_HOST,
    port=DB_PORT,
    user=DB_USER,
    password=DB_PASSWORD,
    dbname=DB_NAME
)

print("Connected to:", DB_NAME)

audit_results = []


# ============================================================
# 4. CORE TABLE COUNT AUDIT
# ============================================================

stage("1. CORE TABLE COUNT AUDIT")

expected_exact = {
    "patient": 1163,
    "organization": 1127,
    "provider": 5056,
    "encounter": 61459,
    "diagnosis": 38094,
    "patient_procedure": 83823,
    "care_plan": 3931,
    "medical_device": 89,
    "patient_allergy": 794,
    "patient_immunization": 17009,
    "prescription": 56430,
    "prescription_item": 56430,
    "observation": 531144,
    "lab_result": 150894,
    "patient_insurance": 53101,
    "claim": 117889,
    "claim_transaction": 711238,
    "loinc_concept": 112405
}

for table, expected in expected_exact.items():

    actual = scalar(
        conn,
        f"SELECT COUNT(*) FROM {table}"
    )

    check(
        f"{table} row count",
        actual == expected,
        f"database={actual:,}, expected={expected:,}"
    )


# ============================================================
# 5. SOURCE VS DATABASE
# ============================================================

stage("2. SYNTHEA SOURCE VS DATABASE")

source_mapping = {
    "patients.csv": "patient",
    "organizations.csv": "organization",
    "providers.csv": "provider",
    "encounters.csv": "encounter",
    "conditions.csv": "diagnosis",
    "procedures.csv": "patient_procedure",
    "careplans.csv": "care_plan",
    "devices.csv": "medical_device",
    "allergies.csv": "patient_allergy",
    "immunizations.csv": "patient_immunization",
    "medications.csv": "prescription",
    "observations.csv": "observation",
    "payer_transitions.csv": "patient_insurance",
    "claims.csv": "claim",
    "claims_transactions.csv": "claim_transaction"
}

source_db_rows = []

for filename, table in source_mapping.items():

    path = SYNTHEA / filename

    source_count = len(
        pd.read_csv(
            path,
            dtype=str,
            low_memory=False
        )
    )

    db_count = scalar(
        conn,
        f"SELECT COUNT(*) FROM {table}"
    )

    difference = source_count - db_count

    source_db_rows.append({
        "source_file": filename,
        "source_rows": source_count,
        "database_table": table,
        "database_rows": db_count,
        "difference": difference
    })

    check(
        f"{filename} -> {table}",
        difference == 0,
        (
            f"source={source_count:,}, "
            f"db={db_count:,}, "
            f"difference={difference:,}"
        )
    )


# ============================================================
# 6. IMAGING STUDY AUDIT
# ============================================================

stage("3. IMAGING STUDY AUDIT")

imaging_path = SYNTHEA / "imaging_studies.csv"

imaging = pd.read_csv(
    imaging_path,
    dtype=str,
    low_memory=False
)

print(
    "Source imaging rows:",
    f"{len(imaging):,}"
)

print("Columns:")
print(list(imaging.columns))

print("\nSource uniqueness analysis:")

for column in [
    "Id",
    "ID",
    "PATIENT",
    "ENCOUNTER",
    "DATE",
    "BODYSITE_CODE",
    "MODALITY_CODE",
    "SOP_CODE"
]:

    if column in imaging.columns:

        unique_count = imaging[
            column
        ].nunique(
            dropna=True
        )

        print(
            f"{column:<25}: "
            f"{unique_count:,} unique values"
        )


db_imaging = scalar(
    conn,
    "SELECT COUNT(*) FROM imaging_study"
)

print(
    "\nDatabase imaging rows:",
    f"{db_imaging:,}"
)


available_cols = [
    c
    for c in [
        "PATIENT",
        "ENCOUNTER",
        "DATE",
        "BODYSITE_CODE",
        "MODALITY_CODE",
        "SOP_CODE"
    ]
    if c in imaging.columns
]

if available_cols:

    unique_combinations = (
        imaging[
            available_cols
        ]
        .drop_duplicates()
        .shape[0]
    )

    print(
        "Unique imaging combinations "
        f"({', '.join(available_cols)}): "
        f"{unique_combinations:,}"
    )


with conn.cursor() as cur:

    cur.execute(
        """
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT patient_id),
            COUNT(DISTINCT encounter_id)
        FROM imaging_study
        """
    )

    row = cur.fetchone()

    print(
        "\nDatabase imaging characteristics:"
    )

    print(
        "Rows              :",
        f"{row[0]:,}"
    )

    print(
        "Distinct patients :",
        f"{row[1]:,}"
    )

    print(
        "Distinct encounters:",
        f"{row[2]:,}"
    )


with conn.cursor() as cur:

    cur.execute(
        """
        SELECT synthea_id::text
        FROM patient
        """
    )

    patient_source_ids = {
        row[0]
        for row in cur.fetchall()
    }


with conn.cursor() as cur:

    cur.execute(
        """
        SELECT synthea_id::text
        FROM encounter
        WHERE synthea_id IS NOT NULL
        """
    )

    encounter_source_ids = {
        row[0]
        for row in cur.fetchall()
    }


if "PATIENT" in imaging.columns:

    valid = (
        imaging["PATIENT"]
        .astype(str)
        .isin(patient_source_ids)
    )

    print(
        "\nImaging rows with valid patients:",
        f"{valid.sum():,}",
        "/",
        f"{len(imaging):,}"
    )


if "ENCOUNTER" in imaging.columns:

    valid = (
        imaging["ENCOUNTER"]
        .astype(str)
        .isin(encounter_source_ids)
    )

    print(
        "Imaging rows with valid encounters:",
        f"{valid.sum():,}",
        "/",
        f"{len(imaging):,}"
    )


# ============================================================
# 7. RXNORM AUDIT
# ============================================================

stage("4. RXNORM DRUG CONCEPT AUDIT")

rxnconso_columns = [
    "RXCUI",
    "LAT",
    "TS",
    "LUI",
    "STT",
    "SUI",
    "ISPREF",
    "RXAUI",
    "SAUI",
    "SCUI",
    "SDUI",
    "SAB",
    "TTY",
    "CODE",
    "STR",
    "SRL",
    "SUPPRESS",
    "CVF",
    "EMPTY"
]

rxnorm_path = (
    RXNORM /
    "rrf" /
    "RXNCONSO.RRF"
)

raw_rxnorm_rows = 0
filtered_rxnorm_rows = 0

filtered_rxcui = set()

for chunk in pd.read_csv(
    rxnorm_path,
    sep="|",
    names=rxnconso_columns,
    dtype=str,
    chunksize=50000,
    low_memory=False
):

    raw_rxnorm_rows += len(chunk)

    filtered = chunk[
        (chunk["LAT"] == "ENG") &
        (chunk["SAB"] == "RXNORM") &
        (chunk["SUPPRESS"] != "O")
    ]

    filtered_rxnorm_rows += len(
        filtered
    )

    filtered_rxcui.update(
        filtered["RXCUI"]
        .dropna()
        .astype(str)
        .tolist()
    )


print(
    "Raw RXNCONSO rows              :",
    f"{raw_rxnorm_rows:,}"
)

print(
    "Filtered canonical rows        :",
    f"{filtered_rxnorm_rows:,}"
)

print(
    "Unique RXCUI after filtering   :",
    f"{len(filtered_rxcui):,}"
)


db_drug_concepts = scalar(
    conn,
    "SELECT COUNT(*) FROM drug_concept"
)

print(
    "Database drug_concept rows     :",
    f"{db_drug_concepts:,}"
)


rxnorm_expected = len(
    filtered_rxcui
)

check(
    "RxNorm unique RXCUI vs drug_concept",
    db_drug_concepts == rxnorm_expected,
    (
        f"filtered unique RXCUI="
        f"{rxnorm_expected:,}, "
        f"database={db_drug_concepts:,}"
    )
)


# ============================================================
# 8. MEDICATION RXNORM COVERAGE
# ============================================================

stage("5. SYNTHEA MEDICATION / RXNORM AUDIT")

medications = pd.read_csv(
    SYNTHEA / "medications.csv",
    dtype=str,
    low_memory=False
)

medication_codes = set(
    medications["CODE"]
    .dropna()
    .astype(str)
    .str.strip()
)

print(
    "Unique Synthea medication codes:",
    f"{len(medication_codes):,}"
)


with conn.cursor() as cur:

    cur.execute(
        """
        SELECT rxcui
        FROM drug_concept
        """
    )

    database_rxcui = {
        str(row[0])
        for row in cur.fetchall()
    }


matched_med_codes = (
    medication_codes &
    database_rxcui
)

missing_med_codes = (
    medication_codes -
    database_rxcui
)

print(
    "Medication codes in drug_concept:",
    f"{len(matched_med_codes):,}"
)

print(
    "Medication codes missing:",
    f"{len(missing_med_codes):,}"
)

coverage = (
    len(matched_med_codes)
    / len(medication_codes)
    * 100
    if medication_codes
    else 0
)

print(
    f"Medication code coverage: "
    f"{coverage:.2f}%"
)

check(
    "Synthea medication RxNorm coverage",
    coverage >= 90,
    f"{coverage:.2f}%"
)


# ============================================================
# 9. FOREIGN KEY ORPHAN AUDIT
# ============================================================

stage("6. FOREIGN KEY ORPHAN AUDIT")

orphan_queries = {

    "encounter_without_patient": """
        SELECT COUNT(*)
        FROM encounter e
        LEFT JOIN patient p
          ON e.patient_id = p.patient_id
        WHERE p.patient_id IS NULL
    """,

    "diagnosis_without_patient": """
        SELECT COUNT(*)
        FROM diagnosis d
        LEFT JOIN patient p
          ON d.patient_id = p.patient_id
        WHERE p.patient_id IS NULL
    """,

    "diagnosis_without_condition": """
        SELECT COUNT(*)
        FROM diagnosis d
        LEFT JOIN condition c
          ON d.condition_id = c.condition_id
        WHERE c.condition_id IS NULL
    """,

    "procedure_without_patient": """
        SELECT COUNT(*)
        FROM patient_procedure pp
        LEFT JOIN patient p
          ON pp.patient_id = p.patient_id
        WHERE p.patient_id IS NULL
    """,

    "prescription_without_patient": """
        SELECT COUNT(*)
        FROM prescription pr
        LEFT JOIN patient p
          ON pr.patient_id = p.patient_id
        WHERE p.patient_id IS NULL
    """,

    "prescription_item_without_prescription": """
        SELECT COUNT(*)
        FROM prescription_item pi
        LEFT JOIN prescription p
          ON pi.prescription_id = p.prescription_id
        WHERE p.prescription_id IS NULL
    """,

    "observation_without_patient": """
        SELECT COUNT(*)
        FROM observation o
        LEFT JOIN patient p
          ON o.patient_id = p.patient_id
        WHERE p.patient_id IS NULL
    """,

    "lab_without_patient": """
        SELECT COUNT(*)
        FROM lab_result l
        LEFT JOIN patient p
          ON l.patient_id = p.patient_id
        WHERE p.patient_id IS NULL
    """,

    "claim_without_patient": """
        SELECT COUNT(*)
        FROM claim c
        LEFT JOIN patient p
          ON c.patient_id = p.patient_id
        WHERE p.patient_id IS NULL
    """,

    "claim_transaction_without_claim": """
        SELECT COUNT(*)
        FROM claim_transaction ct
        LEFT JOIN claim c
          ON ct.claim_id = c.claim_id
        WHERE c.claim_id IS NULL
    """
}

for name, query in orphan_queries.items():

    count = scalar(
        conn,
        query
    )

    check(
        name,
        count == 0,
        f"orphans={count:,}"
    )


# ============================================================
# 10. MAPPING QUALITY
# ============================================================

stage("7. MAPPING QUALITY AUDIT")

mapping_queries = {

    "encounter_missing_provider": """
        SELECT COUNT(*)
        FROM encounter
        WHERE provider_id IS NULL
    """,

    "encounter_missing_organization": """
        SELECT COUNT(*)
        FROM encounter
        WHERE organization_id IS NULL
    """,

    "diagnosis_missing_encounter": """
        SELECT COUNT(*)
        FROM diagnosis
        WHERE encounter_id IS NULL
    """,

    "prescription_missing_encounter": """
        SELECT COUNT(*)
        FROM prescription
        WHERE encounter_id IS NULL
    """,

    "observation_missing_encounter": """
        SELECT COUNT(*)
        FROM observation
        WHERE encounter_id IS NULL
    """,

    "claim_missing_payer": """
        SELECT COUNT(*)
        FROM claim
        WHERE payer_id IS NULL
    """,

    "claim_missing_provider": """
        SELECT COUNT(*)
        FROM claim
        WHERE provider_id IS NULL
    """
}

mapping_results = {}

for name, query in mapping_queries.items():

    count = scalar(
        conn,
        query
    )

    mapping_results[name] = count

    print(
        f"{name:<45}: "
        f"{count:,}"
    )


# ============================================================
# 11. EXPLAIN MISSING OBSERVATION ENCOUNTERS
# ============================================================

stage("7A. OBSERVATION ENCOUNTER MAPPING ANALYSIS")

observations_source = pd.read_csv(
    SYNTHEA / "observations.csv",
    dtype=str,
    low_memory=False
)

if "ENCOUNTER" in observations_source.columns:

    encounter_values = (
        observations_source["ENCOUNTER"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    blank_encounters = (
        encounter_values == ""
    ).sum()

    nonblank_encounters = (
        encounter_values != ""
    ).sum()

    valid_source_encounters = (
        encounter_values.isin(
            encounter_source_ids
        )
    ).sum()

    invalid_nonblank = (
        (
            encounter_values != ""
        ) &
        (
            ~encounter_values.isin(
                encounter_source_ids
            )
        )
    ).sum()

    print(
        "Source observations:",
        f"{len(observations_source):,}"
    )

    print(
        "Blank encounter values:",
        f"{blank_encounters:,}"
    )

    print(
        "Nonblank encounter values:",
        f"{nonblank_encounters:,}"
    )

    print(
        "Valid encounter mappings:",
        f"{valid_source_encounters:,}"
    )

    print(
        "Invalid nonblank mappings:",
        f"{invalid_nonblank:,}"
    )

    print(
        "Database NULL encounter_id:",
        f"{mapping_results['observation_missing_encounter']:,}"
    )

    observation_null_expected = (
        mapping_results[
            "observation_missing_encounter"
        ] == blank_encounters
    )

    check(
        "Observation NULL encounters explained by source",
        observation_null_expected,
        (
            f"source blanks={blank_encounters:,}, "
            f"database NULL={mapping_results['observation_missing_encounter']:,}"
        )
    )


# ============================================================
# 12. CLAIM PAYER MAPPING ANALYSIS
# ============================================================

stage("7B. CLAIM PAYER MAPPING ANALYSIS")

claims_source = pd.read_csv(
    SYNTHEA / "claims.csv",
    dtype=str,
    low_memory=False
)

print(
    "Claims source rows:",
    f"{len(claims_source):,}"
)

print(
    "Claims columns:"
)

print(
    list(claims_source.columns)
)


payer_columns = [
    c
    for c in [
        "PAYERID",
        "PAYER",
        "PRIMARYPATIENTINSURANCEID"
    ]
    if c in claims_source.columns
]

print(
    "Available payer-related columns:",
    payer_columns
)


with conn.cursor() as cur:

    cur.execute(
        """
        SELECT synthea_id::text
        FROM payer
        WHERE synthea_id IS NOT NULL
        """
    )

    payer_source_ids = {
        row[0]
        for row in cur.fetchall()
    }


for column in payer_columns:

    values = (
        claims_source[column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    blank = (
        values == ""
    ).sum()

    zero = (
        values == "0"
    ).sum()

    valid = (
        values.isin(
            payer_source_ids
        )
    ).sum()

    print()
    print(
        f"{column}:"
    )

    print(
        "  blank values :",
        f"{blank:,}"
    )

    print(
        "  zero values  :",
        f"{zero:,}"
    )

    print(
        "  payer matches:",
        f"{valid:,}"
    )


print()
print(
    "Database claims with NULL payer_id:",
    f"{mapping_results['claim_missing_payer']:,}"
)


# ============================================================
# 13. DUPLICATE AUDIT
# ============================================================

stage("8. DUPLICATE AUDIT")

duplicate_queries = {

    "duplicate_patient_synthea_id": """
        SELECT COUNT(*)
        FROM (
            SELECT synthea_id
            FROM patient
            GROUP BY synthea_id
            HAVING COUNT(*) > 1
        ) x
    """,

    "duplicate_encounter_synthea_id": """
        SELECT COUNT(*)
        FROM (
            SELECT synthea_id
            FROM encounter
            WHERE synthea_id IS NOT NULL
            GROUP BY synthea_id
            HAVING COUNT(*) > 1
        ) x
    """,

    "duplicate_claim_synthea_id": """
        SELECT COUNT(*)
        FROM (
            SELECT synthea_id
            FROM claim
            WHERE synthea_id IS NOT NULL
            GROUP BY synthea_id
            HAVING COUNT(*) > 1
        ) x
    """,

    "duplicate_loinc_code": """
        SELECT COUNT(*)
        FROM (
            SELECT loinc_code
            FROM loinc_concept
            GROUP BY loinc_code
            HAVING COUNT(*) > 1
        ) x
    """,

    "duplicate_rxcui": """
        SELECT COUNT(*)
        FROM (
            SELECT rxcui
            FROM drug_concept
            GROUP BY rxcui
            HAVING COUNT(*) > 1
        ) x
    """
}

for name, query in duplicate_queries.items():

    count = scalar(
        conn,
        query
    )

    check(
        name,
        count == 0,
        f"duplicate groups={count:,}"
    )


# ============================================================
# 14. LOGICAL CONSISTENCY
# ============================================================

stage("9. LOGICAL CONSISTENCY AUDIT")

logic_queries = {

    "encounter_end_before_start": """
        SELECT COUNT(*)
        FROM encounter
        WHERE start_time IS NOT NULL
          AND end_time IS NOT NULL
          AND end_time < start_time
    """,

    "diagnosis_stop_before_start": """
        SELECT COUNT(*)
        FROM diagnosis
        WHERE start_date IS NOT NULL
          AND stop_date IS NOT NULL
          AND stop_date < start_date
    """,

    "careplan_stop_before_start": """
        SELECT COUNT(*)
        FROM care_plan
        WHERE start_date IS NOT NULL
          AND stop_date IS NOT NULL
          AND stop_date < start_date
    """
}

for name, query in logic_queries.items():

    count = scalar(
        conn,
        query
    )

    check(
        name,
        count == 0,
        f"invalid rows={count:,}"
    )


# ============================================================
# 15. PATIENT INSURANCE DATE AUDIT
# ============================================================

stage("9A. PATIENT INSURANCE DATE AUDIT")

insurance_columns = get_columns(
    conn,
    "patient_insurance"
)

print(
    "patient_insurance columns:"
)

print(
    insurance_columns
)


possible_date_pairs = [
    ("coverage_start", "coverage_end"),
    ("start_date", "end_date"),
    ("start_year", "end_year"),
    ("coverage_start_date", "coverage_end_date")
]

insurance_date_pair = None

for start_col, end_col in possible_date_pairs:

    if (
        start_col in insurance_columns
        and end_col in insurance_columns
    ):

        insurance_date_pair = (
            start_col,
            end_col
        )

        break


if insurance_date_pair:

    start_col, end_col = (
        insurance_date_pair
    )

    print(
        "Detected insurance date columns:",
        start_col,
        "/",
        end_col
    )

    query = f"""
        SELECT COUNT(*)
        FROM patient_insurance
        WHERE {start_col} IS NOT NULL
          AND {end_col} IS NOT NULL
          AND {end_col} < {start_col}
    """

    count = scalar(
        conn,
        query
    )

    check(
        "insurance_end_before_start",
        count == 0,
        f"invalid rows={count:,}"
    )

else:

    check(
        "insurance_end_before_start",
        False,
        (
            "No recognized insurance "
            "start/end columns found"
        )
    )


# ============================================================
# 16. NEGATIVE FINANCIAL VALUE AUDIT
# ============================================================

stage("9B. FINANCIAL VALUE AUDIT")

financial_checks = {

    "negative_encounter_base_cost": """
        SELECT COUNT(*)
        FROM encounter
        WHERE base_cost < 0
    """,

    "negative_encounter_claim_cost": """
        SELECT COUNT(*)
        FROM encounter
        WHERE total_claim_cost < 0
    """,

    "negative_encounter_payer_coverage": """
        SELECT COUNT(*)
        FROM encounter
        WHERE payer_coverage < 0
    """
}

for name, query in financial_checks.items():

    try:

        count = scalar(
            conn,
            query
        )

        check(
            name,
            count == 0,
            f"negative rows={count:,}"
        )

    except Exception as e:

        conn.rollback()

        check(
            name,
            False,
            str(e)
        )


# ============================================================
# 17. VIEW VALIDATION
# ============================================================

stage("10. VIEW VALIDATION")

views = [
    "patient_summary",
    "patient_condition_history",
    "patient_medication_history",
    "patient_lab_history",
    "slow_query_summary"
]

for view in views:

    try:

        count = scalar(
            conn,
            f"SELECT COUNT(*) FROM {view}"
        )

        check(
            f"View {view}",
            True,
            f"rows={count:,}"
        )

    except Exception as e:

        conn.rollback()

        check(
            f"View {view}",
            False,
            str(e)
        )


# ============================================================
# 18. STORED FUNCTION VALIDATION
# ============================================================

stage("11. STORED FUNCTION VALIDATION")

sample_patient = scalar(
    conn,
    """
    SELECT patient_id
    FROM patient
    ORDER BY patient_id
    LIMIT 1
    """
)

print(
    "Sample patient:",
    sample_patient
)

function_tests = [
    (
        "get_patient_encounter_count",
        """
        SELECT
        get_patient_encounter_count(%s)
        """,
        (sample_patient,)
    ),

    (
        "get_patient_diagnosis_count",
        """
        SELECT
        get_patient_diagnosis_count(%s)
        """,
        (sample_patient,)
    ),

    (
        "get_average_query_latency",
        """
        SELECT
        get_average_query_latency()
        """,
        None
    )
]

for name, query, params in function_tests:

    try:

        result = scalar(
            conn,
            query,
            params
        )

        check(
            f"Function {name}",
            True,
            f"result={result}"
        )

    except Exception as e:

        conn.rollback()

        check(
            f"Function {name}",
            False,
            str(e)
        )


# ============================================================
# 19. DATABASE SIZE
# ============================================================

stage("12. DATABASE SIZE")

database_size = scalar(
    conn,
    """
    SELECT pg_size_pretty(
        pg_database_size(
            current_database()
        )
    )
    """
)

print(
    "Database size:",
    database_size
)


# ============================================================
# 20. TABLE SIZES
# ============================================================

stage("12A. LARGEST TABLES")

with conn.cursor() as cur:

    cur.execute(
        """
        SELECT
            relname,
            pg_total_relation_size(
                relid
            ) AS size_bytes
        FROM pg_catalog.pg_statio_user_tables
        ORDER BY size_bytes DESC
        """
    )

    table_sizes = cur.fetchall()


table_size_rows = []

for table_name, size_bytes in table_sizes:

    size_mb = (
        size_bytes /
        1024 /
        1024
    )

    table_size_rows.append({
        "table": table_name,
        "size_mb": round(
            size_mb,
            2
        )
    })

    print(
        f"{table_name:<35}"
        f"{size_mb:>12.2f} MB"
    )


# ============================================================
# 21. SAVE REPORTS
# ============================================================

stage("13. GENERATING AUDIT REPORTS")

pd.DataFrame(
    source_db_rows
).to_csv(
    RESULTS /
    "phase3_audit_source_vs_db.csv",
    index=False
)


pd.DataFrame({
    "missing_medication_code":
        sorted(
            missing_med_codes
        )
}).to_csv(
    RESULTS /
    "phase3_audit_missing_medication_codes.csv",
    index=False
)


pd.DataFrame(
    table_size_rows
).to_csv(
    RESULTS /
    "phase3_audit_table_sizes.csv",
    index=False
)


audit_df = pd.DataFrame(
    audit_results
)

audit_df.to_csv(
    RESULTS /
    "phase3_integrity_audit.csv",
    index=False
)


passes = int(
    (
        audit_df["status"]
        == "PASS"
    ).sum()
)

checks = int(
    (
        audit_df["status"]
        == "CHECK"
    ).sum()
)


report = {

    "audit_timestamp":
        datetime.now().isoformat(),

    "database":
        DB_NAME,

    "database_size":
        database_size,

    "total_checks":
        len(audit_df),

    "passed":
        passes,

    "requires_review":
        checks,

    "imaging": {
        "source_rows":
            len(imaging),

        "database_rows":
            db_imaging,

        "source_unique_ids":
            int(
                imaging["Id"].nunique()
                if "Id" in imaging.columns
                else 0
            )
    },

    "rxnorm": {
        "raw_rows":
            raw_rxnorm_rows,

        "filtered_rows":
            filtered_rxnorm_rows,

        "unique_filtered_rxcui":
            len(filtered_rxcui),

        "database_drug_concepts":
            db_drug_concepts
    },

    "medication_mapping": {
        "unique_codes":
            len(medication_codes),

        "matched_codes":
            len(matched_med_codes),

        "missing_codes":
            len(missing_med_codes),

        "coverage_percent":
            round(
                coverage,
                2
            )
    },

    "mapping_quality":
        mapping_results
}


with open(
    RESULTS /
    "phase3_integrity_audit.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=4
    )


# ============================================================
# 22. FINAL RESULT
# ============================================================

stage("14. FINAL AUDIT RESULT")

print(
    "PASS checks          :",
    passes
)

print(
    "Checks needing review:",
    checks
)

print(
    "Database size        :",
    database_size
)

print()

print(
    "Imaging source rows  :",
    f"{len(imaging):,}"
)

print(
    "Imaging unique IDs   :",
    (
        f"{imaging['Id'].nunique():,}"
        if "Id" in imaging.columns
        else "N/A"
    )
)

print(
    "Imaging DB rows      :",
    f"{db_imaging:,}"
)

print()

print(
    "RxNorm filtered rows :",
    f"{filtered_rxnorm_rows:,}"
)

print(
    "RxNorm unique RXCUI  :",
    f"{len(filtered_rxcui):,}"
)

print(
    "Drug concepts in DB  :",
    f"{db_drug_concepts:,}"
)

print()

print(
    "Medication coverage  :",
    f"{coverage:.2f}%"
)

print()

print(
    "Reports generated:"
)

print(
    RESULTS /
    "phase3_integrity_audit.csv"
)

print(
    RESULTS /
    "phase3_integrity_audit.json"
)

print(
    RESULTS /
    "phase3_audit_source_vs_db.csv"
)

print(
    RESULTS /
    "phase3_audit_missing_medication_codes.csv"
)

print(
    RESULTS /
    "phase3_audit_table_sizes.csv"
)


# ============================================================
# 23. FINAL DECISION MESSAGE
# ============================================================

print()

if checks == 0:

    print("=" * 100)
    print("PHASE 3 AUDIT STATUS: PASS")
    print("=" * 100)

    print(
        "No critical integrity problems "
        "were detected."
    )

    print(
        "Phase 3 can be frozen and "
        "Phase 4 can begin."
    )

else:

    print("=" * 100)
    print(
        "PHASE 3 AUDIT STATUS: "
        "REVIEW REQUIRED"
    )
    print("=" * 100)

    print(
        "Some audit checks require "
        "review before freezing Phase 3."
    )


conn.close()