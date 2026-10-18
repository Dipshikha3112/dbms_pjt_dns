from pathlib import Path
import os
import re
import json
import random
import hashlib
import getpass
from datetime import datetime

import pandas as pd
import psycopg


# ============================================================
# PHASE 6 - HEALTHCARE WORKLOAD GENERATOR
# Adaptive Healthcare Knowledge Database
# ============================================================

print("=" * 100)
print("PHASE 6 - HEALTHCARE WORKLOAD GENERATOR")
print("Adaptive Healthcare Knowledge Database")
print("=" * 100)


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"
WORKLOAD_DIR = RESULTS_DIR / "phase6_workloads"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
WORKLOAD_DIR.mkdir(parents=True, exist_ok=True)

DB_NAME = "adaptive_healthcare_db"
DB_HOST = os.getenv("PGHOST", "localhost")
DB_PORT = int(os.getenv("PGPORT", "5432"))
DB_USER = os.getenv("PGUSER", "postgres")
DB_PASSWORD = os.getenv("PGPASSWORD")

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

DOCTOR_QUERIES = 250
ADMIN_QUERIES = 200
RESEARCH_QUERIES = 200
MIXED_QUERIES = 300


# ============================================================
# 2. HELPER FUNCTIONS
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
        sql
    ).strip()


def query_hash(sql):
    return hashlib.sha256(
        clean_sql(sql).encode("utf-8")
    ).hexdigest()


def sql_literal(value):

    if value is None:
        return "NULL"

    value = str(value)

    return "'" + value.replace("'", "''") + "'"


def fetch_values(conn, sql):

    with conn.cursor() as cur:

        cur.execute(sql)

        return [
            row[0]
            for row in cur.fetchall()
            if row[0] is not None
        ]


def get_table_columns(conn, table_name):

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

        return {
            row[0]
            for row in cur.fetchall()
        }


def make_record(
    workload_type,
    query_family,
    query_name,
    sql,
    complexity,
    expected_use
):

    sql = clean_sql(sql)

    return {
        "workload_type": workload_type,
        "query_family": query_family,
        "query_name": query_name,
        "complexity": complexity,
        "expected_use": expected_use,
        "query_hash": query_hash(sql),
        "sql": sql
    }


def save_workload(name, records):

    df = pd.DataFrame(records)

    csv_path = (
        WORKLOAD_DIR /
        f"{name}_workload.csv"
    )

    json_path = (
        WORKLOAD_DIR /
        f"{name}_workload.json"
    )

    sql_path = (
        WORKLOAD_DIR /
        f"{name}_workload.sql"
    )

    df.to_csv(
        csv_path,
        index=False
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            records,
            f,
            indent=4,
            default=str
        )

    with open(
        sql_path,
        "w",
        encoding="utf-8"
    ) as f:

        for i, record in enumerate(
            records,
            start=1
        ):

            f.write(
                f"-- Query {i}\n"
            )

            f.write(
                f"-- Workload: "
                f"{record['workload_type']}\n"
            )

            f.write(
                f"-- Family: "
                f"{record['query_family']}\n"
            )

            f.write(
                f"-- Name: "
                f"{record['query_name']}\n"
            )

            f.write(
                f"-- Complexity: "
                f"{record['complexity']}\n"
            )

            f.write(
                record["sql"]
            )

            f.write(";\n\n")

    return df


# ============================================================
# 3. DATABASE CONNECTION
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

print("Connected to PostgreSQL.")
print("Database:", DB_NAME)


# ============================================================
# 4. SCHEMA VALIDATION
# ============================================================

stage("VALIDATING PHASE 3 DATABASE SCHEMA")


EXPECTED_SCHEMA = {

    "patient": {
        "patient_id",
        "birth_date",
        "first_name",
        "last_name",
        "gender"
    },

    "organization": {
        "organization_id",
        "name"
    },

    "provider": {
        "provider_id",
        "organization_id",
        "name",
        "specialty"
    },

    "encounter": {
        "encounter_id",
        "patient_id",
        "provider_id",
        "organization_id",
        "start_time",
        "end_time",
        "encounter_class",
        "description",
        "base_cost",
        "total_claim_cost",
        "payer_coverage"
    },

    "condition": {
        "condition_id",
        "source_code",
        "description"
    },

    "diagnosis": {
        "diagnosis_id",
        "patient_id",
        "encounter_id",
        "condition_id",
        "start_date",
        "stop_date"
    },

    "procedure_definition": {
        "procedure_id",
        "source_code",
        "description"
    },

    "patient_procedure": {
        "patient_procedure_id",
        "patient_id",
        "encounter_id",
        "procedure_id",
        "procedure_date",
        "base_cost"
    },

    "drug_concept": {
        "rxcui",
        "name",
        "concept_type"
    },

    "prescription": {
        "prescription_id",
        "patient_id",
        "encounter_id",
        "start_time",
        "stop_time"
    },

    "prescription_item": {
        "prescription_item_id",
        "prescription_id",
        "rxcui",
        "description",
        "base_cost",
        "payer_coverage",
        "dispenses",
        "total_cost"
    },

    "loinc_concept": {
        "loinc_code",
        "long_common_name"
    },

    "observation": {
        "observation_id",
        "patient_id",
        "encounter_id",
        "observation_time",
        "loinc_code",
        "value_text",
        "value_numeric",
        "unit"
    },

    "lab_result": {
        "lab_result_id",
        "patient_id",
        "encounter_id",
        "loinc_code",
        "result_time",
        "result_value",
        "result_numeric",
        "unit"
    },

    "payer": {
        "payer_id",
        "name"
    },

    "claim": {
        "claim_id",
        "patient_id",
        "encounter_id",
        "payer_id",
        "provider_id",
        "outstanding_1",
        "outstanding_2"
    },

    "claim_transaction": {
        "claim_transaction_id",
        "claim_id",
        "transaction_type",
        "transaction_method",
        "transaction_date",
        "amount",
        "patient_payment"
    }
}


schema_errors = []

for table_name, expected_columns in EXPECTED_SCHEMA.items():

    actual_columns = get_table_columns(
        conn,
        table_name
    )

    if not actual_columns:

        schema_errors.append(
            f"Missing table: {table_name}"
        )

        print(
            f"FAIL  {table_name:<30} "
            "TABLE NOT FOUND"
        )

        continue

    missing_columns = (
        expected_columns -
        actual_columns
    )

    if missing_columns:

        schema_errors.append(
            f"{table_name}: missing "
            f"{sorted(missing_columns)}"
        )

        print(
            f"FAIL  {table_name:<30} "
            f"missing {sorted(missing_columns)}"
        )

    else:

        print(
            f"PASS  {table_name:<30} "
            f"{len(actual_columns)} columns"
        )


if schema_errors:

    print()
    print("=" * 100)
    print("SCHEMA VALIDATION FAILED")
    print("=" * 100)

    for error in schema_errors:
        print(" -", error)

    conn.close()

    raise RuntimeError(
        "Phase 6 stopped because the "
        "database schema does not match."
    )


print()
print(
    "Schema validation passed."
)


# ============================================================
# 5. LOAD REAL DATABASE REFERENCE VALUES
# ============================================================

stage("LOADING WORKLOAD REFERENCE VALUES")


patient_ids = fetch_values(
    conn,
    """
    SELECT patient_id
    FROM patient
    """
)


provider_ids = fetch_values(
    conn,
    """
    SELECT DISTINCT provider_id
    FROM encounter
    WHERE provider_id IS NOT NULL
    """
)


organization_ids = fetch_values(
    conn,
    """
    SELECT DISTINCT organization_id
    FROM encounter
    WHERE organization_id IS NOT NULL
    """
)


payer_ids = fetch_values(
    conn,
    """
    SELECT DISTINCT payer_id
    FROM claim
    WHERE payer_id IS NOT NULL
    """
)


condition_ids = fetch_values(
    conn,
    """
    SELECT DISTINCT condition_id
    FROM diagnosis
    WHERE condition_id IS NOT NULL
    """
)


loinc_codes = fetch_values(
    conn,
    """
    SELECT DISTINCT loinc_code
    FROM lab_result
    WHERE loinc_code IS NOT NULL
    """
)


rxcui_values = fetch_values(
    conn,
    """
    SELECT DISTINCT rxcui
    FROM prescription_item
    WHERE rxcui IS NOT NULL
    """
)


procedure_ids = fetch_values(
    conn,
    """
    SELECT DISTINCT procedure_id
    FROM patient_procedure
    WHERE procedure_id IS NOT NULL
    """
)


print(
    "Patients:",
    f"{len(patient_ids):,}"
)

print(
    "Active providers:",
    f"{len(provider_ids):,}"
)

print(
    "Active organizations:",
    f"{len(organization_ids):,}"
)

print(
    "Payers used in claims:",
    f"{len(payer_ids):,}"
)

print(
    "Conditions used:",
    f"{len(condition_ids):,}"
)

print(
    "LOINC codes used:",
    f"{len(loinc_codes):,}"
)

print(
    "RxNorm codes used:",
    f"{len(rxcui_values):,}"
)

print(
    "Procedure definitions used:",
    f"{len(procedure_ids):,}"
)


if not patient_ids:
    raise RuntimeError(
        "No patients found."
    )

if not provider_ids:
    raise RuntimeError(
        "No providers found in encounters."
    )

if not condition_ids:
    raise RuntimeError(
        "No diagnoses found."
    )

if not loinc_codes:
    raise RuntimeError(
        "No LOINC-coded lab results found."
    )


# ============================================================
# 6. DOCTOR WORKLOAD
# ============================================================

stage("GENERATING DOCTOR WORKLOAD")

doctor_records = []


def doctor_patient_summary():

    patient = random.choice(
        patient_ids
    )

    sql = f"""
        SELECT
            p.patient_id,
            p.first_name,
            p.last_name,
            p.birth_date,
            p.gender,
            p.race,
            p.ethnicity,
            p.marital_status
        FROM patient p
        WHERE p.patient_id =
            {sql_literal(patient)}
    """

    return make_record(
        "doctor",
        "patient_lookup",
        "Patient demographic summary",
        sql,
        "LOW",
        "Retrieve patient demographic information"
    )


def doctor_encounter_history():

    patient = random.choice(
        patient_ids
    )

    sql = f"""
        SELECT
            e.encounter_id,
            e.start_time,
            e.end_time,
            e.encounter_class,
            e.description,
            p.name AS provider_name,
            o.name AS organization_name
        FROM encounter e
        LEFT JOIN provider p
            ON e.provider_id =
               p.provider_id
        LEFT JOIN organization o
            ON e.organization_id =
               o.organization_id
        WHERE e.patient_id =
            {sql_literal(patient)}
        ORDER BY e.start_time DESC
    """

    return make_record(
        "doctor",
        "encounter_history",
        "Patient encounter history",
        sql,
        "MEDIUM",
        "Review previous clinical encounters"
    )


def doctor_diagnosis_history():

    patient = random.choice(
        patient_ids
    )

    sql = f"""
        SELECT
            d.diagnosis_id,
            d.start_date,
            d.stop_date,
            c.source_code,
            c.description
        FROM diagnosis d
        JOIN condition c
            ON d.condition_id =
               c.condition_id
        WHERE d.patient_id =
            {sql_literal(patient)}
        ORDER BY d.start_date DESC
    """

    return make_record(
        "doctor",
        "diagnosis_history",
        "Patient diagnosis history",
        sql,
        "MEDIUM",
        "Review patient diagnoses"
    )


def doctor_medication_history():

    patient = random.choice(
        patient_ids
    )

    sql = f"""
        SELECT
            pr.prescription_id,
            pr.start_time,
            pr.stop_time,
            pi.rxcui,
            COALESCE(
                dc.name,
                pi.description
            ) AS medication,
            pi.dispenses,
            pi.total_cost
        FROM prescription pr
        JOIN prescription_item pi
            ON pr.prescription_id =
               pi.prescription_id
        LEFT JOIN drug_concept dc
            ON pi.rxcui =
               dc.rxcui
        WHERE pr.patient_id =
            {sql_literal(patient)}
        ORDER BY pr.start_time DESC
    """

    return make_record(
        "doctor",
        "medication_history",
        "Patient medication history",
        sql,
        "MEDIUM",
        "Review medications prescribed to patient"
    )


def doctor_lab_history():

    patient = random.choice(
        patient_ids
    )

    sql = f"""
        SELECT
            lr.lab_result_id,
            lr.result_time,
            lr.loinc_code,
            lc.long_common_name,
            lr.result_value,
            lr.result_numeric,
            lr.unit
        FROM lab_result lr
        LEFT JOIN loinc_concept lc
            ON lr.loinc_code =
               lc.loinc_code
        WHERE lr.patient_id =
            {sql_literal(patient)}
        ORDER BY lr.result_time DESC
    """

    return make_record(
        "doctor",
        "lab_history",
        "Patient laboratory history",
        sql,
        "MEDIUM",
        "Review patient laboratory history"
    )


def doctor_recent_labs():

    patient = random.choice(
        patient_ids
    )

    sql = f"""
        SELECT
            lr.result_time,
            lc.long_common_name,
            lr.result_value,
            lr.result_numeric,
            lr.unit
        FROM lab_result lr
        LEFT JOIN loinc_concept lc
            ON lr.loinc_code =
               lc.loinc_code
        WHERE lr.patient_id =
            {sql_literal(patient)}
        ORDER BY lr.result_time DESC
        LIMIT 20
    """

    return make_record(
        "doctor",
        "recent_labs",
        "Recent laboratory results",
        sql,
        "MEDIUM",
        "Retrieve recent laboratory results"
    )


def doctor_procedure_history():

    patient = random.choice(
        patient_ids
    )

    sql = f"""
        SELECT
            pp.patient_procedure_id,
            pp.procedure_date,
            pd.source_code,
            pd.description,
            pp.base_cost,
            pp.reason_description
        FROM patient_procedure pp
        JOIN procedure_definition pd
            ON pp.procedure_id =
               pd.procedure_id
        WHERE pp.patient_id =
            {sql_literal(patient)}
        ORDER BY pp.procedure_date DESC
    """

    return make_record(
        "doctor",
        "procedure_history",
        "Patient procedure history",
        sql,
        "MEDIUM",
        "Review procedures performed"
    )


def doctor_allergy_history():

    patient = random.choice(
        patient_ids
    )

    sql = f"""
        SELECT
            start_date,
            stop_date,
            code,
            description,
            allergy_type,
            category,
            severity_1
        FROM patient_allergy
        WHERE patient_id =
            {sql_literal(patient)}
        ORDER BY start_date DESC
    """

    return make_record(
        "doctor",
        "allergy_history",
        "Patient allergy history",
        sql,
        "LOW",
        "Review known patient allergies"
    )


def doctor_complete_summary():

    patient = random.choice(
        patient_ids
    )

    sql = f"""
        SELECT
            p.patient_id,
            p.first_name,
            p.last_name,

            (
                SELECT COUNT(*)
                FROM encounter e
                WHERE e.patient_id =
                      p.patient_id
            ) AS encounter_count,

            (
                SELECT COUNT(*)
                FROM diagnosis d
                WHERE d.patient_id =
                      p.patient_id
            ) AS diagnosis_count,

            (
                SELECT COUNT(*)
                FROM prescription pr
                WHERE pr.patient_id =
                      p.patient_id
            ) AS prescription_count,

            (
                SELECT COUNT(*)
                FROM lab_result lr
                WHERE lr.patient_id =
                      p.patient_id
            ) AS lab_count,

            (
                SELECT COUNT(*)
                FROM patient_procedure pp
                WHERE pp.patient_id =
                      p.patient_id
            ) AS procedure_count

        FROM patient p

        WHERE p.patient_id =
            {sql_literal(patient)}
    """

    return make_record(
        "doctor",
        "complete_patient_summary",
        "Complete patient clinical summary",
        sql,
        "HIGH",
        "Retrieve consolidated patient history"
    )


doctor_generators = [
    doctor_patient_summary,
    doctor_encounter_history,
    doctor_diagnosis_history,
    doctor_medication_history,
    doctor_lab_history,
    doctor_recent_labs,
    doctor_procedure_history,
    doctor_allergy_history,
    doctor_complete_summary
]


doctor_weights = [
    15,
    14,
    14,
    14,
    14,
    10,
    8,
    5,
    6
]


for _ in range(
    DOCTOR_QUERIES
):

    generator = random.choices(
        doctor_generators,
        weights=doctor_weights,
        k=1
    )[0]

    doctor_records.append(
        generator()
    )


print(
    "Doctor workload generated:",
    len(doctor_records)
)


# ============================================================
# 7. ADMIN WORKLOAD
# ============================================================

stage("GENERATING ADMIN WORKLOAD")

admin_records = []


def admin_provider_workload():

    provider = random.choice(
        provider_ids
    )

    sql = f"""
        SELECT
            p.provider_id,
            p.name,
            p.specialty,
            COUNT(e.encounter_id)
                AS encounter_count,
            COUNT(
                DISTINCT e.patient_id
            ) AS unique_patients
        FROM provider p
        LEFT JOIN encounter e
            ON p.provider_id =
               e.provider_id
        WHERE p.provider_id =
            {sql_literal(provider)}
        GROUP BY
            p.provider_id,
            p.name,
            p.specialty
    """

    return make_record(
        "admin",
        "provider_workload",
        "Provider workload",
        sql,
        "MEDIUM",
        "Measure provider clinical workload"
    )


def admin_organization_activity():

    organization = random.choice(
        organization_ids
    )

    sql = f"""
        SELECT
            o.organization_id,
            o.name,
            COUNT(
                DISTINCT p.provider_id
            ) AS provider_count,
            COUNT(
                DISTINCT e.encounter_id
            ) AS encounter_count,
            COUNT(
                DISTINCT e.patient_id
            ) AS patient_count
        FROM organization o
        LEFT JOIN provider p
            ON o.organization_id =
               p.organization_id
        LEFT JOIN encounter e
            ON o.organization_id =
               e.organization_id
        WHERE o.organization_id =
            {sql_literal(organization)}
        GROUP BY
            o.organization_id,
            o.name
    """

    return make_record(
        "admin",
        "organization_activity",
        "Organization activity",
        sql,
        "HIGH",
        "Measure organization utilization"
    )


def admin_claim_summary():

    sql = """
        SELECT
            COUNT(*) AS total_claims,
            AVG(outstanding_1)
                AS average_outstanding_1,
            AVG(outstanding_2)
                AS average_outstanding_2,
            SUM(outstanding_1)
                AS total_outstanding_1,
            SUM(outstanding_2)
                AS total_outstanding_2
        FROM claim
    """

    return make_record(
        "admin",
        "claim_summary",
        "Global claim summary",
        sql,
        "LOW",
        "Summarize financial claim activity"
    )


def admin_payer_claims():

    payer = random.choice(
        payer_ids
    )

    sql = f"""
        SELECT
            c.payer_id,
            p.name AS payer_name,
            COUNT(*) AS claim_count,
            AVG(c.outstanding_1)
                AS average_outstanding,
            SUM(c.outstanding_1)
                AS total_outstanding
        FROM claim c
        JOIN payer p
            ON c.payer_id =
               p.payer_id
        WHERE c.payer_id =
            {sql_literal(payer)}
        GROUP BY
            c.payer_id,
            p.name
    """

    return make_record(
        "admin",
        "payer_claims",
        "Payer claim activity",
        sql,
        "MEDIUM",
        "Analyze claim workload by payer"
    )


def admin_claim_transactions():

    sql = """
        SELECT
            transaction_type,
            COUNT(*) AS transaction_count,
            SUM(amount)
                AS total_amount,
            AVG(amount)
                AS average_amount,
            SUM(patient_payment)
                AS patient_payments
        FROM claim_transaction
        GROUP BY transaction_type
        ORDER BY transaction_count DESC
    """

    return make_record(
        "admin",
        "claim_transactions",
        "Claim transaction summary",
        sql,
        "MEDIUM",
        "Analyze claim transaction activity"
    )


def admin_encounter_classes():

    sql = """
        SELECT
            encounter_class,
            COUNT(*) AS encounter_count,
            AVG(base_cost)
                AS average_base_cost,
            AVG(total_claim_cost)
                AS average_claim_cost,
            AVG(payer_coverage)
                AS average_payer_coverage
        FROM encounter
        GROUP BY encounter_class
        ORDER BY encounter_count DESC
    """

    return make_record(
        "admin",
        "encounter_statistics",
        "Encounter class statistics",
        sql,
        "MEDIUM",
        "Analyze utilization by encounter class"
    )


def admin_monthly_encounters():

    sql = """
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
        ORDER BY month
    """

    return make_record(
        "admin",
        "temporal_activity",
        "Monthly encounter activity",
        sql,
        "MEDIUM",
        "Monitor encounter volume over time"
    )


def admin_provider_ranking():

    sql = """
        SELECT
            p.provider_id,
            p.name,
            p.specialty,
            COUNT(e.encounter_id)
                AS encounter_count,
            COUNT(
                DISTINCT e.patient_id
            ) AS patient_count
        FROM provider p
        JOIN encounter e
            ON p.provider_id =
               e.provider_id
        GROUP BY
            p.provider_id,
            p.name,
            p.specialty
        ORDER BY encounter_count DESC
        LIMIT 25
    """

    return make_record(
        "admin",
        "provider_ranking",
        "Most active providers",
        sql,
        "MEDIUM",
        "Identify providers with highest workloads"
    )


def admin_transaction_monthly():

    sql = """
        SELECT
            DATE_TRUNC(
                'month',
                transaction_date
            ) AS month,
            COUNT(*) AS transaction_count,
            SUM(amount) AS total_amount,
            SUM(patient_payment)
                AS patient_payment
        FROM claim_transaction
        WHERE transaction_date IS NOT NULL
        GROUP BY
            DATE_TRUNC(
                'month',
                transaction_date
            )
        ORDER BY month
    """

    return make_record(
        "admin",
        "financial_temporal",
        "Monthly financial transactions",
        sql,
        "MEDIUM",
        "Analyze transaction trends over time"
    )


admin_generators = [
    admin_provider_workload,
    admin_organization_activity,
    admin_claim_summary,
    admin_payer_claims,
    admin_claim_transactions,
    admin_encounter_classes,
    admin_monthly_encounters,
    admin_provider_ranking,
    admin_transaction_monthly
]


admin_weights = [
    15,
    10,
    10,
    15,
    10,
    15,
    10,
    10,
    5
]


for _ in range(
    ADMIN_QUERIES
):

    generator = random.choices(
        admin_generators,
        weights=admin_weights,
        k=1
    )[0]

    admin_records.append(
        generator()
    )


print(
    "Admin workload generated:",
    len(admin_records)
)


# ============================================================
# 8. RESEARCH WORKLOAD
# ============================================================

stage("GENERATING RESEARCH WORKLOAD")

research_records = []


def research_condition_prevalence():

    sql = """
        SELECT
            c.source_code,
            c.description,
            COUNT(
                DISTINCT d.patient_id
            ) AS patient_count,
            COUNT(d.diagnosis_id)
                AS diagnosis_count
        FROM diagnosis d
        JOIN condition c
            ON d.condition_id =
               c.condition_id
        GROUP BY
            c.source_code,
            c.description
        ORDER BY patient_count DESC
        LIMIT 50
    """

    return make_record(
        "research",
        "condition_prevalence",
        "Condition prevalence",
        sql,
        "HIGH",
        "Study disease prevalence"
    )


def research_condition_cohort():

    condition = random.choice(
        condition_ids
    )

    sql = f"""
        SELECT DISTINCT
            p.patient_id,
            p.gender,
            p.birth_date,
            p.race,
            p.ethnicity
        FROM patient p
        JOIN diagnosis d
            ON p.patient_id =
               d.patient_id
        WHERE d.condition_id =
            {sql_literal(condition)}
    """

    return make_record(
        "research",
        "cohort_selection",
        "Condition patient cohort",
        sql,
        "MEDIUM",
        "Select patients diagnosed with condition"
    )


def research_drug_usage():

    sql = """
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
            ON pr.prescription_id =
               pi.prescription_id
        LEFT JOIN drug_concept dc
            ON pi.rxcui =
               dc.rxcui
        WHERE pi.rxcui IS NOT NULL
        GROUP BY
            dc.rxcui,
            dc.name
        ORDER BY prescription_count DESC
        LIMIT 50
    """

    return make_record(
        "research",
        "drug_utilization",
        "Drug utilization analysis",
        sql,
        "HIGH",
        "Analyze medication utilization"
    )


def research_specific_drug():

    if rxcui_values:

        rxcui = random.choice(
            rxcui_values
        )

    else:

        rxcui = "UNKNOWN"

    sql = f"""
        SELECT
            pi.rxcui,
            dc.name,
            COUNT(
                DISTINCT pr.patient_id
            ) AS patient_count,
            COUNT(*) AS prescription_count,
            AVG(pi.total_cost)
                AS average_cost
        FROM prescription_item pi
        JOIN prescription pr
            ON pi.prescription_id =
               pr.prescription_id
        LEFT JOIN drug_concept dc
            ON pi.rxcui =
               dc.rxcui
        WHERE pi.rxcui =
            {sql_literal(rxcui)}
        GROUP BY
            pi.rxcui,
            dc.name
    """

    return make_record(
        "research",
        "drug_cohort",
        "Specific drug utilization",
        sql,
        "MEDIUM",
        "Analyze use of a specific RxNorm concept"
    )


def research_lab_distribution():

    loinc = random.choice(
        loinc_codes
    )

    sql = f"""
        SELECT
            lr.loinc_code,
            lc.long_common_name,
            COUNT(*) AS result_count,
            AVG(lr.result_numeric)
                AS average_value,
            MIN(lr.result_numeric)
                AS minimum_value,
            MAX(lr.result_numeric)
                AS maximum_value
        FROM lab_result lr
        LEFT JOIN loinc_concept lc
            ON lr.loinc_code =
               lc.loinc_code
        WHERE lr.loinc_code =
            {sql_literal(loinc)}
          AND lr.result_numeric
              IS NOT NULL
        GROUP BY
            lr.loinc_code,
            lc.long_common_name
    """

    return make_record(
        "research",
        "lab_distribution",
        "Laboratory value distribution",
        sql,
        "MEDIUM",
        "Analyze numeric laboratory results"
    )


def research_age_distribution():

    sql = """
        SELECT
            FLOOR(
                EXTRACT(
                    YEAR FROM AGE(
                        CURRENT_DATE,
                        birth_date
                    )
                ) / 10
            ) * 10 AS age_group,
            COUNT(*) AS patient_count
        FROM patient
        WHERE birth_date IS NOT NULL
        GROUP BY age_group
        ORDER BY age_group
    """

    return make_record(
        "research",
        "demographic_analysis",
        "Patient age distribution",
        sql,
        "LOW",
        "Analyze age distribution"
    )


def research_gender_condition():

    sql = """
        SELECT
            p.gender,
            c.description,
            COUNT(
                DISTINCT p.patient_id
            ) AS patient_count
        FROM patient p
        JOIN diagnosis d
            ON p.patient_id =
               d.patient_id
        JOIN condition c
            ON d.condition_id =
               c.condition_id
        GROUP BY
            p.gender,
            c.description
        ORDER BY patient_count DESC
        LIMIT 100
    """

    return make_record(
        "research",
        "condition_demographics",
        "Condition distribution by gender",
        sql,
        "HIGH",
        "Analyze demographic disease patterns"
    )


def research_condition_medication():

    sql = """
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
            ON d.condition_id =
               c.condition_id
        JOIN prescription pr
            ON d.patient_id =
               pr.patient_id
        JOIN prescription_item pi
            ON pr.prescription_id =
               pi.prescription_id
        LEFT JOIN drug_concept dc
            ON pi.rxcui =
               dc.rxcui
        GROUP BY
            c.description,
            COALESCE(
                dc.name,
                pi.description
            )
        ORDER BY patient_count DESC
        LIMIT 100
    """

    return make_record(
        "research",
        "condition_medication",
        "Condition medication association",
        sql,
        "HIGH",
        "Explore disease-medication relationships"
    )


def research_patient_complexity():

    sql = """
        SELECT
            p.patient_id,

            (
                SELECT COUNT(*)
                FROM diagnosis d
                WHERE d.patient_id =
                      p.patient_id
            ) AS diagnosis_count,

            (
                SELECT COUNT(*)
                FROM prescription pr
                WHERE pr.patient_id =
                      p.patient_id
            ) AS prescription_count,

            (
                SELECT COUNT(*)
                FROM lab_result lr
                WHERE lr.patient_id =
                      p.patient_id
            ) AS lab_count,

            (
                SELECT COUNT(*)
                FROM encounter e
                WHERE e.patient_id =
                      p.patient_id
            ) AS encounter_count

        FROM patient p

        ORDER BY
            diagnosis_count DESC,
            prescription_count DESC

        LIMIT 100
    """

    return make_record(
        "research",
        "patient_complexity",
        "Complex patient cohort",
        sql,
        "HIGH",
        "Identify clinically complex patients"
    )


def research_procedure_usage():

    sql = """
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
            AVG(pp.base_cost)
                AS average_cost
        FROM patient_procedure pp
        JOIN procedure_definition pd
            ON pp.procedure_id =
               pd.procedure_id
        GROUP BY
            pd.procedure_id,
            pd.source_code,
            pd.description
        ORDER BY procedure_count DESC
        LIMIT 50
    """

    return make_record(
        "research",
        "procedure_utilization",
        "Procedure utilization analysis",
        sql,
        "HIGH",
        "Analyze procedure frequency and cost"
    )


def research_encounter_cost():

    sql = """
        SELECT
            encounter_class,
            COUNT(*) AS encounter_count,
            AVG(base_cost)
                AS average_base_cost,
            AVG(total_claim_cost)
                AS average_claim_cost,
            AVG(payer_coverage)
                AS average_coverage
        FROM encounter
        WHERE encounter_class IS NOT NULL
        GROUP BY encounter_class
        ORDER BY encounter_count DESC
    """

    return make_record(
        "research",
        "cost_analysis",
        "Encounter cost analysis",
        sql,
        "MEDIUM",
        "Analyze costs by encounter category"
    )


research_generators = [
    research_condition_prevalence,
    research_condition_cohort,
    research_drug_usage,
    research_specific_drug,
    research_lab_distribution,
    research_age_distribution,
    research_gender_condition,
    research_condition_medication,
    research_patient_complexity,
    research_procedure_usage,
    research_encounter_cost
]


research_weights = [
    12,
    12,
    12,
    8,
    12,
    8,
    8,
    8,
    7,
    7,
    6
]


for _ in range(
    RESEARCH_QUERIES
):

    generator = random.choices(
        research_generators,
        weights=research_weights,
        k=1
    )[0]

    research_records.append(
        generator()
    )


print(
    "Research workload generated:",
    len(research_records)
)


# ============================================================
# 9. MIXED WORKLOAD
# ============================================================

stage("GENERATING MIXED WORKLOAD")

mixed_records = []


for _ in range(
    MIXED_QUERIES
):

    workload = random.choices(
        [
            "doctor",
            "admin",
            "research"
        ],
        weights=[
            50,
            25,
            25
        ],
        k=1
    )[0]

    if workload == "doctor":

        record = random.choice(
            doctor_records
        ).copy()

    elif workload == "admin":

        record = random.choice(
            admin_records
        ).copy()

    else:

        record = random.choice(
            research_records
        ).copy()

    record[
        "source_workload"
    ] = workload

    record[
        "workload_type"
    ] = "mixed"

    mixed_records.append(
        record
    )


print(
    "Mixed workload generated:",
    len(mixed_records)
)


# ============================================================
# 10. VALIDATE GENERATED SQL
# ============================================================

stage("VALIDATING GENERATED SQL WITH POSTGRESQL")


base_records = (
    doctor_records +
    admin_records +
    research_records
)


unique_queries = {}

for record in base_records:

    unique_queries[
        record["query_hash"]
    ] = record


print(
    "Base workload instances:",
    f"{len(base_records):,}"
)

print(
    "Unique generated SQL:",
    f"{len(unique_queries):,}"
)

print()
print(
    "Running EXPLAIN validation..."
)


validation_records = []

valid_count = 0
invalid_count = 0


for i, record in enumerate(
    unique_queries.values(),
    start=1
):

    try:

        with conn.cursor() as cur:

            cur.execute(
                "EXPLAIN " +
                record["sql"]
            )

            cur.fetchall()

        valid = True
        error = ""

        valid_count += 1

    except Exception as exc:

        conn.rollback()

        valid = False
        error = str(exc)

        invalid_count += 1

        print()
        print(
            "INVALID QUERY:"
        )

        print(
            record["query_name"]
        )

        print(
            record["sql"]
        )

        print(
            "ERROR:"
        )

        print(
            error
        )

    validation_records.append({

        "query_hash":
            record["query_hash"],

        "workload_type":
            record["workload_type"],

        "query_family":
            record["query_family"],

        "query_name":
            record["query_name"],

        "valid":
            valid,

        "error":
            error
    })


    if (
        i % 100 == 0
        or
        i == len(unique_queries)
    ):

        print(
            f"Validated "
            f"{i:,}/"
            f"{len(unique_queries):,}"
        )


print()
print(
    "Valid queries:",
    f"{valid_count:,}"
)

print(
    "Invalid queries:",
    f"{invalid_count:,}"
)


# ============================================================
# 11. STOP IF ANY SQL IS INVALID
# ============================================================

validation_df = pd.DataFrame(
    validation_records
)

validation_df.to_csv(
    WORKLOAD_DIR /
    "query_validation.csv",
    index=False
)


if invalid_count > 0:

    print()
    print("=" * 100)
    print(
        "PHASE 6 STATUS: REVIEW REQUIRED"
    )
    print("=" * 100)

    print(
        "One or more generated SQL "
        "queries failed validation."
    )

    print(
        "No workload has been approved "
        "for Phase 7."
    )

    print()
    print(
        "See:"
    )

    print(
        WORKLOAD_DIR /
        "query_validation.csv"
    )

    conn.close()

    raise RuntimeError(
        f"{invalid_count} generated "
        "queries failed validation."
    )


# ============================================================
# 12. SAVE WORKLOADS
# ============================================================

stage("SAVING WORKLOAD DATASETS")


doctor_df = save_workload(
    "doctor",
    doctor_records
)


admin_df = save_workload(
    "admin",
    admin_records
)


research_df = save_workload(
    "research",
    research_records
)


mixed_df = save_workload(
    "mixed",
    mixed_records
)


combined_records = (
    doctor_records +
    admin_records +
    research_records +
    mixed_records
)


combined_df = pd.DataFrame(
    combined_records
)


combined_df.to_csv(
    WORKLOAD_DIR /
    "combined_workload.csv",
    index=False
)


with open(
    WORKLOAD_DIR /
    "combined_workload.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        combined_records,
        f,
        indent=4,
        default=str
    )


# ============================================================
# 13. WORKLOAD SUMMARY
# ============================================================

stage("WORKLOAD SUMMARY")


summary_rows = []


for workload_name, dataframe in [

    (
        "doctor",
        doctor_df
    ),

    (
        "admin",
        admin_df
    ),

    (
        "research",
        research_df
    ),

    (
        "mixed",
        mixed_df
    )

]:

    complexity = (
        dataframe[
            "complexity"
        ]
        .value_counts()
        .to_dict()
    )

    summary_rows.append({

        "workload":
            workload_name,

        "queries":
            len(dataframe),

        "unique_sql":
            dataframe[
                "query_hash"
            ].nunique(),

        "query_families":
            dataframe[
                "query_family"
            ].nunique(),

        "low_complexity":
            complexity.get(
                "LOW",
                0
            ),

        "medium_complexity":
            complexity.get(
                "MEDIUM",
                0
            ),

        "high_complexity":
            complexity.get(
                "HIGH",
                0
            )
    })


summary_df = pd.DataFrame(
    summary_rows
)


summary_df.to_csv(
    WORKLOAD_DIR /
    "workload_summary.csv",
    index=False
)


print(
    summary_df.to_string(
        index=False
    )
)


# ============================================================
# 14. QUERY FAMILY DISTRIBUTION
# ============================================================

stage("QUERY FAMILY DISTRIBUTION")


family_distribution = (

    combined_df

    .groupby(
        [
            "workload_type",
            "query_family"
        ]
    )

    .size()

    .reset_index(
        name="frequency"
    )
)


family_distribution.to_csv(
    WORKLOAD_DIR /
    "query_family_distribution.csv",
    index=False
)


print(
    family_distribution.to_string(
        index=False
    )
)


# ============================================================
# 15. COMPLEXITY DISTRIBUTION
# ============================================================

stage("COMPLEXITY DISTRIBUTION")


complexity_distribution = (

    combined_df[
        "complexity"
    ]

    .value_counts()

    .rename_axis(
        "complexity"
    )

    .reset_index(
        name="frequency"
    )
)


complexity_distribution.to_csv(
    WORKLOAD_DIR /
    "complexity_distribution.csv",
    index=False
)


print(
    complexity_distribution.to_string(
        index=False
    )
)


# ============================================================
# 16. SOURCE WORKLOAD DISTRIBUTION FOR MIXED LOAD
# ============================================================

stage("MIXED WORKLOAD DISTRIBUTION")


mixed_distribution = (

    mixed_df[
        "source_workload"
    ]

    .value_counts()

    .rename_axis(
        "source_workload"
    )

    .reset_index(
        name="frequency"
    )
)


mixed_distribution[
    "percentage"
] = (

    mixed_distribution[
        "frequency"
    ]

    / len(mixed_df)

    * 100
)


mixed_distribution.to_csv(
    WORKLOAD_DIR /
    "mixed_workload_distribution.csv",
    index=False
)


print(
    mixed_distribution.to_string(
        index=False
    )
)


# ============================================================
# 17. PHASE 6 METADATA
# ============================================================

stage("GENERATING PHASE 6 METADATA")


metadata = {

    "phase": 6,

    "phase_name":
        "Healthcare Workload Generator",

    "generated_at":
        datetime.now().isoformat(),

    "database":
        DB_NAME,

    "random_seed":
        RANDOM_SEED,

    "workloads": {

        "doctor": {
            "query_instances":
                len(doctor_records),

            "description":
                "Clinical patient lookup, history, "
                "diagnosis, medication, laboratory "
                "and procedure workload"
        },

        "admin": {
            "query_instances":
                len(admin_records),

            "description":
                "Provider, organization, claims, "
                "financial and utilization workload"
        },

        "research": {
            "query_instances":
                len(research_records),

            "description":
                "Cohort, prevalence, drug, lab, "
                "demographic and analytical workload"
        },

        "mixed": {
            "query_instances":
                len(mixed_records),

            "description":
                "Production-style mixed workload"
        }
    },

    "total_query_instances":
        len(combined_records),

    "base_query_instances":
        len(base_records),

    "unique_base_sql":
        len(unique_queries),

    "sql_validation": {

        "valid":
            valid_count,

        "invalid":
            invalid_count
    },

    "phase7_note":
        (
            "Phase 6 only generates workloads "
            "and validates SQL using EXPLAIN. "
            "Execution-time, EXPLAIN ANALYZE, "
            "I/O, scan and index metrics are "
            "collected in Phase 7."
        )
}


with open(
    WORKLOAD_DIR /
    "phase6_metadata.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )


# ============================================================
# 18. TEXT REPORT
# ============================================================

report = f"""
PHASE 6 - HEALTHCARE WORKLOAD GENERATOR
======================================================================

Database:
{DB_NAME}

Random seed:
{RANDOM_SEED}

WORKLOADS
----------------------------------------------------------------------

Doctor workload:
{len(doctor_records):,}

Admin workload:
{len(admin_records):,}

Research workload:
{len(research_records):,}

Mixed workload:
{len(mixed_records):,}

Total workload instances:
{len(combined_records):,}

Base workload instances:
{len(base_records):,}

Unique base SQL statements:
{len(unique_queries):,}


SQL VALIDATION
----------------------------------------------------------------------

Valid SQL:
{valid_count:,}

Invalid SQL:
{invalid_count:,}


PHASE BOUNDARY
----------------------------------------------------------------------

Phase 6:
Workload generation and SQL validation.

Phase 7:
Query execution and monitoring.

Phase 7 will collect:

- query fingerprint
- execution time
- EXPLAIN ANALYZE
- estimated cost
- actual rows
- joins
- filters
- sequential scans
- index scans
- shared blocks
- read blocks
- query frequency
- query_log records


STATUS
----------------------------------------------------------------------

PHASE 6 COMPLETE
"""


with open(
    WORKLOAD_DIR /
    "phase6_report.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        report.strip()
    )


# ============================================================
# 19. FINAL OUTPUT
# ============================================================

stage("PHASE 6 FINAL RESULT")


print(
    "Doctor queries          :",
    f"{len(doctor_records):,}"
)

print(
    "Admin queries           :",
    f"{len(admin_records):,}"
)

print(
    "Research queries        :",
    f"{len(research_records):,}"
)

print(
    "Mixed queries           :",
    f"{len(mixed_records):,}"
)

print(
    "Total workload queries  :",
    f"{len(combined_records):,}"
)

print(
    "Unique base SQL         :",
    f"{len(unique_queries):,}"
)

print(
    "SQL validation passed   :",
    f"{valid_count:,}"
)

print(
    "SQL validation failed   :",
    f"{invalid_count:,}"
)


print()
print(
    "Output directory:"
)

print(
    WORKLOAD_DIR
)


print()
print(
    "Generated files:"
)


for path in sorted(
    WORKLOAD_DIR.iterdir()
):

    print(
        " -",
        path.name
    )


print()
print("=" * 100)
print("PHASE 6 STATUS: COMPLETE")
print("=" * 100)

print(
    "Database schema validation       PASS"
)

print(
    "Doctor workload generation       COMPLETE"
)

print(
    "Admin workload generation        COMPLETE"
)

print(
    "Research workload generation     COMPLETE"
)

print(
    "Mixed workload generation        COMPLETE"
)

print(
    "PostgreSQL SQL validation        PASS"
)

print(
    "Workload reports                 COMPLETE"
)

print()
print(
    "NEXT:"
)

print(
    "PHASE 7 -> QUERY MONITORING"
)


conn.close()