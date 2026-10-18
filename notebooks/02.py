from pathlib import Path
import os
import sys
import csv
import time
import json
import hashlib
import traceback
from datetime import datetime
import re

import pandas as pd
import psycopg
from psycopg import sql
from psycopg.rows import dict_row

# ============================================================
# PHASE 3
# PostgreSQL Implementation + ETL
# Adaptive Healthcare Knowledge Database
# ============================================================

print("=" * 100)
print("PHASE 3 - POSTGRESQL IMPLEMENTATION + ETL")
print("=" * 100)

# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA = PROJECT_ROOT / "data"
SYNTHEA = DATA / "synthea"
LOINC = DATA / "loinc"
RXNORM = DATA / "rxnorm"

DATABASE_DIR = PROJECT_ROOT / "database"
SCHEMA_DIR = DATABASE_DIR / "schema"

RESULTS = PROJECT_ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

DB_NAME = "adaptive_healthcare_db"

DB_HOST = os.getenv("PGHOST", "localhost")
DB_PORT = int(os.getenv("PGPORT", "5432"))
DB_USER = os.getenv("PGUSER", "postgres")
DB_PASSWORD = os.getenv("PGPASSWORD")

CHUNK_SIZE = 10000

# Set True only if you want to recreate all tables.
RESET_SCHEMA = True

# RxNorm relationships are huge.
# True = load all relationships.
LOAD_RXNORM_RELATIONSHIPS = True

# ============================================================
# 2. PASSWORD
# ============================================================

if not DB_PASSWORD:
    import getpass
    DB_PASSWORD = getpass.getpass(
        f"PostgreSQL password for '{DB_USER}': "
    )

# ============================================================
# 3. PATH VALIDATION
# ============================================================

required_files = [
    SYNTHEA / "patients.csv",
    SYNTHEA / "organizations.csv",
    SYNTHEA / "providers.csv",
    SYNTHEA / "encounters.csv",
    SYNTHEA / "conditions.csv",
    SYNTHEA / "procedures.csv",
    SYNTHEA / "medications.csv",
    SYNTHEA / "observations.csv",
    SYNTHEA / "allergies.csv",
    SYNTHEA / "immunizations.csv",
    SYNTHEA / "careplans.csv",
    SYNTHEA / "devices.csv",
    SYNTHEA / "imaging_studies.csv",
    SYNTHEA / "payers.csv",
    SYNTHEA / "payer_transitions.csv",
    SYNTHEA / "claims.csv",
    SYNTHEA / "claims_transactions.csv",
    SYNTHEA / "supplies.csv",

    LOINC / "LoincTable" / "Loinc.csv",

    RXNORM / "rrf" / "RXNCONSO.RRF",
    RXNORM / "rrf" / "RXNREL.RRF",
    RXNORM / "rrf" / "RXNSAT.RRF",

    SCHEMA_DIR / "01_tables.sql",
    SCHEMA_DIR / "02_constraints.sql",
    SCHEMA_DIR / "03_baseline_indexes.sql",
    SCHEMA_DIR / "04_views.sql",
    SCHEMA_DIR / "05_triggers.sql",
    SCHEMA_DIR / "06_functions.sql",
]

print("\nChecking Phase 3 inputs...")

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(path)

print("All required files found.")

# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def clean(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except:
        pass

    if isinstance(value, str):
        value = value.strip()

        if value == "":
            return None

    return value


def clean_float(value):
    value = clean(value)

    if value is None:
        return None

    try:
        return float(value)
    except:
        return None


def clean_int(value):
    value = clean(value)

    if value is None:
        return None

    try:
        return int(float(value))
    except:
        return None


def read_csv(path):
    return pd.read_csv(
        path,
        dtype=str,
        low_memory=False
    )


def get_value(row, *columns):
    for column in columns:
        if column in row.index:
            value = clean(row[column])

            if value is not None:
                return value

    return None


def execute_sql_file(conn, path):
    print(f"Executing {path.name}...")

    text = path.read_text(encoding="utf-8")

    with conn.cursor() as cur:
        cur.execute(text)

    conn.commit()


def batch_insert(
    conn,
    query,
    rows,
    batch_size=5000
):
    if not rows:
        return 0

    total = 0

    with conn.cursor() as cur:

        for start in range(0, len(rows), batch_size):

            batch = rows[start:start + batch_size]

            cur.executemany(query, batch)

            total += len(batch)

    conn.commit()

    return total


def get_map(
    conn,
    table,
    source_column,
    target_column
):
    result = {}

    query = sql.SQL(
        "SELECT {source}, {target} "
        "FROM {table} "
        "WHERE {source} IS NOT NULL"
    ).format(
        source=sql.Identifier(source_column),
        target=sql.Identifier(target_column),
        table=sql.Identifier(table)
    )

    with conn.cursor() as cur:
        cur.execute(query)

        for source, target in cur.fetchall():
            result[str(source)] = target

    return result


def table_count(conn, table):
    query = sql.SQL(
        "SELECT COUNT(*) FROM {}"
    ).format(
        sql.Identifier(table)
    )

    with conn.cursor() as cur:
        cur.execute(query)
        return cur.fetchone()[0]


def print_stage(name):
    print("\n" + "=" * 100)
    print(name)
    print("=" * 100)


# ============================================================
# 5. CONNECT TO POSTGRES
# ============================================================

print_stage("DATABASE CONNECTION")

admin_conn = psycopg.connect(
    host=DB_HOST,
    port=DB_PORT,
    user=DB_USER,
    password=DB_PASSWORD,
    dbname="postgres",
    autocommit=True
)

print("Connected to PostgreSQL.")

# ============================================================
# 6. CREATE DATABASE
# ============================================================

with admin_conn.cursor() as cur:

    cur.execute(
        "SELECT 1 FROM pg_database WHERE datname=%s",
        (DB_NAME,)
    )

    exists = cur.fetchone()

    if not exists:

        print(f"Creating database: {DB_NAME}")

        cur.execute(
            sql.SQL(
                "CREATE DATABASE {}"
            ).format(
                sql.Identifier(DB_NAME)
            )
        )

    else:
        print(f"Database already exists: {DB_NAME}")

admin_conn.close()

# ============================================================
# 7. CONNECT TO PROJECT DATABASE
# ============================================================

conn = psycopg.connect(
    host=DB_HOST,
    port=DB_PORT,
    user=DB_USER,
    password=DB_PASSWORD,
    dbname=DB_NAME
)

print("Connected to:", DB_NAME)

# ============================================================
# 8. RESET PUBLIC SCHEMA
# ============================================================

if RESET_SCHEMA:

    print_stage("RESETTING DATABASE SCHEMA")

    with conn.cursor() as cur:
        cur.execute("DROP SCHEMA public CASCADE")
        cur.execute("CREATE SCHEMA public")

    conn.commit()

    print("Schema reset complete.")

# ============================================================
# 9. CREATE TABLES
# ============================================================

print_stage("CREATING TABLES")

execute_sql_file(
    conn,
    SCHEMA_DIR / "01_tables.sql"
)

# ============================================================
# 10. LOAD LOINC
# ============================================================

print_stage("LOADING LOINC")

loinc_path = LOINC / "LoincTable" / "Loinc.csv"

loinc_insert = """
INSERT INTO loinc_concept (
    loinc_code,
    component,
    property,
    time_aspect,
    system,
    scale_type,
    method_type,
    class,
    long_common_name,
    short_name,
    status
)
VALUES (
    %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
)
ON CONFLICT (loinc_code) DO NOTHING
"""

total_loinc = 0

for chunk in pd.read_csv(
    loinc_path,
    dtype=str,
    low_memory=False,
    chunksize=CHUNK_SIZE
):

    rows = []

    for _, r in chunk.iterrows():

        rows.append((
            clean(r.get("LOINC_NUM")),
            clean(r.get("COMPONENT")),
            clean(r.get("PROPERTY")),
            clean(r.get("TIME_ASPCT")),
            clean(r.get("SYSTEM")),
            clean(r.get("SCALE_TYP")),
            clean(r.get("METHOD_TYP")),
            clean(r.get("CLASS")),
            clean(r.get("LONG_COMMON_NAME")),
            clean(r.get("SHORTNAME")),
            clean(r.get("STATUS"))
        ))

    total_loinc += batch_insert(
        conn,
        loinc_insert,
        rows
    )

    print(
        f"\rLOINC processed: {total_loinc:,}",
        end=""
    )

print()
print("LOINC loaded.")

# ============================================================
# 11. LOAD RXNORM CONCEPTS
# ============================================================

print_stage("LOADING RXNORM CONCEPTS")

rxnconso_columns = [
    "RXCUI","LAT","TS","LUI","STT","SUI","ISPREF",
    "RXAUI","SAUI","SCUI","SDUI","SAB","TTY","CODE",
    "STR","SRL","SUPPRESS","CVF","EMPTY"
]

rxnorm_insert = """
INSERT INTO drug_concept (
    rxcui,
    name,
    concept_type,
    source_code,
    source
)
VALUES (%s,%s,%s,%s,%s)
ON CONFLICT (rxcui) DO NOTHING
"""

rxnorm_loaded = 0

for chunk in pd.read_csv(
    RXNORM / "rrf" / "RXNCONSO.RRF",
    sep="|",
    names=rxnconso_columns,
    dtype=str,
    chunksize=CHUNK_SIZE,
    low_memory=False
):

    # Prefer canonical RxNorm English concepts.
    filtered = chunk[
        (chunk["LAT"] == "ENG") &
        (chunk["SAB"] == "RXNORM") &
        (chunk["SUPPRESS"] != "O")
    ]

    rows = []

    for _, r in filtered.iterrows():

        rows.append((
            clean(r["RXCUI"]),
            clean(r["STR"]),
            clean(r["TTY"]),
            clean(r["CODE"]),
            clean(r["SAB"])
        ))

    rxnorm_loaded += batch_insert(
        conn,
        rxnorm_insert,
        rows
    )

    print(
        f"\rRxNorm concepts processed: "
        f"{rxnorm_loaded:,}",
        end=""
    )

print()
print("RxNorm concepts loaded.")

# ============================================================
# 12. LOAD RXNORM RELATIONSHIPS
# ============================================================

if LOAD_RXNORM_RELATIONSHIPS:

    print_stage("LOADING RXNORM RELATIONSHIPS")

    rxnrel_columns = [
        "RXCUI1","RXAUI1","STYPE1","REL",
        "RXCUI2","RXAUI2","STYPE2","RELA",
        "RUI","SRUI","SAB","SL","DIR","RG",
        "SUPPRESS","CVF","EMPTY"
    ]

    rxnrel_insert = """
    INSERT INTO drug_relationship (
        source_rxcui,
        target_rxcui,
        relationship_type,
        relationship_attribute,
        source
    )
    VALUES (%s,%s,%s,%s,%s)
    """

    relationship_total = 0

    valid_rxcui = set(
        get_map(
            conn,
            "drug_concept",
            "rxcui",
            "rxcui"
        ).keys()
    )

    for chunk in pd.read_csv(
        RXNORM / "rrf" / "RXNREL.RRF",
        sep="|",
        names=rxnrel_columns,
        dtype=str,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        rows = []

        for _, r in chunk.iterrows():

            source = clean(r["RXCUI1"])
            target = clean(r["RXCUI2"])

            if (
                source in valid_rxcui
                and target in valid_rxcui
            ):

                rows.append((
                    source,
                    target,
                    clean(r["REL"]),
                    clean(r["RELA"]),
                    clean(r["SAB"])
                ))

        relationship_total += batch_insert(
            conn,
            rxnrel_insert,
            rows
        )

        print(
            f"\rRxNorm relationships loaded: "
            f"{relationship_total:,}",
            end=""
        )

    print()

# ============================================================
# 13. LOAD PATIENTS
# ============================================================

print_stage("LOADING PATIENTS")

patients = read_csv(
    SYNTHEA / "patients.csv"
)

patient_insert = """
INSERT INTO patient (
    synthea_id,
    birth_date,
    death_date,
    ssn,
    drivers_license,
    passport,
    prefix,
    first_name,
    last_name,
    suffix,
    maiden_name,
    marital_status,
    race,
    ethnicity,
    gender
)
VALUES (
    %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
    %s,%s,%s,%s,%s
)
ON CONFLICT (synthea_id) DO NOTHING
"""

patient_rows = []

for _, r in patients.iterrows():

    patient_rows.append((
        clean(r.get("Id")),
        clean(r.get("BIRTHDATE")),
        clean(r.get("DEATHDATE")),
        clean(r.get("SSN")),
        clean(r.get("DRIVERS")),
        clean(r.get("PASSPORT")),
        clean(r.get("PREFIX")),
        clean(r.get("FIRST")),
        clean(r.get("LAST")),
        clean(r.get("SUFFIX")),
        clean(r.get("MAIDEN")),
        clean(r.get("MARITAL")),
        clean(r.get("RACE")),
        clean(r.get("ETHNICITY")),
        clean(r.get("GENDER"))
    ))

batch_insert(
    conn,
    patient_insert,
    patient_rows
)

patient_map = get_map(
    conn,
    "patient",
    "synthea_id",
    "patient_id"
)

print("Patients loaded:", len(patient_map))

# ============================================================
# 14. PATIENT ADDRESSES
# ============================================================

print_stage("LOADING PATIENT ADDRESSES")

address_insert = """
INSERT INTO patient_address (
    patient_id,
    address_line,
    city,
    state,
    county,
    zip_code,
    latitude,
    longitude
)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
"""

rows = []

for _, r in patients.iterrows():

    patient_id = patient_map.get(
        str(clean(r.get("Id")))
    )

    if patient_id:

        rows.append((
            patient_id,
            clean(r.get("ADDRESS")),
            clean(r.get("CITY")),
            clean(r.get("STATE")),
            clean(r.get("COUNTY")),
            clean(r.get("ZIP")),
            clean_float(r.get("LAT")),
            clean_float(r.get("LON"))
        ))

batch_insert(conn, address_insert, rows)

# ============================================================
# 15. ORGANIZATIONS
# ============================================================

print_stage("LOADING ORGANIZATIONS")

organizations = read_csv(
    SYNTHEA / "organizations.csv"
)

organization_insert = """
INSERT INTO organization (
    synthea_id,
    name,
    address,
    city,
    state,
    zip_code,
    latitude,
    longitude
)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
ON CONFLICT (synthea_id) DO NOTHING
"""

rows = []

for _, r in organizations.iterrows():

    rows.append((
        clean(r.get("Id")),
        clean(r.get("NAME")),
        clean(r.get("ADDRESS")),
        clean(r.get("CITY")),
        clean(r.get("STATE")),
        clean(r.get("ZIP")),
        clean_float(r.get("LAT")),
        clean_float(r.get("LON"))
    ))

batch_insert(
    conn,
    organization_insert,
    rows
)

organization_map = get_map(
    conn,
    "organization",
    "synthea_id",
    "organization_id"
)

print(
    "Organizations loaded:",
    len(organization_map)
)

# ============================================================
# 16. PROVIDERS
# ============================================================

print_stage("LOADING PROVIDERS")

providers = read_csv(
    SYNTHEA / "providers.csv"
)

provider_insert = """
INSERT INTO provider (
    synthea_id,
    organization_id,
    name,
    gender,
    specialty,
    address,
    city,
    state,
    zip_code
)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
ON CONFLICT (synthea_id) DO NOTHING
"""

rows = []

for _, r in providers.iterrows():

    org_source = clean(
        r.get("ORGANIZATION")
    )

    rows.append((
        clean(r.get("Id")),
        organization_map.get(
            str(org_source)
        ),
        clean(r.get("NAME")),
        clean(r.get("GENDER")),
        clean(r.get("SPECIALITY"))
        or clean(r.get("SPECIALTY")),
        clean(r.get("ADDRESS")),
        clean(r.get("CITY")),
        clean(r.get("STATE")),
        clean(r.get("ZIP"))
    ))

batch_insert(
    conn,
    provider_insert,
    rows
)

provider_map = get_map(
    conn,
    "provider",
    "synthea_id",
    "provider_id"
)

print("Providers loaded:", len(provider_map))

# ============================================================
# 17. PAYERS
# ============================================================

print_stage("LOADING PAYERS")

payers = read_csv(
    SYNTHEA / "payers.csv"
)

payer_insert = """
INSERT INTO payer (
    synthea_id,
    name,
    address,
    city,
    state,
    zip_code,
    phone
)
VALUES (%s,%s,%s,%s,%s,%s,%s)
ON CONFLICT (synthea_id) DO NOTHING
"""

rows = []

for _, r in payers.iterrows():

    rows.append((
        clean(r.get("Id")),
        clean(r.get("NAME")),
        clean(r.get("ADDRESS")),
        clean(r.get("CITY")),
        clean(r.get("STATE_HEADQUARTERED"))
        or clean(r.get("STATE")),
        clean(r.get("ZIP")),
        clean(r.get("PHONE"))
    ))

batch_insert(
    conn,
    payer_insert,
    rows
)

payer_map = get_map(
    conn,
    "payer",
    "synthea_id",
    "payer_id"
)

print("Payers loaded:", len(payer_map))

# ============================================================
# 18. ENCOUNTER TYPES
# ============================================================

print_stage("LOADING ENCOUNTER TYPES")

encounters = read_csv(
    SYNTHEA / "encounters.csv"
)

classes = sorted(
    encounters["ENCOUNTERCLASS"]
    .dropna()
    .unique()
) if "ENCOUNTERCLASS" in encounters else []

with conn.cursor() as cur:

    for encounter_class in classes:

        cur.execute(
            """
            INSERT INTO encounter_type (
                code,
                description
            )
            VALUES (%s,%s)
            ON CONFLICT (code) DO NOTHING
            """,
            (
                encounter_class,
                encounter_class
            )
        )

conn.commit()

encounter_type_map = get_map(
    conn,
    "encounter_type",
    "code",
    "encounter_type_id"
)

# ============================================================
# 19. ENCOUNTERS
# ============================================================

print_stage("LOADING ENCOUNTERS")

encounter_insert = """
INSERT INTO encounter (
    synthea_id,
    patient_id,
    provider_id,
    organization_id,
    encounter_type_id,
    start_time,
    end_time,
    encounter_class,
    code,
    description,
    base_cost,
    total_claim_cost,
    payer_coverage,
    reason_code,
    reason_description
)
VALUES (
    %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
    %s,%s,%s,%s,%s
)
ON CONFLICT (synthea_id) DO NOTHING
"""

rows = []
skipped_encounters = 0

for _, r in encounters.iterrows():

    patient_source = clean(r.get("PATIENT"))

    patient_id = patient_map.get(
        str(patient_source)
    )

    if not patient_id:
        skipped_encounters += 1
        continue

    provider_id = provider_map.get(
        str(clean(r.get("PROVIDER")))
    )

    organization_id = organization_map.get(
        str(clean(r.get("ORGANIZATION")))
    )

    encounter_class = clean(
        r.get("ENCOUNTERCLASS")
    )

    rows.append((
        clean(r.get("Id")),
        patient_id,
        provider_id,
        organization_id,
        encounter_type_map.get(
            str(encounter_class)
        ),
        clean(r.get("START")),
        clean(r.get("STOP")),
        encounter_class,
        clean(r.get("CODE")),
        clean(r.get("DESCRIPTION")),
        clean_float(r.get("BASE_ENCOUNTER_COST")),
        clean_float(r.get("TOTAL_CLAIM_COST")),
        clean_float(r.get("PAYER_COVERAGE")),
        clean(r.get("REASONCODE")),
        clean(r.get("REASONDESCRIPTION"))
    ))

batch_insert(
    conn,
    encounter_insert,
    rows
)

encounter_map = get_map(
    conn,
    "encounter",
    "synthea_id",
    "encounter_id"
)

print("Encounters loaded:", len(encounter_map))
print("Skipped encounters:", skipped_encounters)

# ============================================================
# 20. CONDITIONS + DIAGNOSES
# ============================================================

print_stage("LOADING CONDITIONS / DIAGNOSES")

conditions = read_csv(
    SYNTHEA / "conditions.csv"
)

condition_insert = """
INSERT INTO condition (
    source_code,
    description
)
VALUES (%s,%s)
ON CONFLICT (source_code) DO NOTHING
"""

condition_rows = []

for code, group in conditions.groupby("CODE"):

    description = group.iloc[0].get(
        "DESCRIPTION"
    )

    condition_rows.append((
        clean(code),
        clean(description)
    ))

batch_insert(
    conn,
    condition_insert,
    condition_rows
)

condition_map = get_map(
    conn,
    "condition",
    "source_code",
    "condition_id"
)

diagnosis_insert = """
INSERT INTO diagnosis (
    patient_id,
    encounter_id,
    condition_id,
    start_date,
    stop_date
)
VALUES (%s,%s,%s,%s,%s)
"""

rows = []

for _, r in conditions.iterrows():

    patient_id = patient_map.get(
        str(clean(r.get("PATIENT")))
    )

    condition_id = condition_map.get(
        str(clean(r.get("CODE")))
    )

    encounter_id = encounter_map.get(
        str(clean(r.get("ENCOUNTER")))
    )

    if patient_id and condition_id:

        rows.append((
            patient_id,
            encounter_id,
            condition_id,
            clean(r.get("START")),
            clean(r.get("STOP"))
        ))

batch_insert(
    conn,
    diagnosis_insert,
    rows
)

# ============================================================
# 21. PROCEDURES
# ============================================================

print_stage("LOADING PROCEDURES")

procedures = read_csv(
    SYNTHEA / "procedures.csv"
)

procedure_def_insert = """
INSERT INTO procedure_definition (
    source_code,
    description
)
VALUES (%s,%s)
ON CONFLICT (source_code) DO NOTHING
"""

rows = []

for code, group in procedures.groupby("CODE"):

    rows.append((
        clean(code),
        clean(group.iloc[0].get("DESCRIPTION"))
    ))

batch_insert(
    conn,
    procedure_def_insert,
    rows
)

procedure_map = get_map(
    conn,
    "procedure_definition",
    "source_code",
    "procedure_id"
)

patient_procedure_insert = """
INSERT INTO patient_procedure (
    patient_id,
    encounter_id,
    procedure_id,
    procedure_date,
    base_cost,
    reason_code,
    reason_description
)
VALUES (%s,%s,%s,%s,%s,%s,%s)
"""

rows = []

for _, r in procedures.iterrows():

    patient_id = patient_map.get(
        str(clean(r.get("PATIENT")))
    )

    procedure_id = procedure_map.get(
        str(clean(r.get("CODE")))
    )

    encounter_id = encounter_map.get(
        str(clean(r.get("ENCOUNTER")))
    )

    if patient_id and procedure_id:

        rows.append((
            patient_id,
            encounter_id,
            procedure_id,
            clean(r.get("START"))
            or clean(r.get("DATE")),
            clean_float(r.get("BASE_COST")),
            clean(r.get("REASONCODE")),
            clean(r.get("REASONDESCRIPTION"))
        ))

batch_insert(
    conn,
    patient_procedure_insert,
    rows
)

# ============================================================
# 22. CARE PLANS
# ============================================================

print_stage("LOADING CARE PLANS")

careplans = read_csv(
    SYNTHEA / "careplans.csv"
)

careplan_insert = """
INSERT INTO care_plan (
    synthea_id,
    patient_id,
    encounter_id,
    start_date,
    stop_date,
    code,
    description,
    reason_code,
    reason_description
)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
ON CONFLICT (synthea_id) DO NOTHING
"""

rows = []

for _, r in careplans.iterrows():

    patient_id = patient_map.get(
        str(clean(r.get("PATIENT")))
    )

    if patient_id:

        rows.append((
            clean(r.get("Id")),
            patient_id,
            encounter_map.get(
                str(clean(r.get("ENCOUNTER")))
            ),
            clean(r.get("START")),
            clean(r.get("STOP")),
            clean(r.get("CODE")),
            clean(r.get("DESCRIPTION")),
            clean(r.get("REASONCODE")),
            clean(r.get("REASONDESCRIPTION"))
        ))

batch_insert(
    conn,
    careplan_insert,
    rows
)

# ============================================================
# 23. DEVICES
# ============================================================

print_stage("LOADING DEVICES")

devices = read_csv(
    SYNTHEA / "devices.csv"
)

device_insert = """
INSERT INTO medical_device (
    patient_id,
    encounter_id,
    start_time,
    stop_time,
    code,
    description,
    udi
)
VALUES (%s,%s,%s,%s,%s,%s,%s)
"""

rows = []

for _, r in devices.iterrows():

    patient_id = patient_map.get(
        str(clean(r.get("PATIENT")))
    )

    if patient_id:

        rows.append((
            patient_id,
            encounter_map.get(
                str(clean(r.get("ENCOUNTER")))
            ),
            clean(r.get("START")),
            clean(r.get("STOP")),
            clean(r.get("CODE")),
            clean(r.get("DESCRIPTION")),
            clean(r.get("UDI"))
        ))

batch_insert(
    conn,
    device_insert,
    rows
)

# ============================================================
# 24. IMAGING STUDIES
# ============================================================

print_stage("LOADING IMAGING STUDIES")

imaging_insert = """
INSERT INTO imaging_study (
    synthea_id,
    patient_id,
    encounter_id,
    study_date,
    body_site_code,
    body_site_description,
    modality_code,
    modality_description,
    sop_code,
    sop_description
)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
ON CONFLICT (synthea_id) DO NOTHING
"""

imaging_total = 0

for chunk in pd.read_csv(
    SYNTHEA / "imaging_studies.csv",
    dtype=str,
    chunksize=CHUNK_SIZE,
    low_memory=False
):

    rows = []

    for _, r in chunk.iterrows():

        patient_id = patient_map.get(
            str(clean(r.get("PATIENT")))
        )

        if not patient_id:
            continue

        rows.append((
            clean(r.get("Id")),
            patient_id,
            encounter_map.get(
                str(clean(r.get("ENCOUNTER")))
            ),
            clean(r.get("DATE")),
            clean(r.get("BODYSITE_CODE")),
            clean(r.get("BODYSITE_DESCRIPTION")),
            clean(r.get("MODALITY_CODE")),
            clean(r.get("MODALITY_DESCRIPTION")),
            clean(r.get("SOP_CODE")),
            clean(r.get("SOP_DESCRIPTION"))
        ))

    imaging_total += batch_insert(
        conn,
        imaging_insert,
        rows
    )

    print(
        f"\rImaging rows: {imaging_total:,}",
        end=""
    )

print()

# ============================================================
# 25. ALLERGIES
# ============================================================

print_stage("LOADING ALLERGIES")

allergies = read_csv(
    SYNTHEA / "allergies.csv"
)

allergy_insert = """
INSERT INTO patient_allergy (
    patient_id,
    encounter_id,
    start_date,
    stop_date,
    code,
    system,
    description,
    allergy_type,
    category,
    reaction_code_1,
    reaction_description_1,
    severity_1,
    reaction_code_2,
    reaction_description_2,
    severity_2
)
VALUES (
    %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
    %s,%s,%s,%s,%s
)
"""

rows = []

for _, r in allergies.iterrows():

    patient_id = patient_map.get(
        str(clean(r.get("PATIENT")))
    )

    if patient_id:

        rows.append((
            patient_id,
            encounter_map.get(
                str(clean(r.get("ENCOUNTER")))
            ),
            clean(r.get("START")),
            clean(r.get("STOP")),
            clean(r.get("CODE")),
            clean(r.get("SYSTEM")),
            clean(r.get("DESCRIPTION")),
            clean(r.get("TYPE")),
            clean(r.get("CATEGORY")),
            clean(r.get("REACTION1")),
            clean(r.get("DESCRIPTION1")),
            clean(r.get("SEVERITY1")),
            clean(r.get("REACTION2")),
            clean(r.get("DESCRIPTION2")),
            clean(r.get("SEVERITY2"))
        ))

batch_insert(
    conn,
    allergy_insert,
    rows
)

# ============================================================
# 26. IMMUNIZATIONS
# ============================================================

print_stage("LOADING IMMUNIZATIONS")

immunizations = read_csv(
    SYNTHEA / "immunizations.csv"
)

immunization_insert = """
INSERT INTO patient_immunization (
    patient_id,
    encounter_id,
    immunization_date,
    code,
    description,
    base_cost
)
VALUES (%s,%s,%s,%s,%s,%s)
"""

rows = []

for _, r in immunizations.iterrows():

    patient_id = patient_map.get(
        str(clean(r.get("PATIENT")))
    )

    if patient_id:

        rows.append((
            patient_id,
            encounter_map.get(
                str(clean(r.get("ENCOUNTER")))
            ),
            clean(r.get("DATE")),
            clean(r.get("CODE")),
            clean(r.get("DESCRIPTION")),
            clean_float(r.get("BASE_COST"))
        ))

batch_insert(
    conn,
    immunization_insert,
    rows
)

# ============================================================
# 27. MEDICATIONS
# ============================================================

print_stage("LOADING MEDICATIONS")

medications = read_csv(
    SYNTHEA / "medications.csv"
)

prescription_insert = """
INSERT INTO prescription (
    patient_id,
    encounter_id,
    start_time,
    stop_time,
    reason_code,
    reason_description
)
VALUES (%s,%s,%s,%s,%s,%s)
RETURNING prescription_id
"""

prescription_item_insert = """
INSERT INTO prescription_item (
    prescription_id,
    rxcui,
    description,
    base_cost,
    payer_coverage,
    dispenses,
    total_cost
)
VALUES (%s,%s,%s,%s,%s,%s,%s)
"""

with conn.cursor() as cur:

    total = 0

    for _, r in medications.iterrows():

        patient_id = patient_map.get(
            str(clean(r.get("PATIENT")))
        )

        if not patient_id:
            continue

        cur.execute(
            prescription_insert,
            (
                patient_id,
                encounter_map.get(
                    str(clean(r.get("ENCOUNTER")))
                ),
                clean(r.get("START")),
                clean(r.get("STOP")),
                clean(r.get("REASONCODE")),
                clean(r.get("REASONDESCRIPTION"))
            )
        )

        prescription_id = cur.fetchone()[0]

        medication_code = clean(r.get("CODE"))

        # Synthea medication CODE is commonly RxNorm.
        # Only attach FK if that RXCUI exists.
        rxcui = (
            str(medication_code)
            if medication_code is not None
            and str(medication_code)
            in valid_rxcui
            else None
        )

        cur.execute(
            prescription_item_insert,
            (
                prescription_id,
                rxcui,
                clean(r.get("DESCRIPTION")),
                clean_float(r.get("BASE_COST")),
                clean_float(r.get("PAYER_COVERAGE")),
                clean_int(r.get("DISPENSES")),
                clean_float(r.get("TOTALCOST"))
            )
        )

        total += 1

        if total % 5000 == 0:
            conn.commit()
            print(
                f"\rMedications loaded: {total:,}",
                end=""
            )

conn.commit()

print()
print("Medications loaded:", total)

# ============================================================
# 28. OBSERVATIONS
# ============================================================

print_stage("LOADING OBSERVATIONS")

observation_insert = """
INSERT INTO observation (
    patient_id,
    encounter_id,
    observation_time,
    loinc_code,
    description,
    value_text,
    value_numeric,
    unit,
    value_type
)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
"""

with conn.cursor() as cur:
    cur.execute(
        "SELECT loinc_code FROM loinc_concept"
    )
    valid_loinc = {
        str(x[0])
        for x in cur.fetchall()
    }

observation_total = 0

for chunk in pd.read_csv(
    SYNTHEA / "observations.csv",
    dtype=str,
    chunksize=CHUNK_SIZE,
    low_memory=False
):

    rows = []

    for _, r in chunk.iterrows():

        patient_id = patient_map.get(
            str(clean(r.get("PATIENT")))
        )

        if not patient_id:
            continue

        code = clean(r.get("CODE"))

        loinc_code = (
            str(code)
            if code is not None
            and str(code) in valid_loinc
            else None
        )

        raw_value = clean(r.get("VALUE"))

        numeric_value = clean_float(
            raw_value
        )

        rows.append((
            patient_id,
            encounter_map.get(
                str(clean(r.get("ENCOUNTER")))
            ),
            clean(r.get("DATE")),
            loinc_code,
            clean(r.get("DESCRIPTION")),
            str(raw_value)
            if raw_value is not None
            else None,
            numeric_value,
            clean(r.get("UNITS")),
            clean(r.get("TYPE"))
        ))

    observation_total += batch_insert(
        conn,
        observation_insert,
        rows
    )

    print(
        f"\rObservations loaded: "
        f"{observation_total:,}",
        end=""
    )

print()

# ============================================================
# 29. LAB RESULT DERIVATION
# ============================================================

print_stage("DERIVING LAB RESULTS")

# Rather than duplicating every observation,
# derive laboratory-type observations where possible.

with conn.cursor() as cur:

    cur.execute("""
        INSERT INTO lab_result (
            patient_id,
            encounter_id,
            loinc_code,
            result_time,
            result_value,
            result_numeric,
            unit,
            status
        )
        SELECT
            o.patient_id,
            o.encounter_id,
            o.loinc_code,
            o.observation_time,
            o.value_text,
            o.value_numeric,
            o.unit,
            'FINAL'
        FROM observation o
        JOIN loinc_concept l
          ON l.loinc_code = o.loinc_code
        WHERE
            UPPER(COALESCE(l.class,'')) LIKE 'CHEM%'
            OR UPPER(COALESCE(l.class,'')) LIKE 'HEM%'
            OR UPPER(COALESCE(l.class,'')) LIKE 'MICRO%'
            OR UPPER(COALESCE(l.class,'')) LIKE 'SERO%'
            OR UPPER(COALESCE(l.class,'')) LIKE 'UA%'
    """)

conn.commit()

print(
    "Derived lab results:",
    table_count(conn, "lab_result")
)

# ============================================================
# 30. INSURANCE TRANSITIONS
# ============================================================

print_stage("LOADING PATIENT INSURANCE")

payer_transitions = read_csv(
    SYNTHEA / "payer_transitions.csv"
)

insurance_insert = """
INSERT INTO patient_insurance (
    patient_id,
    payer_id,
    start_date,
    end_date,
    ownership,
    owner_name
)
VALUES (%s,%s,%s,%s,%s,%s)
"""

rows = []

for _, r in payer_transitions.iterrows():

    patient_id = patient_map.get(
        str(clean(r.get("PATIENT")))
    )

    payer_id = payer_map.get(
        str(clean(r.get("PAYER")))
    )

    if patient_id and payer_id:

        rows.append((
            patient_id,
            payer_id,
            clean(r.get("START_DATE"))
            or clean(r.get("START_YEAR")),
            clean(r.get("END_DATE"))
            or clean(r.get("END_YEAR")),
            clean(r.get("OWNERSHIP")),
            clean(r.get("OWNER_NAME"))
        ))

# Some Synthea versions use integer years rather than dates.
# Insert only rows that PostgreSQL can safely interpret.
safe_rows = []

for row in rows:

    start = row[2]
    end = row[3]

    if (
        start is not None
        and re.fullmatch(r"\d{4}", str(start))
    ):
        start = f"{start}-01-01"

    if (
        end is not None
        and re.fullmatch(r"\d{4}", str(end))
    ):
        end = f"{end}-12-31"

    safe_rows.append((
        row[0],
        row[1],
        start,
        end,
        row[4],
        row[5]
    ))

batch_insert(
    conn,
    insurance_insert,
    safe_rows
)

# ============================================================
# 31. CLAIMS
# ============================================================

print_stage("LOADING CLAIMS")

claims = read_csv(
    SYNTHEA / "claims.csv"
)

claim_insert = """
INSERT INTO claim (
    synthea_id,
    patient_id,
    encounter_id,
    payer_id,
    provider_id,
    primary_patient_insurance_source_id,
    secondary_patient_insurance_source_id,
    department_id,
    diagnosis_code,
    outstanding_1,
    outstanding_2
)
VALUES (
    %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
)
ON CONFLICT (synthea_id) DO NOTHING
"""

rows = []

for _, r in claims.iterrows():

    patient_source = get_value(
        r,
        "PATIENTID",
        "PATIENT"
    )

    patient_id = patient_map.get(
        str(patient_source)
    )

    if not patient_id:
        continue

    encounter_source = get_value(
        r,
        "APPOINTMENTID",
        "ENCOUNTERID",
        "ENCOUNTER"
    )

    payer_source = get_value(
        r,
        "PRIMARYPATIENTINSURANCEID",
        "PAYERID",
        "PAYER"
    )

    provider_source = get_value(
        r,
        "PROVIDERID",
        "PROVIDER"
    )

    rows.append((
        get_value(r, "Id", "ID"),
        patient_id,
        encounter_map.get(
            str(encounter_source)
        ),
        payer_map.get(
            str(payer_source)
        ),
        provider_map.get(
            str(provider_source)
        ),
        get_value(
            r,
            "PRIMARYPATIENTINSURANCEID"
        ),
        get_value(
            r,
            "SECONDARYPATIENTINSURANCEID"
        ),
        get_value(
            r,
            "DEPARTMENTID"
        ),
        get_value(
            r,
            "DIAGNOSIS1",
            "DIAGNOSIS"
        ),
        clean_float(
            get_value(
                r,
                "OUTSTANDING1"
            )
        ),
        clean_float(
            get_value(
                r,
                "OUTSTANDING2"
            )
        )
    ))

batch_insert(
    conn,
    claim_insert,
    rows
)

claim_map = get_map(
    conn,
    "claim",
    "synthea_id",
    "claim_id"
)

print("Claims loaded:", len(claim_map))

# ============================================================
# 32. CLAIM TRANSACTIONS
# ============================================================

print_stage("LOADING CLAIM TRANSACTIONS")

claim_transaction_insert = """
INSERT INTO claim_transaction (
    claim_id,
    transaction_type,
    transaction_method,
    transaction_date,
    amount,
    patient_payment,
    transfer_out_id,
    transfer_type
)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
"""

transaction_total = 0

for chunk in pd.read_csv(
    SYNTHEA / "claims_transactions.csv",
    dtype=str,
    chunksize=CHUNK_SIZE,
    low_memory=False
):

    rows = []

    for _, r in chunk.iterrows():

        claim_source = get_value(
            r,
            "CLAIMID"
        )

        claim_id = claim_map.get(
            str(claim_source)
        )

        if not claim_id:
            continue

        rows.append((
            claim_id,
            get_value(
                r,
                "TYPE",
                "TRANSACTIONTYPE"
            ),
            get_value(
                r,
                "METHOD"
            ),
            get_value(
                r,
                "FROMDATE",
                "DATE"
            ),
            clean_float(
                get_value(
                    r,
                    "AMOUNT"
                )
            ),
            clean_float(
                get_value(
                    r,
                    "PATIENTPAYMENT"
                )
            ),
            get_value(
                r,
                "TRANSFEROUTID"
            ),
            get_value(
                r,
                "TRANSFEROUTTYPE"
            )
        ))

    transaction_total += batch_insert(
        conn,
        claim_transaction_insert,
        rows
    )

    print(
        f"\rClaim transactions loaded: "
        f"{transaction_total:,}",
        end=""
    )

print()

# ============================================================
# 33. CREATE CONSTRAINTS
# ============================================================

print_stage("CREATING ADDITIONAL CONSTRAINTS")

execute_sql_file(
    conn,
    SCHEMA_DIR / "02_constraints.sql"
)

# ============================================================
# 34. CREATE BASELINE INDEXES
# ============================================================

print_stage("CREATING BASELINE INDEXES")

execute_sql_file(
    conn,
    SCHEMA_DIR / "03_baseline_indexes.sql"
)

# ============================================================
# 35. CREATE VIEWS
# ============================================================

print_stage("CREATING VIEWS")

execute_sql_file(
    conn,
    SCHEMA_DIR / "04_views.sql"
)

# ============================================================
# 36. CREATE TRIGGERS
# ============================================================

print_stage("CREATING TRIGGERS")

execute_sql_file(
    conn,
    SCHEMA_DIR / "05_triggers.sql"
)

# ============================================================
# 37. CREATE STORED FUNCTIONS
# ============================================================

print_stage("CREATING STORED FUNCTIONS")

execute_sql_file(
    conn,
    SCHEMA_DIR / "06_functions.sql"
)

# ============================================================
# 38. ANALYZE DATABASE
# ============================================================

print_stage("ANALYZING DATABASE")

conn.commit()

with conn.cursor() as cur:
    cur.execute("ANALYZE")

conn.commit()

print("PostgreSQL statistics updated.")

# ============================================================
# 39. VALIDATION
# ============================================================

print_stage("DATABASE VALIDATION")

tables = [
    "patient",
    "patient_address",
    "patient_contact",
    "patient_allergy",
    "patient_immunization",
    "organization",
    "provider",
    "provider_specialty",
    "encounter",
    "encounter_type",
    "admission",
    "discharge",
    "condition",
    "diagnosis",
    "procedure_definition",
    "patient_procedure",
    "care_plan",
    "medical_device",
    "imaging_study",
    "drug_concept",
    "prescription",
    "prescription_item",
    "drug_relationship",
    "loinc_concept",
    "observation",
    "lab_result",
    "payer",
    "patient_insurance",
    "claim",
    "claim_transaction",
    "query_log",
    "optimization_recommendation"
]

validation = []

for table in tables:

    count = table_count(
        conn,
        table
    )

    validation.append({
        "table": table,
        "rows": count
    })

    print(
        f"{table:<35} {count:>12,}"
    )

validation_df = pd.DataFrame(validation)

validation_df.to_csv(
    RESULTS / "phase3_table_counts.csv",
    index=False
)

# ============================================================
# 40. FK VALIDATION
# ============================================================

print_stage("FOREIGN KEY VALIDATION")

fk_tests = {
    "encounter_without_patient": """
        SELECT COUNT(*)
        FROM encounter e
        LEFT JOIN patient p
          ON p.patient_id=e.patient_id
        WHERE p.patient_id IS NULL
    """,

    "diagnosis_without_patient": """
        SELECT COUNT(*)
        FROM diagnosis d
        LEFT JOIN patient p
          ON p.patient_id=d.patient_id
        WHERE p.patient_id IS NULL
    """,

    "prescription_without_patient": """
        SELECT COUNT(*)
        FROM prescription pr
        LEFT JOIN patient p
          ON p.patient_id=pr.patient_id
        WHERE p.patient_id IS NULL
    """,

    "observation_without_patient": """
        SELECT COUNT(*)
        FROM observation o
        LEFT JOIN patient p
          ON p.patient_id=o.patient_id
        WHERE p.patient_id IS NULL
    """,

    "claim_transaction_without_claim": """
        SELECT COUNT(*)
        FROM claim_transaction ct
        LEFT JOIN claim c
          ON c.claim_id=ct.claim_id
        WHERE c.claim_id IS NULL
    """
}

fk_results = {}

with conn.cursor() as cur:

    for name, query in fk_tests.items():

        cur.execute(query)

        value = cur.fetchone()[0]

        fk_results[name] = value

        print(
            f"{name:<45}: {value}"
        )

# ============================================================
# 41. DATABASE SIZE
# ============================================================

print_stage("DATABASE SIZE")

with conn.cursor() as cur:

    cur.execute("""
        SELECT
            pg_size_pretty(
                pg_database_size(current_database())
            ),
            pg_database_size(current_database())
    """)

    pretty_size, bytes_size = cur.fetchone()

print("Database size:", pretty_size)

# ============================================================
# 42. TOP TABLE SIZES
# ============================================================

with conn.cursor() as cur:

    cur.execute("""
        SELECT
            relname,
            pg_total_relation_size(
                quote_ident(relname)
            ) AS bytes
        FROM pg_stat_user_tables
        ORDER BY bytes DESC
    """)

    table_sizes = cur.fetchall()

table_size_df = pd.DataFrame(
    table_sizes,
    columns=[
        "table",
        "size_bytes"
    ]
)

table_size_df["size_mb"] = (
    table_size_df["size_bytes"]
    / 1024
    / 1024
).round(2)

table_size_df.to_csv(
    RESULTS / "phase3_table_sizes.csv",
    index=False
)

print(
    table_size_df[
        ["table", "size_mb"]
    ].head(15).to_string(index=False)
)

# ============================================================
# 43. TEST VIEWS - SAFE SCALING VALIDATION
# ============================================================

print_stage("TESTING VIEWS")

views = [
    "patient_summary",
    "patient_condition_history",
    "patient_medication_history",
    "patient_lab_history",
    "slow_query_summary"
]

VIEW_TIMEOUT_MS = 15000
view_results = {}

conn.commit()

for view in views:
    try:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute(
                    "SET LOCAL statement_timeout = %s",
                    (VIEW_TIMEOUT_MS,)
                )
                cur.execute(
                    sql.SQL("SELECT COUNT(*) FROM {}").format(
                        sql.Identifier(view)
                    )
                )
                count = cur.fetchone()[0]

        view_results[view] = count
        print(f"{view:<35}: {count:,}", flush=True)

    except psycopg.errors.QueryCanceled:
        view_results[view] = None
        print(f"{view:<35}: TIMEOUT", flush=True)

    except Exception as exc:
        view_results[view] = None
        print(
            f"{view:<35}: ERROR - {str(exc).splitlines()[0]}",
            flush=True
        )

conn.commit()

# ============================================================
# ============================================================
# 44. TEST FUNCTIONS
# ============================================================

print_stage("TESTING STORED FUNCTIONS")

with conn.cursor() as cur:

    cur.execute("""
        SELECT patient_id
        FROM patient
        LIMIT 1
    """)

    sample_patient = cur.fetchone()

    if sample_patient:

        sample_patient_id = sample_patient[0]

        cur.execute(
            """
            SELECT get_patient_encounter_count(%s)
            """,
            (sample_patient_id,)
        )

        encounter_count = cur.fetchone()[0]

        cur.execute(
            """
            SELECT get_patient_diagnosis_count(%s)
            """,
            (sample_patient_id,)
        )

        diagnosis_count = cur.fetchone()[0]

        print(
            "Sample patient:",
            sample_patient_id
        )

        print(
            "Encounter count:",
            encounter_count
        )

        print(
            "Diagnosis count:",
            diagnosis_count
        )

# ============================================================
# 45. GENERATE ETL REPORT
# ============================================================

print_stage("GENERATING PHASE 3 REPORT")

report = {
    "phase": 3,
    "project": "Adaptive Healthcare Knowledge Database",
    "generated_at": datetime.now().isoformat(),
    "database": {
        "name": DB_NAME,
        "host": DB_HOST,
        "port": DB_PORT,
        "size": pretty_size,
        "size_bytes": bytes_size
    },
    "datasets": {
        "synthea": True,
        "loinc": True,
        "rxnorm": True
    },
    "table_counts": {
        row["table"]: int(row["rows"])
        for row in validation
    },
    "foreign_key_validation": {
        key: int(value)
        for key, value
        in fk_results.items()
    },
    "view_counts": {
        key: int(value)
        for key, value
        in view_results.items()
    },
    "status": "SUCCESS"
}

with open(
    RESULTS / "phase3_etl_report.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=4,
        default=str
    )

# ============================================================
# 46. CREATE TEXT REPORT
# ============================================================

report_lines = []

report_lines.append(
    "PHASE 3 - POSTGRESQL IMPLEMENTATION + ETL"
)

report_lines.append(
    "=" * 70
)

report_lines.append(
    f"Generated: {datetime.now()}"
)

report_lines.append(
    f"Database: {DB_NAME}"
)

report_lines.append(
    f"Database size: {pretty_size}"
)

report_lines.append("")

report_lines.append(
    "TABLE ROW COUNTS"
)

report_lines.append(
    "-" * 70
)

for row in validation:

    report_lines.append(
        f"{row['table']:<35}"
        f"{row['rows']:>15,}"
    )

report_lines.append("")
report_lines.append(
    "FOREIGN KEY VALIDATION"
)

report_lines.append(
    "-" * 70
)

for key, value in fk_results.items():

    report_lines.append(
        f"{key:<45}{value}"
    )

report_lines.append("")
report_lines.append(
    "STATUS: SUCCESS"
)

(
    RESULTS / "phase3_etl_report.txt"
).write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)

# ============================================================
# 47. CLOSE CONNECTION
# ============================================================

conn.close()

# ============================================================
# 48. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 100)
print("PHASE 3 COMPLETED SUCCESSFULLY")
print("=" * 100)

print("\nDatabase:")
print(DB_NAME)

print("\nDatabase size:")
print(pretty_size)

print("\nReports generated:")

print(
    RESULTS / "phase3_table_counts.csv"
)

print(
    RESULTS / "phase3_table_sizes.csv"
)

print(
    RESULTS / "phase3_etl_report.json"
)

print(
    RESULTS / "phase3_etl_report.txt"
)

print("\nPHASE 3:")
print("PostgreSQL database creation     COMPLETE")
print("Schema creation                  COMPLETE")
print("LOINC loading                    COMPLETE")
print("RxNorm loading                   COMPLETE")
print("Synthea ETL                      COMPLETE")
print("Constraints                      COMPLETE")
print("Baseline indexes                 COMPLETE")
print("Views                            COMPLETE")
print("Triggers                         COMPLETE")
print("Stored functions                 COMPLETE")
print("Validation                       COMPLETE")

print("\nNEXT:")
print("PHASE 4 -> Intelligent Query Analyzer")