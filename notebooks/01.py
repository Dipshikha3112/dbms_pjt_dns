from pathlib import Path
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import json
import re
import os
from datetime import datetime

# ============================================================
# PHASE 2
# DATABASE DESIGN + DATASET ANALYSIS
# Adaptive Healthcare Knowledge Database
# ============================================================

print("=" * 90)
print("PHASE 2 - DATABASE DESIGN")
print("Adaptive Healthcare Knowledge Database")
print("=" * 90)

# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path("..").resolve()

DATA = PROJECT_ROOT / "data"
SYNTHEA = DATA / "synthea"
LOINC = DATA / "loinc"
RXNORM = DATA / "rxnorm"
MESH = DATA / "mesh"

NOTEBOOKS = PROJECT_ROOT / "notebooks"

DATABASE = PROJECT_ROOT / "database"
SCHEMA_DIR = DATABASE / "schema"
DESIGN_DIR = DATABASE / "design"
ETL_DIR = DATABASE / "etl"

DOCS = PROJECT_ROOT / "docs"
DB_DOCS = DOCS / "database_design"

RESULTS = PROJECT_ROOT / "results"

for directory in [
    NOTEBOOKS,
    DATABASE,
    SCHEMA_DIR,
    DESIGN_DIR,
    ETL_DIR,
    DOCS,
    DB_DOCS,
    RESULTS
]:
    directory.mkdir(parents=True, exist_ok=True)

print("\nProject root:", PROJECT_ROOT)

# ============================================================
# 2. VERIFY DATASETS
# ============================================================

required_paths = {
    "Synthea": SYNTHEA,
    "LOINC": LOINC / "LoincTable" / "Loinc.csv",
    "RxNorm Concepts": RXNORM / "rrf" / "RXNCONSO.RRF",
    "RxNorm Relationships": RXNORM / "rrf" / "RXNREL.RRF",
    "RxNorm Attributes": RXNORM / "rrf" / "RXNSAT.RRF",
    "MeSH": MESH / "desc2026.xml"
}

print("\n" + "=" * 90)
print("DATASET VERIFICATION")
print("=" * 90)

missing = []

for name, path in required_paths.items():
    status = "FOUND" if path.exists() else "MISSING"
    print(f"{name:<25} : {status}")
    if not path.exists():
        missing.append(str(path))

if missing:
    raise FileNotFoundError(
        "\nMissing required files:\n" + "\n".join(missing)
    )

# ============================================================
# 3. INSPECT ALL SYNTHEA CSV FILES
# ============================================================

print("\n" + "=" * 90)
print("SYNTHEA DATASET ANALYSIS")
print("=" * 90)

synthea_files = sorted(SYNTHEA.glob("*.csv"))

synthea_summary = []
synthea_dictionary = []

for file in synthea_files:

    try:
        df = pd.read_csv(file, low_memory=False)

        synthea_summary.append({
            "file": file.name,
            "rows": len(df),
            "columns": len(df.columns),
            "memory_mb": round(
                df.memory_usage(deep=True).sum() / 1024**2, 2
            )
        })

        print(
            f"{file.name:<30}"
            f" rows={len(df):<10}"
            f" columns={len(df.columns):<5}"
        )

        for column in df.columns:

            series = df[column]

            synthea_dictionary.append({
                "source_file": file.name,
                "column": column,
                "pandas_dtype": str(series.dtype),
                "nullable": bool(series.isnull().any()),
                "null_count": int(series.isnull().sum()),
                "unique_values": int(series.nunique(dropna=True)),
                "sample_value": (
                    str(series.dropna().iloc[0])
                    if not series.dropna().empty
                    else ""
                )
            })

    except Exception as e:
        print("ERROR:", file.name, e)

synthea_summary_df = pd.DataFrame(synthea_summary)
synthea_dictionary_df = pd.DataFrame(synthea_dictionary)

synthea_summary_df.to_csv(
    DB_DOCS / "synthea_file_summary.csv",
    index=False
)

synthea_dictionary_df.to_csv(
    DB_DOCS / "synthea_data_dictionary.csv",
    index=False
)

# ============================================================
# 4. DETECT POSSIBLE SYNTHEA KEYS
# ============================================================

print("\n" + "=" * 90)
print("SYNTHEA KEY / RELATIONSHIP ANALYSIS")
print("=" * 90)

relationship_candidates = []

common_fk_names = [
    "PATIENT",
    "ENCOUNTER",
    "PROVIDER",
    "ORGANIZATION",
    "PAYER",
    "CLAIMID"
]

for file in synthea_files:

    df = pd.read_csv(file, low_memory=False)

    for col in df.columns:

        if col.upper() in common_fk_names:

            relationship_candidates.append({
                "source_file": file.name,
                "column": col,
                "possible_reference": {
                    "PATIENT": "patients.csv.ID",
                    "ENCOUNTER": "encounters.csv.Id",
                    "PROVIDER": "providers.csv.Id",
                    "ORGANIZATION": "organizations.csv.Id",
                    "PAYER": "payers.csv.Id",
                    "CLAIMID": "claims.csv.Id"
                }.get(col.upper(), "")
            })

relationship_df = pd.DataFrame(relationship_candidates)

relationship_df.to_csv(
    DB_DOCS / "synthea_relationship_candidates.csv",
    index=False
)

print(relationship_df.to_string(index=False))

# ============================================================
# 5. LOINC ANALYSIS
# ============================================================

print("\n" + "=" * 90)
print("LOINC ANALYSIS")
print("=" * 90)

loinc_path = LOINC / "LoincTable" / "Loinc.csv"

loinc = pd.read_csv(
    loinc_path,
    low_memory=False
)

print("LOINC rows    :", len(loinc))
print("LOINC columns :", len(loinc.columns))

loinc_dictionary = []

for col in loinc.columns:

    loinc_dictionary.append({
        "column": col,
        "dtype": str(loinc[col].dtype),
        "nullable": bool(loinc[col].isnull().any()),
        "null_count": int(loinc[col].isnull().sum()),
        "unique_values": int(loinc[col].nunique(dropna=True))
    })

pd.DataFrame(loinc_dictionary).to_csv(
    DB_DOCS / "loinc_data_dictionary.csv",
    index=False
)

important_loinc_columns = [
    "LOINC_NUM",
    "COMPONENT",
    "PROPERTY",
    "TIME_ASPCT",
    "SYSTEM",
    "SCALE_TYP",
    "METHOD_TYP",
    "CLASS",
    "LONG_COMMON_NAME",
    "SHORTNAME",
    "STATUS"
]

existing_loinc_columns = [
    c for c in important_loinc_columns
    if c in loinc.columns
]

print("\nImportant LOINC columns:")
print(existing_loinc_columns)

print("\nLOINC sample:")
print(
    loinc[existing_loinc_columns]
    .head(5)
    .to_string(index=False)
)

# ============================================================
# 6. RXNORM ANALYSIS
# ============================================================

print("\n" + "=" * 90)
print("RXNORM ANALYSIS")
print("=" * 90)

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

rxnconso = pd.read_csv(
    RXNORM / "rrf" / "RXNCONSO.RRF",
    sep="|",
    names=rxnconso_columns,
    dtype=str,
    low_memory=False
)

print("RXNCONSO rows:", len(rxnconso))

print(
    rxnconso[
        ["RXCUI", "LAT", "SAB", "TTY", "CODE", "STR"]
    ].head(10).to_string(index=False)
)

rxnrel_columns = [
    "RXCUI1",
    "RXAUI1",
    "STYPE1",
    "REL",
    "RXCUI2",
    "RXAUI2",
    "STYPE2",
    "RELA",
    "RUI",
    "SRUI",
    "SAB",
    "SL",
    "DIR",
    "RG",
    "SUPPRESS",
    "CVF",
    "EMPTY"
]

rxnrel = pd.read_csv(
    RXNORM / "rrf" / "RXNREL.RRF",
    sep="|",
    names=rxnrel_columns,
    dtype=str,
    low_memory=False
)

print("RXNREL rows :", len(rxnrel))

rxnsat_columns = [
    "RXCUI",
    "LUI",
    "SUI",
    "RXAUI",
    "STYPE",
    "CODE",
    "ATUI",
    "SATUI",
    "ATN",
    "SAB",
    "ATV",
    "SUPPRESS",
    "CVF",
    "EMPTY"
]

rxnsat = pd.read_csv(
    RXNORM / "rrf" / "RXNSAT.RRF",
    sep="|",
    names=rxnsat_columns,
    dtype=str,
    low_memory=False
)

print("RXNSAT rows :", len(rxnsat))

rxnorm_summary = pd.DataFrame([
    {
        "file": "RXNCONSO.RRF",
        "rows": len(rxnconso),
        "columns": len(rxnconso.columns)
    },
    {
        "file": "RXNREL.RRF",
        "rows": len(rxnrel),
        "columns": len(rxnrel.columns)
    },
    {
        "file": "RXNSAT.RRF",
        "rows": len(rxnsat),
        "columns": len(rxnsat.columns)
    }
])

rxnorm_summary.to_csv(
    DB_DOCS / "rxnorm_summary.csv",
    index=False
)

# ============================================================
# 7. MESH ANALYSIS
# ============================================================

print("\n" + "=" * 90)
print("MESH ANALYSIS")
print("=" * 90)

mesh_path = MESH / "desc2026.xml"

mesh_records = []
mesh_count = 0

for event, elem in ET.iterparse(
    mesh_path,
    events=("end",)
):

    if elem.tag == "DescriptorRecord":

        mesh_count += 1

        descriptor_ui = elem.findtext("DescriptorUI")
        descriptor_name = elem.findtext(
            "DescriptorName/String"
        )

        tree_numbers = [
            node.text
            for node in elem.findall(
                "TreeNumberList/TreeNumber"
            )
            if node.text
        ]

        if len(mesh_records) < 100:

            mesh_records.append({
                "descriptor_ui": descriptor_ui,
                "descriptor_name": descriptor_name,
                "tree_numbers": "|".join(tree_numbers)
            })

        elem.clear()

print("MeSH descriptor records:", mesh_count)

mesh_sample_df = pd.DataFrame(mesh_records)

mesh_sample_df.to_csv(
    DB_DOCS / "mesh_sample.csv",
    index=False
)

print(mesh_sample_df.head(10).to_string(index=False))

# ============================================================
# 8. STORAGE BOUNDARY DESIGN
# ============================================================

storage_design = pd.DataFrame([
    ["Patient", "PostgreSQL", "Transactional"],
    ["Provider", "PostgreSQL", "Transactional"],
    ["Organization", "PostgreSQL", "Transactional"],
    ["Encounter", "PostgreSQL", "Transactional"],
    ["Diagnosis", "PostgreSQL", "Transactional"],
    ["Medication", "PostgreSQL", "Transactional"],
    ["Observation", "PostgreSQL", "Transactional"],
    ["Lab Result", "PostgreSQL", "Transactional"],
    ["Procedure", "PostgreSQL", "Transactional"],
    ["Claim", "PostgreSQL", "Transactional"],
    ["Payer", "PostgreSQL", "Transactional"],
    ["LOINC concepts", "PostgreSQL + Neo4j", "Terminology"],
    ["RxNorm concepts", "PostgreSQL + Neo4j", "Terminology"],
    ["MeSH concepts", "Neo4j", "Knowledge graph"],
    ["Biomedical relationships", "Neo4j", "Knowledge graph"],
    ["Research paper chunks", "pgvector", "Semantic retrieval"],
    ["Drug label chunks", "pgvector", "Semantic retrieval"]
], columns=[
    "data_type",
    "storage",
    "purpose"
])

storage_design.to_csv(
    DB_DOCS / "storage_boundary.csv",
    index=False
)

# ============================================================
# 9. FINAL PROPOSED RELATIONAL TABLES
# ============================================================

table_design = [
    ("patient", "Patient", "Core patient demographic record"),
    ("patient_address", "Patient", "Patient address"),
    ("patient_contact", "Patient", "Patient contact information"),
    ("patient_allergy", "Patient", "Patient allergies"),
    ("patient_immunization", "Patient", "Patient immunizations"),

    ("organization", "Provider", "Healthcare organization"),
    ("provider", "Provider", "Healthcare provider"),
    ("provider_specialty", "Provider", "Provider specialty"),

    ("encounter", "Encounter", "Clinical encounter"),
    ("encounter_type", "Encounter", "Encounter classification"),
    ("admission", "Encounter", "Admission details"),
    ("discharge", "Encounter", "Discharge details"),

    ("condition", "Clinical", "Condition terminology"),
    ("diagnosis", "Clinical", "Patient diagnosis"),
    ("procedure_definition", "Clinical", "Procedure terminology"),
    ("patient_procedure", "Clinical", "Performed procedures"),
    ("care_plan", "Clinical", "Patient care plan"),
    ("medical_device", "Clinical", "Patient medical device"),
    ("imaging_study", "Clinical", "Imaging metadata"),

    ("drug_concept", "Medication", "RxNorm drug concept"),
    ("prescription", "Medication", "Prescription"),
    ("prescription_item", "Medication", "Prescription drug item"),
    ("drug_relationship", "Medication", "RxNorm drug relationship"),

    ("loinc_concept", "Laboratory", "LOINC terminology"),
    ("observation", "Laboratory", "Clinical observation"),
    ("lab_result", "Laboratory", "Laboratory result"),

    ("payer", "Financial", "Insurance payer"),
    ("patient_insurance", "Financial", "Patient payer coverage"),
    ("claim", "Financial", "Healthcare claim"),
    ("claim_transaction", "Financial", "Claim transaction"),

    ("query_log", "Optimization", "Database workload log"),
    ("optimization_recommendation", "Optimization", "AI optimization recommendation")
]

table_design_df = pd.DataFrame(
    table_design,
    columns=["table_name", "module", "purpose"]
)

table_design_df.to_csv(
    DB_DOCS / "table_design.csv",
    index=False
)

print("\nProposed relational tables:", len(table_design_df))
print(table_design_df.to_string(index=False))

# ============================================================
# 10. RELATIONSHIP DESIGN
# ============================================================

relationships = [
    ("patient", "encounter", "1:N"),
    ("patient", "patient_address", "1:N"),
    ("patient", "patient_contact", "1:N"),
    ("patient", "patient_allergy", "1:N"),
    ("patient", "patient_immunization", "1:N"),

    ("organization", "provider", "1:N"),
    ("organization", "encounter", "1:N"),
    ("provider", "encounter", "1:N"),
    ("provider", "provider_specialty", "1:N"),

    ("encounter", "diagnosis", "1:N"),
    ("condition", "diagnosis", "1:N"),

    ("encounter", "patient_procedure", "1:N"),
    ("procedure_definition", "patient_procedure", "1:N"),

    ("patient", "care_plan", "1:N"),
    ("patient", "medical_device", "1:N"),
    ("encounter", "imaging_study", "1:N"),

    ("patient", "prescription", "1:N"),
    ("encounter", "prescription", "1:N"),
    ("prescription", "prescription_item", "1:N"),
    ("drug_concept", "prescription_item", "1:N"),

    ("drug_concept", "drug_relationship", "1:N"),

    ("patient", "observation", "1:N"),
    ("encounter", "observation", "1:N"),
    ("loinc_concept", "observation", "1:N"),

    ("patient", "lab_result", "1:N"),
    ("encounter", "lab_result", "1:N"),
    ("loinc_concept", "lab_result", "1:N"),

    ("payer", "patient_insurance", "1:N"),
    ("patient", "patient_insurance", "1:N"),

    ("patient", "claim", "1:N"),
    ("payer", "claim", "1:N"),
    ("claim", "claim_transaction", "1:N")
]

relationship_design_df = pd.DataFrame(
    relationships,
    columns=[
        "parent_table",
        "child_table",
        "cardinality"
    ]
)

relationship_design_df.to_csv(
    DB_DOCS / "relationship_design.csv",
    index=False
)

# ============================================================
# 11. SOURCE -> TARGET MAPPING
# ============================================================

source_mapping = [
    ("patients.csv", "Id", "patient", "synthea_id"),
    ("patients.csv", "BIRTHDATE", "patient", "birth_date"),
    ("patients.csv", "DEATHDATE", "patient", "death_date"),
    ("patients.csv", "SSN", "patient", "ssn"),
    ("patients.csv", "DRIVERS", "patient", "drivers_license"),
    ("patients.csv", "PASSPORT", "patient", "passport"),
    ("patients.csv", "PREFIX", "patient", "prefix"),
    ("patients.csv", "FIRST", "patient", "first_name"),
    ("patients.csv", "LAST", "patient", "last_name"),
    ("patients.csv", "SUFFIX", "patient", "suffix"),
    ("patients.csv", "MAIDEN", "patient", "maiden_name"),
    ("patients.csv", "MARITAL", "patient", "marital_status"),
    ("patients.csv", "RACE", "patient", "race"),
    ("patients.csv", "ETHNICITY", "patient", "ethnicity"),
    ("patients.csv", "GENDER", "patient", "gender"),

    ("organizations.csv", "Id", "organization", "synthea_id"),
    ("organizations.csv", "NAME", "organization", "name"),

    ("providers.csv", "Id", "provider", "synthea_id"),
    ("providers.csv", "ORGANIZATION", "provider", "organization_source_id"),
    ("providers.csv", "NAME", "provider", "name"),

    ("encounters.csv", "Id", "encounter", "synthea_id"),
    ("encounters.csv", "PATIENT", "encounter", "patient_source_id"),
    ("encounters.csv", "ORGANIZATION", "encounter", "organization_source_id"),
    ("encounters.csv", "PROVIDER", "encounter", "provider_source_id"),
    ("encounters.csv", "START", "encounter", "start_time"),
    ("encounters.csv", "STOP", "encounter", "end_time"),

    ("conditions.csv", "PATIENT", "diagnosis", "patient_source_id"),
    ("conditions.csv", "ENCOUNTER", "diagnosis", "encounter_source_id"),
    ("conditions.csv", "CODE", "condition", "source_code"),
    ("conditions.csv", "DESCRIPTION", "condition", "description"),

    ("procedures.csv", "PATIENT", "patient_procedure", "patient_source_id"),
    ("procedures.csv", "ENCOUNTER", "patient_procedure", "encounter_source_id"),
    ("procedures.csv", "CODE", "procedure_definition", "source_code"),
    ("procedures.csv", "DESCRIPTION", "procedure_definition", "description"),

    ("medications.csv", "PATIENT", "prescription", "patient_source_id"),
    ("medications.csv", "ENCOUNTER", "prescription", "encounter_source_id"),
    ("medications.csv", "CODE", "drug_concept", "source_code"),
    ("medications.csv", "DESCRIPTION", "drug_concept", "source_name"),

    ("observations.csv", "PATIENT", "observation", "patient_source_id"),
    ("observations.csv", "ENCOUNTER", "observation", "encounter_source_id"),
    ("observations.csv", "CODE", "observation", "loinc_code"),
    ("observations.csv", "VALUE", "observation", "value"),
    ("observations.csv", "UNITS", "observation", "unit"),

    ("payers.csv", "Id", "payer", "synthea_id"),
    ("payers.csv", "NAME", "payer", "name"),

    ("claims.csv", "Id", "claim", "synthea_id"),
    ("claims.csv", "PATIENTID", "claim", "patient_source_id"),
    ("claims.csv", "PRIMARYPATIENTINSURANCEID", "claim", "primary_patient_insurance_source_id"),
    ("claims.csv", "SECONDARYPATIENTINSURANCEID", "claim", "secondary_patient_insurance_source_id"),

    ("claims_transactions.csv", "CLAIMID", "claim_transaction", "claim_source_id"),

    ("Loinc.csv", "LOINC_NUM", "loinc_concept", "loinc_code"),
    ("Loinc.csv", "COMPONENT", "loinc_concept", "component"),
    ("Loinc.csv", "PROPERTY", "loinc_concept", "property"),
    ("Loinc.csv", "SYSTEM", "loinc_concept", "system"),
    ("Loinc.csv", "LONG_COMMON_NAME", "loinc_concept", "long_common_name"),

    ("RXNCONSO.RRF", "RXCUI", "drug_concept", "rxcui"),
    ("RXNCONSO.RRF", "STR", "drug_concept", "name"),
    ("RXNCONSO.RRF", "TTY", "drug_concept", "concept_type"),

    ("RXNREL.RRF", "RXCUI1", "drug_relationship", "source_rxcui"),
    ("RXNREL.RRF", "RXCUI2", "drug_relationship", "target_rxcui"),
    ("RXNREL.RRF", "REL", "drug_relationship", "relationship_type"),
    ("RXNREL.RRF", "RELA", "drug_relationship", "relationship_attribute")
]

source_mapping_df = pd.DataFrame(
    source_mapping,
    columns=[
        "source",
        "source_field",
        "target_table",
        "target_field"
    ]
)

source_mapping_df.to_csv(
    DB_DOCS / "source_mapping.csv",
    index=False
)

# ============================================================
# 12. CREATE NORMALIZATION DOCUMENT
# ============================================================

normalization_document = """
# Database Normalization

## First Normal Form (1NF)

All relations contain atomic values.
Repeating groups are separated into independent relations.

Examples:

Patient addresses are stored in patient_address instead of
repeating multiple address fields inside unrelated clinical records.

Prescription medications are separated using prescription and
prescription_item.

## Second Normal Form (2NF)

All non-key attributes depend on the complete primary key.

Many-to-many relationships are represented using associative
relations where appropriate.

Examples:

prescription_item links prescriptions and drug concepts.

diagnosis links clinical encounters with condition concepts.

## Third Normal Form (3NF)

Non-key attributes do not depend transitively on other non-key
attributes.

Example:

Provider organization information is not repeatedly stored in
encounter.

Instead:

organization
    -> provider
    -> encounter

Drug names and terminology are not repeatedly stored in every
prescription item.

Instead:

drug_concept
    -> prescription_item

LOINC descriptions are not repeatedly stored in every laboratory
result.

Instead:

loinc_concept
    -> observation/lab_result

This reduces redundancy and update anomalies and produces a
normalized relational healthcare database.
"""

(DB_DOCS / "normalization.md").write_text(
    normalization_document,
    encoding="utf-8"
)

# ============================================================
# 13. CREATE RELATIONAL SCHEMA DOCUMENT
# ============================================================

relational_document = """
# Adaptive Healthcare Knowledge Database
## Relational Schema

### Patient Module

patient
patient_address
patient_contact
patient_allergy
patient_immunization

### Provider Module

organization
provider
provider_specialty

### Encounter Module

encounter
encounter_type
admission
discharge

### Clinical Module

condition
diagnosis
procedure_definition
patient_procedure
care_plan
medical_device
imaging_study

### Medication Module

drug_concept
prescription
prescription_item
drug_relationship

### Laboratory Module

loinc_concept
observation
lab_result

### Financial Module

payer
patient_insurance
claim
claim_transaction

### Adaptive Optimization Module

query_log
optimization_recommendation
"""

(DB_DOCS / "relational_schema.md").write_text(
    relational_document,
    encoding="utf-8"
)

# ============================================================
# 14. GENERATE SQL TABLE DESIGN
# ============================================================

tables_sql = r"""
-- ============================================================
-- Adaptive Healthcare Knowledge Database
-- Phase 2 Generated PostgreSQL Schema
-- ============================================================

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE patient (
    patient_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    synthea_id UUID UNIQUE NOT NULL,
    birth_date DATE,
    death_date DATE,
    ssn VARCHAR(32),
    drivers_license VARCHAR(64),
    passport VARCHAR(64),
    prefix VARCHAR(20),
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    suffix VARCHAR(20),
    maiden_name VARCHAR(100),
    marital_status VARCHAR(50),
    race VARCHAR(100),
    ethnicity VARCHAR(100),
    gender VARCHAR(30),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE patient_address (
    address_id BIGSERIAL PRIMARY KEY,
    patient_id UUID NOT NULL REFERENCES patient(patient_id),
    address_line VARCHAR(255),
    city VARCHAR(100),
    state VARCHAR(100),
    county VARCHAR(100),
    zip_code VARCHAR(20),
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION
);

CREATE TABLE patient_contact (
    contact_id BIGSERIAL PRIMARY KEY,
    patient_id UUID NOT NULL REFERENCES patient(patient_id),
    contact_type VARCHAR(30),
    contact_value VARCHAR(255)
);

CREATE TABLE organization (
    organization_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    synthea_id UUID UNIQUE,
    name VARCHAR(255) NOT NULL,
    address VARCHAR(255),
    city VARCHAR(100),
    state VARCHAR(100),
    zip_code VARCHAR(20),
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION
);

CREATE TABLE provider (
    provider_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    synthea_id UUID UNIQUE,
    organization_id UUID REFERENCES organization(organization_id),
    name VARCHAR(255),
    gender VARCHAR(30),
    specialty VARCHAR(255),
    address VARCHAR(255),
    city VARCHAR(100),
    state VARCHAR(100),
    zip_code VARCHAR(20)
);

CREATE TABLE provider_specialty (
    provider_specialty_id BIGSERIAL PRIMARY KEY,
    provider_id UUID NOT NULL REFERENCES provider(provider_id),
    specialty VARCHAR(255) NOT NULL
);

CREATE TABLE encounter_type (
    encounter_type_id BIGSERIAL PRIMARY KEY,
    code VARCHAR(100) UNIQUE,
    description VARCHAR(255)
);

CREATE TABLE encounter (
    encounter_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    synthea_id UUID UNIQUE,
    patient_id UUID NOT NULL REFERENCES patient(patient_id),
    provider_id UUID REFERENCES provider(provider_id),
    organization_id UUID REFERENCES organization(organization_id),
    encounter_type_id BIGINT REFERENCES encounter_type(encounter_type_id),
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,
    encounter_class VARCHAR(100),
    code VARCHAR(100),
    description VARCHAR(255),
    base_cost NUMERIC(14,2),
    total_claim_cost NUMERIC(14,2),
    payer_coverage NUMERIC(14,2),
    reason_code VARCHAR(100),
    reason_description VARCHAR(255)
);

CREATE TABLE admission (
    admission_id BIGSERIAL PRIMARY KEY,
    encounter_id UUID UNIQUE NOT NULL REFERENCES encounter(encounter_id),
    admission_time TIMESTAMPTZ,
    admission_type VARCHAR(100)
);

CREATE TABLE discharge (
    discharge_id BIGSERIAL PRIMARY KEY,
    encounter_id UUID UNIQUE NOT NULL REFERENCES encounter(encounter_id),
    discharge_time TIMESTAMPTZ,
    disposition VARCHAR(100)
);

CREATE TABLE condition (
    condition_id BIGSERIAL PRIMARY KEY,
    source_code VARCHAR(100) UNIQUE,
    description VARCHAR(500)
);

CREATE TABLE diagnosis (
    diagnosis_id BIGSERIAL PRIMARY KEY,
    patient_id UUID NOT NULL REFERENCES patient(patient_id),
    encounter_id UUID REFERENCES encounter(encounter_id),
    condition_id BIGINT NOT NULL REFERENCES condition(condition_id),
    start_date TIMESTAMPTZ,
    stop_date TIMESTAMPTZ
);

CREATE TABLE procedure_definition (
    procedure_id BIGSERIAL PRIMARY KEY,
    source_code VARCHAR(100) UNIQUE,
    description VARCHAR(500)
);

CREATE TABLE patient_procedure (
    patient_procedure_id BIGSERIAL PRIMARY KEY,
    patient_id UUID NOT NULL REFERENCES patient(patient_id),
    encounter_id UUID REFERENCES encounter(encounter_id),
    procedure_id BIGINT NOT NULL REFERENCES procedure_definition(procedure_id),
    procedure_date TIMESTAMPTZ,
    base_cost NUMERIC(14,2),
    reason_code VARCHAR(100),
    reason_description VARCHAR(500)
);

CREATE TABLE care_plan (
    care_plan_id BIGSERIAL PRIMARY KEY,
    synthea_id UUID UNIQUE,
    patient_id UUID NOT NULL REFERENCES patient(patient_id),
    encounter_id UUID REFERENCES encounter(encounter_id),
    start_date DATE,
    stop_date DATE,
    code VARCHAR(100),
    description VARCHAR(500),
    reason_code VARCHAR(100),
    reason_description VARCHAR(500)
);

CREATE TABLE medical_device (
    device_id BIGSERIAL PRIMARY KEY,
    patient_id UUID NOT NULL REFERENCES patient(patient_id),
    encounter_id UUID REFERENCES encounter(encounter_id),
    start_time TIMESTAMPTZ,
    stop_time TIMESTAMPTZ,
    code VARCHAR(100),
    description VARCHAR(500),
    udi VARCHAR(255)
);

CREATE TABLE imaging_study (
    imaging_study_id BIGSERIAL PRIMARY KEY,
    synthea_id UUID UNIQUE,
    patient_id UUID NOT NULL REFERENCES patient(patient_id),
    encounter_id UUID REFERENCES encounter(encounter_id),
    study_date TIMESTAMPTZ,
    body_site_code VARCHAR(100),
    body_site_description VARCHAR(255),
    modality_code VARCHAR(100),
    modality_description VARCHAR(255),
    sop_code VARCHAR(100),
    sop_description VARCHAR(255)
);

CREATE TABLE drug_concept (
    rxcui VARCHAR(50) PRIMARY KEY,
    name TEXT NOT NULL,
    concept_type VARCHAR(50),
    source_code VARCHAR(100),
    source VARCHAR(50)
);

CREATE TABLE prescription (
    prescription_id BIGSERIAL PRIMARY KEY,
    patient_id UUID NOT NULL REFERENCES patient(patient_id),
    encounter_id UUID REFERENCES encounter(encounter_id),
    start_time TIMESTAMPTZ,
    stop_time TIMESTAMPTZ,
    reason_code VARCHAR(100),
    reason_description VARCHAR(500)
);

CREATE TABLE prescription_item (
    prescription_item_id BIGSERIAL PRIMARY KEY,
    prescription_id BIGINT NOT NULL REFERENCES prescription(prescription_id),
    rxcui VARCHAR(50) REFERENCES drug_concept(rxcui),
    description TEXT,
    base_cost NUMERIC(14,2),
    payer_coverage NUMERIC(14,2),
    dispenses INTEGER,
    total_cost NUMERIC(14,2)
);

CREATE TABLE drug_relationship (
    drug_relationship_id BIGSERIAL PRIMARY KEY,
    source_rxcui VARCHAR(50) REFERENCES drug_concept(rxcui),
    target_rxcui VARCHAR(50) REFERENCES drug_concept(rxcui),
    relationship_type VARCHAR(100),
    relationship_attribute VARCHAR(255),
    source VARCHAR(50)
);

CREATE TABLE loinc_concept (
    loinc_code VARCHAR(30) PRIMARY KEY,
    component TEXT,
    property VARCHAR(100),
    time_aspect VARCHAR(100),
    system TEXT,
    scale_type VARCHAR(100),
    method_type TEXT,
    class VARCHAR(100),
    long_common_name TEXT,
    short_name TEXT,
    status VARCHAR(50)
);

CREATE TABLE observation (
    observation_id BIGSERIAL PRIMARY KEY,
    patient_id UUID NOT NULL REFERENCES patient(patient_id),
    encounter_id UUID REFERENCES encounter(encounter_id),
    observation_time TIMESTAMPTZ,
    loinc_code VARCHAR(30) REFERENCES loinc_concept(loinc_code),
    description TEXT,
    value_text TEXT,
    value_numeric DOUBLE PRECISION,
    unit VARCHAR(100),
    value_type VARCHAR(50)
);

CREATE TABLE lab_result (
    lab_result_id BIGSERIAL PRIMARY KEY,
    patient_id UUID NOT NULL REFERENCES patient(patient_id),
    encounter_id UUID REFERENCES encounter(encounter_id),
    loinc_code VARCHAR(30) REFERENCES loinc_concept(loinc_code),
    result_time TIMESTAMPTZ,
    result_value TEXT,
    result_numeric DOUBLE PRECISION,
    unit VARCHAR(100),
    reference_range VARCHAR(255),
    status VARCHAR(50)
);

CREATE TABLE patient_allergy (
    allergy_id BIGSERIAL PRIMARY KEY,
    patient_id UUID NOT NULL REFERENCES patient(patient_id),
    encounter_id UUID REFERENCES encounter(encounter_id),
    start_date DATE,
    stop_date DATE,
    code VARCHAR(100),
    system VARCHAR(100),
    description TEXT,
    allergy_type VARCHAR(100),
    category VARCHAR(100),
    reaction_code_1 VARCHAR(100),
    reaction_description_1 TEXT,
    severity_1 VARCHAR(50),
    reaction_code_2 VARCHAR(100),
    reaction_description_2 TEXT,
    severity_2 VARCHAR(50)
);

CREATE TABLE patient_immunization (
    immunization_id BIGSERIAL PRIMARY KEY,
    patient_id UUID NOT NULL REFERENCES patient(patient_id),
    encounter_id UUID REFERENCES encounter(encounter_id),
    immunization_date TIMESTAMPTZ,
    code VARCHAR(100),
    description TEXT,
    base_cost NUMERIC(14,2)
);

CREATE TABLE payer (
    payer_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    synthea_id UUID UNIQUE,
    name VARCHAR(255),
    address VARCHAR(255),
    city VARCHAR(100),
    state VARCHAR(100),
    zip_code VARCHAR(20),
    phone VARCHAR(100)
);

CREATE TABLE patient_insurance (
    patient_insurance_id BIGSERIAL PRIMARY KEY,
    patient_id UUID NOT NULL REFERENCES patient(patient_id),
    payer_id UUID NOT NULL REFERENCES payer(payer_id),
    start_date DATE,
    end_date DATE,
    ownership VARCHAR(100),
    owner_name VARCHAR(255)
);

CREATE TABLE claim (
    claim_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    synthea_id UUID UNIQUE,
    patient_id UUID NOT NULL REFERENCES patient(patient_id),
    encounter_id UUID REFERENCES encounter(encounter_id),
    payer_id UUID REFERENCES payer(payer_id),
    provider_id UUID REFERENCES provider(provider_id),
    primary_patient_insurance_source_id VARCHAR(100),
    secondary_patient_insurance_source_id VARCHAR(100),
    department_id VARCHAR(100),
    diagnosis_code VARCHAR(100),
    outstanding_1 NUMERIC(14,2),
    outstanding_2 NUMERIC(14,2)
);

CREATE TABLE claim_transaction (
    claim_transaction_id BIGSERIAL PRIMARY KEY,
    claim_id UUID NOT NULL REFERENCES claim(claim_id),
    transaction_type VARCHAR(100),
    transaction_method VARCHAR(100),
    transaction_date TIMESTAMPTZ,
    amount NUMERIC(14,2),
    patient_payment NUMERIC(14,2),
    transfer_out_id VARCHAR(255),
    transfer_type VARCHAR(100)
);

CREATE TABLE query_log (
    query_log_id BIGSERIAL PRIMARY KEY,
    query_hash VARCHAR(128),
    query_fingerprint TEXT,
    query_text TEXT,
    query_type VARCHAR(50),
    tables_accessed TEXT,
    columns_accessed TEXT,
    join_count INTEGER DEFAULT 0,
    filter_count INTEGER DEFAULT 0,
    execution_time_ms DOUBLE PRECISION,
    estimated_cost DOUBLE PRECISION,
    actual_rows BIGINT,
    index_used BOOLEAN,
    cpu_usage DOUBLE PRECISION,
    io_read_blocks BIGINT,
    io_write_blocks BIGINT,
    executed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE optimization_recommendation (
    recommendation_id BIGSERIAL PRIMARY KEY,
    query_log_id BIGINT REFERENCES query_log(query_log_id),
    recommendation_type VARCHAR(100),
    target_table VARCHAR(255),
    target_columns TEXT,
    recommendation_sql TEXT,
    reason TEXT,
    estimated_before_cost DOUBLE PRECISION,
    estimated_after_cost DOUBLE PRECISION,
    expected_improvement_percent DOUBLE PRECISION,
    actual_before_ms DOUBLE PRECISION,
    actual_after_ms DOUBLE PRECISION,
    actual_improvement_percent DOUBLE PRECISION,
    status VARCHAR(30) DEFAULT 'PENDING',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    evaluated_at TIMESTAMPTZ
);
"""

(SCHEMA_DIR / "01_tables.sql").write_text(
    tables_sql,
    encoding="utf-8"
)

# ============================================================
# 15. CONSTRAINTS
# ============================================================

constraints_sql = r"""
-- Additional Phase 2 constraints

ALTER TABLE encounter
ADD CONSTRAINT chk_encounter_dates
CHECK (
    end_time IS NULL
    OR start_time IS NULL
    OR end_time >= start_time
);

ALTER TABLE diagnosis
ADD CONSTRAINT chk_diagnosis_dates
CHECK (
    stop_date IS NULL
    OR start_date IS NULL
    OR stop_date >= start_date
);

ALTER TABLE care_plan
ADD CONSTRAINT chk_care_plan_dates
CHECK (
    stop_date IS NULL
    OR start_date IS NULL
    OR stop_date >= start_date
);

ALTER TABLE patient_insurance
ADD CONSTRAINT chk_insurance_dates
CHECK (
    end_date IS NULL
    OR start_date IS NULL
    OR end_date >= start_date
);

ALTER TABLE patient_procedure
ADD CONSTRAINT chk_procedure_cost
CHECK (
    base_cost IS NULL
    OR base_cost >= 0
);

ALTER TABLE prescription_item
ADD CONSTRAINT chk_medication_cost
CHECK (
    total_cost IS NULL
    OR total_cost >= 0
);

ALTER TABLE optimization_recommendation
ADD CONSTRAINT chk_recommendation_status
CHECK (
    status IN (
        'PENDING',
        'APPROVED',
        'REJECTED',
        'APPLIED',
        'FAILED'
    )
);
"""

(SCHEMA_DIR / "02_constraints.sql").write_text(
    constraints_sql,
    encoding="utf-8"
)

# ============================================================
# 16. BASELINE INDEX STRATEGY
# ============================================================

baseline_indexes_sql = r"""
-- ============================================================
-- BASELINE INDEXES ONLY
--
-- Deliberately minimal.
-- Composite workload indexes will be recommended later by
-- the adaptive optimization engine.
-- ============================================================

CREATE INDEX idx_encounter_patient_fk
ON encounter(patient_id);

CREATE INDEX idx_encounter_provider_fk
ON encounter(provider_id);

CREATE INDEX idx_diagnosis_patient_fk
ON diagnosis(patient_id);

CREATE INDEX idx_diagnosis_encounter_fk
ON diagnosis(encounter_id);

CREATE INDEX idx_prescription_patient_fk
ON prescription(patient_id);

CREATE INDEX idx_prescription_encounter_fk
ON prescription(encounter_id);

CREATE INDEX idx_observation_patient_fk
ON observation(patient_id);

CREATE INDEX idx_observation_encounter_fk
ON observation(encounter_id);

CREATE INDEX idx_lab_result_patient_fk
ON lab_result(patient_id);

CREATE INDEX idx_claim_patient_fk
ON claim(patient_id);

CREATE INDEX idx_claim_transaction_claim_fk
ON claim_transaction(claim_id);

CREATE INDEX idx_query_log_executed_at
ON query_log(executed_at);

-- IMPORTANT:
-- Do not add workload-specific composite indexes here.
-- Those are reserved for the AI optimization experiments.
"""

(SCHEMA_DIR / "03_baseline_indexes.sql").write_text(
    baseline_indexes_sql,
    encoding="utf-8"
)

# ============================================================
# 17. VIEWS
# ============================================================

views_sql = r"""
CREATE OR REPLACE VIEW patient_summary AS
SELECT
    p.patient_id,
    p.synthea_id,
    p.first_name,
    p.last_name,
    p.birth_date,
    p.gender,
    COUNT(DISTINCT e.encounter_id) AS encounter_count,
    COUNT(DISTINCT d.diagnosis_id) AS diagnosis_count,
    COUNT(DISTINCT pr.prescription_id) AS prescription_count
FROM patient p
LEFT JOIN encounter e
    ON e.patient_id = p.patient_id
LEFT JOIN diagnosis d
    ON d.patient_id = p.patient_id
LEFT JOIN prescription pr
    ON pr.patient_id = p.patient_id
GROUP BY
    p.patient_id,
    p.synthea_id,
    p.first_name,
    p.last_name,
    p.birth_date,
    p.gender;


CREATE OR REPLACE VIEW patient_condition_history AS
SELECT
    p.patient_id,
    p.first_name,
    p.last_name,
    c.source_code,
    c.description,
    d.start_date,
    d.stop_date
FROM diagnosis d
JOIN patient p
    ON p.patient_id = d.patient_id
JOIN condition c
    ON c.condition_id = d.condition_id;


CREATE OR REPLACE VIEW patient_medication_history AS
SELECT
    p.patient_id,
    p.first_name,
    p.last_name,
    pr.prescription_id,
    dc.rxcui,
    dc.name AS drug_name,
    pr.start_time,
    pr.stop_time
FROM prescription pr
JOIN patient p
    ON p.patient_id = pr.patient_id
JOIN prescription_item pi
    ON pi.prescription_id = pr.prescription_id
LEFT JOIN drug_concept dc
    ON dc.rxcui = pi.rxcui;


CREATE OR REPLACE VIEW patient_lab_history AS
SELECT
    p.patient_id,
    p.first_name,
    p.last_name,
    lr.result_time,
    lr.loinc_code,
    lc.long_common_name,
    lr.result_value,
    lr.result_numeric,
    lr.unit
FROM lab_result lr
JOIN patient p
    ON p.patient_id = lr.patient_id
LEFT JOIN loinc_concept lc
    ON lc.loinc_code = lr.loinc_code;


CREATE OR REPLACE VIEW slow_query_summary AS
SELECT
    query_hash,
    query_fingerprint,
    COUNT(*) AS execution_count,
    AVG(execution_time_ms) AS avg_execution_time_ms,
    MAX(execution_time_ms) AS max_execution_time_ms,
    AVG(estimated_cost) AS avg_estimated_cost
FROM query_log
GROUP BY
    query_hash,
    query_fingerprint;
"""

(SCHEMA_DIR / "04_views.sql").write_text(
    views_sql,
    encoding="utf-8"
)

# ============================================================
# 18. TRIGGERS
# ============================================================

triggers_sql = r"""
CREATE OR REPLACE FUNCTION update_patient_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;


CREATE TRIGGER trg_patient_updated_at
BEFORE UPDATE ON patient
FOR EACH ROW
EXECUTE FUNCTION update_patient_timestamp();


CREATE OR REPLACE FUNCTION calculate_actual_improvement()
RETURNS TRIGGER AS $$
BEGIN

    IF NEW.actual_before_ms IS NOT NULL
       AND NEW.actual_after_ms IS NOT NULL
       AND NEW.actual_before_ms > 0 THEN

        NEW.actual_improvement_percent =
        (
            (NEW.actual_before_ms - NEW.actual_after_ms)
            / NEW.actual_before_ms
        ) * 100;

    END IF;

    RETURN NEW;

END;
$$ LANGUAGE plpgsql;


CREATE TRIGGER trg_calculate_optimization_improvement
BEFORE INSERT OR UPDATE
ON optimization_recommendation
FOR EACH ROW
EXECUTE FUNCTION calculate_actual_improvement();
"""

(SCHEMA_DIR / "05_triggers.sql").write_text(
    triggers_sql,
    encoding="utf-8"
)

# ============================================================
# 19. STORED FUNCTIONS
# ============================================================

functions_sql = r"""
CREATE OR REPLACE FUNCTION get_patient_encounter_count(
    p_patient_id UUID
)
RETURNS BIGINT
LANGUAGE SQL
AS $$
    SELECT COUNT(*)
    FROM encounter
    WHERE patient_id = p_patient_id;
$$;


CREATE OR REPLACE FUNCTION get_patient_diagnosis_count(
    p_patient_id UUID
)
RETURNS BIGINT
LANGUAGE SQL
AS $$
    SELECT COUNT(*)
    FROM diagnosis
    WHERE patient_id = p_patient_id;
$$;


CREATE OR REPLACE FUNCTION get_average_query_latency()
RETURNS DOUBLE PRECISION
LANGUAGE SQL
AS $$
    SELECT AVG(execution_time_ms)
    FROM query_log;
$$;


CREATE OR REPLACE FUNCTION get_slow_queries(
    threshold_ms DOUBLE PRECISION
)
RETURNS TABLE (
    query_log_id BIGINT,
    query_fingerprint TEXT,
    execution_time_ms DOUBLE PRECISION,
    executed_at TIMESTAMPTZ
)
LANGUAGE SQL
AS $$
    SELECT
        q.query_log_id,
        q.query_fingerprint,
        q.execution_time_ms,
        q.executed_at
    FROM query_log q
    WHERE q.execution_time_ms >= threshold_ms
    ORDER BY q.execution_time_ms DESC;
$$;
"""

(SCHEMA_DIR / "06_functions.sql").write_text(
    functions_sql,
    encoding="utf-8"
)

# ============================================================
# 20. CREATE ER DIAGRAM USING MERMAID TEXT
# ============================================================

er_diagram = """
erDiagram

    PATIENT ||--o{ ENCOUNTER : has
    PATIENT ||--o{ PATIENT_ADDRESS : has
    PATIENT ||--o{ PATIENT_CONTACT : has
    PATIENT ||--o{ PATIENT_ALLERGY : has
    PATIENT ||--o{ PATIENT_IMMUNIZATION : receives

    ORGANIZATION ||--o{ PROVIDER : employs
    ORGANIZATION ||--o{ ENCOUNTER : hosts

    PROVIDER ||--o{ ENCOUNTER : conducts
    PROVIDER ||--o{ PROVIDER_SPECIALTY : has

    ENCOUNTER ||--o{ DIAGNOSIS : contains
    CONDITION ||--o{ DIAGNOSIS : classifies

    ENCOUNTER ||--o{ PATIENT_PROCEDURE : contains
    PROCEDURE_DEFINITION ||--o{ PATIENT_PROCEDURE : defines

    PATIENT ||--o{ CARE_PLAN : has
    PATIENT ||--o{ MEDICAL_DEVICE : uses
    ENCOUNTER ||--o{ IMAGING_STUDY : contains

    PATIENT ||--o{ PRESCRIPTION : receives
    ENCOUNTER ||--o{ PRESCRIPTION : generates

    PRESCRIPTION ||--o{ PRESCRIPTION_ITEM : contains
    DRUG_CONCEPT ||--o{ PRESCRIPTION_ITEM : identifies

    DRUG_CONCEPT ||--o{ DRUG_RELATIONSHIP : participates

    PATIENT ||--o{ OBSERVATION : has
    ENCOUNTER ||--o{ OBSERVATION : contains
    LOINC_CONCEPT ||--o{ OBSERVATION : identifies

    PATIENT ||--o{ LAB_RESULT : has
    ENCOUNTER ||--o{ LAB_RESULT : contains
    LOINC_CONCEPT ||--o{ LAB_RESULT : identifies

    PATIENT ||--o{ PATIENT_INSURANCE : has
    PAYER ||--o{ PATIENT_INSURANCE : provides

    PATIENT ||--o{ CLAIM : has
    PAYER ||--o{ CLAIM : processes
    CLAIM ||--o{ CLAIM_TRANSACTION : contains

    QUERY_LOG ||--o{ OPTIMIZATION_RECOMMENDATION : generates
"""

(DB_DOCS / "er_diagram_mermaid.md").write_text(
    "```mermaid\n" + er_diagram + "\n```",
    encoding="utf-8"
)

# ============================================================
# 21. GENERATE PHASE 2 SUMMARY
# ============================================================

phase2_summary = f"""
# Phase 2 - Database Design Summary

Generated: {datetime.now().isoformat()}

## Datasets

Synthea files: {len(synthea_files)}
LOINC records: {len(loinc)}
RxNorm concept rows: {len(rxnconso)}
RxNorm relationship rows: {len(rxnrel)}
RxNorm attribute rows: {len(rxnsat)}
MeSH descriptor records: {mesh_count}

## PostgreSQL Design

Relational tables proposed: {len(table_design_df)}

Modules:

- Patient
- Provider
- Encounter
- Clinical
- Medication
- Laboratory
- Financial
- Adaptive Optimization

## Storage Architecture

PostgreSQL:
Transactional and structured healthcare data.

Neo4j:
Biomedical concepts and relationships.

pgvector:
Biomedical document embeddings and semantic retrieval.

## Index Strategy

Only baseline PK/FK/supporting indexes are initially created.

Workload-specific composite, partial, GIN and BRIN indexes
are deliberately reserved for the adaptive optimization phase.

## Normalization

Operational relational schema designed toward Third Normal Form.

## Phase 2 Output

- Dataset summaries
- Data dictionaries
- Relationship candidates
- Source-to-target mappings
- Storage boundaries
- Table design
- Relationship design
- Normalization documentation
- ER diagram definition
- PostgreSQL table DDL
- Constraints
- Baseline indexes
- Views
- Triggers
- Stored functions
"""

(DB_DOCS / "phase2_summary.md").write_text(
    phase2_summary,
    encoding="utf-8"
)

# ============================================================
# 22. FINAL VALIDATION
# ============================================================

print("\n" + "=" * 90)
print("PHASE 2 OUTPUT VALIDATION")
print("=" * 90)

expected_outputs = [
    DB_DOCS / "synthea_file_summary.csv",
    DB_DOCS / "synthea_data_dictionary.csv",
    DB_DOCS / "synthea_relationship_candidates.csv",
    DB_DOCS / "loinc_data_dictionary.csv",
    DB_DOCS / "rxnorm_summary.csv",
    DB_DOCS / "mesh_sample.csv",
    DB_DOCS / "storage_boundary.csv",
    DB_DOCS / "table_design.csv",
    DB_DOCS / "relationship_design.csv",
    DB_DOCS / "source_mapping.csv",
    DB_DOCS / "normalization.md",
    DB_DOCS / "relational_schema.md",
    DB_DOCS / "er_diagram_mermaid.md",
    DB_DOCS / "phase2_summary.md",

    SCHEMA_DIR / "01_tables.sql",
    SCHEMA_DIR / "02_constraints.sql",
    SCHEMA_DIR / "03_baseline_indexes.sql",
    SCHEMA_DIR / "04_views.sql",
    SCHEMA_DIR / "05_triggers.sql",
    SCHEMA_DIR / "06_functions.sql"
]

success = True

for output in expected_outputs:

    exists = output.exists()

    print(
        f"{'OK' if exists else 'MISSING':<10}",
        output.relative_to(PROJECT_ROOT)
    )

    if not exists:
        success = False

print("\n" + "=" * 90)

if success:

    print("PHASE 2 DESIGN GENERATION COMPLETED SUCCESSFULLY")

else:

    print("PHASE 2 COMPLETED WITH MISSING OUTPUTS")

print("=" * 90)

print("\nGenerated relational tables:", len(table_design_df))
print("Synthea files inspected:", len(synthea_files))
print("LOINC concepts inspected:", len(loinc))
print("RxNorm concept rows inspected:", len(rxnconso))
print("RxNorm relationship rows inspected:", len(rxnrel))
print("MeSH descriptors inspected:", mesh_count)

print("\nNEXT PHASE:")
print("PHASE 3 -> PostgreSQL implementation and ETL")