
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
