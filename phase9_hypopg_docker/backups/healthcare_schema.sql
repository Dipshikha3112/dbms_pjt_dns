--
-- PostgreSQL database dump
--

\restrict i15N56GapnQAiZ6fd0oiTEhdo05tBa4ZbnvNqS575y2vxVeel1UhWKqmrfODd5d

-- Dumped from database version 17.11
-- Dumped by pg_dump version 17.11

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: public; Type: SCHEMA; Schema: -; Owner: -
--

-- *not* creating schema, since initdb creates it


--
-- Name: SCHEMA public; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON SCHEMA public IS '';


--
-- Name: pgcrypto; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA public;


--
-- Name: EXTENSION pgcrypto; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION pgcrypto IS 'cryptographic functions';


--
-- Name: calculate_actual_improvement(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.calculate_actual_improvement() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
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
$$;


--
-- Name: get_average_query_latency(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.get_average_query_latency() RETURNS double precision
    LANGUAGE sql
    AS $$
    SELECT AVG(execution_time_ms)
    FROM query_log;
$$;


--
-- Name: get_patient_diagnosis_count(uuid); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.get_patient_diagnosis_count(p_patient_id uuid) RETURNS bigint
    LANGUAGE sql
    AS $$
    SELECT COUNT(*)
    FROM diagnosis
    WHERE patient_id = p_patient_id;
$$;


--
-- Name: get_patient_encounter_count(uuid); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.get_patient_encounter_count(p_patient_id uuid) RETURNS bigint
    LANGUAGE sql
    AS $$
    SELECT COUNT(*)
    FROM encounter
    WHERE patient_id = p_patient_id;
$$;


--
-- Name: get_slow_queries(double precision); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.get_slow_queries(threshold_ms double precision) RETURNS TABLE(query_log_id bigint, query_fingerprint text, execution_time_ms double precision, executed_at timestamp with time zone)
    LANGUAGE sql
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


--
-- Name: update_patient_timestamp(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.update_patient_timestamp() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$;


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: admission; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.admission (
    admission_id bigint NOT NULL,
    encounter_id uuid NOT NULL,
    admission_time timestamp with time zone,
    admission_type character varying(100)
);


--
-- Name: admission_admission_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.admission_admission_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: admission_admission_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.admission_admission_id_seq OWNED BY public.admission.admission_id;


--
-- Name: care_plan; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.care_plan (
    care_plan_id bigint NOT NULL,
    synthea_id uuid,
    patient_id uuid NOT NULL,
    encounter_id uuid,
    start_date date,
    stop_date date,
    code character varying(100),
    description character varying(500),
    reason_code character varying(100),
    reason_description character varying(500),
    CONSTRAINT chk_care_plan_dates CHECK (((stop_date IS NULL) OR (start_date IS NULL) OR (stop_date >= start_date)))
);


--
-- Name: care_plan_care_plan_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.care_plan_care_plan_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: care_plan_care_plan_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.care_plan_care_plan_id_seq OWNED BY public.care_plan.care_plan_id;


--
-- Name: claim; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.claim (
    claim_id uuid DEFAULT gen_random_uuid() NOT NULL,
    synthea_id uuid,
    patient_id uuid NOT NULL,
    encounter_id uuid,
    payer_id uuid,
    provider_id uuid,
    primary_patient_insurance_source_id character varying(100),
    secondary_patient_insurance_source_id character varying(100),
    department_id character varying(100),
    diagnosis_code character varying(100),
    outstanding_1 numeric(14,2),
    outstanding_2 numeric(14,2)
);


--
-- Name: claim_transaction; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.claim_transaction (
    claim_transaction_id bigint NOT NULL,
    claim_id uuid NOT NULL,
    transaction_type character varying(100),
    transaction_method character varying(100),
    transaction_date timestamp with time zone,
    amount numeric(14,2),
    patient_payment numeric(14,2),
    transfer_out_id character varying(255),
    transfer_type character varying(100)
);


--
-- Name: claim_transaction_claim_transaction_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.claim_transaction_claim_transaction_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: claim_transaction_claim_transaction_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.claim_transaction_claim_transaction_id_seq OWNED BY public.claim_transaction.claim_transaction_id;


--
-- Name: condition; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.condition (
    condition_id bigint NOT NULL,
    source_code character varying(100),
    description character varying(500)
);


--
-- Name: condition_condition_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.condition_condition_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: condition_condition_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.condition_condition_id_seq OWNED BY public.condition.condition_id;


--
-- Name: diagnosis; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.diagnosis (
    diagnosis_id bigint NOT NULL,
    patient_id uuid NOT NULL,
    encounter_id uuid,
    condition_id bigint NOT NULL,
    start_date timestamp with time zone,
    stop_date timestamp with time zone,
    CONSTRAINT chk_diagnosis_dates CHECK (((stop_date IS NULL) OR (start_date IS NULL) OR (stop_date >= start_date)))
);


--
-- Name: diagnosis_diagnosis_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.diagnosis_diagnosis_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: diagnosis_diagnosis_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.diagnosis_diagnosis_id_seq OWNED BY public.diagnosis.diagnosis_id;


--
-- Name: discharge; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.discharge (
    discharge_id bigint NOT NULL,
    encounter_id uuid NOT NULL,
    discharge_time timestamp with time zone,
    disposition character varying(100)
);


--
-- Name: discharge_discharge_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.discharge_discharge_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: discharge_discharge_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.discharge_discharge_id_seq OWNED BY public.discharge.discharge_id;


--
-- Name: drug_concept; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.drug_concept (
    rxcui character varying(50) NOT NULL,
    name text NOT NULL,
    concept_type character varying(50),
    source_code character varying(100),
    source character varying(50)
);


--
-- Name: drug_relationship; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.drug_relationship (
    drug_relationship_id bigint NOT NULL,
    source_rxcui character varying(50),
    target_rxcui character varying(50),
    relationship_type character varying(100),
    relationship_attribute character varying(255),
    source character varying(50)
);


--
-- Name: drug_relationship_drug_relationship_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.drug_relationship_drug_relationship_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: drug_relationship_drug_relationship_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.drug_relationship_drug_relationship_id_seq OWNED BY public.drug_relationship.drug_relationship_id;


--
-- Name: encounter; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.encounter (
    encounter_id uuid DEFAULT gen_random_uuid() NOT NULL,
    synthea_id uuid,
    patient_id uuid NOT NULL,
    provider_id uuid,
    organization_id uuid,
    encounter_type_id bigint,
    start_time timestamp with time zone,
    end_time timestamp with time zone,
    encounter_class character varying(100),
    code character varying(100),
    description character varying(255),
    base_cost numeric(14,2),
    total_claim_cost numeric(14,2),
    payer_coverage numeric(14,2),
    reason_code character varying(100),
    reason_description character varying(255),
    CONSTRAINT chk_encounter_dates CHECK (((end_time IS NULL) OR (start_time IS NULL) OR (end_time >= start_time)))
);


--
-- Name: encounter_type; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.encounter_type (
    encounter_type_id bigint NOT NULL,
    code character varying(100),
    description character varying(255)
);


--
-- Name: encounter_type_encounter_type_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.encounter_type_encounter_type_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: encounter_type_encounter_type_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.encounter_type_encounter_type_id_seq OWNED BY public.encounter_type.encounter_type_id;


--
-- Name: imaging_study; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.imaging_study (
    imaging_study_id bigint NOT NULL,
    synthea_id uuid,
    patient_id uuid NOT NULL,
    encounter_id uuid,
    study_date timestamp with time zone,
    body_site_code character varying(100),
    body_site_description character varying(255),
    modality_code character varying(100),
    modality_description character varying(255),
    sop_code character varying(100),
    sop_description character varying(255)
);


--
-- Name: imaging_study_imaging_study_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.imaging_study_imaging_study_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: imaging_study_imaging_study_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.imaging_study_imaging_study_id_seq OWNED BY public.imaging_study.imaging_study_id;


--
-- Name: lab_result; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.lab_result (
    lab_result_id bigint NOT NULL,
    patient_id uuid NOT NULL,
    encounter_id uuid,
    loinc_code character varying(30),
    result_time timestamp with time zone,
    result_value text,
    result_numeric double precision,
    unit character varying(100),
    reference_range character varying(255),
    status character varying(50)
);


--
-- Name: lab_result_lab_result_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.lab_result_lab_result_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: lab_result_lab_result_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.lab_result_lab_result_id_seq OWNED BY public.lab_result.lab_result_id;


--
-- Name: loinc_concept; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.loinc_concept (
    loinc_code character varying(30) NOT NULL,
    component text,
    property character varying(100),
    time_aspect character varying(100),
    system text,
    scale_type character varying(100),
    method_type text,
    class character varying(100),
    long_common_name text,
    short_name text,
    status character varying(50)
);


--
-- Name: medical_device; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.medical_device (
    device_id bigint NOT NULL,
    patient_id uuid NOT NULL,
    encounter_id uuid,
    start_time timestamp with time zone,
    stop_time timestamp with time zone,
    code character varying(100),
    description character varying(500),
    udi character varying(255)
);


--
-- Name: medical_device_device_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.medical_device_device_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: medical_device_device_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.medical_device_device_id_seq OWNED BY public.medical_device.device_id;


--
-- Name: observation; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.observation (
    observation_id bigint NOT NULL,
    patient_id uuid NOT NULL,
    encounter_id uuid,
    observation_time timestamp with time zone,
    loinc_code character varying(30),
    description text,
    value_text text,
    value_numeric double precision,
    unit character varying(100),
    value_type character varying(50)
);


--
-- Name: observation_observation_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.observation_observation_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: observation_observation_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.observation_observation_id_seq OWNED BY public.observation.observation_id;


--
-- Name: optimization_recommendation; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.optimization_recommendation (
    recommendation_id bigint NOT NULL,
    query_log_id bigint,
    recommendation_type character varying(100),
    target_table character varying(255),
    target_columns text,
    recommendation_sql text,
    reason text,
    estimated_before_cost double precision,
    estimated_after_cost double precision,
    expected_improvement_percent double precision,
    actual_before_ms double precision,
    actual_after_ms double precision,
    actual_improvement_percent double precision,
    status character varying(30) DEFAULT 'PENDING'::character varying,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    evaluated_at timestamp with time zone,
    CONSTRAINT chk_recommendation_status CHECK (((status)::text = ANY ((ARRAY['PENDING'::character varying, 'APPROVED'::character varying, 'REJECTED'::character varying, 'APPLIED'::character varying, 'FAILED'::character varying])::text[])))
);


--
-- Name: optimization_recommendation_recommendation_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.optimization_recommendation_recommendation_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: optimization_recommendation_recommendation_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.optimization_recommendation_recommendation_id_seq OWNED BY public.optimization_recommendation.recommendation_id;


--
-- Name: organization; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.organization (
    organization_id uuid DEFAULT gen_random_uuid() NOT NULL,
    synthea_id uuid,
    name character varying(255) NOT NULL,
    address character varying(255),
    city character varying(100),
    state character varying(100),
    zip_code character varying(20),
    latitude double precision,
    longitude double precision
);


--
-- Name: patient; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.patient (
    patient_id uuid DEFAULT gen_random_uuid() NOT NULL,
    synthea_id uuid NOT NULL,
    birth_date date,
    death_date date,
    ssn character varying(32),
    drivers_license character varying(64),
    passport character varying(64),
    prefix character varying(20),
    first_name character varying(100),
    last_name character varying(100),
    suffix character varying(20),
    maiden_name character varying(100),
    marital_status character varying(50),
    race character varying(100),
    ethnicity character varying(100),
    gender character varying(30),
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- Name: patient_address; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.patient_address (
    address_id bigint NOT NULL,
    patient_id uuid NOT NULL,
    address_line character varying(255),
    city character varying(100),
    state character varying(100),
    county character varying(100),
    zip_code character varying(20),
    latitude double precision,
    longitude double precision
);


--
-- Name: patient_address_address_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.patient_address_address_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: patient_address_address_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.patient_address_address_id_seq OWNED BY public.patient_address.address_id;


--
-- Name: patient_allergy; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.patient_allergy (
    allergy_id bigint NOT NULL,
    patient_id uuid NOT NULL,
    encounter_id uuid,
    start_date date,
    stop_date date,
    code character varying(100),
    system character varying(100),
    description text,
    allergy_type character varying(100),
    category character varying(100),
    reaction_code_1 character varying(100),
    reaction_description_1 text,
    severity_1 character varying(50),
    reaction_code_2 character varying(100),
    reaction_description_2 text,
    severity_2 character varying(50)
);


--
-- Name: patient_allergy_allergy_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.patient_allergy_allergy_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: patient_allergy_allergy_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.patient_allergy_allergy_id_seq OWNED BY public.patient_allergy.allergy_id;


--
-- Name: patient_condition_history; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.patient_condition_history AS
 SELECT p.patient_id,
    p.first_name,
    p.last_name,
    c.source_code,
    c.description,
    d.start_date,
    d.stop_date
   FROM ((public.diagnosis d
     JOIN public.patient p ON ((p.patient_id = d.patient_id)))
     JOIN public.condition c ON ((c.condition_id = d.condition_id)));


--
-- Name: patient_contact; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.patient_contact (
    contact_id bigint NOT NULL,
    patient_id uuid NOT NULL,
    contact_type character varying(30),
    contact_value character varying(255)
);


--
-- Name: patient_contact_contact_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.patient_contact_contact_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: patient_contact_contact_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.patient_contact_contact_id_seq OWNED BY public.patient_contact.contact_id;


--
-- Name: patient_immunization; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.patient_immunization (
    immunization_id bigint NOT NULL,
    patient_id uuid NOT NULL,
    encounter_id uuid,
    immunization_date timestamp with time zone,
    code character varying(100),
    description text,
    base_cost numeric(14,2)
);


--
-- Name: patient_immunization_immunization_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.patient_immunization_immunization_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: patient_immunization_immunization_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.patient_immunization_immunization_id_seq OWNED BY public.patient_immunization.immunization_id;


--
-- Name: patient_insurance; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.patient_insurance (
    patient_insurance_id bigint NOT NULL,
    patient_id uuid NOT NULL,
    payer_id uuid NOT NULL,
    start_date date,
    end_date date,
    ownership character varying(100),
    owner_name character varying(255),
    CONSTRAINT chk_insurance_dates CHECK (((end_date IS NULL) OR (start_date IS NULL) OR (end_date >= start_date)))
);


--
-- Name: patient_insurance_patient_insurance_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.patient_insurance_patient_insurance_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: patient_insurance_patient_insurance_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.patient_insurance_patient_insurance_id_seq OWNED BY public.patient_insurance.patient_insurance_id;


--
-- Name: patient_lab_history; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.patient_lab_history AS
 SELECT p.patient_id,
    p.first_name,
    p.last_name,
    lr.result_time,
    lr.loinc_code,
    lc.long_common_name,
    lr.result_value,
    lr.result_numeric,
    lr.unit
   FROM ((public.lab_result lr
     JOIN public.patient p ON ((p.patient_id = lr.patient_id)))
     LEFT JOIN public.loinc_concept lc ON (((lc.loinc_code)::text = (lr.loinc_code)::text)));


--
-- Name: prescription; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.prescription (
    prescription_id bigint NOT NULL,
    patient_id uuid NOT NULL,
    encounter_id uuid,
    start_time timestamp with time zone,
    stop_time timestamp with time zone,
    reason_code character varying(100),
    reason_description character varying(500)
);


--
-- Name: prescription_item; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.prescription_item (
    prescription_item_id bigint NOT NULL,
    prescription_id bigint NOT NULL,
    rxcui character varying(50),
    description text,
    base_cost numeric(14,2),
    payer_coverage numeric(14,2),
    dispenses integer,
    total_cost numeric(14,2),
    CONSTRAINT chk_medication_cost CHECK (((total_cost IS NULL) OR (total_cost >= (0)::numeric)))
);


--
-- Name: patient_medication_history; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.patient_medication_history AS
 SELECT p.patient_id,
    p.first_name,
    p.last_name,
    pr.prescription_id,
    dc.rxcui,
    dc.name AS drug_name,
    pr.start_time,
    pr.stop_time
   FROM (((public.prescription pr
     JOIN public.patient p ON ((p.patient_id = pr.patient_id)))
     JOIN public.prescription_item pi ON ((pi.prescription_id = pr.prescription_id)))
     LEFT JOIN public.drug_concept dc ON (((dc.rxcui)::text = (pi.rxcui)::text)));


--
-- Name: patient_procedure; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.patient_procedure (
    patient_procedure_id bigint NOT NULL,
    patient_id uuid NOT NULL,
    encounter_id uuid,
    procedure_id bigint NOT NULL,
    procedure_date timestamp with time zone,
    base_cost numeric(14,2),
    reason_code character varying(100),
    reason_description character varying(500),
    CONSTRAINT chk_procedure_cost CHECK (((base_cost IS NULL) OR (base_cost >= (0)::numeric)))
);


--
-- Name: patient_procedure_patient_procedure_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.patient_procedure_patient_procedure_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: patient_procedure_patient_procedure_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.patient_procedure_patient_procedure_id_seq OWNED BY public.patient_procedure.patient_procedure_id;


--
-- Name: patient_summary; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.patient_summary AS
 SELECT p.patient_id,
    p.synthea_id,
    p.first_name,
    p.last_name,
    p.birth_date,
    p.gender,
    count(DISTINCT e.encounter_id) AS encounter_count,
    count(DISTINCT d.diagnosis_id) AS diagnosis_count,
    count(DISTINCT pr.prescription_id) AS prescription_count
   FROM (((public.patient p
     LEFT JOIN public.encounter e ON ((e.patient_id = p.patient_id)))
     LEFT JOIN public.diagnosis d ON ((d.patient_id = p.patient_id)))
     LEFT JOIN public.prescription pr ON ((pr.patient_id = p.patient_id)))
  GROUP BY p.patient_id, p.synthea_id, p.first_name, p.last_name, p.birth_date, p.gender;


--
-- Name: payer; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.payer (
    payer_id uuid DEFAULT gen_random_uuid() NOT NULL,
    synthea_id uuid,
    name character varying(255),
    address character varying(255),
    city character varying(100),
    state character varying(100),
    zip_code character varying(20),
    phone character varying(100)
);


--
-- Name: prescription_item_prescription_item_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.prescription_item_prescription_item_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: prescription_item_prescription_item_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.prescription_item_prescription_item_id_seq OWNED BY public.prescription_item.prescription_item_id;


--
-- Name: prescription_prescription_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.prescription_prescription_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: prescription_prescription_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.prescription_prescription_id_seq OWNED BY public.prescription.prescription_id;


--
-- Name: procedure_definition; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.procedure_definition (
    procedure_id bigint NOT NULL,
    source_code character varying(100),
    description character varying(500)
);


--
-- Name: procedure_definition_procedure_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.procedure_definition_procedure_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: procedure_definition_procedure_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.procedure_definition_procedure_id_seq OWNED BY public.procedure_definition.procedure_id;


--
-- Name: provider; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.provider (
    provider_id uuid DEFAULT gen_random_uuid() NOT NULL,
    synthea_id uuid,
    organization_id uuid,
    name character varying(255),
    gender character varying(30),
    specialty character varying(255),
    address character varying(255),
    city character varying(100),
    state character varying(100),
    zip_code character varying(20)
);


--
-- Name: provider_specialty; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.provider_specialty (
    provider_specialty_id bigint NOT NULL,
    provider_id uuid NOT NULL,
    specialty character varying(255) NOT NULL
);


--
-- Name: provider_specialty_provider_specialty_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.provider_specialty_provider_specialty_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: provider_specialty_provider_specialty_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.provider_specialty_provider_specialty_id_seq OWNED BY public.provider_specialty.provider_specialty_id;


--
-- Name: query_log; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.query_log (
    query_log_id bigint NOT NULL,
    query_hash character varying(128),
    query_fingerprint text,
    query_text text,
    query_type character varying(50),
    tables_accessed text,
    columns_accessed text,
    join_count integer DEFAULT 0,
    filter_count integer DEFAULT 0,
    execution_time_ms double precision,
    estimated_cost double precision,
    actual_rows bigint,
    index_used boolean,
    cpu_usage double precision,
    io_read_blocks bigint,
    io_write_blocks bigint,
    executed_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- Name: query_log_query_log_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.query_log_query_log_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: query_log_query_log_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.query_log_query_log_id_seq OWNED BY public.query_log.query_log_id;


--
-- Name: slow_query_summary; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.slow_query_summary AS
 SELECT query_hash,
    query_fingerprint,
    count(*) AS execution_count,
    avg(execution_time_ms) AS avg_execution_time_ms,
    max(execution_time_ms) AS max_execution_time_ms,
    avg(estimated_cost) AS avg_estimated_cost
   FROM public.query_log
  GROUP BY query_hash, query_fingerprint;


--
-- Name: admission admission_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.admission ALTER COLUMN admission_id SET DEFAULT nextval('public.admission_admission_id_seq'::regclass);


--
-- Name: care_plan care_plan_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.care_plan ALTER COLUMN care_plan_id SET DEFAULT nextval('public.care_plan_care_plan_id_seq'::regclass);


--
-- Name: claim_transaction claim_transaction_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.claim_transaction ALTER COLUMN claim_transaction_id SET DEFAULT nextval('public.claim_transaction_claim_transaction_id_seq'::regclass);


--
-- Name: condition condition_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.condition ALTER COLUMN condition_id SET DEFAULT nextval('public.condition_condition_id_seq'::regclass);


--
-- Name: diagnosis diagnosis_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.diagnosis ALTER COLUMN diagnosis_id SET DEFAULT nextval('public.diagnosis_diagnosis_id_seq'::regclass);


--
-- Name: discharge discharge_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.discharge ALTER COLUMN discharge_id SET DEFAULT nextval('public.discharge_discharge_id_seq'::regclass);


--
-- Name: drug_relationship drug_relationship_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.drug_relationship ALTER COLUMN drug_relationship_id SET DEFAULT nextval('public.drug_relationship_drug_relationship_id_seq'::regclass);


--
-- Name: encounter_type encounter_type_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.encounter_type ALTER COLUMN encounter_type_id SET DEFAULT nextval('public.encounter_type_encounter_type_id_seq'::regclass);


--
-- Name: imaging_study imaging_study_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.imaging_study ALTER COLUMN imaging_study_id SET DEFAULT nextval('public.imaging_study_imaging_study_id_seq'::regclass);


--
-- Name: lab_result lab_result_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.lab_result ALTER COLUMN lab_result_id SET DEFAULT nextval('public.lab_result_lab_result_id_seq'::regclass);


--
-- Name: medical_device device_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.medical_device ALTER COLUMN device_id SET DEFAULT nextval('public.medical_device_device_id_seq'::regclass);


--
-- Name: observation observation_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.observation ALTER COLUMN observation_id SET DEFAULT nextval('public.observation_observation_id_seq'::regclass);


--
-- Name: optimization_recommendation recommendation_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.optimization_recommendation ALTER COLUMN recommendation_id SET DEFAULT nextval('public.optimization_recommendation_recommendation_id_seq'::regclass);


--
-- Name: patient_address address_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.patient_address ALTER COLUMN address_id SET DEFAULT nextval('public.patient_address_address_id_seq'::regclass);


--
-- Name: patient_allergy allergy_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.patient_allergy ALTER COLUMN allergy_id SET DEFAULT nextval('public.patient_allergy_allergy_id_seq'::regclass);


--
-- Name: patient_contact contact_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.patient_contact ALTER COLUMN contact_id SET DEFAULT nextval('public.patient_contact_contact_id_seq'::regclass);


--
-- Name: patient_immunization immunization_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.patient_immunization ALTER COLUMN immunization_id SET DEFAULT nextval('public.patient_immunization_immunization_id_seq'::regclass);


--
-- Name: patient_insurance patient_insurance_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.patient_insurance ALTER COLUMN patient_insurance_id SET DEFAULT nextval('public.patient_insurance_patient_insurance_id_seq'::regclass);


--
-- Name: patient_procedure patient_procedure_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.patient_procedure ALTER COLUMN patient_procedure_id SET DEFAULT nextval('public.patient_procedure_patient_procedure_id_seq'::regclass);


--
-- Name: prescription prescription_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prescription ALTER COLUMN prescription_id SET DEFAULT nextval('public.prescription_prescription_id_seq'::regclass);


--
-- Name: prescription_item prescription_item_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prescription_item ALTER COLUMN prescription_item_id SET DEFAULT nextval('public.prescription_item_prescription_item_id_seq'::regclass);


--
-- Name: procedure_definition procedure_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.procedure_definition ALTER COLUMN procedure_id SET DEFAULT nextval('public.procedure_definition_procedure_id_seq'::regclass);


--
-- Name: provider_specialty provider_specialty_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.provider_specialty ALTER COLUMN provider_specialty_id SET DEFAULT nextval('public.provider_specialty_provider_specialty_id_seq'::regclass);


--
-- Name: query_log query_log_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.query_log ALTER COLUMN query_log_id SET DEFAULT nextval('public.query_log_query_log_id_seq'::regclass);


--
-- Name: admission admission_encounter_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.admission
    ADD CONSTRAINT admission_encounter_id_key UNIQUE (encounter_id);


--
-- Name: admission admission_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.admission
    ADD CONSTRAINT admission_pkey PRIMARY KEY (admission_id);


--
-- Name: care_plan care_plan_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.care_plan
    ADD CONSTRAINT care_plan_pkey PRIMARY KEY (care_plan_id);


--
-- Name: care_plan care_plan_synthea_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.care_plan
    ADD CONSTRAINT care_plan_synthea_id_key UNIQUE (synthea_id);


--
-- Name: claim claim_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.claim
    ADD CONSTRAINT claim_pkey PRIMARY KEY (claim_id);


--
-- Name: claim claim_synthea_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.claim
    ADD CONSTRAINT claim_synthea_id_key UNIQUE (synthea_id);


--
-- Name: claim_transaction claim_transaction_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.claim_transaction
    ADD CONSTRAINT claim_transaction_pkey PRIMARY KEY (claim_transaction_id);


--
-- Name: condition condition_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.condition
    ADD CONSTRAINT condition_pkey PRIMARY KEY (condition_id);


--
-- Name: condition condition_source_code_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.condition
    ADD CONSTRAINT condition_source_code_key UNIQUE (source_code);


--
-- Name: diagnosis diagnosis_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.diagnosis
    ADD CONSTRAINT diagnosis_pkey PRIMARY KEY (diagnosis_id);


--
-- Name: discharge discharge_encounter_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.discharge
    ADD CONSTRAINT discharge_encounter_id_key UNIQUE (encounter_id);


--
-- Name: discharge discharge_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.discharge
    ADD CONSTRAINT discharge_pkey PRIMARY KEY (discharge_id);


--
-- Name: drug_concept drug_concept_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.drug_concept
    ADD CONSTRAINT drug_concept_pkey PRIMARY KEY (rxcui);


--
-- Name: drug_relationship drug_relationship_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.drug_relationship
    ADD CONSTRAINT drug_relationship_pkey PRIMARY KEY (drug_relationship_id);


--
-- Name: encounter encounter_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.encounter
    ADD CONSTRAINT encounter_pkey PRIMARY KEY (encounter_id);


--
-- Name: encounter encounter_synthea_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.encounter
    ADD CONSTRAINT encounter_synthea_id_key UNIQUE (synthea_id);


--
-- Name: encounter_type encounter_type_code_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.encounter_type
    ADD CONSTRAINT encounter_type_code_key UNIQUE (code);


--
-- Name: encounter_type encounter_type_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.encounter_type
    ADD CONSTRAINT encounter_type_pkey PRIMARY KEY (encounter_type_id);


--
-- Name: imaging_study imaging_study_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.imaging_study
    ADD CONSTRAINT imaging_study_pkey PRIMARY KEY (imaging_study_id);


--
-- Name: imaging_study imaging_study_synthea_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.imaging_study
    ADD CONSTRAINT imaging_study_synthea_id_key UNIQUE (synthea_id);


--
-- Name: lab_result lab_result_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.lab_result
    ADD CONSTRAINT lab_result_pkey PRIMARY KEY (lab_result_id);


--
-- Name: loinc_concept loinc_concept_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.loinc_concept
    ADD CONSTRAINT loinc_concept_pkey PRIMARY KEY (loinc_code);


--
-- Name: medical_device medical_device_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.medical_device
    ADD CONSTRAINT medical_device_pkey PRIMARY KEY (device_id);


--
-- Name: observation observation_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.observation
    ADD CONSTRAINT observation_pkey PRIMARY KEY (observation_id);


--
-- Name: optimization_recommendation optimization_recommendation_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.optimization_recommendation
    ADD CONSTRAINT optimization_recommendation_pkey PRIMARY KEY (recommendation_id);


--
-- Name: organization organization_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.organization
    ADD CONSTRAINT organization_pkey PRIMARY KEY (organization_id);


--
-- Name: organization organization_synthea_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.organization
    ADD CONSTRAINT organization_synthea_id_key UNIQUE (synthea_id);


--
-- Name: patient_address patient_address_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.patient_address
    ADD CONSTRAINT patient_address_pkey PRIMARY KEY (address_id);


--
-- Name: patient_allergy patient_allergy_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.patient_allergy
    ADD CONSTRAINT patient_allergy_pkey PRIMARY KEY (allergy_id);


--
-- Name: patient_contact patient_contact_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.patient_contact
    ADD CONSTRAINT patient_contact_pkey PRIMARY KEY (contact_id);


--
-- Name: patient_immunization patient_immunization_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.patient_immunization
    ADD CONSTRAINT patient_immunization_pkey PRIMARY KEY (immunization_id);


--
-- Name: patient_insurance patient_insurance_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.patient_insurance
    ADD CONSTRAINT patient_insurance_pkey PRIMARY KEY (patient_insurance_id);


--
-- Name: patient patient_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.patient
    ADD CONSTRAINT patient_pkey PRIMARY KEY (patient_id);


--
-- Name: patient_procedure patient_procedure_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.patient_procedure
    ADD CONSTRAINT patient_procedure_pkey PRIMARY KEY (patient_procedure_id);


--
-- Name: patient patient_synthea_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.patient
    ADD CONSTRAINT patient_synthea_id_key UNIQUE (synthea_id);


--
-- Name: payer payer_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.payer
    ADD CONSTRAINT payer_pkey PRIMARY KEY (payer_id);


--
-- Name: payer payer_synthea_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.payer
    ADD CONSTRAINT payer_synthea_id_key UNIQUE (synthea_id);


--
-- Name: prescription_item prescription_item_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prescription_item
    ADD CONSTRAINT prescription_item_pkey PRIMARY KEY (prescription_item_id);


--
-- Name: prescription prescription_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prescription
    ADD CONSTRAINT prescription_pkey PRIMARY KEY (prescription_id);


--
-- Name: procedure_definition procedure_definition_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.procedure_definition
    ADD CONSTRAINT procedure_definition_pkey PRIMARY KEY (procedure_id);


--
-- Name: procedure_definition procedure_definition_source_code_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.procedure_definition
    ADD CONSTRAINT procedure_definition_source_code_key UNIQUE (source_code);


--
-- Name: provider provider_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.provider
    ADD CONSTRAINT provider_pkey PRIMARY KEY (provider_id);


--
-- Name: provider_specialty provider_specialty_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.provider_specialty
    ADD CONSTRAINT provider_specialty_pkey PRIMARY KEY (provider_specialty_id);


--
-- Name: provider provider_synthea_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.provider
    ADD CONSTRAINT provider_synthea_id_key UNIQUE (synthea_id);


--
-- Name: query_log query_log_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.query_log
    ADD CONSTRAINT query_log_pkey PRIMARY KEY (query_log_id);


--
-- Name: idx_claim_patient_fk; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_claim_patient_fk ON public.claim USING btree (patient_id);


--
-- Name: idx_claim_transaction_claim_fk; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_claim_transaction_claim_fk ON public.claim_transaction USING btree (claim_id);


--
-- Name: idx_diagnosis_encounter_fk; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_diagnosis_encounter_fk ON public.diagnosis USING btree (encounter_id);


--
-- Name: idx_diagnosis_patient_fk; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_diagnosis_patient_fk ON public.diagnosis USING btree (patient_id);


--
-- Name: idx_encounter_patient_fk; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_encounter_patient_fk ON public.encounter USING btree (patient_id);


--
-- Name: idx_encounter_provider_fk; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_encounter_provider_fk ON public.encounter USING btree (provider_id);


--
-- Name: idx_lab_result_patient_fk; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_lab_result_patient_fk ON public.lab_result USING btree (patient_id);


--
-- Name: idx_observation_encounter_fk; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_observation_encounter_fk ON public.observation USING btree (encounter_id);


--
-- Name: idx_observation_patient_fk; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_observation_patient_fk ON public.observation USING btree (patient_id);


--
-- Name: idx_prescription_encounter_fk; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_prescription_encounter_fk ON public.prescription USING btree (encounter_id);


--
-- Name: idx_prescription_patient_fk; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_prescription_patient_fk ON public.prescription USING btree (patient_id);


--
-- Name: idx_query_log_executed_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_query_log_executed_at ON public.query_log USING btree (executed_at);


--
-- Name: optimization_recommendation trg_calculate_optimization_improvement; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER trg_calculate_optimization_improvement BEFORE INSERT OR UPDATE ON public.optimization_recommendation FOR EACH ROW EXECUTE FUNCTION public.calculate_actual_improvement();


--
-- Name: patient trg_patient_updated_at; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER trg_patient_updated_at BEFORE UPDATE ON public.patient FOR EACH ROW EXECUTE FUNCTION public.update_patient_timestamp();


--
-- Name: admission admission_encounter_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.admission
    ADD CONSTRAINT admission_encounter_id_fkey FOREIGN KEY (encounter_id) REFERENCES public.encounter(encounter_id);


--
-- Name: care_plan care_plan_encounter_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.care_plan
    ADD CONSTRAINT care_plan_encounter_id_fkey FOREIGN KEY (encounter_id) REFERENCES public.encounter(encounter_id);


--
-- Name: care_plan care_plan_patient_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.care_plan
    ADD CONSTRAINT care_plan_patient_id_fkey FOREIGN KEY (patient_id) REFERENCES public.patient(patient_id);


--
-- Name: claim claim_encounter_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.claim
    ADD CONSTRAINT claim_encounter_id_fkey FOREIGN KEY (encounter_id) REFERENCES public.encounter(encounter_id);


--
-- Name: claim claim_patient_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.claim
    ADD CONSTRAINT claim_patient_id_fkey FOREIGN KEY (patient_id) REFERENCES public.patient(patient_id);


--
-- Name: claim claim_payer_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.claim
    ADD CONSTRAINT claim_payer_id_fkey FOREIGN KEY (payer_id) REFERENCES public.payer(payer_id);


--
-- Name: claim claim_provider_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.claim
    ADD CONSTRAINT claim_provider_id_fkey FOREIGN KEY (provider_id) REFERENCES public.provider(provider_id);


--
-- Name: claim_transaction claim_transaction_claim_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.claim_transaction
    ADD CONSTRAINT claim_transaction_claim_id_fkey FOREIGN KEY (claim_id) REFERENCES public.claim(claim_id);


--
-- Name: diagnosis diagnosis_condition_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.diagnosis
    ADD CONSTRAINT diagnosis_condition_id_fkey FOREIGN KEY (condition_id) REFERENCES public.condition(condition_id);


--
-- Name: diagnosis diagnosis_encounter_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.diagnosis
    ADD CONSTRAINT diagnosis_encounter_id_fkey FOREIGN KEY (encounter_id) REFERENCES public.encounter(encounter_id);


--
-- Name: diagnosis diagnosis_patient_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.diagnosis
    ADD CONSTRAINT diagnosis_patient_id_fkey FOREIGN KEY (patient_id) REFERENCES public.patient(patient_id);


--
-- Name: discharge discharge_encounter_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.discharge
    ADD CONSTRAINT discharge_encounter_id_fkey FOREIGN KEY (encounter_id) REFERENCES public.encounter(encounter_id);


--
-- Name: drug_relationship drug_relationship_source_rxcui_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.drug_relationship
    ADD CONSTRAINT drug_relationship_source_rxcui_fkey FOREIGN KEY (source_rxcui) REFERENCES public.drug_concept(rxcui);


--
-- Name: drug_relationship drug_relationship_target_rxcui_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.drug_relationship
    ADD CONSTRAINT drug_relationship_target_rxcui_fkey FOREIGN KEY (target_rxcui) REFERENCES public.drug_concept(rxcui);


--
-- Name: encounter encounter_encounter_type_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.encounter
    ADD CONSTRAINT encounter_encounter_type_id_fkey FOREIGN KEY (encounter_type_id) REFERENCES public.encounter_type(encounter_type_id);


--
-- Name: encounter encounter_organization_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.encounter
    ADD CONSTRAINT encounter_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES public.organization(organization_id);


--
-- Name: encounter encounter_patient_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.encounter
    ADD CONSTRAINT encounter_patient_id_fkey FOREIGN KEY (patient_id) REFERENCES public.patient(patient_id);


--
-- Name: encounter encounter_provider_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.encounter
    ADD CONSTRAINT encounter_provider_id_fkey FOREIGN KEY (provider_id) REFERENCES public.provider(provider_id);


--
-- Name: imaging_study imaging_study_encounter_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.imaging_study
    ADD CONSTRAINT imaging_study_encounter_id_fkey FOREIGN KEY (encounter_id) REFERENCES public.encounter(encounter_id);


--
-- Name: imaging_study imaging_study_patient_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.imaging_study
    ADD CONSTRAINT imaging_study_patient_id_fkey FOREIGN KEY (patient_id) REFERENCES public.patient(patient_id);


--
-- Name: lab_result lab_result_encounter_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.lab_result
    ADD CONSTRAINT lab_result_encounter_id_fkey FOREIGN KEY (encounter_id) REFERENCES public.encounter(encounter_id);


--
-- Name: lab_result lab_result_loinc_code_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.lab_result
    ADD CONSTRAINT lab_result_loinc_code_fkey FOREIGN KEY (loinc_code) REFERENCES public.loinc_concept(loinc_code);


--
-- Name: lab_result lab_result_patient_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.lab_result
    ADD CONSTRAINT lab_result_patient_id_fkey FOREIGN KEY (patient_id) REFERENCES public.patient(patient_id);


--
-- Name: medical_device medical_device_encounter_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.medical_device
    ADD CONSTRAINT medical_device_encounter_id_fkey FOREIGN KEY (encounter_id) REFERENCES public.encounter(encounter_id);


--
-- Name: medical_device medical_device_patient_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.medical_device
    ADD CONSTRAINT medical_device_patient_id_fkey FOREIGN KEY (patient_id) REFERENCES public.patient(patient_id);


--
-- Name: observation observation_encounter_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.observation
    ADD CONSTRAINT observation_encounter_id_fkey FOREIGN KEY (encounter_id) REFERENCES public.encounter(encounter_id);


--
-- Name: observation observation_loinc_code_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.observation
    ADD CONSTRAINT observation_loinc_code_fkey FOREIGN KEY (loinc_code) REFERENCES public.loinc_concept(loinc_code);


--
-- Name: observation observation_patient_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.observation
    ADD CONSTRAINT observation_patient_id_fkey FOREIGN KEY (patient_id) REFERENCES public.patient(patient_id);


--
-- Name: optimization_recommendation optimization_recommendation_query_log_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.optimization_recommendation
    ADD CONSTRAINT optimization_recommendation_query_log_id_fkey FOREIGN KEY (query_log_id) REFERENCES public.query_log(query_log_id);


--
-- Name: patient_address patient_address_patient_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.patient_address
    ADD CONSTRAINT patient_address_patient_id_fkey FOREIGN KEY (patient_id) REFERENCES public.patient(patient_id);


--
-- Name: patient_allergy patient_allergy_encounter_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.patient_allergy
    ADD CONSTRAINT patient_allergy_encounter_id_fkey FOREIGN KEY (encounter_id) REFERENCES public.encounter(encounter_id);


--
-- Name: patient_allergy patient_allergy_patient_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.patient_allergy
    ADD CONSTRAINT patient_allergy_patient_id_fkey FOREIGN KEY (patient_id) REFERENCES public.patient(patient_id);


--
-- Name: patient_contact patient_contact_patient_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.patient_contact
    ADD CONSTRAINT patient_contact_patient_id_fkey FOREIGN KEY (patient_id) REFERENCES public.patient(patient_id);


--
-- Name: patient_immunization patient_immunization_encounter_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.patient_immunization
    ADD CONSTRAINT patient_immunization_encounter_id_fkey FOREIGN KEY (encounter_id) REFERENCES public.encounter(encounter_id);


--
-- Name: patient_immunization patient_immunization_patient_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.patient_immunization
    ADD CONSTRAINT patient_immunization_patient_id_fkey FOREIGN KEY (patient_id) REFERENCES public.patient(patient_id);


--
-- Name: patient_insurance patient_insurance_patient_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.patient_insurance
    ADD CONSTRAINT patient_insurance_patient_id_fkey FOREIGN KEY (patient_id) REFERENCES public.patient(patient_id);


--
-- Name: patient_insurance patient_insurance_payer_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.patient_insurance
    ADD CONSTRAINT patient_insurance_payer_id_fkey FOREIGN KEY (payer_id) REFERENCES public.payer(payer_id);


--
-- Name: patient_procedure patient_procedure_encounter_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.patient_procedure
    ADD CONSTRAINT patient_procedure_encounter_id_fkey FOREIGN KEY (encounter_id) REFERENCES public.encounter(encounter_id);


--
-- Name: patient_procedure patient_procedure_patient_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.patient_procedure
    ADD CONSTRAINT patient_procedure_patient_id_fkey FOREIGN KEY (patient_id) REFERENCES public.patient(patient_id);


--
-- Name: patient_procedure patient_procedure_procedure_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.patient_procedure
    ADD CONSTRAINT patient_procedure_procedure_id_fkey FOREIGN KEY (procedure_id) REFERENCES public.procedure_definition(procedure_id);


--
-- Name: prescription prescription_encounter_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prescription
    ADD CONSTRAINT prescription_encounter_id_fkey FOREIGN KEY (encounter_id) REFERENCES public.encounter(encounter_id);


--
-- Name: prescription_item prescription_item_prescription_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prescription_item
    ADD CONSTRAINT prescription_item_prescription_id_fkey FOREIGN KEY (prescription_id) REFERENCES public.prescription(prescription_id);


--
-- Name: prescription_item prescription_item_rxcui_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prescription_item
    ADD CONSTRAINT prescription_item_rxcui_fkey FOREIGN KEY (rxcui) REFERENCES public.drug_concept(rxcui);


--
-- Name: prescription prescription_patient_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prescription
    ADD CONSTRAINT prescription_patient_id_fkey FOREIGN KEY (patient_id) REFERENCES public.patient(patient_id);


--
-- Name: provider provider_organization_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.provider
    ADD CONSTRAINT provider_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES public.organization(organization_id);


--
-- Name: provider_specialty provider_specialty_provider_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.provider_specialty
    ADD CONSTRAINT provider_specialty_provider_id_fkey FOREIGN KEY (provider_id) REFERENCES public.provider(provider_id);


--
-- PostgreSQL database dump complete
--

\unrestrict i15N56GapnQAiZ6fd0oiTEhdo05tBa4ZbnvNqS575y2vxVeel1UhWKqmrfODd5d

