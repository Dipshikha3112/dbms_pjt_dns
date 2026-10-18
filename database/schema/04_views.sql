CREATE OR REPLACE VIEW patient_summary AS
SELECT
    p.patient_id,
    p.synthea_id,
    p.first_name,
    p.last_name,
    p.birth_date,
    p.gender,
    COALESCE(e.encounter_count, 0) AS encounter_count,
    COALESCE(d.diagnosis_count, 0) AS diagnosis_count,
    COALESCE(pr.prescription_count, 0) AS prescription_count
FROM patient p
LEFT JOIN (
    SELECT
        patient_id,
        COUNT(*) AS encounter_count
    FROM encounter
    GROUP BY patient_id
) e
    ON e.patient_id = p.patient_id
LEFT JOIN (
    SELECT
        patient_id,
        COUNT(*) AS diagnosis_count
    FROM diagnosis
    GROUP BY patient_id
) d
    ON d.patient_id = p.patient_id
LEFT JOIN (
    SELECT
        patient_id,
        COUNT(*) AS prescription_count
    FROM prescription
    GROUP BY patient_id
) pr
    ON pr.patient_id = p.patient_id;


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