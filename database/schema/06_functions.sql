
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
