
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
