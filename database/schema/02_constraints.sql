
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
