
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
