
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
