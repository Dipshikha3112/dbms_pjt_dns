
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
