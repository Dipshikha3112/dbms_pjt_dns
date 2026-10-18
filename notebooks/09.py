
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
import os
import re
import json
import hashlib
import getpass
import random

import pandas as pd
import psycopg
import sqlglot
from sqlglot import exp

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "results" / "phase6_workloads" / "combined_workload.csv"
OUT = ROOT / "results" / "phase8_2_expanded"
OUT.mkdir(parents=True, exist_ok=True)

DB_NAME = os.getenv("PGDATABASE", "adaptive_healthcare_scale_10k")
DB_HOST = os.getenv("PGHOST", "localhost")
DB_PORT = int(os.getenv("PGPORT", "5432"))
DB_USER = os.getenv("PGUSER", "postgres")
DB_PASSWORD = os.getenv("PGPASSWORD")

SEED = 42
RNG = random.Random(SEED)
ROLES = ("doctor", "admin", "research")


def section(title):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75, flush=True)


def clean(sql):
    return re.sub(r"\s+", " ", str(sql)).strip().rstrip(";")


def sha(sql):
    return hashlib.sha256(clean(sql).encode()).hexdigest()


def fingerprint(sql):
    try:
        tree = sqlglot.parse_one(sql, read="postgres")

        # Replace literal values while preserving SQL structure.
        for node in list(tree.find_all(exp.Literal)):
            node.replace(exp.Literal.string("VALUE"))

        return hashlib.sha256(
            tree.sql(dialect="postgres").lower().encode()
        ).hexdigest()

    except Exception:
        text = re.sub(r"'(?:''|[^'])*'", "'VALUE'", sql)
        text = re.sub(r"\b\d+(?:\.\d+)?\b", "0", text)
        return sha(text.lower())


def load_original():
    section("1. LOAD ORIGINAL WORKLOAD")

    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    df = pd.read_csv(SOURCE, low_memory=False)

    required = {
        "sql", "workload_type", "query_hash",
        "query_family"
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print("Original records:", len(df))
    print("Original labels:", df.workload_type.value_counts().to_dict())

    df["original_workload_type"] = df["workload_type"]
    df["sql"] = df["sql"].map(clean)

    # A mixed workload is a mixture of role-specific queries.
    # It is not a fourth SQL query class.
    def recover_role(row):
        role = str(row["workload_type"]).strip().lower()

        if role in ROLES:
            return role

        if role == "mixed":
            source = str(
                row.get("source_workload", "")
            ).strip().lower()

            if source in ROLES:
                return source

        return None

    df["role_label"] = df.apply(recover_role, axis=1)

    unknown = int(df["role_label"].isna().sum())
    if unknown:
        raise ValueError(
            f"{unknown} rows lack recoverable source roles. "
            "Do not guess their labels."
        )

    df["workload_context"] = df["original_workload_type"]
    df["fingerprint_hash"] = df["sql"].map(fingerprint)

    conflicts = (
        df.groupby("query_hash")["role_label"]
        .nunique()
    )

    print(
        "Conflicting role labels after correction:",
        int((conflicts > 1).sum())
    )

    # Preserve any remaining conflicts for investigation.
    bad_hashes = set(conflicts[conflicts > 1].index)

    df[df.query_hash.isin(bad_hashes)].to_csv(
        OUT / "remaining_label_conflicts.csv",
        index=False
    )

    if bad_hashes:
        raise RuntimeError(
            "Some identical SQL queries still have conflicting "
            "roles. Review remaining_label_conflicts.csv."
        )

    return df


# Each family below represents a distinct analytical question.
# Parameter values are intentionally not used to manufacture
# additional fingerprints.

DOCTOR_TEMPLATES = {
    "patient_activity": [
        """
        SELECT p.patient_id, COUNT(e.encounter_id) AS visits
        FROM patient p
        LEFT JOIN encounter e ON e.patient_id = p.patient_id
        WHERE p.patient_id = {patient}
        GROUP BY p.patient_id
        """,
        """
        SELECT e.encounter_class, COUNT(*) AS visits
        FROM encounter e
        WHERE e.patient_id = {patient}
        GROUP BY e.encounter_class
        """,
        """
        SELECT e.encounter_id, e.start_time, e.description
        FROM encounter e
        WHERE e.patient_id = {patient}
          AND e.end_time IS NULL
        ORDER BY e.start_time DESC
        LIMIT 25
        """
    ],
    "diagnosis_review": [
        """
        SELECT d.start_date, c.description
        FROM diagnosis d
        JOIN condition c ON c.condition_id = d.condition_id
        WHERE d.patient_id = {patient}
          AND d.stop_date IS NULL
        ORDER BY d.start_date DESC
        """,
        """
        SELECT c.description, COUNT(*) AS occurrences
        FROM diagnosis d
        JOIN condition c ON c.condition_id = d.condition_id
        WHERE d.patient_id = {patient}
        GROUP BY c.description
        ORDER BY occurrences DESC
        """,
        """
        SELECT EXISTS (
            SELECT 1 FROM diagnosis d
            WHERE d.patient_id = {patient}
              AND d.condition_id = {condition}
        ) AS has_diagnosis
        """
    ],
    "medication_review": [
        """
        SELECT pr.start_time, pi.description, pi.dispenses
        FROM prescription pr
        JOIN prescription_item pi
          ON pi.prescription_id = pr.prescription_id
        WHERE pr.patient_id = {patient}
          AND pr.stop_time IS NULL
        """,
        """
        SELECT pi.rxcui, COUNT(*) AS prescriptions
        FROM prescription pr
        JOIN prescription_item pi
          ON pi.prescription_id = pr.prescription_id
        WHERE pr.patient_id = {patient}
        GROUP BY pi.rxcui
        ORDER BY prescriptions DESC
        """,
        """
        SELECT COUNT(DISTINCT pi.rxcui) AS unique_drugs
        FROM prescription pr
        JOIN prescription_item pi
          ON pi.prescription_id = pr.prescription_id
        WHERE pr.patient_id = {patient}
        """
    ],
    "lab_review": [
        """
        SELECT lr.loinc_code, MAX(lr.result_time) AS latest_test
        FROM lab_result lr
        WHERE lr.patient_id = {patient}
        GROUP BY lr.loinc_code
        """,
        """
        SELECT lr.result_time, lr.result_numeric, lr.unit
        FROM lab_result lr
        WHERE lr.patient_id = {patient}
          AND lr.loinc_code = {loinc}
          AND lr.result_numeric IS NOT NULL
        ORDER BY lr.result_time DESC
        LIMIT 10
        """,
        """
        SELECT lr.loinc_code, AVG(lr.result_numeric) AS avg_value
        FROM lab_result lr
        WHERE lr.patient_id = {patient}
          AND lr.result_numeric IS NOT NULL
        GROUP BY lr.loinc_code
        """
    ],
    "procedure_review": [
        """
        SELECT pp.procedure_date, pd.description
        FROM patient_procedure pp
        JOIN procedure_definition pd
          ON pd.procedure_id = pp.procedure_id
        WHERE pp.patient_id = {patient}
        ORDER BY pp.procedure_date DESC
        LIMIT 20
        """,
        """
        SELECT COUNT(*) AS procedure_count
        FROM patient_procedure
        WHERE patient_id = {patient}
        """,
        """
        SELECT pd.description, COUNT(*) AS times_performed
        FROM patient_procedure pp
        JOIN procedure_definition pd
          ON pd.procedure_id = pp.procedure_id
        WHERE pp.patient_id = {patient}
        GROUP BY pd.description
        """
    ]
}

ADMIN_TEMPLATES = {
    "provider_operations": [
        """
        SELECT provider_id, COUNT(*) AS visits
        FROM encounter
        GROUP BY provider_id
        HAVING COUNT(*) > 10
        ORDER BY visits DESC
        LIMIT 30
        """,
        """
        SELECT organization_id,
               COUNT(DISTINCT provider_id) AS providers
        FROM encounter
        GROUP BY organization_id
        ORDER BY providers DESC
        """,
        """
        SELECT provider_id, AVG(base_cost) AS avg_cost
        FROM encounter
        WHERE provider_id IS NOT NULL
        GROUP BY provider_id
        """,
        """
        SELECT organization_id, SUM(total_claim_cost) AS claims
        FROM encounter
        GROUP BY organization_id
        ORDER BY claims DESC
        LIMIT 20
        """
    ],
    "claims_operations": [
        """
        SELECT payer_id, COUNT(*) AS claims
        FROM claim
        GROUP BY payer_id
        ORDER BY claims DESC
        """,
        """
        SELECT payer_id, SUM(outstanding_1) AS balance
        FROM claim
        GROUP BY payer_id
        HAVING SUM(outstanding_1) > 0
        """,
        """
        SELECT transaction_method, SUM(amount) AS total
        FROM claim_transaction
        GROUP BY transaction_method
        """,
        """
        SELECT transaction_type,
               AVG(patient_payment) AS avg_payment
        FROM claim_transaction
        GROUP BY transaction_type
        """
    ],
    "utilization_operations": [
        """
        SELECT encounter_class, COUNT(*) AS visits
        FROM encounter
        GROUP BY encounter_class
        ORDER BY visits DESC
        """,
        """
        SELECT DATE_TRUNC('year', start_time) AS year,
               COUNT(*) AS visits
        FROM encounter
        WHERE start_time IS NOT NULL
        GROUP BY DATE_TRUNC('year', start_time)
        ORDER BY year
        """,
        """
        SELECT organization_id,
               AVG(total_claim_cost) AS avg_claim
        FROM encounter
        GROUP BY organization_id
        """,
        """
        SELECT provider_id, COUNT(DISTINCT patient_id) AS patients
        FROM encounter
        WHERE provider_id IS NOT NULL
        GROUP BY provider_id
        HAVING COUNT(DISTINCT patient_id) >= 5
        """
    ]
}

RESEARCH_TEMPLATES = {
    "cohort_analysis": [
        """
        SELECT p.gender, COUNT(DISTINCT d.patient_id) AS patients
        FROM patient p
        JOIN diagnosis d ON d.patient_id = p.patient_id
        WHERE d.condition_id = {condition}
        GROUP BY p.gender
        """,
        """
        SELECT COUNT(DISTINCT d.patient_id) AS cohort_size
        FROM diagnosis d
        WHERE d.condition_id = {condition}
          AND d.stop_date IS NULL
        """,
        """
        SELECT c.description, COUNT(DISTINCT d.patient_id) AS patients
        FROM diagnosis d
        JOIN condition c ON c.condition_id = d.condition_id
        GROUP BY c.description
        HAVING COUNT(DISTINCT d.patient_id) >= 10
        ORDER BY patients DESC
        LIMIT 50
        """,
        """
        SELECT p.gender, COUNT(*) AS diagnoses
        FROM diagnosis d
        JOIN patient p ON p.patient_id = d.patient_id
        GROUP BY p.gender
        """
    ],
    "medication_epidemiology": [
        """
        SELECT pi.rxcui, COUNT(DISTINCT pr.patient_id) AS patients
        FROM prescription pr
        JOIN prescription_item pi
          ON pi.prescription_id = pr.prescription_id
        GROUP BY pi.rxcui
        ORDER BY patients DESC
        LIMIT 50
        """,
        """
        SELECT pi.rxcui, AVG(pi.total_cost) AS avg_cost
        FROM prescription_item pi
        WHERE pi.rxcui IS NOT NULL
        GROUP BY pi.rxcui
        """,
        """
        SELECT d.condition_id, COUNT(DISTINCT pr.patient_id) AS treated
        FROM diagnosis d
        JOIN prescription pr ON pr.patient_id = d.patient_id
        GROUP BY d.condition_id
        ORDER BY treated DESC
        LIMIT 50
        """
    ],
    "laboratory_epidemiology": [
        """
        SELECT loinc_code, COUNT(*) AS samples,
               AVG(result_numeric) AS mean_value
        FROM lab_result
        WHERE result_numeric IS NOT NULL
        GROUP BY loinc_code
        """,
        """
        SELECT loinc_code, MIN(result_numeric) AS min_value,
               MAX(result_numeric) AS max_value
        FROM lab_result
        WHERE result_numeric IS NOT NULL
        GROUP BY loinc_code
        """,
        """
        SELECT loinc_code, COUNT(DISTINCT patient_id) AS patients
        FROM lab_result
        GROUP BY loinc_code
        HAVING COUNT(DISTINCT patient_id) > 10
        """,
        """
        SELECT DATE_TRUNC('year', result_time) AS year,
               COUNT(*) AS tests
        FROM lab_result
        WHERE result_time IS NOT NULL
        GROUP BY DATE_TRUNC('year', result_time)
        ORDER BY year
        """
    ],
    "procedure_epidemiology": [
        """
        SELECT procedure_id, COUNT(DISTINCT patient_id) AS patients
        FROM patient_procedure
        GROUP BY procedure_id
        ORDER BY patients DESC
        LIMIT 50
        """,
        """
        SELECT procedure_id, AVG(base_cost) AS avg_cost
        FROM patient_procedure
        GROUP BY procedure_id
        """,
        """
        SELECT DATE_TRUNC('year', procedure_date) AS year,
               COUNT(*) AS procedures
        FROM patient_procedure
        WHERE procedure_date IS NOT NULL
        GROUP BY DATE_TRUNC('year', procedure_date)
        ORDER BY year
        """
    ]
}

TEMPLATES = {
    "doctor": DOCTOR_TEMPLATES,
    "admin": ADMIN_TEMPLATES,
    "research": RESEARCH_TEMPLATES
}


def get_reference_values(conn):
    section("2. LOAD DATABASE REFERENCE VALUES")

    queries = {
        "patient": "SELECT patient_id FROM patient LIMIT 500",
        "condition": "SELECT condition_id FROM condition LIMIT 500",
        "loinc": "SELECT loinc_code FROM loinc_concept LIMIT 500"
    }

    values = {}

    for key, query in queries.items():
        with conn.cursor() as cur:
            cur.execute(query)
            values[key] = [
                row[0] for row in cur.fetchall()
                if row[0] is not None
            ]

        if not values[key]:
            raise RuntimeError(
                f"No reference values available for {key}"
            )

        print(key, len(values[key]))

    return values


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def instantiate(template, values):
    substitutions = {
        key: literal(RNG.choice(items))
        for key, items in values.items()
    }

    return clean(
        template.format(**substitutions)
    )


def validate_sql(conn, sql):
    try:
        # A read-only transaction prevents accidental modifications.
        # EXPLAIN without ANALYZE does not run the SELECT.
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute(
                    "SET TRANSACTION READ ONLY"
                )
                cur.execute(
                    "EXPLAIN " + sql
                )
                cur.fetchall()

        return True, ""

    except Exception as exc:
        return False, str(exc)


def generate_expanded(conn, values, original):
    section("3. GENERATE AND VALIDATE NEW TEMPLATES")

    original_hashes = set(original["query_hash"].astype(str))
    original_fps = set(original["fingerprint_hash"].astype(str))

    rows = []
    validation = []

    for role, families in TEMPLATES.items():
        for family, templates in families.items():
            for template_id, template in enumerate(
                templates, start=1
            ):
                sql = instantiate(template, values)
                qhash = sha(sql)
                fp = fingerprint(sql)

                valid, error = validate_sql(conn, sql)

                validation.append({
                    "role": role,
                    "family": family,
                    "template_id": template_id,
                    "query_hash": qhash,
                    "fingerprint_hash": fp,
                    "valid": valid,
                    "error": error
                })

                if not valid:
                    print(
                        "INVALID:",
                        role, family, template_id,
                        error[:150]
                    )
                    continue

                # New templates remain distinguishable from originals.
                if fp in original_fps:
                    print(
                        "EXISTING FINGERPRINT:",
                        role, family, template_id
                    )
                    continue

                # Generate multiple parameterized instances
                # without pretending they are new templates.
                for instance in range(8):
                    instance_sql = instantiate(template, values)
                    instance_hash = sha(instance_sql)

                    rows.append({
                        "workload_type": role,
                        "role_label": role,
                        "workload_context": "role_specific",
                        "source_workload": role,
                        "query_family": family,
                        "query_name": (
                            f"{family}_{template_id}"
                        ),
                        "complexity": "UNASSESSED",
                        "expected_use": family.replace("_", " "),
                        "query_hash": instance_hash,
                        "fingerprint_hash": fp,
                        "sql": instance_sql,
                        "source": "phase82_expansion",
                        "template_id": f"{role}_{family}_{template_id}",
                        "instance_id": instance
                    })

    pd.DataFrame(validation).to_csv(
        OUT / "new_query_validation.csv",
        index=False
    )

    new_df = pd.DataFrame(rows)

    if new_df.empty:
        raise RuntimeError(
            "No valid novel templates generated. "
            "Inspect new_query_validation.csv."
        )

    # Ensure that generated query hashes do not conflict
    # with original labels.
    original_role_by_hash = (
        original.groupby("query_hash")["role_label"]
        .first()
        .to_dict()
    )

    collision = new_df.apply(
        lambda r: (
            r["query_hash"] in original_role_by_hash
            and original_role_by_hash[r["query_hash"]]
            != r["role_label"]
        ),
        axis=1
    )

    if collision.any():
        raise RuntimeError(
            "Generated queries conflict with existing role labels."
        )

    print("New query instances:", len(new_df))
    print(
        "New unique fingerprints:",
        new_df.fingerprint_hash.nunique()
    )

    return new_df


def save_outputs(original, expanded):
    section("4. SAVE SEPARATE PHASE 8.2 DATASET")

    original = original.copy()
    original["source"] = "phase6_original"

    combined = pd.concat(
        [original, expanded],
        ignore_index=True,
        sort=False
    )

    combined.to_csv(
        OUT / "expanded_workload.csv",
        index=False
    )

    expanded.to_csv(
        OUT / "new_templates_only.csv",
        index=False
    )

    # Training-ready role labels, with duplicate exact SQL
    # instances retained for workload-frequency analysis.
    training = combined.copy()
    training["workload_type"] = training["role_label"]

    training.to_csv(
        OUT / "phase82_training.csv",
        index=False
    )

    summary = {
        "phase": "8.2",
        "created_at": datetime.now().isoformat(),
        "original_instances": len(original),
        "new_instances": len(expanded),
        "total_instances": len(combined),
        "original_fingerprints": int(
            original.fingerprint_hash.nunique()
        ),
        "new_fingerprints": int(
            expanded.fingerprint_hash.nunique()
        ),
        "total_fingerprints": int(
            combined.fingerprint_hash.nunique()
        ),
        "role_distribution": {
            str(k): int(v)
            for k, v in combined.role_label.value_counts().items()
        },
        "notes": [
            "Original Phase 6 and Phase 7 files are unchanged.",
            "Mixed workload is retained as context, not a SQL role.",
            "New SQL was validated with EXPLAIN, not executed.",
            "Generated examples have no measured execution latency.",
            "Original and synthetic instances must be reported separately.",
            "New fingerprints do not guarantee better unseen-template accuracy."
        ]
    }

    (OUT / "expansion_report.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    print("\nSaved:", OUT)


def main():
    section("PHASE 8.2 - WORKLOAD EXPANSION")

    original = load_original()

    if not DB_PASSWORD:
        password = getpass.getpass(
            f"PostgreSQL password for {DB_USER}: "
        )
    else:
        password = DB_PASSWORD

    with psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=password,
        autocommit=True
    ) as conn:
        values = get_reference_values(conn)
        expanded = generate_expanded(
            conn, values, original
        )

    save_outputs(original, expanded)

    section("PHASE 8.2 EXPANSION COMPLETE")


if __name__ == "__main__":
    main()
