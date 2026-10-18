
from pathlib import Path
from datetime import datetime
import argparse
import csv
import getpass
import json
import os
import re
import shutil
import subprocess
import sys
import time

import psycopg
from psycopg import sql


ROOT = Path(__file__).resolve().parent.parent
ORIGINAL_ETL = ROOT / "notebooks" / "02.py"
SOURCE_DATA = ROOT / "data"
SOURCE_SCHEMA = ROOT / "database" / "schema"

WORK_ROOT = ROOT / "phase5_scaling"
RESULTS = ROOT / "results" / "phase5_scaling"

RESULTS.mkdir(parents=True, exist_ok=True)

REQUIRED_CSV = [
    "patients.csv",
    "organizations.csv",
    "providers.csv",
    "encounters.csv",
    "conditions.csv",
    "procedures.csv",
    "medications.csv",
    "observations.csv",
    "allergies.csv",
    "immunizations.csv",
    "careplans.csv",
    "devices.csv",
    "imaging_studies.csv",
    "payers.csv",
    "payer_transitions.csv",
    "claims.csv",
    "claims_transactions.csv",
    "supplies.csv",
]

PATIENT_TABLES = [
    "patient",
    "encounter",
    "diagnosis",
    "prescription",
    "prescription_item",
    "lab_result",
]

SCALE_DATABASES = {
    10000: "adaptive_healthcare_scale_10k",
    50000: "adaptive_healthcare_scale_50k",
    100000: "adaptive_healthcare_scale_100k",
}


def heading(message):
    print("\n" + "=" * 75)
    print(message)
    print("=" * 75)


def run(command, cwd=None, env=None):
    print("\nCOMMAND:", subprocess.list2cmdline(
        [str(x) for x in command]
    ))

    subprocess.run(
        [str(x) for x in command],
        cwd=cwd,
        env=env,
        check=True
    )


def count_csv(path):
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        return max(
            sum(1 for _ in csv.reader(file)) - 1,
            0
        )


def validate_csv(csv_dir):
    heading("PHASE 5A — CSV VALIDATION")

    if not csv_dir.is_dir():
        raise FileNotFoundError(csv_dir)

    missing = [
        name for name in REQUIRED_CSV
        if not (csv_dir / name).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing Synthea files:\n" +
            "\n".join(missing)
        )

    counts = {}

    for name in REQUIRED_CSV:
        path = csv_dir / name
        count = count_csv(path)
        counts[name] = count
        print(f"{name:<30} {count:>12,}")

    if counts["patients.csv"] == 0:
        raise ValueError("No patients found.")

    return counts


def generate_synthea(jar, patients, destination):
    heading("PHASE 5B — GENERATE SYNTHEA DATA")

    if not jar.is_file():
        raise FileNotFoundError(jar)

    destination.mkdir(
        parents=True,
        exist_ok=True
    )

    config = destination / "synthea.properties"

    output_dir = destination / "generated"

    config.write_text(
        "exporter.csv.export=true\n"
        "exporter.fhir.export=false\n"
        "exporter.ccda.export=false\n"
        f"exporter.baseDirectory="
        f"{output_dir.as_posix()}\n",
        encoding="utf-8"
    )

    run([
        "java",
        "-jar",
        str(jar),
        "-p",
        str(patients),
        "-c",
        str(config),
    ])

    csv_dir = output_dir / "csv"

    if not csv_dir.is_dir():
        raise RuntimeError(
            "Synthea did not produce the expected "
            f"CSV directory: {csv_dir}. "
            "Check the generator output."
        )

    return csv_dir


def copy_or_link(source, destination):
    if not source.exists():
        raise FileNotFoundError(source)

    if destination.exists():
        return

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if os.name == "nt" and source.is_dir():
        try:
            subprocess.run(
                [
                    "cmd", "/c", "mklink", "/J",
                    str(destination),
                    str(source),
                ],
                check=True,
                capture_output=True,
                text=True
            )
            return
        except subprocess.CalledProcessError:
            pass

    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)


def prepare_workspace(csv_dir, patients):
    heading("PHASE 5C — PREPARE ISOLATED WORKSPACE")

    workspace = WORK_ROOT / f"scale_{patients}"

    data_dir = workspace / "data"
    notebooks_dir = workspace / "notebooks"
    schema_dir = workspace / "database" / "schema"

    notebooks_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    schema_dir.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if not ORIGINAL_ETL.is_file():
        raise FileNotFoundError(
            f"Original ETL missing: {ORIGINAL_ETL}"
        )

    copy_or_link(
        SOURCE_DATA / "loinc",
        data_dir / "loinc"
    )

    copy_or_link(
        SOURCE_DATA / "rxnorm",
        data_dir / "rxnorm"
    )

    copy_or_link(
        SOURCE_SCHEMA,
        schema_dir
    )

    destination_csv = data_dir / "synthea"

    if destination_csv.exists():
        if destination_csv.is_symlink():
            raise RuntimeError(
                "Refusing to replace a linked CSV directory."
            )

        shutil.rmtree(destination_csv)

    shutil.copytree(csv_dir, destination_csv)

    print("Workspace:", workspace)
    print("Staged CSV:", destination_csv)

    return workspace


def prepare_etl(workspace, database_name):
    heading("PHASE 5D — PREPARE EXISTING ETL")

    original_code = ORIGINAL_ETL.read_text(
        encoding="utf-8-sig"
    )

    database_pattern = (
        r'(?m)^DB_NAME\s*=\s*["\'][^"\']+["\']'
    )

    reset_pattern = (
        r"(?m)^RESET_SCHEMA\s*=\s*(True|False)"
    )

    code, db_changes = re.subn(
        database_pattern,
        f'DB_NAME = "{database_name}"',
        original_code,
        count=1
    )

    code, reset_changes = re.subn(
        reset_pattern,
        "RESET_SCHEMA = True",
        code,
        count=1
    )

    if db_changes != 1 or reset_changes != 1:
        raise RuntimeError(
            "Cannot safely patch the original ETL. "
            "Check DB_NAME and RESET_SCHEMA in 02.py."
        )

    if database_name == "adaptive_healthcare_db":
        raise RuntimeError(
            "Original database is protected."
        )

    destination = workspace / "notebooks" / "02.py"

    destination.write_text(
        code,
        encoding="utf-8"
    )

    print("Original ETL:", ORIGINAL_ETL)
    print("Isolated ETL:", destination)
    print("Target database:", database_name)
    print("Original 02.py remains unchanged.")

    return destination


def connect_postgres(password, database="postgres"):
    return psycopg.connect(
        dbname=database,
        host=os.getenv("PGHOST", "localhost"),
        port=int(os.getenv("PGPORT", "5432")),
        user=os.getenv("PGUSER", "postgres"),
        password=password,
    )


def create_database(database_name, password):
    heading("PHASE 5E — CREATE TEST DATABASE")

    if database_name == "adaptive_healthcare_db":
        raise RuntimeError(
            "Refusing to modify original database."
        )

    conn = connect_postgres(password)
    conn.autocommit = True

    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM pg_database "
                "WHERE datname = %s",
                (database_name,)
            )

            if cur.fetchone():
                raise RuntimeError(
                    f"Database {database_name} already "
                    "exists. It will not be overwritten. "
                    "Use a fresh test database name."
                )

            cur.execute(
                sql.SQL("CREATE DATABASE {}").format(
                    sql.Identifier(database_name)
                )
            )
    finally:
        conn.close()

    print("Created:", database_name)


def execute_etl(etl_path, password):
    heading("PHASE 5F — EXECUTE POSTGRESQL ETL")

    environment = os.environ.copy()
    environment["PGPASSWORD"] = password

    start = time.perf_counter()

    run(
        [sys.executable, str(etl_path)],
        cwd=etl_path.parent,
        env=environment
    )

    elapsed = time.perf_counter() - start

    print(f"ETL time: {elapsed:.2f} seconds")

    return elapsed


def validate_database(database_name, password):
    heading("PHASE 5G — DATABASE VALIDATION")

    conn = connect_postgres(
        password,
        database_name
    )

    counts = {}

    try:
        with conn.cursor() as cur:
            for table in PATIENT_TABLES:
                cur.execute(
                    """
                    SELECT EXISTS (
                        SELECT 1
                        FROM information_schema.tables
                        WHERE table_schema = 'public'
                        AND table_name = %s
                    )
                    """,
                    (table,)
                )

                if not cur.fetchone()[0]:
                    counts[table] = None
                    print(f"{table:<25} MISSING")
                    continue

                cur.execute(
                    sql.SQL("SELECT COUNT(*) FROM {}").format(
                        sql.Identifier(table)
                    )
                )

                count = cur.fetchone()[0]
                counts[table] = count

                print(f"{table:<25} {count:>12,}")

            cur.execute(
                """
                SELECT COUNT(*)
                FROM (
                    SELECT patient_id
                    FROM patient
                    GROUP BY patient_id
                    HAVING COUNT(*) > 1
                ) AS duplicates
                """
            )

            counts["duplicate_patient_ids"] = (
                cur.fetchone()[0]
            )

            cur.execute(
                """
                SELECT COUNT(*)
                FROM diagnosis d
                LEFT JOIN patient p
                  ON d.patient_id = p.patient_id
                WHERE p.patient_id IS NULL
                """
            )

            counts["orphan_diagnoses"] = (
                cur.fetchone()[0]
            )

    finally:
        conn.close()

    print(
        "Duplicate patient IDs:",
        counts["duplicate_patient_ids"]
    )

    print(
        "Orphan diagnoses:",
        counts["orphan_diagnoses"]
    )

    counts["integrity_pass"] = (
        counts["duplicate_patient_ids"] == 0
        and counts["orphan_diagnoses"] == 0
        and counts["patient"] is not None
        and counts["patient"] > 0
    )

    return counts


def export_report(patients, database_name, csv_counts,
                  db_counts, etl_seconds):
    heading("PHASE 5H — EXPORT RESULTS")

    report = {
        "timestamp": datetime.now().isoformat(),
        "requested_population": patients,
        "database": database_name,
        "source_csv_counts": csv_counts,
        "database_counts": db_counts,
        "etl_seconds": etl_seconds,
        "integrity_pass": db_counts["integrity_pass"],
    }

    destination = (
        RESULTS / f"phase5_{patients}_report.json"
    )

    destination.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    print("Report:", destination)

    return report


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--patients",
        type=int,
        choices=[10000, 50000, 100000],
        required=True
    )

    source = parser.add_mutually_exclusive_group(
        required=True
    )

    source.add_argument(
        "--synthea-jar",
        type=Path
    )

    source.add_argument(
        "--csv-dir",
        type=Path
    )

    parser.add_argument(
        "--prepare-only",
        action="store_true"
    )

    args = parser.parse_args()

    patients = args.patients
    database_name = SCALE_DATABASES[patients]

    heading("PHASE 5 — DATASET SCALING")

    print("Target patients:", patients)
    print("Target database:", database_name)
    print("Original database: PROTECTED")
    print("Neo4j graph: NOT MODIFIED")

    if args.synthea_jar:
        generation_dir = (
            WORK_ROOT / f"generation_{patients}"
        )

        csv_dir = generate_synthea(
            args.synthea_jar.resolve(),
            patients,
            generation_dir
        )
    else:
        csv_dir = args.csv_dir.resolve()

    csv_counts = validate_csv(csv_dir)

    actual_patients = csv_counts["patients.csv"]

    print(
        "\nGenerated patient rows:",
        actual_patients
    )

    if actual_patients < patients:
        print(
            "WARNING: Generated fewer patient rows "
            "than requested. Synthea's output may "
            "depend on its generation settings."
        )

    workspace = prepare_workspace(
        csv_dir,
        patients
    )

    etl_path = prepare_etl(
        workspace,
        database_name
    )

    if args.prepare_only:
        print(
            "\nPREPARATION COMPLETE."
            "\nNo PostgreSQL database was created."
        )
        return

    password = (
        os.getenv("PGPASSWORD")
        or getpass.getpass(
            "PostgreSQL password: "
        )
    )

    create_database(
        database_name,
        password
    )

    etl_seconds = execute_etl(
        etl_path,
        password
    )

    db_counts = validate_database(
        database_name,
        password
    )

    report = export_report(
        patients,
        database_name,
        csv_counts,
        db_counts,
        etl_seconds
    )

    heading("PHASE 5 FINAL STATUS")

    print(
        "Database:",
        database_name
    )

    print(
        "Patients:",
        db_counts["patient"]
    )

    print(
        "ETL seconds:",
        round(etl_seconds, 2)
    )

    print(
        "Integrity:",
        "PASS" if report["integrity_pass"]
        else "REVIEW REQUIRED"
    )

    print(
        "Original PostgreSQL database: UNCHANGED"
    )

    print(
        "Original Neo4j graph: UNCHANGED"
    )


if __name__ == "__main__":
    main()
