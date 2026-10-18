
import argparse
import csv
import getpass
import json
import logging
import statistics
import sys
import time
from collections import Counter
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from neo4j import GraphDatabase, READ_ACCESS


ROOT = Path(__file__).resolve().parent.parent
EXPECTED_DATABASE = "adaptive_healthcare_db"


def serializable(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
            default=serializable,
        )


def save_csv(path, rows):
    if not rows:
        return

    columns = sorted({
        key
        for row in rows
        for key in row.keys()
    })

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for row in rows:
            writer.writerow({
                key: (
                    json.dumps(value, default=serializable)
                    if isinstance(value, (list, dict, set))
                    else value
                )
                for key, value in row.items()
            })


def percentile(values, p):
    if not values:
        return None

    ordered = sorted(values)
    index = (len(ordered) - 1) * p / 100
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower

    return (
        ordered[lower] * (1 - weight)
        + ordered[upper] * weight
    )


class HybridEngine:
    def __init__(self, pg, neo4j_session, output, limit):
        self.pg = pg
        self.neo4j = neo4j_session
        self.output = output
        self.limit = limit
        self.timings = []
        self.bridge_verified = False

    def sql(self, label, query, params=None):
        started = time.perf_counter()

        with self.pg.cursor(row_factory=dict_row) as cur:
            cur.execute(query, params or ())
            rows = [dict(row) for row in cur.fetchall()]

        elapsed = (time.perf_counter() - started) * 1000

        self.timings.append({
            "engine": "PostgreSQL",
            "label": label,
            "latency_ms": round(elapsed, 3),
            "rows": len(rows),
        })

        logging.info(
            "SQL %-32s rows=%d latency=%.3f ms",
            label,
            len(rows),
            elapsed,
        )

        return rows

    def cypher(self, label, query, params=None):
        started = time.perf_counter()

        result = self.neo4j.run(query, params or {})
        rows = [dict(row) for row in result]

        elapsed = (time.perf_counter() - started) * 1000

        self.timings.append({
            "engine": "Neo4j",
            "label": label,
            "latency_ms": round(elapsed, 3),
            "rows": len(rows),
        })

        logging.info(
            "CYPHER %-29s rows=%d latency=%.3f ms",
            label,
            len(rows),
            elapsed,
        )

        return rows

    def verify_bridge(self):
        logging.info("Verifying original PostgreSQL/Neo4j bridge")

        pg_db = self.sql(
            "database_identity",
            "SELECT current_database() AS database_name",
        )[0]["database_name"]

        if pg_db != EXPECTED_DATABASE:
            raise RuntimeError(
                f"Unsafe database: {pg_db}. "
                f"Phase 11 requires {EXPECTED_DATABASE}."
            )

        pg_ids = {
            str(row["patient_id"])
            for row in self.sql(
                "postgres_patient_ids",
                """
                SELECT patient_id::text AS patient_id
                FROM public.patient
                """,
            )
        }

        graph_rows = self.cypher(
            "neo4j_patient_ids",
            """
            MATCH (p:Patient)
            RETURN p.patient_id AS patient_id
            """,
        )

        graph_ids = [
            str(row["patient_id"])
            for row in graph_rows
        ]

        graph_set = set(graph_ids)

        result = {
            "postgres_database": pg_db,
            "postgres_unique_patients": len(pg_ids),
            "graph_patient_nodes": len(graph_ids),
            "graph_unique_patients": len(graph_set),
            "missing_from_graph": len(pg_ids - graph_set),
            "missing_from_postgres": len(graph_set - pg_ids),
            "graph_duplicate_ids": (
                len(graph_ids) - len(graph_set)
            ),
        }

        result["passed"] = (
            pg_ids == graph_set
            and len(graph_ids) == len(graph_set)
            and len(pg_ids) == 1163
        )

        save_json(
            self.output / "bridge_validation.json",
            result,
        )

        if not result["passed"]:
            raise RuntimeError(
                "Patient bridge validation failed. "
                "Hybrid execution has been blocked."
            )

        self.bridge_verified = True
        logging.info("Patient bridge verified: 1163/1163")

        return result

    def require_bridge(self):
        if not self.bridge_verified:
            raise RuntimeError(
                "Patient bridge must be verified first."
            )

    def sample_patient_ids(self, count):
        self.require_bridge()

        rows = self.sql(
            "sample_patients",
            """
            SELECT patient_id::text AS patient_id
            FROM public.patient
            ORDER BY patient_id
            LIMIT %s
            """,
            (count,),
        )

        return [row["patient_id"] for row in rows]

    def patient_profile(self, patient_id):
        self.require_bridge()

        started = time.perf_counter()

        patient = self.sql(
            "patient_demographics",
            """
            SELECT *
            FROM public.patient
            WHERE patient_id::text = %s
            """,
            (patient_id,),
        )

        if not patient:
            raise ValueError(
                f"Patient not found in original PostgreSQL: "
                f"{patient_id}"
            )

        diagnoses = self.sql(
            "patient_diagnoses",
            """
            SELECT DISTINCT
                c.source_code::text AS disease_code,
                c.description::text AS disease_name
            FROM public.diagnosis d
            JOIN public.condition c
              ON c.condition_id = d.condition_id
            WHERE d.patient_id::text = %s
            ORDER BY disease_code
            LIMIT %s
            """,
            (patient_id, self.limit),
        )

        labs = self.sql(
            "patient_labs",
            """
            SELECT
                lr.loinc_code::text AS loinc_code,
                COUNT(*)::bigint AS observation_count
            FROM public.lab_result lr
            WHERE lr.patient_id::text = %s
              AND lr.loinc_code IS NOT NULL
            GROUP BY lr.loinc_code
            ORDER BY observation_count DESC, loinc_code
            LIMIT %s
            """,
            (patient_id, self.limit),
        )

        medications = self.sql(
            "patient_medications",
            """
            SELECT
                pi.rxcui::text AS rxcui,
                COUNT(*)::bigint AS prescription_item_count
            FROM public.prescription p
            JOIN public.prescription_item pi
              ON p.prescription_id = pi.prescription_id
            WHERE p.patient_id::text = %s
              AND pi.rxcui IS NOT NULL
            GROUP BY pi.rxcui
            ORDER BY prescription_item_count DESC, rxcui
            LIMIT %s
            """,
            (patient_id, self.limit),
        )

        graph_patient = self.cypher(
            "graph_patient_profile",
            """
            MATCH (p:Patient {patient_id: $patient_id})
            RETURN
                p.patient_id AS patient_id,
                properties(p) AS graph_properties
            """,
            {"patient_id": patient_id},
        )

        graph_diseases = self.cypher(
            "graph_patient_conditions",
            """
            MATCH (p:Patient {patient_id: $patient_id})
                  -[:HAS_CONDITION]->(d:Disease)
            RETURN DISTINCT
                d.source_code AS disease_code,
                d.name AS graph_disease_name
            ORDER BY disease_code
            LIMIT $limit
            """,
            {
                "patient_id": patient_id,
                "limit": self.limit,
            },
        )

        graph_labs = self.cypher(
            "graph_patient_labs",
            """
            MATCH (p:Patient {patient_id: $patient_id})
                  -[:HAS_LAB]->(l:LabTest)
            RETURN DISTINCT
                l.loinc_code AS loinc_code,
                l.name AS graph_lab_name
            ORDER BY loinc_code
            LIMIT $limit
            """,
            {
                "patient_id": patient_id,
                "limit": self.limit,
            },
        )

        graph_drugs = self.cypher(
            "graph_patient_medications",
            """
            MATCH (p:Patient {patient_id: $patient_id})
                  -[:HAS_MEDICATION]->(d:Drug)
            RETURN DISTINCT
                d.rxcui AS rxcui,
                d.name AS graph_drug_name
            ORDER BY rxcui
            LIMIT $limit
            """,
            {
                "patient_id": patient_id,
                "limit": self.limit,
            },
        )

        def comparison(sql_rows, graph_rows, sql_key, graph_key):
            sql_codes = {
                str(r[sql_key])
                for r in sql_rows
                if r[sql_key] is not None
            }

            graph_codes = {
                str(r[graph_key])
                for r in graph_rows
                if r[graph_key] is not None
            }

            return {
                "sql_returned_codes": len(sql_codes),
                "graph_returned_codes": len(graph_codes),
                "shared_codes": len(sql_codes & graph_codes),
                "sql_only_codes": sorted(sql_codes - graph_codes),
                "graph_only_codes": sorted(graph_codes - sql_codes),
                "equal_within_returned_limits": (
                    sql_codes == graph_codes
                ),
                "comparison_is_truncated_if_limit_reached": (
                    len(sql_rows) >= self.limit
                    or len(graph_rows) >= self.limit
                ),
            }

        result = {
            "query_type": "patient_profile",
            "patient_id": patient_id,
            "postgres_patient": patient[0],
            "neo4j_patient": graph_patient,
            "postgres_diagnoses": diagnoses,
            "neo4j_diagnoses": graph_diseases,
            "postgres_labs": labs,
            "neo4j_labs": graph_labs,
            "postgres_medications": medications,
            "neo4j_medications": graph_drugs,
            "comparisons": {
                "conditions": comparison(
                    diagnoses,
                    graph_diseases,
                    "disease_code",
                    "disease_code",
                ),
                "labs": comparison(
                    labs,
                    graph_labs,
                    "loinc_code",
                    "loinc_code",
                ),
                "medications": comparison(
                    medications,
                    graph_drugs,
                    "rxcui",
                    "rxcui",
                ),
            },
            "total_elapsed_ms": round(
                (time.perf_counter() - started) * 1000,
                3,
            ),
        }

        return result

    def disease_labs(self, disease_code):
        self.require_bridge()

        started = time.perf_counter()

        sql_disease = self.sql(
            "sql_disease_lookup",
            """
            SELECT
                source_code::text AS disease_code,
                description::text AS disease_name
            FROM public.condition
            WHERE source_code::text = %s
            """,
            (disease_code,),
        )

        graph_disease = self.cypher(
            "graph_disease_lookup",
            """
            MATCH (d:Disease {source_code: $code})
            RETURN
                d.source_code AS disease_code,
                d.name AS graph_name
            """,
            {"code": disease_code},
        )

        graph_labs = self.cypher(
            "disease_associated_labs",
            """
            MATCH (d:Disease {source_code: $code})
                  -[:ASSOCIATED_WITH]->(l:LabTest)
            RETURN DISTINCT
                l.loinc_code AS loinc_code,
                l.name AS lab_name
            ORDER BY loinc_code
            LIMIT $limit
            """,
            {
                "code": disease_code,
                "limit": self.limit,
            },
        )

        codes = sorted({
            str(row["loinc_code"])
            for row in graph_labs
            if row["loinc_code"] is not None
        })

        sql_lab_reference = []

        if codes:
            sql_lab_reference = self.sql(
                "sql_lab_reference",
                """
                SELECT
                    loinc_code::text AS loinc_code,
                    long_common_name::text AS lab_name
                FROM public.loinc_concept
                WHERE loinc_code::text = ANY(%s)
                """,
                (codes,),
            )

        return {
            "query_type": "disease_labs",
            "disease_code": disease_code,
            "postgres_disease": sql_disease,
            "graph_disease": graph_disease,
            "graph_associated_labs": graph_labs,
            "postgres_lab_reference": sql_lab_reference,
            "note": (
                "Graph associations are knowledge links, "
                "not proof that a laboratory test is clinically "
                "indicated for a particular patient."
            ),
            "total_elapsed_ms": round(
                (time.perf_counter() - started) * 1000,
                3,
            ),
        }

    def drug_mesh(self, rxcui):
        self.require_bridge()

        started = time.perf_counter()

        sql_drug = self.sql(
            "sql_drug_lookup",
            """
            SELECT
                rxcui::text AS rxcui,
                name::text AS name
            FROM public.drug_concept
            WHERE rxcui::text = %s
            """,
            (rxcui,),
        )

        graph_drug = self.cypher(
            "graph_drug_lookup",
            """
            MATCH (d:Drug {rxcui: $rxcui})
            RETURN
                d.rxcui AS rxcui,
                d.name AS name,
                d.source AS source,
                d.concept_type AS concept_type
            """,
            {"rxcui": rxcui},
        )

        mesh = self.cypher(
            "drug_mesh_knowledge",
            """
            MATCH (d:Drug {rxcui: $rxcui})-[r]->(m:MeSHConcept)
            WHERE type(r) IN ['MAPPED_TO', 'ABOUT_MESH']
            RETURN DISTINCT
                type(r) AS relationship,
                properties(m) AS mesh_properties
            LIMIT $limit
            """,
            {
                "rxcui": rxcui,
                "limit": self.limit,
            },
        )

        related_drugs = self.cypher(
            "related_drug_knowledge",
            """
            MATCH (d:Drug {rxcui: $rxcui})
                  -[:RELATED_TO]-(other:Drug)
            WHERE other.rxcui <> d.rxcui
            RETURN DISTINCT
                other.rxcui AS related_rxcui,
                other.name AS related_name
            ORDER BY related_rxcui
            LIMIT $limit
            """,
            {
                "rxcui": rxcui,
                "limit": self.limit,
            },
        )

        return {
            "query_type": "drug_mesh",
            "rxcui": rxcui,
            "postgres_drug": sql_drug,
            "graph_drug": graph_drug,
            "mesh_knowledge": mesh,
            "related_drugs": related_drugs,
            "note": (
                "RELATED_TO and MeSH links represent graph "
                "knowledge; they do not establish therapeutic "
                "interchangeability or clinical suitability."
            ),
            "total_elapsed_ms": round(
                (time.perf_counter() - started) * 1000,
                3,
            ),
        }

    def disease_drugs(self, disease_code):
        self.require_bridge()

        started = time.perf_counter()

        sql_disease = self.sql(
            "sql_disease_drug_lookup",
            """
            SELECT
                source_code::text AS disease_code,
                description::text AS disease_name
            FROM public.condition
            WHERE source_code::text = %s
            """,
            (disease_code,),
        )

        graph_drugs = self.cypher(
            "disease_observed_drugs",
            """
            MATCH (d:Disease {source_code: $code})
                  -[:OBSERVED_WITH_MEDICATION]->(drug:Drug)
            RETURN DISTINCT
                drug.rxcui AS rxcui,
                drug.name AS drug_name
            ORDER BY rxcui
            LIMIT $limit
            """,
            {
                "code": disease_code,
                "limit": self.limit,
            },
        )

        codes = sorted({
            str(row["rxcui"])
            for row in graph_drugs
            if row["rxcui"] is not None
        })

        sql_drugs = []

        if codes:
            sql_drugs = self.sql(
                "sql_observed_drug_reference",
                """
                SELECT
                    rxcui::text AS rxcui,
                    name::text AS name
                FROM public.drug_concept
                WHERE rxcui::text = ANY(%s)
                """,
                (codes,),
            )

        return {
            "query_type": "disease_drugs",
            "disease_code": disease_code,
            "postgres_disease": sql_disease,
            "graph_observed_drugs": graph_drugs,
            "postgres_drug_reference": sql_drugs,
            "note": (
                "OBSERVED_WITH_MEDICATION is observational "
                "and must not be interpreted as a treatment "
                "recommendation, causal relationship, or "
                "evidence of effectiveness."
            ),
            "total_elapsed_ms": round(
                (time.perf_counter() - started) * 1000,
                3,
            ),
        }

    def get_sample_disease(self):
        rows = self.sql(
            "sample_disease",
            """
            SELECT c.source_code::text AS disease_code
            FROM public.condition c
            WHERE c.source_code IS NOT NULL
              AND EXISTS (
                  SELECT 1
                  FROM public.diagnosis d
                  WHERE d.condition_id = c.condition_id
              )
            ORDER BY c.source_code
            LIMIT 1
            """,
        )

        if not rows:
            raise RuntimeError(
                "No disease code available for sampling."
            )

        return rows[0]["disease_code"]

    def get_sample_drug(self):
        rows = self.sql(
            "sample_drug",
            """
            SELECT dc.rxcui::text AS rxcui
            FROM public.drug_concept dc
            WHERE EXISTS (
                SELECT 1
                FROM public.prescription_item pi
                WHERE pi.rxcui = dc.rxcui
            )
            ORDER BY dc.rxcui
            LIMIT 1
            """,
        )

        if not rows:
            raise RuntimeError(
                "No prescribed drug available for sampling."
            )

        return rows[0]["rxcui"]

    def run_benchmark(self, patient_count, repeats):
        self.require_bridge()

        patient_ids = self.sample_patient_ids(patient_count)
        disease_code = self.get_sample_disease()
        rxcui = self.get_sample_drug()

        workloads = []

        for patient_id in patient_ids:
            workloads.append((
                "patient_profile",
                patient_id,
            ))

        workloads.extend([
            ("disease_labs", disease_code),
            ("drug_mesh", rxcui),
            ("disease_drugs", disease_code),
        ])

        results = []

        for repeat in range(1, repeats + 1):
            for query_type, identifier in workloads:
                started = time.perf_counter()

                try:
                    if query_type == "patient_profile":
                        result = self.patient_profile(identifier)
                    elif query_type == "disease_labs":
                        result = self.disease_labs(identifier)
                    elif query_type == "drug_mesh":
                        result = self.drug_mesh(identifier)
                    else:
                        result = self.disease_drugs(identifier)

                    status = "SUCCESS"
                    error = ""
                    result_size = len(
                        json.dumps(
                            result,
                            default=serializable,
                        )
                    )

                except Exception as exc:
                    status = "ERROR"
                    error = str(exc)
                    result_size = 0

                elapsed = (
                    time.perf_counter() - started
                ) * 1000

                row = {
                    "repeat": repeat,
                    "query_type": query_type,
                    "identifier": identifier,
                    "status": status,
                    "latency_ms": round(elapsed, 3),
                    "result_json_characters": result_size,
                    "error": error,
                }

                results.append(row)

                logging.info(
                    "BENCHMARK repeat=%d type=%s status=%s "
                    "latency=%.3f ms",
                    repeat,
                    query_type,
                    status,
                    elapsed,
                )

        save_csv(
            self.output / "benchmark_runs.csv",
            results,
        )

        grouped = {}

        for query_type in sorted({
            row["query_type"]
            for row in results
        }):
            subset = [
                row for row in results
                if row["query_type"] == query_type
            ]

            successful = [
                row["latency_ms"]
                for row in subset
                if row["status"] == "SUCCESS"
            ]

            grouped[query_type] = {
                "runs": len(subset),
                "successes": len(successful),
                "errors": len(subset) - len(successful),
                "mean_ms": (
                    round(statistics.mean(successful), 3)
                    if successful else None
                ),
                "median_ms": (
                    round(statistics.median(successful), 3)
                    if successful else None
                ),
                "p95_ms": (
                    round(percentile(successful, 95), 3)
                    if successful else None
                ),
                "max_ms": (
                    round(max(successful), 3)
                    if successful else None
                ),
            }

        return {
            "query_type": "benchmark",
            "patient_count": len(patient_ids),
            "repeats": repeats,
            "total_runs": len(results),
            "successful_runs": sum(
                row["status"] == "SUCCESS"
                for row in results
            ),
            "failed_runs": sum(
                row["status"] != "SUCCESS"
                for row in results
            ),
            "workload_summary": grouped,
            "limitations": [
                "Benchmarks are sequential, not concurrent.",
                "No SQL-only versus hybrid control is included.",
                "The first run may include cache warm-up.",
                "Sampled patients are ordered, not random.",
                "The returned results are bounded by --limit.",
            ],
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Phase 11: Read-only PostgreSQL + Neo4j "
            "hybrid clinical query execution"
        )
    )

    parser.add_argument(
        "--pg-original",
        default=EXPECTED_DATABASE,
    )
    parser.add_argument("--pg-host", default="127.0.0.1")
    parser.add_argument("--pg-port", type=int, default=5432)
    parser.add_argument("--pg-user", default="postgres")

    parser.add_argument(
        "--neo4j-uri",
        default="bolt://127.0.0.1:7687",
    )
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-database", default="neo4j")

    parser.add_argument(
        "--ask-pg-password",
        action="store_true",
    )
    parser.add_argument(
        "--ask-neo4j-password",
        action="store_true",
    )

    parser.add_argument(
        "--mode",
        choices=[
            "all",
            "patient",
            "disease-labs",
            "drug-mesh",
            "disease-drugs",
            "benchmark",
        ],
        default="all",
    )

    parser.add_argument("--patient-id")
    parser.add_argument("--disease-code")
    parser.add_argument("--rxcui")

    parser.add_argument(
        "--limit",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--benchmark-patients",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.pg_original != EXPECTED_DATABASE:
        print(
            "ERROR: Phase 11 only supports "
            f"{EXPECTED_DATABASE}. "
            "Scaled populations have not been graph-verified."
        )
        return 1

    if not 1 <= args.limit <= 1000:
        print("ERROR: --limit must be between 1 and 1000")
        return 1

    if not 1 <= args.benchmark_patients <= 100:
        print(
            "ERROR: --benchmark-patients "
            "must be between 1 and 100"
        )
        return 1

    if not 1 <= args.repeats <= 20:
        print("ERROR: --repeats must be between 1 and 20")
        return 1

    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    output = (
        ROOT
        / "results"
        / "phase11_hybrid_execution"
        / stamp
    )

    output.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s %(levelname)s: %(message)s"
        ),
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                output / "execution.log",
                encoding="utf-8",
            ),
        ],
    )

    pg_password = (
        getpass.getpass("PostgreSQL password: ")
        if args.ask_pg_password
        else None
    )

    neo_password = (
        getpass.getpass("Neo4j password: ")
        if args.ask_neo4j_password
        else None
    )

    summary = {
        "phase": "11",
        "mode": args.mode,
        "status": "STARTED",
        "postgres_database": args.pg_original,
        "neo4j_database": args.neo4j_database,
        "database_modifications": "NONE",
        "errors": [],
        "completed_queries": [],
    }

    try:
        with psycopg.connect(
            host=args.pg_host,
            port=args.pg_port,
            dbname=args.pg_original,
            user=args.pg_user,
            password=pg_password,
            options=(
                "-c default_transaction_read_only=on "
                "-c statement_timeout=120000"
            ),
        ) as pg:

            with GraphDatabase.driver(
                args.neo4j_uri,
                auth=(args.neo4j_user, neo_password),
            ) as driver:

                driver.verify_connectivity()

                with driver.session(
                    database=args.neo4j_database,
                    default_access_mode=READ_ACCESS,
                ) as neo:

                    engine = HybridEngine(
                        pg,
                        neo,
                        output,
                        args.limit,
                    )

                    summary["bridge"] = (
                        engine.verify_bridge()
                    )

                    if args.mode in ("all", "patient"):
                        patient_id = args.patient_id

                        if not patient_id:
                            patient_id = (
                                engine.sample_patient_ids(1)[0]
                            )

                        result = engine.patient_profile(
                            patient_id
                        )

                        save_json(
                            output / "patient_profile.json",
                            result,
                        )

                        summary["completed_queries"].append(
                            "patient_profile"
                        )

                        summary["patient_id"] = patient_id
                        summary["patient_comparisons"] = (
                            result["comparisons"]
                        )

                    if args.mode in (
                        "all",
                        "disease-labs",
                    ):
                        disease_code = (
                            args.disease_code
                            or engine.get_sample_disease()
                        )

                        result = engine.disease_labs(
                            disease_code
                        )

                        save_json(
                            output / "disease_labs.json",
                            result,
                        )

                        summary["completed_queries"].append(
                            "disease_labs"
                        )
                        summary["disease_code"] = disease_code

                    if args.mode in (
                        "all",
                        "drug-mesh",
                    ):
                        rxcui = (
                            args.rxcui
                            or engine.get_sample_drug()
                        )

                        result = engine.drug_mesh(rxcui)

                        save_json(
                            output / "drug_mesh.json",
                            result,
                        )

                        summary["completed_queries"].append(
                            "drug_mesh"
                        )
                        summary["rxcui"] = rxcui

                    if args.mode in (
                        "all",
                        "disease-drugs",
                    ):
                        disease_code = (
                            args.disease_code
                            or engine.get_sample_disease()
                        )

                        result = engine.disease_drugs(
                            disease_code
                        )

                        save_json(
                            output / "disease_drugs.json",
                            result,
                        )

                        summary["completed_queries"].append(
                            "disease_drugs"
                        )

                    if args.mode in (
                        "all",
                        "benchmark",
                    ):
                        benchmark = engine.run_benchmark(
                            args.benchmark_patients,
                            args.repeats,
                        )

                        save_json(
                            output / "benchmark_summary.json",
                            benchmark,
                        )

                        summary["benchmark"] = benchmark
                        summary["completed_queries"].append(
                            "benchmark"
                        )

                    save_csv(
                        output / "engine_timings.csv",
                        engine.timings,
                    )

                    summary["engine_calls"] = len(
                        engine.timings
                    )

                    summary["status"] = (
                        "COMPLETED_WITH_BENCHMARK_ERRORS"
                        if (
                            "benchmark" in summary
                            and summary["benchmark"]["failed_runs"] > 0
                        )
                        else "PASS"
                    )

    except Exception as exc:
        summary["status"] = "ERRORS_PRESENT"
        summary["errors"].append(
            f"{type(exc).__name__}: {exc}"
        )
        logging.exception("Phase 11 failed")

    finally:
        summary["output_directory"] = str(output)
        summary["completed_at"] = (
            datetime.now().isoformat()
        )

        save_json(
            output / "summary.json",
            summary,
        )

        print("\n" + "=" * 72)
        print("PHASE 11 - HYBRID SQL-CYPHER EXECUTION")
        print("=" * 72)

        print("Status:", summary["status"])
        print("Mode:", summary["mode"])

        if "bridge" in summary:
            print(
                "Patient bridge:",
                summary["bridge"]["passed"],
            )

        print(
            "Completed queries:",
            summary["completed_queries"],
        )

        if "patient_comparisons" in summary:
            print("\nPatient clinical-code comparisons:")

            for key, value in (
                summary["patient_comparisons"].items()
            ):
                print(
                    f"  {key}: "
                    f"{value['shared_codes']} shared, "
                    f"{len(value['sql_only_codes'])} SQL-only, "
                    f"{len(value['graph_only_codes'])} graph-only"
                )

        if "benchmark" in summary:
            benchmark = summary["benchmark"]

            print("\nBenchmark:")
            print(
                "  Total runs:",
                benchmark["total_runs"],
            )
            print(
                "  Successful:",
                benchmark["successful_runs"],
            )
            print(
                "  Failed:",
                benchmark["failed_runs"],
            )

            for name, metrics in (
                benchmark["workload_summary"].items()
            ):
                print(
                    f"  {name}: "
                    f"mean={metrics['mean_ms']} ms, "
                    f"p95={metrics['p95_ms']} ms"
                )

        print("\nErrors:", len(summary["errors"]))
        print("Output directory:", output)
        print("Database modifications: NONE")

    return (
        0
        if summary["status"] == "PASS"
        else 1
    )


if __name__ == "__main__":
    sys.exit(main())
