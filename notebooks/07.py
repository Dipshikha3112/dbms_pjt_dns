from pathlib import Path
from collections import defaultdict
from datetime import datetime
import xml.etree.ElementTree as ET
import csv
import getpass
import json
import os
import re
import statistics
import time

import pandas as pd
import psycopg
from neo4j import GraphDatabase


# =============================================================================
# PHASE 10 — BIOMEDICAL KNOWLEDGE GRAPH
# FINAL DATASET-AWARE SEMANTIC ALIGNMENT VERSION (v2 — improved coverage)
# =============================================================================

print("=" * 120)
print("PHASE 10 — BIOMEDICAL KNOWLEDGE GRAPH")
print("FINAL DATASET-AWARE SEMANTIC ALIGNMENT VERSION (v2)")
print("PostgreSQL + LOINC + RxNorm + MeSH -> Neo4j")
print("=" * 120)


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

LOINC_FILE = (
    DATA_DIR
    / "loinc"
    / "LoincTable"
    / "Loinc.csv"
)

RXNORM_DIR = (
    DATA_DIR
    / "rxnorm"
    / "rrf"
)

RXNCONSO_FILE = RXNORM_DIR / "RXNCONSO.RRF"
RXNREL_FILE = RXNORM_DIR / "RXNREL.RRF"
RXNSAT_FILE = RXNORM_DIR / "RXNSAT.RRF"

MESH_FILE = (
    DATA_DIR
    / "mesh"
    / "desc2026.xml"
)

# NEW: MeSH Supplementary Concepts (SCRs) — hugely improves drug/lab coverage
MESH_SUPP_FILE = (
    DATA_DIR
    / "mesh"
    / "supp2026.1.xml"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "phase10_knowledge_graph"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


NODE_STATS_FILE = RESULTS_DIR / "node_statistics.csv"
REL_STATS_FILE = RESULTS_DIR / "relationship_statistics.csv"
GRAPH_STATS_FILE = RESULTS_DIR / "graph_statistics.csv"
MAPPING_FILE = RESULTS_DIR / "mapping_coverage.csv"
SEMANTIC_COVERAGE_FILE = RESULTS_DIR / "semantic_coverage.csv"
VALIDATION_FILE = RESULTS_DIR / "graph_validation.csv"
BENCHMARK_FILE = RESULTS_DIR / "graph_query_benchmarks.csv"
METADATA_FILE = RESULTS_DIR / "phase10_metadata.json"
REPORT_FILE = RESULTS_DIR / "phase10_report.txt"

DRUG_DIAGNOSTICS_FILE = (
    RESULTS_DIR
    / "clinical_drug_mapping_diagnostics.csv"
)

LAB_DIAGNOSTICS_FILE = (
    RESULTS_DIR
    / "clinical_lab_mapping_diagnostics.csv"
)

DISEASE_DIAGNOSTICS_FILE = (
    RESULTS_DIR
    / "clinical_disease_mapping_diagnostics.csv"
)

UNMAPPED_DISEASE_FILE = (
    RESULTS_DIR
    / "unmapped_disease_analysis.csv"
)

UNMAPPED_DRUG_FILE = (
    RESULTS_DIR
    / "unmapped_drug_analysis.csv"
)

UNMAPPED_LAB_FILE = (
    RESULTS_DIR
    / "unmapped_lab_analysis.csv"
)

MAPPING_METHOD_FILE = (
    RESULTS_DIR
    / "mapping_method_statistics.csv"
)


POSTGRES_CONFIG = {
    "dbname": "adaptive_healthcare_db",
    "host": "localhost",
    "port": 5432,
    "user": "postgres",
}


NEO4J_URI = os.getenv(
    "NEO4J_URI",
    "bolt://localhost:7687"
)

NEO4J_USER = os.getenv(
    "NEO4J_USER",
    "neo4j"
)

NEO4J_DATABASE = os.getenv(
    "NEO4J_DATABASE",
    "neo4j"
)


BATCH_SIZE = 2000
BENCHMARK_RUNS = 10

# True = rebuild Phase 10 graph completely
RESET_GRAPH = True


# =============================================================================
# HELPERS
# =============================================================================

def stage(title):
    print()
    print("=" * 120)
    print(title)
    print("=" * 120)


def clean_text(value):

    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    return value


def normalize_basic(value):

    value = clean_text(value)

    if not value:
        return None

    value = value.lower()

    value = value.replace(
        "’",
        "'"
    )

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


def normalize_rxcui(value):

    value = clean_text(value)

    if not value:
        return None

    value = re.sub(
        r"\.0$",
        "",
        value
    )

    return value


def normalize_loinc(value):

    value = clean_text(value)

    if not value:
        return None

    return value.upper()


def safe_percentage(
    numerator,
    denominator
):

    if not denominator:
        return 0.0

    return (
        100.0
        * numerator
        / denominator
    )


def chunks(
    rows,
    size=BATCH_SIZE
):

    for i in range(
        0,
        len(rows),
        size
    ):
        yield rows[
            i:i + size
        ]


def serialize_nested_rows(rows):

    output = []

    for row in rows:

        new_row = {}

        for key, value in row.items():

            if isinstance(
                value,
                (list, dict)
            ):

                new_row[key] = json.dumps(
                    value,
                    default=str
                )

            else:

                new_row[key] = value

        output.append(
            new_row
        )

    return output


# =============================================================================
# SOURCE VALIDATION
# =============================================================================

stage(
    "SUBPHASE 10A — VERIFY + PREPARE KNOWLEDGE SOURCES"
)


required_files = {
    "LOINC":
        LOINC_FILE,

    "RxNorm RXNCONSO":
        RXNCONSO_FILE,

    "RxNorm RXNREL":
        RXNREL_FILE,

    "RxNorm RXNSAT":
        RXNSAT_FILE,

    "MeSH Descriptors":
        MESH_FILE,
}


optional_files = {
    "MeSH Supplementary (supp2026.xml)":
        MESH_SUPP_FILE,
}


missing_files = []


for name, path in required_files.items():

    if path.exists():

        print(
            f"{name:<35}",
            "PASS ->",
            path
        )

    else:

        print(
            f"{name:<35}",
            "MISSING ->",
            path
        )

        missing_files.append(
            str(path)
        )


for name, path in optional_files.items():

    if path.exists():

        print(
            f"{name:<35}",
            "FOUND ->",
            path
        )

    else:

        print(
            f"{name:<35}",
            "NOT FOUND (optional) ->",
            path
        )


if missing_files:

    raise FileNotFoundError(
        "\nMissing required source files:\n"
        + "\n".join(
            missing_files
        )
    )


# =============================================================================
# DATABASE PASSWORDS
# =============================================================================

pg_password = os.getenv(
    "POSTGRES_PASSWORD"
)

if not pg_password:

    pg_password = getpass.getpass(
        "PostgreSQL password: "
    )


neo4j_password = os.getenv(
    "NEO4J_PASSWORD"
)

if not neo4j_password:

    neo4j_password = getpass.getpass(
        "Neo4j password: "
    )


# =============================================================================
# POSTGRESQL CONNECTION
# =============================================================================

pg_conn = psycopg.connect(
    **POSTGRES_CONFIG,
    password=pg_password
)


print()
print(
    "PostgreSQL connection: PASS"
)


# =============================================================================
# NEO4J CONNECTION
# =============================================================================

neo4j_driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(
        NEO4J_USER,
        neo4j_password
    )
)


neo4j_driver.verify_connectivity()


print(
    "Neo4j connection: PASS"
)


# =============================================================================
# DATABASE HELPERS
# =============================================================================

def pg_records(
    query,
    params=None
):

    with pg_conn.cursor() as cur:

        cur.execute(
            query,
            params
        )

        columns = [
            desc.name
            for desc
            in cur.description
        ]

        return [
            dict(
                zip(
                    columns,
                    row
                )
            )
            for row
            in cur.fetchall()
        ]


def neo4j_records(
    query,
    **params
):

    with neo4j_driver.session(
        database=NEO4J_DATABASE
    ) as session:

        result = session.run(
            query,
            **params
        )

        return [
            record.data()
            for record
            in result
        ]


def neo4j_scalar(
    query,
    **params
):

    with neo4j_driver.session(
        database=NEO4J_DATABASE
    ) as session:

        record = session.run(
            query,
            **params
        ).single()

        if not record:
            return 0

        return record[
            "value"
        ]


def neo4j_batch_write(
    query,
    rows,
    label
):

    if not rows:

        print(
            f"{label:<40}",
            "0"
        )

        return


    total = 0


    with neo4j_driver.session(
        database=NEO4J_DATABASE
    ) as session:

        for batch in chunks(
            rows
        ):

            session.run(
                query,
                rows=batch
            ).consume()

            total += len(
                batch
            )


    print(
        f"{label:<40}",
        f"{total:,}"
    )


# =============================================================================
# RESET GRAPH
# =============================================================================

if RESET_GRAPH:

    stage(
        "RESETTING PHASE 10 NEO4J GRAPH"
    )

    with neo4j_driver.session(
        database=NEO4J_DATABASE
    ) as session:

        session.run(
            """
            MATCH ()-[r]->()
            CALL (r) {
                DELETE r
            } IN TRANSACTIONS OF 10000 ROWS
            """
        ).consume()

        session.run(
            """
            MATCH (n)
            CALL (n) {
                DELETE n
            } IN TRANSACTIONS OF 10000 ROWS
            """
        ).consume()

    print(
        "Existing graph removed: PASS"
    )


# =============================================================================
# EXTRACT POSTGRESQL CLINICAL DATA
# =============================================================================

stage(
    "SUBPHASE 10B — EXTRACT + TRANSFORM CLINICAL DATA"
)


# -----------------------------------------------------------------------------
# PATIENTS
# -----------------------------------------------------------------------------

patients = pg_records(
    """
    SELECT
        patient_id::text AS patient_id,
        first_name,
        last_name,
        gender,
        birth_date::text AS birth_date,
        death_date::text AS death_date,
        race,
        ethnicity
    FROM patient
    """
)


print(
    "Patients:",
    f"{len(patients):,}"
)


# -----------------------------------------------------------------------------
# DISEASE CONCEPTS
# -----------------------------------------------------------------------------

diseases = pg_records(
    """
    SELECT
        condition_id::text
            AS condition_id,

        source_code::text
            AS source_code,

        description
            AS description

    FROM condition
    """
)


for row in diseases:

    row["name"] = row.get(
        "description"
    )


print(
    "Disease concepts:",
    f"{len(diseases):,}"
)


# -----------------------------------------------------------------------------
# PATIENT CONDITIONS
# -----------------------------------------------------------------------------

patient_diseases = pg_records(
    """
    SELECT
        d.patient_id::text
            AS patient_id,

        d.condition_id::text
            AS condition_id,

        COUNT(*)
            AS diagnosis_count,

        MIN(d.start_date)::text
            AS first_diagnosed,

        MAX(d.start_date)::text
            AS latest_diagnosed,

        MAX(d.stop_date)::text
            AS latest_stop

    FROM diagnosis d

    GROUP BY
        d.patient_id,
        d.condition_id
    """
)


print(
    "Patient-condition links:",
    f"{len(patient_diseases):,}"
)


# -----------------------------------------------------------------------------
# DRUG CONCEPTS
# -----------------------------------------------------------------------------

clinical_drugs = pg_records(
    """
    SELECT
        rxcui::text
            AS rxcui,

        name,

        concept_type

    FROM drug_concept
    """
)


for row in clinical_drugs:

    row["rxcui"] = normalize_rxcui(
        row.get(
            "rxcui"
        )
    )


print(
    "Clinical drug concepts:",
    f"{len(clinical_drugs):,}"
)


# -----------------------------------------------------------------------------
# PATIENT MEDICATIONS
# -----------------------------------------------------------------------------

patient_drugs = pg_records(
    """
    SELECT
        p.patient_id::text
            AS patient_id,

        pi.rxcui::text
            AS rxcui,

        COUNT(*)
            AS prescription_count,

        MIN(p.start_time)::text
            AS first_prescribed,

        MAX(p.start_time)::text
            AS latest_prescribed,

        MAX(p.stop_time)::text
            AS latest_stop

    FROM prescription p

    JOIN prescription_item pi
        ON pi.prescription_id =
           p.prescription_id

    WHERE
        pi.rxcui IS NOT NULL

    GROUP BY
        p.patient_id,
        pi.rxcui
    """
)


for row in patient_drugs:

    row["rxcui"] = normalize_rxcui(
        row.get(
            "rxcui"
        )
    )


print(
    "Patient-drug links:",
    f"{len(patient_drugs):,}"
)


# -----------------------------------------------------------------------------
# PATIENT LABS
# -----------------------------------------------------------------------------

patient_labs = pg_records(
    """
    SELECT
        lr.patient_id::text
            AS patient_id,

        lr.loinc_code::text
            AS loinc_code,

        COUNT(*)
            AS result_count,

        MAX(lr.result_time)::text
            AS latest_result_time

    FROM lab_result lr

    WHERE
        lr.loinc_code IS NOT NULL

    GROUP BY
        lr.patient_id,
        lr.loinc_code
    """
)


for row in patient_labs:

    row["loinc_code"] = normalize_loinc(
        row.get(
            "loinc_code"
        )
    )


print(
    "Patient-lab links:",
    f"{len(patient_labs):,}"
)


# -----------------------------------------------------------------------------
# LOINC CONCEPTS FROM POSTGRESQL
# -----------------------------------------------------------------------------

loinc_concepts = pg_records(
    """
    SELECT
        loinc_code::text
            AS loinc_code,

        component,
        property,
        time_aspect,
        system,
        scale_type,
        method_type,
        class,
        long_common_name,
        short_name

    FROM loinc_concept
    """
)


for row in loinc_concepts:

    row["loinc_code"] = normalize_loinc(
        row.get(
            "loinc_code"
        )
    )

    row["name"] = (
        row.get(
            "long_common_name"
        )
        or
        row.get(
            "short_name"
        )
        or
        row.get(
            "loinc_code"
        )
    )


print(
    "LOINC concepts:",
    f"{len(loinc_concepts):,}"
)


# =============================================================================
# GRAPH SCHEMA
# =============================================================================

stage(
    "SUBPHASE 10C — GRAPH SCHEMA"
)


GRAPH_SCHEMA = {

    "nodes": [
        "Patient",
        "Disease",
        "Drug",
        "LabTest",
        "MeSHConcept",
    ],

    "relationships": [
        "HAS_CONDITION",
        "HAS_MEDICATION",
        "HAS_LAB",
        "OBSERVED_WITH_MEDICATION",
        "ASSOCIATED_WITH",
        "RELATED_TO",
        "MAPPED_TO",
        "ABOUT_MESH",
        "BROADER_THAN",
    ],

    "reserved_authoritative_relationships": [
        "TREATED_BY",
        "INTERACTS_WITH",
    ]
}


print(
    json.dumps(
        GRAPH_SCHEMA,
        indent=4
    )
)


# =============================================================================
# CONSTRAINTS + INDEXES
# =============================================================================

stage(
    "SUBPHASE 10I — CONSTRAINTS + INDEXES"
)


constraint_queries = [

    """
    CREATE CONSTRAINT patient_id_unique
    IF NOT EXISTS
    FOR (n:Patient)
    REQUIRE n.patient_id IS UNIQUE
    """,

    """
    CREATE CONSTRAINT disease_id_unique
    IF NOT EXISTS
    FOR (n:Disease)
    REQUIRE n.condition_id IS UNIQUE
    """,

    """
    CREATE CONSTRAINT drug_rxcui_unique
    IF NOT EXISTS
    FOR (n:Drug)
    REQUIRE n.rxcui IS UNIQUE
    """,

    """
    CREATE CONSTRAINT lab_loinc_unique
    IF NOT EXISTS
    FOR (n:LabTest)
    REQUIRE n.loinc_code IS UNIQUE
    """,

    """
    CREATE CONSTRAINT mesh_ui_unique
    IF NOT EXISTS
    FOR (n:MeSHConcept)
    REQUIRE n.descriptor_ui IS UNIQUE
    """
]


index_queries = [

    """
    CREATE INDEX disease_name_index
    IF NOT EXISTS
    FOR (n:Disease)
    ON (n.name)
    """,

    """
    CREATE INDEX drug_name_index
    IF NOT EXISTS
    FOR (n:Drug)
    ON (n.name)
    """,

    """
    CREATE INDEX lab_name_index
    IF NOT EXISTS
    FOR (n:LabTest)
    ON (n.name)
    """,

    """
    CREATE INDEX mesh_name_index
    IF NOT EXISTS
    FOR (n:MeSHConcept)
    ON (n.name)
    """
]


with neo4j_driver.session(
    database=NEO4J_DATABASE
) as session:

    for query in constraint_queries:
        session.run(
            query
        ).consume()

    for query in index_queries:
        session.run(
            query
        ).consume()


print(
    "Constraints:",
    len(
        constraint_queries
    )
)

print(
    "Indexes:",
    len(
        index_queries
    )
)

print(
    "Constraint/index creation: PASS"
)


# =============================================================================
# CORE NODES
# =============================================================================

stage(
    "SUBPHASE 10D — CORE CLINICAL NODE TYPES"
)


neo4j_batch_write(

    """
    UNWIND $rows AS row

    MERGE (
        p:Patient {
            patient_id:
                row.patient_id
        }
    )

    SET
        p.first_name =
            row.first_name,

        p.last_name =
            row.last_name,

        p.gender =
            row.gender,

        p.birth_date =
            row.birth_date,

        p.death_date =
            row.death_date,

        p.race =
            row.race,

        p.ethnicity =
            row.ethnicity,

        p.source =
            'PostgreSQL/Synthea'
    """,

    patients,

    "Patient nodes"
)


neo4j_batch_write(

    """
    UNWIND $rows AS row

    MERGE (
        d:Disease {
            condition_id:
                row.condition_id
        }
    )

    SET
        d.source_code =
            row.source_code,

        d.name =
            row.name,

        d.description =
            row.description,

        d.source =
            'PostgreSQL/Synthea'
    """,

    diseases,

    "Disease nodes"
)


neo4j_batch_write(

    """
    UNWIND $rows AS row

    MERGE (
        d:Drug {
            rxcui:
                row.rxcui
        }
    )

    SET
        d.name =
            row.name,

        d.concept_type =
            row.concept_type,

        d.source =
            'PostgreSQL/RxNorm'
    """,

    [
        row
        for row
        in clinical_drugs
        if row.get(
            "rxcui"
        )
    ],

    "Initial Drug nodes"
)


neo4j_batch_write(

    """
    UNWIND $rows AS row

    MERGE (
        l:LabTest {
            loinc_code:
                row.loinc_code
        }
    )

    SET
        l.name =
            row.name,

        l.component =
            row.component,

        l.property =
            row.property,

        l.time_aspect =
            row.time_aspect,

        l.system =
            row.system,

        l.scale_type =
            row.scale_type,

        l.method_type =
            row.method_type,

        l.class =
            row.class,

        l.long_common_name =
            row.long_common_name,

        l.short_name =
            row.short_name,

        l.source =
            'LOINC'
    """,

    [
        row
        for row
        in loinc_concepts
        if row.get(
            "loinc_code"
        )
    ],

    "LabTest nodes"
)


# =============================================================================
# RXNORM CONCEPT LOADING
# =============================================================================

stage(
    "SUBPHASE 10F — RXNORM TERMINOLOGY"
)


rxnorm_concepts = {}

rxnorm_aliases = defaultdict(
    set
)

rxnorm_original_terms = defaultdict(
    set
)

rxnorm_ttys = defaultdict(
    set
)


with open(
    RXNCONSO_FILE,
    "r",
    encoding="utf-8",
    errors="ignore"
) as f:

    reader = csv.reader(
        f,
        delimiter="|"
    )


    for fields in reader:

        if len(fields) < 17:
            continue


        rxcui = normalize_rxcui(
            fields[0]
        )

        language = clean_text(
            fields[1]
        )

        tty = clean_text(
            fields[12]
        )

        term = clean_text(
            fields[14]
        )

        suppress = clean_text(
            fields[16]
        )


        if not rxcui:
            continue


        if language != "ENG":
            continue


        if suppress == "Y":
            continue


        if term:

            normalized = normalize_basic(
                term
            )

            if normalized:

                rxnorm_aliases[
                    rxcui
                ].add(
                    normalized
                )

                rxnorm_original_terms[
                    rxcui
                ].add(
                    term
                )


        if tty:

            rxnorm_ttys[
                rxcui
            ].add(
                tty
            )


        if rxcui not in rxnorm_concepts:

            rxnorm_concepts[
                rxcui
            ] = {
                "rxcui":
                    rxcui,

                "name":
                    term,

                "concept_type":
                    tty,

                "source":
                    "RxNorm"
            }


print(
    "RxNorm concepts processed:",
    f"{len(rxnorm_concepts):,}"
)


neo4j_batch_write(

    """
    UNWIND $rows AS row

    MERGE (
        d:Drug {
            rxcui:
                row.rxcui
        }
    )

    ON CREATE SET
        d.name =
            row.name,

        d.concept_type =
            row.concept_type,

        d.source =
            row.source

    ON MATCH SET
        d.name =
            CASE
                WHEN d.name IS NULL
                THEN row.name
                ELSE d.name
            END,

        d.concept_type =
            CASE
                WHEN d.concept_type IS NULL
                THEN row.concept_type
                ELSE d.concept_type
            END
    """,

    list(
        rxnorm_concepts.values()
    ),

    "RxNorm Drug nodes"
)


# =============================================================================
# RXNORM RELATIONSHIPS
# =============================================================================

print()
print(
    "Loading RxNorm semantic relationships..."
)


rxnorm_relationships = []


with open(
    RXNREL_FILE,
    "r",
    encoding="utf-8",
    errors="ignore"
) as f:

    reader = csv.reader(
        f,
        delimiter="|"
    )


    for fields in reader:

        if len(fields) < 16:
            continue


        rxcui1 = normalize_rxcui(
            fields[0]
        )

        rel = clean_text(
            fields[3]
        )

        rxcui2 = normalize_rxcui(
            fields[4]
        )

        rela = clean_text(
            fields[7]
        )

        suppress = clean_text(
            fields[14]
        )


        if not rxcui1:
            continue

        if not rxcui2:
            continue

        if rxcui1 == rxcui2:
            continue

        if suppress == "Y":
            continue


        rxnorm_relationships.append(
            {
                "source_rxcui":
                    rxcui1,

                "target_rxcui":
                    rxcui2,

                "relationship_type":
                    rel,

                "relationship_attribute":
                    rela,

                "source":
                    "RxNorm RXNREL"
            }
        )


print(
    "RxNorm relationship records:",
    f"{len(rxnorm_relationships):,}"
)


neo4j_batch_write(

    """
    UNWIND $rows AS row

    MATCH (
        source:Drug {
            rxcui:
                row.source_rxcui
        }
    )

    MATCH (
        target:Drug {
            rxcui:
                row.target_rxcui
        }
    )

    MERGE (
        source
    )-[r:RELATED_TO {
        relationship_type:
            coalesce(
                row.relationship_type,
                ''
            ),

        relationship_attribute:
            coalesce(
                row.relationship_attribute,
                ''
            )
    }]->(
        target
    )

    SET
        r.source =
            row.source,

        r.relationship_semantics =
            'rxnorm_terminology_relationship_not_drug_interaction'
    """,

    rxnorm_relationships,

    "RELATED_TO relationships"
)


print(
    "INTERACTS_WITH inferred from RxNorm:",
    0
)

print(
    "Generic RxNorm relationships are NOT treated as drug-drug interactions."
)


# =============================================================================
# RXNORM INGREDIENT RESOLUTION (NEW — fixes clinical drug direct mapping)
# =============================================================================
# Build ingredient index from RXNREL:
#   RO = has_ingredient (product -> ingredient)
#   RN = ingredient_of  (ingredient -> product)
# Also collect ingredient display names from RXNCONSO (TTY in IN/PIN/MIN)

print()
print(
    "Building RxNorm ingredient index..."
)

rxcui_to_ingredients = defaultdict(set)
ingredient_to_products = defaultdict(set)

for rel in rxnorm_relationships:

    rtype = (
        rel.get("relationship_type") or ""
    ).upper()

    src = rel["source_rxcui"]
    tgt = rel["target_rxcui"]

    if rtype == "RO":
        rxcui_to_ingredients[src].add(tgt)
        ingredient_to_products[tgt].add(src)

    elif rtype == "RN":
        rxcui_to_ingredients[tgt].add(src)
        ingredient_to_products[src].add(tgt)


INGREDIENT_TTYS = {"IN", "PIN", "MIN"}

ingredient_name_index = defaultdict(set)


with open(
    RXNCONSO_FILE,
    "r",
    encoding="utf-8",
    errors="ignore"
) as f:

    reader = csv.reader(
        f,
        delimiter="|"
    )

    for fields in reader:

        if len(fields) < 17:
            continue

        rxcui = normalize_rxcui(fields[0])
        language = clean_text(fields[1])
        tty = clean_text(fields[12])
        term = clean_text(fields[14])
        suppress = clean_text(fields[16])

        if not rxcui or language != "ENG" or suppress == "Y":
            continue

        if tty in INGREDIENT_TTYS and term:
            ingredient_name_index[rxcui].add(term)


print(
    "RxNorm ingredient products indexed:",
    f"{len(rxcui_to_ingredients):,}"
)

print(
    "RxNorm ingredient names indexed:",
    f"{len(ingredient_name_index):,}"
)


# =============================================================================
# MESH LOADING + LEXICAL INDEX
# =============================================================================

stage(
    "SUBPHASE 10F — MESH TERMINOLOGY"
)


mesh_nodes = []
mesh_hierarchy = []

mesh_exact_index = defaultdict(
    set
)

mesh_descriptor_names = {}

mesh_descriptor_terms = defaultdict(
    set
)


context = ET.iterparse(
    MESH_FILE,
    events=(
        "end",
    )
)


for event, elem in context:

    if not elem.tag.endswith(
        "DescriptorRecord"
    ):
        continue


    descriptor_ui = None
    descriptor_name = None

    terms = []

    tree_numbers = []


    for child in elem.iter():

        tag = child.tag.split(
            "}"
        )[-1]


        if (
            tag == "DescriptorUI"
            and descriptor_ui is None
        ):

            descriptor_ui = clean_text(
                child.text
            )


        elif tag == "DescriptorName":

            for node in child.iter():

                if (
                    node.tag.split(
                        "}"
                    )[-1]
                    == "String"
                ):

                    descriptor_name = clean_text(
                        node.text
                    )

                    break


        elif tag == "Term":

            for node in child.iter():

                if (
                    node.tag.split(
                        "}"
                    )[-1]
                    == "String"
                ):

                    term = clean_text(
                        node.text
                    )

                    if term:
                        terms.append(
                            term
                        )

                    break


        elif tag == "TreeNumber":

            tree_number = clean_text(
                child.text
            )

            if tree_number:

                tree_numbers.append(
                    tree_number
                )


    if descriptor_ui:

        mesh_descriptor_names[
            descriptor_ui
        ] = descriptor_name


        all_terms = []


        if descriptor_name:

            all_terms.append(
                (
                    descriptor_name,
                    "descriptor"
                )
            )


        for term in terms:

            all_terms.append(
                (
                    term,
                    "entry_term"
                )
            )


        for term, term_type in all_terms:

            normalized = normalize_basic(
                term
            )

            if not normalized:
                continue


            mesh_exact_index[
                normalized
            ].add(
                (
                    descriptor_ui,
                    term_type
                )
            )


            mesh_descriptor_terms[
                descriptor_ui
            ].add(
                normalized
            )


        mesh_nodes.append(
            {
                "descriptor_ui":
                    descriptor_ui,

                "name":
                    descriptor_name,

                "tree_numbers":
                    tree_numbers,

                "source":
                    "MeSH 2026"
            }
        )


    elem.clear()


print(
    "MeSH descriptors:",
    f"{len(mesh_nodes):,}"
)

print(
    "Normalized lexical terms:",
    f"{len(mesh_exact_index):,}"
)


neo4j_batch_write(

    """
    UNWIND $rows AS row

    MERGE (
        m:MeSHConcept {
            descriptor_ui:
                row.descriptor_ui
        }
    )

    SET
        m.name =
            row.name,

        m.tree_numbers =
            row.tree_numbers,

        m.source =
            row.source
    """,

    mesh_nodes,

    "MeSHConcept nodes"
)


# =============================================================================
# MESH SUPPLEMENTARY CONCEPTS (SCRs) — NEW
# =============================================================================
# Loads supp2026.xml and merges into the same lexical index used for matching.
# These add tens of thousands of drug / chemical / lab terms.

if MESH_SUPP_FILE.exists():

    supp_nodes = []

    supp_context = ET.iterparse(
        MESH_SUPP_FILE,
        events=("end",)
    )

    for event, elem in supp_context:

        if not elem.tag.endswith(
            "SupplementalRecord"
        ):
            continue

        supp_ui = None
        supp_name = None
        terms = []

        for child in elem.iter():

            tag = child.tag.split("}")[-1]

            if (
                tag == "SupplementalRecordUI"
                and supp_ui is None
            ):
                supp_ui = clean_text(child.text)

            elif tag == "SupplementalRecordName":

                for node in child.iter():
                    if (
                        node.tag.split("}")[-1]
                        == "String"
                    ):
                        supp_name = clean_text(
                            node.text
                        )
                        break

            elif tag == "Term":

                for node in child.iter():
                    if (
                        node.tag.split("}")[-1]
                        == "String"
                    ):
                        term = clean_text(node.text)
                        if term:
                            terms.append(term)
                        break

        if supp_ui:

            mesh_descriptor_names[supp_ui] = supp_name

            all_terms = []

            if supp_name:
                all_terms.append(
                    (supp_name, "descriptor")
                )

            for term in terms:
                all_terms.append(
                    (term, "entry_term")
                )

            for term, term_type in all_terms:

                normalized = normalize_basic(term)

                if not normalized:
                    continue

                mesh_exact_index[
                    normalized
                ].add((supp_ui, term_type))

                mesh_descriptor_terms[
                    supp_ui
                ].add(normalized)

            supp_nodes.append(
                {
                    "descriptor_ui": supp_ui,
                    "name": supp_name,
                    "tree_numbers": [],
                    "source":
                        "MeSH 2026 Supplementary",
                }
            )

        elem.clear()

    print(
        "MeSH supplementary records:",
        f"{len(supp_nodes):,}"
    )

    print(
        "Normalized lexical terms (with supp):",
        f"{len(mesh_exact_index):,}"
    )

    neo4j_batch_write(
        """
        UNWIND $rows AS row
        MERGE (m:MeSHConcept {
            descriptor_ui: row.descriptor_ui
        })
        SET m.name = row.name,
            m.tree_numbers = row.tree_numbers,
            m.source = row.source
        """,
        supp_nodes,
        "MeSH Supplementary nodes"
    )

    mesh_nodes.extend(supp_nodes)

else:

    print()
    print(
        "MeSH supplementary file not found — skipping."
    )
    print(
        "Download supp2026.xml from "
        "https://nlmpubs.nlm.nih.gov/projects/mesh/"
        "MESH_FILES/xmlmesh/supp2026.xml"
    )
    print(
        "and place it in data/mesh/ to improve "
        "drug and lab coverage."
    )


# =============================================================================
# MESH HIERARCHY
# =============================================================================

tree_to_descriptor = {}


for row in mesh_nodes:

    for tree_number in (
        row.get(
            "tree_numbers"
        )
        or []
    ):

        tree_to_descriptor[
            tree_number
        ] = row[
            "descriptor_ui"
        ]


for tree_number, child_ui in tree_to_descriptor.items():

    if "." not in tree_number:
        continue


    parent_tree = tree_number.rsplit(
        ".",
        1
    )[0]


    parent_ui = tree_to_descriptor.get(
        parent_tree
    )


    if not parent_ui:
        continue


    if parent_ui == child_ui:
        continue


    mesh_hierarchy.append(
        {
            "child_ui":
                child_ui,

            "parent_ui":
                parent_ui,

            "source":
                "MeSH tree hierarchy"
        }
    )


print(
    "MeSH hierarchy source links:",
    f"{len(mesh_hierarchy):,}"
)


neo4j_batch_write(

    """
    UNWIND $rows AS row

    MATCH (
        child:MeSHConcept {
            descriptor_ui:
                row.child_ui
        }
    )

    MATCH (
        parent:MeSHConcept {
            descriptor_ui:
                row.parent_ui
        }
    )

    MERGE (
        child
    )-[r:BROADER_THAN]->(
        parent
    )

    SET
        r.source =
            row.source
    """,

    mesh_hierarchy,

    "BROADER_THAN source load"
)


# =============================================================================
# CONDITION NORMALIZATION
# =============================================================================

SNOMED_SEMANTIC_TAGS = {
    "disorder",
    "finding",
    "situation",
    "procedure",
    "observable entity",
    "event",
    "regime therapy",
    "person",
    "physical object",
}


def strip_snomed_semantic_tag(
    value
):

    value = clean_text(
        value
    )

    if not value:
        return None


    match = re.search(
        r"\(([^()]*)\)\s*$",
        value
    )


    if match:

        tag = normalize_basic(
            match.group(
                1
            )
        )


        if tag in SNOMED_SEMANTIC_TAGS:

            value = value[
                :match.start()
            ].strip()


    return value


def condition_variants(
    value
):

    value = clean_text(
        value
    )

    if not value:
        return set()


    variants = set()


    normalized = normalize_basic(
        value
    )

    if normalized:

        variants.add(
            normalized
        )


    stripped = strip_snomed_semantic_tag(
        value
    )


    if stripped:

        normalized_stripped = normalize_basic(
            stripped
        )

        if normalized_stripped:

            variants.add(
                normalized_stripped
            )


    return variants


# =============================================================================
# DRUG NORMALIZATION
# =============================================================================

DOSE_FORM_WORDS = {
    "oral",
    "tablet",
    "tablets",
    "capsule",
    "capsules",
    "solution",
    "suspension",
    "injection",
    "injectable",
    "cream",
    "ointment",
    "patch",
    "spray",
    "inhaler",
    "powder",
    "chewable",
    "extended",
    "release",
    "delayed",
    "sublingual",
    "topical",
    "ophthalmic",
    "nasal",
    "auto",
    "injector",
    "ml",
    "mg",
    "mcg",
    "g",
    "unit",
    "units",
}


def remove_strength_and_dose_form(
    value
):

    value = clean_text(
        value
    )

    if not value:
        return None


    value = value.lower()


    value = re.sub(
        r"\bnda\d+\b",
        " ",
        value
    )


    value = re.sub(
        r"\b\d+(?:\.\d+)?\s*(?:ml|mg|mcg|g|units?)\b",
        " ",
        value,
        flags=re.I
    )


    value = re.sub(
        r"\b\d+(?:\.\d+)?\s*mg\s*/\s*ml\b",
        " ",
        value,
        flags=re.I
    )


    value = re.sub(
        r"\b\d+(?:\.\d+)?\b",
        " ",
        value
    )


    normalized = normalize_basic(
        value
    )


    if not normalized:
        return None


    tokens = [
        token
        for token
        in normalized.split()
        if token not in DOSE_FORM_WORDS
    ]


    result = " ".join(
        tokens
    ).strip()


    return result or None


def drug_variants(
    value
):

    value = clean_text(
        value
    )

    if not value:
        return set()


    variants = set()


    normalized = normalize_basic(
        value
    )


    if normalized:

        variants.add(
            normalized
        )


    ingredient_form = (
        remove_strength_and_dose_form(
            value
        )
    )


    if ingredient_form:

        variants.add(
            normalize_basic(
                ingredient_form
            )
        )


    pieces = re.split(
        r"\s*/\s*|\s+\+\s+",
        value
    )


    for piece in pieces:

        cleaned = (
            remove_strength_and_dose_form(
                piece
            )
        )


        if cleaned:

            normalized_piece = normalize_basic(
                cleaned
            )


            if normalized_piece:

                variants.add(
                    normalized_piece
                )


    return {
        value
        for value
        in variants
        if value
    }


# =============================================================================
# LAB NORMALIZATION
# =============================================================================

def lab_name_variants(
    row
):

    variants = set()


    fields = [
        row.get(
            "name"
        ),

        row.get(
            "long_common_name"
        ),

        row.get(
            "short_name"
        ),
    ]


    for value in fields:

        normalized = normalize_basic(
            value
        )


        if normalized:

            variants.add(
                normalized
            )


    return variants


def lab_component_variants(
    row
):

    variants = set()


    component = clean_text(
        row.get(
            "component"
        )
    )


    if not component:
        return variants


    normalized = normalize_basic(
        component
    )


    if normalized:

        variants.add(
            normalized
        )


    simplified = re.sub(
        r"\b(total|mass|number|concentration|presence)\b",
        " ",
        normalized or ""
    )


    simplified = re.sub(
        r"\s+",
        " ",
        simplified
    ).strip()


    if simplified:

        variants.add(
            simplified
        )


    return variants


def lab_component_only_variants(
    row
):
    """
    NEW: strip brackets and qualifier clauses from component,
    producing a clean analyte name suitable for MeSH matching.
    Example: "Glucose [Mass/volume] in Serum or Plasma" -> "glucose"
    """

    component = clean_text(row.get("component"))

    if not component:
        return set()

    # Remove bracketed segments (units, specimen)
    cleaned = re.sub(r"\[[^\]]*\]", " ", component)

    # Remove "in <specimen>", "by <method>", "of <thing>"
    cleaned = re.sub(
        r"\b(?:in|by|of|on)\b.*$",
        " ",
        cleaned,
        flags=re.I
    )

    normalized = normalize_basic(cleaned)

    if not normalized:
        return set()

    return {normalized}


# =============================================================================
# MESH MATCHING
# =============================================================================

def find_mesh_matches(
    variants
):

    matches = defaultdict(
        set
    )


    for variant in variants:

        if not variant:
            continue


        for (
            descriptor_ui,
            term_type
        ) in mesh_exact_index.get(
            variant,
            set()
        ):

            matches[
                descriptor_ui
            ].add(
                term_type
            )


    return matches


def select_unique_mesh_match(
    variants
):

    matches = find_mesh_matches(
        variants
    )


    if len(matches) != 1:

        return None


    descriptor_ui = next(
        iter(
            matches
        )
    )


    term_types = matches[
        descriptor_ui
    ]


    if "descriptor" in term_types:

        method = (
            "descriptor_exact"
        )

        confidence = 1.00

    else:

        method = (
            "entry_term_exact"
        )

        confidence = 0.98


    return {
        "mesh_ui":
            descriptor_ui,

        "mesh_name":
            mesh_descriptor_names.get(
                descriptor_ui
            ),

        "method":
            method,

        "confidence":
            confidence,
    }


def select_best_mesh_match(
    variants,
    prefer_descriptor_name=None
):
    """
    NEW: like select_unique_mesh_match but when multiple descriptors
    match, prefer one whose descriptor_name exactly equals the
    provided preferred name. Falls back to None if no clean tie-break.
    """

    matches = find_mesh_matches(variants)

    if not matches:
        return None

    if len(matches) == 1:
        descriptor_ui = next(iter(matches))
        term_types = matches[descriptor_ui]
        method = (
            "descriptor_exact"
            if "descriptor" in term_types
            else "entry_term_exact"
        )
        confidence = 1.00 if method == "descriptor_exact" else 0.98
        return {
            "mesh_ui": descriptor_ui,
            "mesh_name": mesh_descriptor_names.get(descriptor_ui),
            "method": method,
            "confidence": confidence,
        }

    # Multiple candidates — try descriptor-name tie-break
    if prefer_descriptor_name:

        target_norm = normalize_basic(prefer_descriptor_name)

        surviving = [
            ui
            for ui in matches
            if normalize_basic(
                mesh_descriptor_names.get(ui) or ""
            ) == target_norm
        ]

        if len(surviving) == 1:
            ui = surviving[0]
            return {
                "mesh_ui": ui,
                "mesh_name": mesh_descriptor_names.get(ui),
                "method": "descriptor_name_tiebreak",
                "confidence": 0.95,
            }

    return None


# =============================================================================
# DIRECT + SEMANTIC TERMINOLOGY ALIGNMENT
# =============================================================================

stage(
    "SUBPHASE 10G — DATASET-AWARE TERMINOLOGY ALIGNMENT"
)


clinical_disease_ids = {
    row[
        "condition_id"
    ]
    for row
    in patient_diseases
}


clinical_drug_ids = {
    normalize_rxcui(
        row[
            "rxcui"
        ]
    )
    for row
    in patient_drugs
}


clinical_lab_ids = {
    normalize_loinc(
        row[
            "loinc_code"
        ]
    )
    for row
    in patient_labs
}


print(
    "Clinically used diseases:",
    f"{len(clinical_disease_ids):,}"
)

print(
    "Clinically used drugs:",
    f"{len(clinical_drug_ids):,}"
)

print(
    "Clinically used labs:",
    f"{len(clinical_lab_ids):,}"
)


# =============================================================================
# DISEASE -> MESH
# =============================================================================

disease_mapping_rows = []
disease_unmapped_rows = []
disease_ambiguous_rows = []


for disease in diseases:

    condition_id = disease.get(
        "condition_id"
    )

    description = disease.get(
        "description"
    )


    variants = condition_variants(
        description
    )


    # NEW: try tie-break using stripped description as preferred name
    result = select_best_mesh_match(
        variants,
        prefer_descriptor_name=(
            strip_snomed_semantic_tag(description)
        )
    )


    if result:

        disease_mapping_rows.append(
            {
                "condition_id":
                    condition_id,

                "mesh_ui":
                    result[
                        "mesh_ui"
                    ],

                "mesh_name":
                    result[
                        "mesh_name"
                    ],

                "mapping_method":
                    (
                        "synthea_condition_"
                        + result[
                            "method"
                        ]
                    ),

                "confidence":
                    result[
                        "confidence"
                    ],

                "source":
                    (
                        "Synthea condition description "
                        "+ MeSH 2026"
                    ),

                "provenance":
                    "lexical_terminology_alignment",

                "relationship_semantics":
                    (
                        "terminology_alignment_"
                        "not_official_crosswalk"
                    ),
            }
        )


    else:

        matches = find_mesh_matches(
            variants
        )


        diagnostic = {
            "condition_id":
                condition_id,

            "source_code":
                disease.get(
                    "source_code"
                ),

            "description":
                description,

            "variants":
                " | ".join(
                    sorted(
                        variants
                    )
                ),

            "candidate_count":
                len(
                    matches
                ),

            "clinical_use":
                (
                    condition_id
                    in
                    clinical_disease_ids
                ),
        }


        if len(matches) > 1:

            disease_ambiguous_rows.append(
                diagnostic
            )

        else:

            disease_unmapped_rows.append(
                diagnostic
            )


print()
print(
    "Disease direct mappings:",
    f"{len(disease_mapping_rows):,}"
)

print(
    "Disease ambiguous:",
    f"{len(disease_ambiguous_rows):,}"
)

print(
    "Disease unmapped:",
    f"{len(disease_unmapped_rows):,}"
)


# =============================================================================
# DRUG -> MESH
# =============================================================================

drug_mapping_rows = []
drug_about_rows = []
drug_unmapped_rows = []
drug_ambiguous_rows = []


for drug in clinical_drugs:

    rxcui = normalize_rxcui(
        drug.get(
            "rxcui"
        )
    )


    if not rxcui:
        continue


    drug_name = clean_text(
        drug.get(
            "name"
        )
    )


    exact_variants = set()


    normalized_name = normalize_basic(
        drug_name
    )

    if normalized_name:

        exact_variants.add(
            normalized_name
        )


    exact_variants.update(
        rxnorm_aliases.get(
            rxcui,
            set()
        )
    )


    # NEW: pull ingredient names and aliases from RxNorm traversal
    for ing_rxcui in rxcui_to_ingredients.get(rxcui, set()):

        for ing_name in ingredient_name_index.get(ing_rxcui, set()):
            norm = normalize_basic(ing_name)
            if norm:
                exact_variants.add(norm)

        exact_variants.update(
            rxnorm_aliases.get(ing_rxcui, set())
        )


    direct_result = (
        select_unique_mesh_match(
            exact_variants
        )
    )


    if direct_result:

        drug_mapping_rows.append(
            {
                "rxcui":
                    rxcui,

                "mesh_ui":
                    direct_result[
                        "mesh_ui"
                    ],

                "mesh_name":
                    direct_result[
                        "mesh_name"
                    ],

                "mapping_method":
                    (
                        "rxnorm_same_rxcui_"
                        + direct_result[
                            "method"
                        ]
                    ),

                "confidence":
                    direct_result[
                        "confidence"
                    ],

                "source":
                    (
                        "RxNorm RXNCONSO "
                        "+ MeSH 2026"
                    ),

                "provenance":
                    "same_rxcui_lexical_alignment",

                "relationship_semantics":
                    "terminology_alignment",
            }
        )

        continue


    semantic_variants = set()


    semantic_variants.update(
        drug_variants(
            drug_name
        )
    )


    for alias in rxnorm_original_terms.get(
        rxcui,
        set()
    ):

        semantic_variants.update(
            drug_variants(
                alias
            )
        )


    # Also pull ingredient names into semantic variants
    for ing_rxcui in rxcui_to_ingredients.get(rxcui, set()):

        for ing_name in ingredient_name_index.get(ing_rxcui, set()):
            semantic_variants.update(drug_variants(ing_name))


    semantic_matches = (
        find_mesh_matches(
            semantic_variants
        )
    )


    if len(
        semantic_matches
    ) == 1:

        mesh_ui = next(
            iter(
                semantic_matches
            )
        )


        drug_about_rows.append(
            {
                "rxcui":
                    rxcui,

                "mesh_ui":
                    mesh_ui,

                "mesh_name":
                    mesh_descriptor_names.get(
                        mesh_ui
                    ),

                "semantic_method":
                    (
                        "rxnorm_product_to_"
                        "ingredient_lexical_context"
                    ),

                "confidence":
                    0.85,

                "source":
                    (
                        "RxNorm product terminology "
                        "+ MeSH 2026"
                    ),

                "provenance":
                    "dose_form_strength_removed",

                "relationship_semantics":
                    (
                        "semantic_context_"
                        "not_concept_equivalence"
                    ),
            }
        )


    elif len(
        semantic_matches
    ) > 1:

        drug_ambiguous_rows.append(
            {
                "rxcui":
                    rxcui,

                "name":
                    drug_name,

                "candidate_count":
                    len(
                        semantic_matches
                    ),

                "clinical_use":
                    (
                        rxcui
                        in
                        clinical_drug_ids
                    ),
            }
        )


    else:

        drug_unmapped_rows.append(
            {
                "rxcui":
                    rxcui,

                "name":
                    drug_name,

                "concept_type":
                    drug.get(
                        "concept_type"
                    ),

                "clinical_use":
                    (
                        rxcui
                        in
                        clinical_drug_ids
                    ),
            }
        )


print()
print(
    "Drug direct mappings:",
    f"{len(drug_mapping_rows):,}"
)

print(
    "Drug semantic mappings:",
    f"{len(drug_about_rows):,}"
)


# =============================================================================
# LAB -> MESH
# =============================================================================

lab_mapping_rows = []
lab_about_rows = []
lab_unmapped_rows = []


for lab in loinc_concepts:

    loinc_code = normalize_loinc(
        lab.get(
            "loinc_code"
        )
    )


    if not loinc_code:
        continue


    direct_variants = (
        lab_name_variants(
            lab
        )
    )

    # NEW: add strict component-only variant (best for MeSH matching)
    direct_variants.update(
        lab_component_only_variants(lab)
    )


    direct_result = (
        select_unique_mesh_match(
            direct_variants
        )
    )


    if direct_result:

        lab_mapping_rows.append(
            {
                "loinc_code":
                    loinc_code,

                "mesh_ui":
                    direct_result[
                        "mesh_ui"
                    ],

                "mesh_name":
                    direct_result[
                        "mesh_name"
                    ],

                "mapping_method":
                    (
                        "loinc_test_"
                        + direct_result[
                            "method"
                        ]
                    ),

                "confidence":
                    direct_result[
                        "confidence"
                    ],

                "source":
                    "LOINC + MeSH 2026",

                "provenance":
                    (
                        "full_loinc_test_"
                        "lexical_alignment"
                    ),

                "relationship_semantics":
                    "terminology_alignment",
            }
        )


    component_variants = (
        lab_component_variants(
            lab
        )
    )

    component_variants.update(
        lab_component_only_variants(lab)
    )


    component_result = (
        select_unique_mesh_match(
            component_variants
        )
    )


    if component_result:

        lab_about_rows.append(
            {
                "loinc_code":
                    loinc_code,

                "mesh_ui":
                    component_result[
                        "mesh_ui"
                    ],

                "mesh_name":
                    component_result[
                        "mesh_name"
                    ],

                "semantic_method":
                    (
                        "loinc_component_"
                        "mesh_alignment"
                    ),

                "confidence":
                    0.90,

                "source":
                    (
                        "LOINC COMPONENT "
                        "+ MeSH 2026"
                    ),

                "provenance":
                    (
                        "loinc_analyte_"
                        "semantic_context"
                    ),

                "relationship_semantics":
                    (
                        "semantic_context_"
                        "not_concept_equivalence"
                    ),
            }
        )


    elif not direct_result:

        lab_unmapped_rows.append(
            {
                "loinc_code":
                    loinc_code,

                "name":
                    lab.get(
                        "name"
                    ),

                "component":
                    lab.get(
                        "component"
                    ),

                "system":
                    lab.get(
                        "system"
                    ),

                "class":
                    lab.get(
                        "class"
                    ),

                "clinical_use":
                    (
                        loinc_code
                        in
                        clinical_lab_ids
                    ),
            }
        )


unique_lab_about = {}


for row in lab_about_rows:

    unique_lab_about[
        (
            row[
                "loinc_code"
            ],

            row[
                "mesh_ui"
            ]
        )
    ] = row


lab_about_rows = list(
    unique_lab_about.values()
)


print()
print(
    "Lab direct mappings:",
    f"{len(lab_mapping_rows):,}"
)

print(
    "Lab semantic mappings:",
    f"{len(lab_about_rows):,}"
)


# =============================================================================
# LOAD MAPPINGS
# =============================================================================

neo4j_batch_write(

    """
    UNWIND $rows AS row

    MATCH (
        d:Disease {
            condition_id:
                row.condition_id
        }
    )

    MATCH (
        m:MeSHConcept {
            descriptor_ui:
                row.mesh_ui
        }
    )

    MERGE (
        d
    )-[r:MAPPED_TO]->(
        m
    )

    SET
        r.mapping_method =
            row.mapping_method,

        r.confidence =
            row.confidence,

        r.source =
            row.source,

        r.provenance =
            row.provenance,

        r.relationship_semantics =
            row.relationship_semantics
    """,

    disease_mapping_rows,

    "Disease -> MeSH MAPPED_TO"
)


neo4j_batch_write(

    """
    UNWIND $rows AS row

    MATCH (
        d:Drug {
            rxcui:
                row.rxcui
        }
    )

    MATCH (
        m:MeSHConcept {
            descriptor_ui:
                row.mesh_ui
        }
    )

    MERGE (
        d
    )-[r:MAPPED_TO]->(
        m
    )

    SET
        r.mapping_method =
            row.mapping_method,

        r.confidence =
            row.confidence,

        r.source =
            row.source,

        r.provenance =
            row.provenance,

        r.relationship_semantics =
            row.relationship_semantics
    """,

    drug_mapping_rows,

    "Drug -> MeSH MAPPED_TO"
)


neo4j_batch_write(

    """
    UNWIND $rows AS row

    MATCH (
        d:Drug {
            rxcui:
                row.rxcui
        }
    )

    MATCH (
        m:MeSHConcept {
            descriptor_ui:
                row.mesh_ui
        }
    )

    MERGE (
        d
    )-[r:ABOUT_MESH]->(
        m
    )

    SET
        r.semantic_method =
            row.semantic_method,

        r.confidence =
            row.confidence,

        r.source =
            row.source,

        r.provenance =
            row.provenance,

        r.relationship_semantics =
            row.relationship_semantics
    """,

    drug_about_rows,

    "Drug -> MeSH ABOUT_MESH"
)


neo4j_batch_write(

    """
    UNWIND $rows AS row

    MATCH (
        l:LabTest {
            loinc_code:
                row.loinc_code
        }
    )

    MATCH (
        m:MeSHConcept {
            descriptor_ui:
                row.mesh_ui
        }
    )

    MERGE (
        l
    )-[r:MAPPED_TO]->(
        m
    )

    SET
        r.mapping_method =
            row.mapping_method,

        r.confidence =
            row.confidence,

        r.source =
            row.source,

        r.provenance =
            row.provenance,

        r.relationship_semantics =
            row.relationship_semantics
    """,

    lab_mapping_rows,

    "Lab -> MeSH MAPPED_TO"
)


neo4j_batch_write(

    """
    UNWIND $rows AS row

    MATCH (
        l:LabTest {
            loinc_code:
                row.loinc_code
        }
    )

    MATCH (
        m:MeSHConcept {
            descriptor_ui:
                row.mesh_ui
        }
    )

    MERGE (
        l
    )-[r:ABOUT_MESH]->(
        m
    )

    SET
        r.semantic_method =
            row.semantic_method,

        r.confidence =
            row.confidence,

        r.source =
            row.source,

        r.provenance =
            row.provenance,

        r.relationship_semantics =
            row.relationship_semantics
    """,

    lab_about_rows,

    "Lab -> MeSH ABOUT_MESH"
)


# =============================================================================
# PATIENT KNOWLEDGE BRIDGE
# =============================================================================

stage(
    "SUBPHASE 10H — PATIENT KNOWLEDGE BRIDGE"
)


neo4j_batch_write(

    """
    UNWIND $rows AS row

    MATCH (
        p:Patient {
            patient_id:
                row.patient_id
        }
    )

    MATCH (
        d:Disease {
            condition_id:
                row.condition_id
        }
    )

    MERGE (
        p
    )-[r:HAS_CONDITION]->(
        d
    )

    SET
        r.diagnosis_count =
            row.diagnosis_count,

        r.first_diagnosed =
            row.first_diagnosed,

        r.latest_diagnosed =
            row.latest_diagnosed,

        r.latest_stop =
            row.latest_stop,

        r.source =
            'PostgreSQL diagnosis',

        r.provenance =
            'synthetic_clinical_record'
    """,

    patient_diseases,

    "HAS_CONDITION"
)


neo4j_batch_write(

    """
    UNWIND $rows AS row

    MATCH (
        p:Patient {
            patient_id:
                row.patient_id
        }
    )

    MATCH (
        d:Drug {
            rxcui:
                row.rxcui
        }
    )

    MERGE (
        p
    )-[r:HAS_MEDICATION]->(
        d
    )

    SET
        r.prescription_count =
            row.prescription_count,

        r.first_prescribed =
            row.first_prescribed,

        r.latest_prescribed =
            row.latest_prescribed,

        r.latest_stop =
            row.latest_stop,

        r.source =
            'PostgreSQL prescription',

        r.provenance =
            'synthetic_clinical_record'
    """,

    patient_drugs,

    "HAS_MEDICATION"
)


neo4j_batch_write(

    """
    UNWIND $rows AS row

    MATCH (
        p:Patient {
            patient_id:
                row.patient_id
        }
    )

    MATCH (
        l:LabTest {
            loinc_code:
                row.loinc_code
        }
    )

    MERGE (
        p
    )-[r:HAS_LAB]->(
        l
    )

    SET
        r.result_count =
            row.result_count,

        r.latest_result_time =
            row.latest_result_time,

        r.source =
            'PostgreSQL lab_result',

        r.provenance =
            'synthetic_clinical_record'
    """,

    patient_labs,

    "HAS_LAB"
)


print()
print(
    "Patient knowledge bridge: PASS"
)


# =============================================================================
# OBSERVATIONAL DISEASE -> MEDICATION RELATIONSHIPS
# =============================================================================

stage(
    "SUBPHASE 10E — OBSERVATIONAL CLINICAL RELATIONSHIPS"
)


with neo4j_driver.session(
    database=NEO4J_DATABASE
) as session:

    result = session.run(
        """
        MATCH
            (p:Patient)
            -[:HAS_CONDITION]->
            (d:Disease)

        MATCH
            (p)
            -[:HAS_MEDICATION]->
            (drug:Drug)

        WITH
            d,
            drug,
            count(DISTINCT p)
                AS patient_count

        MERGE (
            d
        )-[r:OBSERVED_WITH_MEDICATION]->(
            drug
        )

        SET
            r.patient_count =
                patient_count,

            r.source =
                'synthetic patient co-occurrence',

            r.provenance =
                'PostgreSQL patient history',

            r.relationship_semantics =
                'observational_not_treatment_recommendation'

        RETURN
            count(r)
            AS value
        """
    ).single()


condition_drug_count = (
    result[
        "value"
    ]
    if result
    else 0
)


print(
    "OBSERVED_WITH_MEDICATION:",
    f"{condition_drug_count:,}"
)


# =============================================================================
# OBSERVATIONAL DISEASE -> LAB RELATIONSHIPS
# =============================================================================

with neo4j_driver.session(
    database=NEO4J_DATABASE
) as session:

    result = session.run(
        """
        MATCH
            (p:Patient)
            -[:HAS_CONDITION]->
            (d:Disease)

        MATCH
            (p)
            -[:HAS_LAB]->
            (lab:LabTest)

        WITH
            d,
            lab,
            count(DISTINCT p)
                AS patient_count

        MERGE (
            d
        )-[r:ASSOCIATED_WITH]->(
            lab
        )

        SET
            r.patient_count =
                patient_count,

            r.source =
                'synthetic patient co-occurrence',

            r.provenance =
                'PostgreSQL patient history',

            r.relationship_semantics =
                'observational_association'

        RETURN
            count(r)
            AS value
        """
    ).single()


condition_lab_count = (
    result[
        "value"
    ]
    if result
    else 0
)


print(
    "ASSOCIATED_WITH:",
    f"{condition_lab_count:,}"
)


# =============================================================================
# RXNORM NEIGHBOUR SEMANTIC ENRICHMENT
# =============================================================================

stage(
    "SUBPHASE 10N — SEMANTIC CONTEXT ENRICHMENT"
)


with neo4j_driver.session(
    database=NEO4J_DATABASE
) as session:

    record = session.run(
        """
        MATCH
            (drug:Drug)
            -[:RELATED_TO]-
            (related:Drug)
            -[:MAPPED_TO]->
            (mesh:MeSHConcept)

        WHERE NOT EXISTS {
            MATCH
                (drug)
                -[:MAPPED_TO]->
                (:MeSHConcept)
        }

        WITH
            drug,
            mesh,
            count(*)
                AS evidence_count

        MERGE (
            drug
        )-[r:ABOUT_MESH]->(
            mesh
        )

        ON CREATE SET
            r.semantic_method =
                'rxnorm_related_concept_mesh_context',

            r.confidence =
                0.70,

            r.source =
                'RxNorm RELATED_TO + MeSH 2026',

            r.provenance =
                'one_hop_rxnorm_semantic_expansion',

            r.relationship_semantics =
                'semantic_context_not_concept_equivalence',

            r.evidence_count =
                evidence_count

        RETURN
            count(r)
            AS value
        """
    ).single()


drug_about_count = (
    record[
        "value"
    ]
    if record
    else 0
)


print(
    "Drug ABOUT_MESH links:",
    f"{drug_about_count:,}"
)


# =============================================================================
# VALIDATION
# =============================================================================

stage(
    "SUBPHASE 10J — GRAPH QUALITY + INTEGRITY VALIDATION"
)


validation_results = []


def validate(
    name,
    query,
    expected_zero=True,
    category="structural"
):

    value = neo4j_scalar(
        query
    )


    if expected_zero:

        status = (
            "PASS"
            if value == 0
            else "REVIEW"
        )

    else:

        status = (
            "PASS"
            if value
            and value > 0
            else "REVIEW"
        )


    validation_results.append(
        {
            "category":
                category,

            "validation":
                name,

            "value":
                value,

            "status":
                status,
        }
    )


    print(
        f"{name:<60}",
        f"{value:<10}",
        status
    )


validate(
    "Patient nodes exist",
    """
    MATCH (n:Patient)
    RETURN count(n) AS value
    """,
    False
)


validate(
    "Disease nodes exist",
    """
    MATCH (n:Disease)
    RETURN count(n) AS value
    """,
    False
)


validate(
    "Drug nodes exist",
    """
    MATCH (n:Drug)
    RETURN count(n) AS value
    """,
    False
)


validate(
    "LabTest nodes exist",
    """
    MATCH (n:LabTest)
    RETURN count(n) AS value
    """,
    False
)


validate(
    "MeSH nodes exist",
    """
    MATCH (n:MeSHConcept)
    RETURN count(n) AS value
    """,
    False
)


validate(
    "Duplicate Patient IDs",
    """
    MATCH (n:Patient)
    WITH n.patient_id AS id, count(*) AS c
    WHERE c > 1
    RETURN count(*) AS value
    """
)


validate(
    "Duplicate Disease IDs",
    """
    MATCH (n:Disease)
    WITH n.condition_id AS id, count(*) AS c
    WHERE c > 1
    RETURN count(*) AS value
    """
)


validate(
    "Duplicate Drug RXCUIs",
    """
    MATCH (n:Drug)
    WITH n.rxcui AS id, count(*) AS c
    WHERE c > 1
    RETURN count(*) AS value
    """
)


validate(
    "Duplicate LOINC codes",
    """
    MATCH (n:LabTest)
    WITH n.loinc_code AS id, count(*) AS c
    WHERE c > 1
    RETURN count(*) AS value
    """
)


validate(
    "Duplicate MeSH descriptor UIs",
    """
    MATCH (n:MeSHConcept)
    WITH n.descriptor_ui AS id, count(*) AS c
    WHERE c > 1
    RETURN count(*) AS value
    """
)


validate(
    "Patient nodes without ID",
    """
    MATCH (n:Patient)
    WHERE n.patient_id IS NULL
    RETURN count(n) AS value
    """
)


validate(
    "Disease nodes without ID",
    """
    MATCH (n:Disease)
    WHERE n.condition_id IS NULL
    RETURN count(n) AS value
    """
)


validate(
    "Drug nodes without RXCUI",
    """
    MATCH (n:Drug)
    WHERE n.rxcui IS NULL
    RETURN count(n) AS value
    """
)


validate(
    "LabTest nodes without LOINC",
    """
    MATCH (n:LabTest)
    WHERE n.loinc_code IS NULL
    RETURN count(n) AS value
    """
)


validate(
    "MeSH nodes without UI",
    """
    MATCH (n:MeSHConcept)
    WHERE n.descriptor_ui IS NULL
    RETURN count(n) AS value
    """
)


validate(
    "MAPPED_TO missing mapping_method",
    """
    MATCH ()-[r:MAPPED_TO]->()
    WHERE r.mapping_method IS NULL    RETURN count(r) AS value
    """,
    True,
    "semantic"
)


validate(
    "MAPPED_TO missing confidence",
    """
    MATCH ()-[r:MAPPED_TO]->()
    WHERE r.confidence IS NULL
    RETURN count(r) AS value
    """,
    True,
    "semantic"
)


validate(
    "MAPPED_TO missing provenance",
    """
    MATCH ()-[r:MAPPED_TO]->()
    WHERE r.provenance IS NULL
    RETURN count(r) AS value
    """,
    True,
    "semantic"
)


validate(
    "ABOUT_MESH missing semantic method",
    """
    MATCH ()-[r:ABOUT_MESH]->()
    WHERE r.semantic_method IS NULL
    RETURN count(r) AS value
    """,
    True,
    "semantic"
)


validate(
    "ABOUT_MESH missing confidence",
    """
    MATCH ()-[r:ABOUT_MESH]->()
    WHERE r.confidence IS NULL
    RETURN count(r) AS value
    """,
    True,
    "semantic"
)


validate(
    "ABOUT_MESH missing provenance",
    """
    MATCH ()-[r:ABOUT_MESH]->()
    WHERE r.provenance IS NULL
    RETURN count(r) AS value
    """,
    True,
    "semantic"
)


validate(
    "ABOUT_MESH wrong semantic disclaimer",
    """
    MATCH ()-[r:ABOUT_MESH]->()

    WHERE
        r.relationship_semantics <>
        'semantic_context_not_concept_equivalence'

    RETURN count(r) AS value
    """,
    True,
    "semantic"
)


validate(
    "Observed medication edges missing evidence",
    """
    MATCH
        (:Disease)
        -[r:OBSERVED_WITH_MEDICATION]->
        (:Drug)

    WHERE r.patient_count IS NULL

    RETURN count(r) AS value
    """,
    True,
    "semantic"
)


validate(
    "Observed medication edges missing semantics",
    """
    MATCH
        (:Disease)
        -[r:OBSERVED_WITH_MEDICATION]->
        (:Drug)

    WHERE
        r.relationship_semantics IS NULL
        OR
        r.relationship_semantics <>
        'observational_not_treatment_recommendation'

    RETURN count(r) AS value
    """,
    True,
    "semantic"
)


validate(
    "Disease-lab edges missing evidence",
    """
    MATCH
        (:Disease)
        -[r:ASSOCIATED_WITH]->
        (:LabTest)

    WHERE r.patient_count IS NULL

    RETURN count(r) AS value
    """,
    True,
    "semantic"
)


validate(
    "Disease-lab edges missing semantics",
    """
    MATCH
        (:Disease)
        -[r:ASSOCIATED_WITH]->
        (:LabTest)

    WHERE
        r.relationship_semantics IS NULL
        OR
        r.relationship_semantics <>
        'observational_association'

    RETURN count(r) AS value
    """,
    True,
    "semantic"
)


validate(
    "HAS_CONDITION missing aggregate metadata",
    """
    MATCH
        (:Patient)
        -[r:HAS_CONDITION]->
        (:Disease)

    WHERE r.diagnosis_count IS NULL

    RETURN count(r) AS value
    """,
    True,
    "clinical_bridge"
)


validate(
    "HAS_MEDICATION missing aggregate metadata",
    """
    MATCH
        (:Patient)
        -[r:HAS_MEDICATION]->
        (:Drug)

    WHERE r.prescription_count IS NULL

    RETURN count(r) AS value
    """,
    True,
    "clinical_bridge"
)


validate(
    "HAS_LAB missing aggregate metadata",
    """
    MATCH
        (:Patient)
        -[r:HAS_LAB]->
        (:LabTest)

    WHERE r.result_count IS NULL

    RETURN count(r) AS value
    """,
    True,
    "clinical_bridge"
)


validate(
    "RxNorm RELATED_TO missing semantics",
    """
    MATCH
        (:Drug)
        -[r:RELATED_TO]->
        (:Drug)

    WHERE
        r.relationship_type IS NULL
        AND
        r.relationship_attribute IS NULL

    RETURN count(r) AS value
    """,
    True,
    "semantic"
)


validate(
    "Self-referential RxNorm drug edges",
    """
    MATCH
        (d:Drug)
        -[:RELATED_TO]->
        (d)

    RETURN count(*) AS value
    """,
    True,
    "semantic"
)


validate(
    "Clinically used diseases exist",
    """
    MATCH
        (:Patient)
        -[:HAS_CONDITION]->
        (d:Disease)

    RETURN
        count(DISTINCT d)
        AS value
    """,
    False,
    "clinical_bridge"
)


validate(
    "Clinically used drugs exist",
    """
    MATCH
        (:Patient)
        -[:HAS_MEDICATION]->
        (d:Drug)

    RETURN
        count(DISTINCT d)
        AS value
    """,
    False,
    "clinical_bridge"
)


validate(
    "Clinically used labs exist",
    """
    MATCH
        (:Patient)
        -[:HAS_LAB]->
        (l:LabTest)

    RETURN
        count(DISTINCT l)
        AS value
    """,
    False,
    "clinical_bridge"
)


# =============================================================================
# RESERVED AUTHORITATIVE RELATIONSHIPS
# =============================================================================

try:

    records = neo4j_records(
        """
        SHOW RELATIONSHIP TYPES
        YIELD relationshipType
        RETURN relationshipType
        """
    )

except Exception:

    records = neo4j_records(
        """
        CALL db.relationshipTypes()
        YIELD relationshipType
        RETURN relationshipType
        """
    )


relationship_types = {
    row[
        "relationshipType"
    ]
    for row
    in records
}


if (
    "TREATED_BY"
    in relationship_types
):

    treated_count = neo4j_scalar(
        """
        MATCH ()-[r:TREATED_BY]->()
        RETURN count(r) AS value
        """
    )

else:

    treated_count = 0


if (
    "INTERACTS_WITH"
    in relationship_types
):

    interaction_count = neo4j_scalar(
        """
        MATCH ()-[r:INTERACTS_WITH]->()
        RETURN count(r) AS value
        """
    )

else:

    interaction_count = 0


print()
print(
    "Reserved authoritative relationships:"
)

print(
    "TREATED_BY:",
    treated_count,
    "(expected 0 before authoritative treatment evidence)"
)

print(
    "INTERACTS_WITH:",
    interaction_count,
    "(expected 0 before authoritative DDI evidence)"
)


# =============================================================================
# COVERAGE
# =============================================================================

stage(
    "SUBPHASE 10N — DIRECT + SEMANTIC CLINICAL COVERAGE"
)


coverage_config = {
    "Disease":
        "HAS_CONDITION",

    "Drug":
        "HAS_MEDICATION",

    "LabTest":
        "HAS_LAB",
}


mapping_coverage = []


for (
    label,
    patient_relationship
) in coverage_config.items():

    global_record = neo4j_records(
        f"""
        MATCH (n:`{label}`)

        WITH
            count(n)
                AS total,

            count(
                CASE
                    WHEN EXISTS {{
                        MATCH
                            (n)
                            -[:MAPPED_TO]->
                            (:MeSHConcept)
                    }}
                    THEN 1
                END
            )
                AS mapped,

            count(
                CASE
                    WHEN EXISTS {{
                        MATCH
                            (n)
                            -[:MAPPED_TO|ABOUT_MESH]->
                            (:MeSHConcept)
                    }}
                    THEN 1
                END
            )
                AS semantic

        RETURN
            total,
            mapped,
            semantic
        """
    )[0]


    clinical_record = neo4j_records(
        f"""
        MATCH
            (:Patient)
            -[:`{patient_relationship}`]->
            (n:`{label}`)

        WITH DISTINCT n

        WITH
            count(n)
                AS clinical_total,

            count(
                CASE
                    WHEN EXISTS {{
                        MATCH
                            (n)
                            -[:MAPPED_TO]->
                            (:MeSHConcept)
                    }}
                    THEN 1
                END
            )
                AS clinical_mapped,

            count(
                CASE
                    WHEN EXISTS {{
                        MATCH
                            (n)
                            -[:MAPPED_TO|ABOUT_MESH]->
                            (:MeSHConcept)
                    }}
                    THEN 1
                END
            )
                AS clinical_semantic

        RETURN
            clinical_total,
            clinical_mapped,
            clinical_semantic
        """
    )[0]


    total = global_record[
        "total"
    ]

    mapped = global_record[
        "mapped"
    ]

    global_semantic = global_record[
        "semantic"
    ]

    clinical_total = clinical_record[
        "clinical_total"
    ]

    clinical_mapped = clinical_record[
        "clinical_mapped"
    ]

    clinical_semantic = clinical_record[
        "clinical_semantic"
    ]


    row = {
        "entity_type":
            label,

        "global_total":
            total,

        "global_mapped":
            mapped,

        "global_coverage_percent":
            safe_percentage(
                mapped,
                total
            ),

        "global_semantic_total":
            global_semantic,

        "global_semantic_coverage_percent":
            safe_percentage(
                global_semantic,
                total
            ),

        "clinical_use_total":
            clinical_total,

        "clinical_use_mapped":
            clinical_mapped,

        "clinical_direct_coverage_percent":
            safe_percentage(
                clinical_mapped,
                clinical_total
            ),

        "clinical_semantic_total":
            clinical_semantic,

        "clinical_semantic_coverage_percent":
            safe_percentage(
                clinical_semantic,
                clinical_total
            ),
    }


    mapping_coverage.append(
        row
    )


    print()
    print(
        label
    )

    print(
        "  Global direct     :",
        f"{mapped:,}/{total:,}",
        f"({row['global_coverage_percent']:.2f}%)"
    )

    print(
        "  Global semantic   :",
        f"{global_semantic:,}/{total:,}",
        f"({row['global_semantic_coverage_percent']:.2f}%)"
    )

    print(
        "  Clinical direct   :",
        f"{clinical_mapped:,}/{clinical_total:,}",
        f"({row['clinical_direct_coverage_percent']:.2f}%)"
    )

    print(
        "  Clinical semantic :",
        f"{clinical_semantic:,}/{clinical_total:,}",
        f"({row['clinical_semantic_coverage_percent']:.2f}%)"
    )


# =============================================================================
# COVERAGE STATUS
# =============================================================================

for row in mapping_coverage:

    if (
        row[
            "clinical_use_total"
        ]
        == 0
    ):

        status = "REVIEW"

    elif (
        row[
            "clinical_use_mapped"
        ]
        == 0
        and
        row[
            "clinical_semantic_total"
        ]
        == 0
    ):

        status = "REVIEW"

    elif (
        row[
            "clinical_use_mapped"
        ]
        == 0
        and
        row[
            "clinical_semantic_total"
        ]
        > 0
    ):

        status = "INFO"

    else:

        status = "PASS"


    validation_results.append(
        {
            "category":
                "coverage",

            "validation":
                (
                    row[
                        "entity_type"
                    ]
                    + " clinical semantic coverage"
                ),

            "value":
                row[
                    "clinical_semantic_coverage_percent"
                ],

            "status":
                status,
        }
    )


# =============================================================================
# CLINICAL DIAGNOSTICS
# =============================================================================

stage(
    "SUBPHASE 10N — CLINICAL MAPPING DIAGNOSTICS"
)


clinical_drug_diagnostics = neo4j_records(
    """
    MATCH
        (p:Patient)
        -[:HAS_MEDICATION]->
        (d:Drug)

    WITH
        d,
        count(DISTINCT p)
            AS patient_count

    OPTIONAL MATCH
        (d)
        -[direct:MAPPED_TO]->
        (direct_mesh:MeSHConcept)

    WITH
        d,
        patient_count,

        collect(
            DISTINCT {
                mesh_ui:
                    direct_mesh.descriptor_ui,

                mesh_name:
                    direct_mesh.name,

                method:
                    direct.mapping_method,

                confidence:
                    direct.confidence
            }
        )
            AS direct_mappings

    OPTIONAL MATCH
        (d)
        -[semantic:ABOUT_MESH]->
        (semantic_mesh:MeSHConcept)

    WITH
        d,
        patient_count,
        direct_mappings,

        collect(
            DISTINCT {
                mesh_ui:
                    semantic_mesh.descriptor_ui,

                mesh_name:
                    semantic_mesh.name,

                method:
                    semantic.semantic_method,

                confidence:
                    semantic.confidence
            }
        )
            AS semantic_mappings

    RETURN
        d.rxcui
            AS rxcui,

        d.name
            AS drug_name,

        patient_count,

        size(
            [
                x IN direct_mappings
                WHERE x.mesh_ui IS NOT NULL
            ]
        )
            AS direct_mapping_count,

        size(
            [
                x IN semantic_mappings
                WHERE x.mesh_ui IS NOT NULL
            ]
        )
            AS semantic_mapping_count,

        direct_mappings,

        semantic_mappings

    ORDER BY
        patient_count DESC,
        drug_name
    """
)


clinical_lab_diagnostics = neo4j_records(
    """
    MATCH
        (p:Patient)
        -[:HAS_LAB]->
        (l:LabTest)

    WITH
        l,
        count(DISTINCT p)
            AS patient_count

    OPTIONAL MATCH
        (l)
        -[direct:MAPPED_TO]->
        (direct_mesh:MeSHConcept)

    WITH
        l,
        patient_count,

        collect(
            DISTINCT {
                mesh_ui:
                    direct_mesh.descriptor_ui,

                mesh_name:
                    direct_mesh.name,

                method:
                    direct.mapping_method,

                confidence:
                    direct.confidence
            }
        )
            AS direct_mappings

    OPTIONAL MATCH
        (l)
        -[semantic:ABOUT_MESH]->
        (semantic_mesh:MeSHConcept)

    WITH
        l,
        patient_count,
        direct_mappings,

        collect(
            DISTINCT {
                mesh_ui:
                    semantic_mesh.descriptor_ui,

                mesh_name:
                    semantic_mesh.name,

                method:
                    semantic.semantic_method,

                confidence:
                    semantic.confidence
            }
        )
            AS semantic_mappings

    RETURN
        l.loinc_code
            AS loinc_code,

        l.name
            AS lab_name,

        l.component
            AS component,

        l.system
            AS system,

        patient_count,

        size(
            [
                x IN direct_mappings
                WHERE x.mesh_ui IS NOT NULL
            ]
        )
            AS direct_mapping_count,

        size(
            [
                x IN semantic_mappings
                WHERE x.mesh_ui IS NOT NULL
            ]
        )
            AS semantic_mapping_count,

        direct_mappings,

        semantic_mappings

    ORDER BY
        patient_count DESC,
        lab_name
    """
)


clinical_disease_diagnostics = neo4j_records(
    """
    MATCH
        (p:Patient)
        -[:HAS_CONDITION]->
        (d:Disease)

    WITH
        d,
        count(DISTINCT p)
            AS patient_count

    OPTIONAL MATCH
        (d)
        -[mapping:MAPPED_TO]->
        (mesh:MeSHConcept)

    WITH
        d,
        patient_count,

        collect(
            DISTINCT {
                mesh_ui:
                    mesh.descriptor_ui,

                mesh_name:
                    mesh.name,

                method:
                    mapping.mapping_method,

                confidence:
                    mapping.confidence
            }
        )
            AS mappings

    RETURN
        d.condition_id
            AS condition_id,

        d.source_code
            AS source_code,

        d.name
            AS disease_name,

        patient_count,

        size(
            [
                x IN mappings
                WHERE x.mesh_ui IS NOT NULL
            ]
        )
            AS mapping_count,

        mappings

    ORDER BY
        patient_count DESC,
        disease_name
    """
)


pd.DataFrame(
    serialize_nested_rows(
        clinical_drug_diagnostics
    )
).to_csv(
    DRUG_DIAGNOSTICS_FILE,
    index=False
)


pd.DataFrame(
    serialize_nested_rows(
        clinical_lab_diagnostics
    )
).to_csv(
    LAB_DIAGNOSTICS_FILE,
    index=False
)


pd.DataFrame(
    serialize_nested_rows(
        clinical_disease_diagnostics
    )
).to_csv(
    DISEASE_DIAGNOSTICS_FILE,
    index=False
)


pd.DataFrame(
    disease_unmapped_rows
).to_csv(
    UNMAPPED_DISEASE_FILE,
    index=False
)


pd.DataFrame(
    drug_unmapped_rows
).to_csv(
    UNMAPPED_DRUG_FILE,
    index=False
)


pd.DataFrame(
    lab_unmapped_rows
).to_csv(
    UNMAPPED_LAB_FILE,
    index=False
)


# =============================================================================
# MAPPING METHOD STATISTICS
# =============================================================================

mapping_method_statistics = neo4j_records(
    """
    MATCH
        (concept)
        -[mapping:MAPPED_TO]->
        (:MeSHConcept)

    RETURN
        labels(concept)[0]
            AS entity_type,

        mapping.mapping_method
            AS mapping_method,

        count(mapping)
            AS relationship_count

    UNION ALL

    MATCH
        (concept)
        -[mapping:ABOUT_MESH]->
        (:MeSHConcept)

    RETURN
        labels(concept)[0]
            AS entity_type,

        mapping.semantic_method
            AS mapping_method,

        count(mapping)
            AS relationship_count
    """
)


pd.DataFrame(
    mapping_method_statistics
).to_csv(
    MAPPING_METHOD_FILE,
    index=False
)


# =============================================================================
# REPRESENTATIVE PATIENT
# =============================================================================

patient_candidate = neo4j_records(
    """
    MATCH (p:Patient)

    OPTIONAL MATCH
        (p)
        -[:HAS_CONDITION]->
        (d:Disease)

    OPTIONAL MATCH
        (p)
        -[:HAS_MEDICATION]->
        (drug:Drug)

    OPTIONAL MATCH
        (p)
        -[:HAS_LAB]->
        (lab:LabTest)

    WITH
        p,

        count(
            DISTINCT d
        )
            AS condition_count,

        count(
            DISTINCT drug
        )
            AS drug_count,

        count(
            DISTINCT lab
        )
            AS lab_count

    WHERE
        condition_count > 0
        AND
        drug_count > 0
        AND
        lab_count > 0

    RETURN
        p.patient_id
            AS patient_id,

        condition_count,
        drug_count,
        lab_count,

        (
            condition_count
            + drug_count
            + lab_count
        )
            AS total_context

    ORDER BY
        total_context DESC

    LIMIT 1
    """
)


TEST_PATIENT_ID = (
    patient_candidate[
        0
    ][
        "patient_id"
    ]
    if patient_candidate
    else None
)


print()
print(
    "Representative patient:",
    TEST_PATIENT_ID
)


# =============================================================================
# GRAPH-RAG / CLINICAL QUERIES
# =============================================================================

stage(
    "SUBPHASE 10K — CLINICAL + GRAPH-RAG QUERY VALIDATION"
)


CLINICAL_QUERIES = {

    "patient_conditions":
    """
    MATCH
        (
            p:Patient {
                patient_id:
                    $patient_id
            }
        )
        -[r:HAS_CONDITION]->
        (d:Disease)

    RETURN
        d.name
            AS condition,

        d.source_code
            AS code,

        r.diagnosis_count
            AS diagnosis_count

    ORDER BY
        diagnosis_count DESC

    LIMIT 100
    """,


    "patient_medications":
    """
    MATCH
        (
            p:Patient {
                patient_id:
                    $patient_id
            }
        )
        -[r:HAS_MEDICATION]->
        (d:Drug)

    RETURN
        d.name
            AS medication,

        d.rxcui
            AS rxcui,

        r.prescription_count
            AS prescription_count

    ORDER BY
        prescription_count DESC

    LIMIT 100
    """,


    "patient_labs":
    """
    MATCH
        (
            p:Patient {
                patient_id:
                    $patient_id
            }
        )
        -[r:HAS_LAB]->
        (l:LabTest)

    RETURN
        l.name
            AS lab_test,

        l.loinc_code
            AS loinc_code,

        l.component
            AS component,

        r.result_count
            AS result_count

    ORDER BY
        result_count DESC

    LIMIT 100
    """,


    "condition_to_drugs":
    """
    MATCH
        (
            p:Patient {
                patient_id:
                    $patient_id
            }
        )
        -[:HAS_CONDITION]->
        (d:Disease)

    MATCH
        (d)
        -[r:OBSERVED_WITH_MEDICATION]->
        (drug:Drug)

    RETURN
        d.name
            AS condition,

        drug.name
            AS medication,

        r.patient_count
            AS observational_support

    ORDER BY
        observational_support DESC

    LIMIT 100
    """,


    "condition_to_labs":
    """
    MATCH
        (
            p:Patient {
                patient_id:
                    $patient_id
            }
        )
        -[:HAS_CONDITION]->
        (d:Disease)

    MATCH
        (d)
        -[r:ASSOCIATED_WITH]->
        (lab:LabTest)

    RETURN
        d.name
            AS condition,

        lab.name
            AS lab_test,

        r.patient_count
            AS observational_support

    ORDER BY
        observational_support DESC

    LIMIT 100
    """,


    "patient_mesh_context":
    """
    MATCH
        (
            p:Patient {
                patient_id:
                    $patient_id
            }
        )
        -[:HAS_CONDITION]->
        (d:Disease)

    OPTIONAL MATCH
        (d)
        -[:MAPPED_TO]->
        (mesh:MeSHConcept)

    RETURN DISTINCT
        d.name
            AS condition,

        mesh.name
            AS mesh_context

    LIMIT 100
    """,


    "patient_condition_lab_context":
    """
    MATCH
        (
            p:Patient {
                patient_id:
                    $patient_id
            }
        )
        -[:HAS_CONDITION]->
        (d:Disease)

    OPTIONAL MATCH
        (d)
        -[association:ASSOCIATED_WITH]->
        (lab:LabTest)

    OPTIONAL MATCH
        (lab)
        -[:MAPPED_TO|ABOUT_MESH]->
        (mesh:MeSHConcept)

    RETURN
        d.name
            AS condition,

        lab.name
            AS lab_test,

        association.patient_count
            AS observational_support,

        mesh.name
            AS mesh_context

    ORDER BY
        observational_support DESC

    LIMIT 100
    """,


    "patient_condition_mesh_hierarchy":
    """
    MATCH
        (
            p:Patient {
                patient_id:
                    $patient_id
            }
        )
        -[:HAS_CONDITION]->
        (d:Disease)

    MATCH
        (d)
        -[:MAPPED_TO]->
        (mesh:MeSHConcept)

    OPTIONAL MATCH
        (mesh)
        -[:BROADER_THAN*1..2]->
        (parent:MeSHConcept)

    RETURN DISTINCT
        d.name
            AS condition,

        mesh.name
            AS mesh_concept,

        parent.name
            AS broader_context

    LIMIT 100
    """,


    "patient_condition_medication_context":
    """
    MATCH
        (
            p:Patient {
                patient_id:
                    $patient_id
            }
        )
        -[:HAS_CONDITION]->
        (d:Disease)

    OPTIONAL MATCH
        (d)
        -[obs:OBSERVED_WITH_MEDICATION]->
        (drug:Drug)

    OPTIONAL MATCH
        (p)
        -[med:HAS_MEDICATION]->
        (drug)

    OPTIONAL MATCH
        (drug)
        -[direct:MAPPED_TO]->
        (direct_mesh:MeSHConcept)

    OPTIONAL MATCH
        (drug)
        -[context:ABOUT_MESH]->
        (semantic_mesh:MeSHConcept)

    RETURN
        d.name
            AS condition,

        drug.name
            AS medication,

        obs.patient_count
            AS observational_support,

        med.prescription_count
            AS patient_prescription_count,

        direct_mesh.name
            AS direct_mesh_concept,

        direct.mapping_method
            AS direct_mapping_method,

        semantic_mesh.name
            AS semantic_mesh_context,

        context.semantic_method
            AS semantic_method

    ORDER BY
        observational_support DESC

    LIMIT 100
    """,


    "patient_semantic_retrieval_context":
    """
    MATCH
        (
            p:Patient {
                patient_id:
                    $patient_id
            }
        )

    OPTIONAL MATCH
        (p)
        -[:HAS_CONDITION]->
        (d:Disease)

    OPTIONAL MATCH
        (d)
        -[:MAPPED_TO]->
        (dmesh:MeSHConcept)

    OPTIONAL MATCH
        (p)
        -[:HAS_MEDICATION]->
        (drug:Drug)

    OPTIONAL MATCH
        (drug)
        -[:MAPPED_TO|ABOUT_MESH]->
        (drugmesh:MeSHConcept)

    OPTIONAL MATCH
        (p)
        -[:HAS_LAB]->
        (lab:LabTest)

    OPTIONAL MATCH
        (lab)
        -[:MAPPED_TO|ABOUT_MESH]->
        (labmesh:MeSHConcept)

    RETURN DISTINCT
        d.name
            AS condition,

        dmesh.name
            AS condition_mesh,

        drug.name
            AS medication,

        drugmesh.name
            AS drug_mesh_context,

        lab.name
            AS lab_test,

        labmesh.name
            AS lab_mesh_context

    LIMIT 150
    """
}


clinical_query_results = []


if TEST_PATIENT_ID:

    with neo4j_driver.session(
        database=NEO4J_DATABASE
    ) as session:

        for (
            query_name,
            query
        ) in CLINICAL_QUERIES.items():

            start = time.perf_counter()


            records = list(
                session.run(
                    query,
                    patient_id=
                        TEST_PATIENT_ID
                )
            )


            elapsed_ms = (
                time.perf_counter()
                - start
            ) * 1000


            clinical_query_results.append(
                {
                    "query":
                        query_name,

                    "rows":
                        len(
                            records
                        ),

                    "elapsed_ms":
                        elapsed_ms,

                    "status":
                        "PASS",
                }
            )


            print(
                f"{query_name:<42}",
                f"rows={len(records):<6}",
                f"time={elapsed_ms:.3f} ms",
                "PASS"
            )


# =============================================================================
# BENCHMARKING
# =============================================================================

stage(
    "SUBPHASE 10L — GRAPH QUERY PERFORMANCE VALIDATION"
)


benchmark_results = []


if TEST_PATIENT_ID:

    for (
        query_name,
        query
    ) in CLINICAL_QUERIES.items():

        times = []
        result_counts = []


        with neo4j_driver.session(
            database=NEO4J_DATABASE
        ) as session:

            list(
                session.run(
                    query,
                    patient_id=
                        TEST_PATIENT_ID
                )
            )


            for run_number in range(
                BENCHMARK_RUNS
            ):

                start = (
                    time.perf_counter()
                )


                records = list(
                    session.run(
                        query,
                        patient_id=
                            TEST_PATIENT_ID
                    )
                )


                elapsed_ms = (
                    time.perf_counter()
                    - start
                ) * 1000


                times.append(
                    elapsed_ms
                )

                result_counts.append(
                    len(
                        records
                    )
                )


        benchmark_row = {
            "query":
                query_name,

            "runs":
                BENCHMARK_RUNS,

            "mean_ms":
                statistics.mean(
                    times
                ),

            "median_ms":
                statistics.median(
                    times
                ),

            "min_ms":
                min(
                    times
                ),

            "max_ms":
                max(
                    times
                ),

            "stddev_ms":
                (
                    statistics.stdev(
                        times
                    )
                    if len(
                        times
                    ) > 1
                    else 0.0
                ),

            "result_count":
                int(
                    statistics.median(
                        result_counts
                    )
                ),
        }


        benchmark_results.append(
            benchmark_row
        )


        print(
            f"{query_name:<42}",
            f"median={benchmark_row['median_ms']:.3f} ms",
            f"mean={benchmark_row['mean_ms']:.3f} ms",
            f"std={benchmark_row['stddev_ms']:.3f}"
        )


# =============================================================================
# GRAPH STATISTICS
# =============================================================================

stage(
    "SUBPHASE 10J — GRAPH STATISTICS"
)


node_statistics = []


for label in GRAPH_SCHEMA[
    "nodes"
]:

    count = neo4j_scalar(
        f"""
        MATCH (n:`{label}`)
        RETURN count(n) AS value
        """
    )


    node_statistics.append(
        {
            "label":
                label,

            "count":
                count,
        }
    )


    print(
        f"{label:<20}",
        f"{count:,}"
    )


relationship_statistics = neo4j_records(
    """
    MATCH ()-[r]->()

    RETURN
        type(r)
            AS relationship_type,

        count(r)
            AS count

    ORDER BY
        count DESC
    """
)


print()
print(
    "Relationships:"
)


for row in relationship_statistics:

    print(
        f"{row['relationship_type']:<35}",
        f"{row['count']:,}"
    )


total_nodes = neo4j_scalar(
    """
    MATCH (n)
    RETURN count(n) AS value
    """
)


total_relationships = neo4j_scalar(
    """
    MATCH ()-[r]->()
    RETURN count(r) AS value
    """
)


graph_statistics = [
    {
        "metric":
            "total_nodes",

        "value":
            total_nodes,
    },

    {
        "metric":
            "total_relationships",

        "value":
            total_relationships,
    },
]


for row in node_statistics:

    graph_statistics.append(
        {
            "metric":
                (
                    row[
                        "label"
                    ].lower()
                    + "_nodes"
                ),

            "value":
                row[
                    "count"
                ],
        }
    )


# =============================================================================
# EXPORT
# =============================================================================

stage(
    "SUBPHASE 10M — EXPORT PHASE 10 RESEARCH RESULTS"
)


pd.DataFrame(
    node_statistics
).to_csv(
    NODE_STATS_FILE,
    index=False
)


pd.DataFrame(
    relationship_statistics
).to_csv(
    REL_STATS_FILE,
    index=False
)


pd.DataFrame(
    graph_statistics
).to_csv(
    GRAPH_STATS_FILE,
    index=False
)


pd.DataFrame(
    mapping_coverage
).to_csv(
    MAPPING_FILE,
    index=False
)


pd.DataFrame(
    mapping_coverage
).to_csv(
    SEMANTIC_COVERAGE_FILE,
    index=False
)


pd.DataFrame(
    validation_results
).to_csv(
    VALIDATION_FILE,
    index=False
)


pd.DataFrame(
    benchmark_results
).to_csv(
    BENCHMARK_FILE,
    index=False
)


# =============================================================================
# VALIDATION COUNTS
# =============================================================================

validation_passes = sum(
    1
    for row
    in validation_results
    if row[
        "status"
    ] == "PASS"
)


validation_info = sum(
    1
    for row
    in validation_results
    if row[
        "status"
    ] == "INFO"
)


validation_reviews = sum(
    1
    for row
    in validation_results
    if row[
        "status"
    ] == "REVIEW"
)


# =============================================================================
# METADATA
# =============================================================================

metadata = {
    "phase":
        10,

    "phase_name":
        "Biomedical Knowledge Graph",

    "version":
        "v2_improved_coverage",

    "generated_at":
        datetime.now().isoformat(),

    "project_root":
        str(
            PROJECT_ROOT
        ),

    "loinc_file":
        str(
            LOINC_FILE
        ),

    "rxnconso_file":
        str(
            RXNCONSO_FILE
        ),

    "rxnrel_file":
        str(
            RXNREL_FILE
        ),

    "mesh_file":
        str(
            MESH_FILE
        ),

    "mesh_supp_file":
        str(
            MESH_SUPP_FILE
        ),

    "mesh_supp_loaded":
        MESH_SUPP_FILE.exists(),

    "postgres_database":
        POSTGRES_CONFIG[
            "dbname"
        ],

    "neo4j_uri":
        NEO4J_URI,

    "neo4j_database":
        NEO4J_DATABASE,

    "batch_size":
        BATCH_SIZE,

    "benchmark_runs":
        BENCHMARK_RUNS,

    "total_nodes":
        total_nodes,

    "total_relationships":
        total_relationships,

    "validation_passes":
        validation_passes,

    "validation_info":
        validation_info,

    "validation_reviews":
        validation_reviews,

    "representative_patient":
        TEST_PATIENT_ID,

    "node_statistics":
        node_statistics,

    "relationship_statistics":
        relationship_statistics,

    "mapping_coverage":
        mapping_coverage,

    "mapping_method_statistics":
        mapping_method_statistics,

    "benchmark_results":
        benchmark_results,

    "semantic_model": {
        "MAPPED_TO":
            (
                "Conservative direct terminology "
                "alignment."
            ),

        "ABOUT_MESH":
            (
                "Semantic retrieval context; "
                "not concept equivalence."
            ),

        "OBSERVED_WITH_MEDICATION":
            (
                "Patient-level observational "
                "co-occurrence; not treatment."
            ),

        "ASSOCIATED_WITH":
            (
                "Patient-level observational "
                "disease-lab association."
            ),

        "TREATED_BY":
            (
                "Reserved for future authoritative "
                "treatment evidence."
            ),

        "INTERACTS_WITH":
            (
                "Reserved for authoritative "
                "drug-interaction evidence."
            ),
    },

    "clinical_safety_note":
        (
            "Relationships derived from synthetic patient "
            "co-occurrence are observational and must not "
            "be interpreted as diagnosis, causation, "
            "treatment recommendation, or prescribing advice."
        ),
}


with open(
    METADATA_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=4,
        default=str
    )


# =============================================================================
# TEXT REPORT
# =============================================================================

report_lines = [
    "PHASE 10 — BIOMEDICAL KNOWLEDGE GRAPH (v2)",

    "=" * 80,

    "",

    "Adaptive Healthcare Knowledge Database",

    "",

    "GRAPH SUMMARY",

    "-" * 80,

    f"Total nodes: {total_nodes:,}",

    (
        "Total relationships: "
        f"{total_relationships:,}"
    ),

    "",

    "NODE STATISTICS",

    "-" * 80,
]


for row in node_statistics:

    report_lines.append(
        (
            f"{row['label']}: "
            f"{row['count']:,}"
        )
    )


report_lines.extend(
    [
        "",
        "RELATIONSHIP STATISTICS",
        "-" * 80,
    ]
)


for row in relationship_statistics:

    report_lines.append(
        (
            f"{row['relationship_type']}: "
            f"{row['count']:,}"
        )
    )


report_lines.extend(
    [
        "",
        "MAPPING + SEMANTIC COVERAGE",
        "-" * 80,
    ]
)


for row in mapping_coverage:

    report_lines.extend(
        [
            "",

            row[
                "entity_type"
            ],

            (
                "  Global direct     : "
                f"{row['global_mapped']:,}/"
                f"{row['global_total']:,} "
                f"("
                f"{row['global_coverage_percent']:.2f}%"
                f")"
            ),

            (
                "  Global semantic   : "
                f"{row['global_semantic_total']:,}/"
                f"{row['global_total']:,} "
                f"("
                f"{row['global_semantic_coverage_percent']:.2f}%"
                f")"
            ),

            (
                "  Clinical direct   : "
                f"{row['clinical_use_mapped']:,}/"
                f"{row['clinical_use_total']:,} "
                f"("
                f"{row['clinical_direct_coverage_percent']:.2f}%"
                f")"
            ),

            (
                "  Clinical semantic : "
                f"{row['clinical_semantic_total']:,}/"
                f"{row['clinical_use_total']:,} "
                f"("
                f"{row['clinical_semantic_coverage_percent']:.2f}%"
                f")"
            ),
        ]
    )


report_lines.extend(
    [
        "",
        "MAPPING METHODS",
        "-" * 80,
    ]
)


for row in mapping_method_statistics:

    report_lines.append(
        (
            f"{row['entity_type']:<15} "
            f"{str(row['mapping_method']):<45} "
            f"{row['relationship_count']:,}"
        )
    )


report_lines.extend(
    [
        "",
        "VALIDATION",
        "-" * 80,

        f"PASS   : {validation_passes}",
        f"INFO   : {validation_info}",
        f"REVIEW : {validation_reviews}",

        "",
        "GRAPH QUERY PERFORMANCE",
        "-" * 80,
    ]
)


for row in benchmark_results:

    report_lines.append(
        (
            f"{row['query']:<42} "
            f"mean={row['mean_ms']:.3f} ms "
            f"median={row['median_ms']:.3f} ms "
            f"min={row['min_ms']:.3f} ms "
            f"max={row['max_ms']:.3f} ms "
            f"std={row['stddev_ms']:.3f} ms"
        )
    )


report_lines.extend(
    [
        "",
        "SEMANTIC INTERPRETATION",
        "-" * 80,

        (
            "MAPPED_TO represents conservative direct "
            "terminology alignment."
        ),

        (
            "ABOUT_MESH represents useful semantic context "
            "without claiming concept equivalence."
        ),

        (
            "OBSERVED_WITH_MEDICATION represents patient "
            "co-occurrence and is not a treatment recommendation."
        ),

        (
            "ASSOCIATED_WITH represents observational "
            "disease-lab association and does not imply causation."
        ),

        (
            "TREATED_BY is intentionally absent until "
            "authoritative treatment evidence is integrated."
        ),

        (
            "INTERACTS_WITH is intentionally not inferred from "
            "generic RxNorm terminology relationships."
        ),

        "",
        "PHASE 10 STATUS",
        "-" * 80,

        (
            "READY TO FREEZE"
            if validation_reviews == 0
            else
            "REQUIRES REVIEW BEFORE FREEZE"
        ),
    ]
)


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(
            report_lines
        )
    )


# =============================================================================
# FINAL SUMMARY
# =============================================================================

stage(
    "PHASE 10 FINAL RESULT"
)


print(
    "Total graph nodes             :",
    f"{total_nodes:,}"
)

print(
    "Total graph relationships     :",
    f"{total_relationships:,}"
)

print(
    "Validation PASS               :",
    validation_passes
)

print(
    "Validation INFO               :",
    validation_info
)

print(
    "Validation REVIEW             :",
    validation_reviews
)


print()
print(
    "Node populations:"
)


for row in node_statistics:

    print(
        f"  {row['label']:<18}",
        f"{row['count']:,}"
    )


print()
print(
    "Semantic coverage:"
)


for row in mapping_coverage:

    print()
    print(
        f"  {row['entity_type']}"
    )

    print(
        "    Global direct     :",
        f"{row['global_mapped']:,}/"
        f"{row['global_total']:,}",
        f"({row['global_coverage_percent']:.2f}%)"
    )

    print(
        "    Global semantic   :",
        f"{row['global_semantic_total']:,}/"
        f"{row['global_total']:,}",
        f"({row['global_semantic_coverage_percent']:.2f}%)"
    )

    print(
        "    Clinical direct   :",
        f"{row['clinical_use_mapped']:,}/"
        f"{row['clinical_use_total']:,}",
        f"({row['clinical_direct_coverage_percent']:.2f}%)"
    )

    print(
        "    Clinical semantic :",
        f"{row['clinical_semantic_total']:,}/"
        f"{row['clinical_use_total']:,}",
        f"({row['clinical_semantic_coverage_percent']:.2f}%)"
    )


print()
print(
    "Graph query performance:"
)


for row in benchmark_results:

    print(
        f"  {row['query']:<42}",
        f"median={row['median_ms']:.3f} ms",
        f"std={row['stddev_ms']:.3f}"
    )


print()
print(
    "Diagnostic outputs:"
)


diagnostic_files = [
    DRUG_DIAGNOSTICS_FILE,
    LAB_DIAGNOSTICS_FILE,
    DISEASE_DIAGNOSTICS_FILE,
    UNMAPPED_DISEASE_FILE,
    UNMAPPED_DRUG_FILE,
    UNMAPPED_LAB_FILE,
    MAPPING_METHOD_FILE,
    SEMANTIC_COVERAGE_FILE,
]


for file in diagnostic_files:

    print(
        " -",
        file.name
    )


print()
print(
    "Results directory:"
)

print(
    RESULTS_DIR
)


print()
print("=" * 120)


if validation_reviews == 0:

    print(
        "PHASE 10 STATUS: READY TO FREEZE"
    )

else:

    print(
        "PHASE 10 STATUS: REVIEW REQUIRED"
    )


print("=" * 120)


print(
    "10A Source validation                    COMPLETE"
)

print(
    "10B Extraction/transformation            COMPLETE"
)

print(
    "10C Graph schema                         COMPLETE"
)

print(
    "10D Core node types                      COMPLETE"
)

print(
    "10E Relationship model                   COMPLETE"
)

print(
    "10F Biomedical terminology               COMPLETE"
)

print(
    "10G Disease/terminology knowledge        COMPLETE"
)

print(
    "10H Patient knowledge bridge             COMPLETE"
)

print(
    "10I Constraints/indexes                  COMPLETE"
)

print(
    "10J Integrity + semantic validation      COMPLETE"
)

print(
    "10K Clinical/Graph-RAG queries           COMPLETE"
)

print(
    "10L Performance validation               COMPLETE"
)

print(
    "10M Research output                      COMPLETE"
)

print(
    "10N Semantic enrichment + coverage       COMPLETE"
)


print()


if validation_reviews == 0:

    print(
        "NEXT -> FREEZE PHASE 10, THEN PHASE 11"
    )

else:

    print(
        "NEXT -> INSPECT DIAGNOSTIC CSV FILES "
        "BEFORE FREEZING PHASE 10"
    )


# =============================================================================
# CLEAN SHUTDOWN
# =============================================================================

pg_conn.close()

neo4j_driver.close()


print()
print(
    "PostgreSQL connection closed."
)

print(
    "Neo4j connection closed."
)