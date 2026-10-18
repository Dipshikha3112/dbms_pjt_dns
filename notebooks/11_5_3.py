
import argparse
import csv
import getpass
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    import psycopg
    from psycopg import sql
    from neo4j import GraphDatabase, READ_ACCESS
except ImportError as exc:
    sys.exit(
        f"Missing dependency: {exc}\n"
        "Install: pip install 'psycopg[binary]' neo4j"
    )


ROOT = Path(__file__).resolve().parent.parent
LOG = logging.getLogger("phase10_8")


RELATIONSHIPS = {
    "conditions": {
        "relationship": "HAS_CONDITION",
        "label": "Disease",
        "graph_key": "source_code",
        "tables": ["diagnosis", "patient_condition", "patient_diagnosis"],
        "direct": ["source_code", "code", "snomed_code"],
        "foreign": ["condition_id"],
        "reference": "condition",
        "reference_code": "source_code",
    },
    "labs": {
        "relationship": "HAS_LAB",
        "label": "LabTest",
        "graph_key": "loinc_code",
        "tables": ["lab_result", "patient_lab", "observation"],
        "direct": ["loinc_code", "loinc", "code"],
        "foreign": ["lab_test_id", "lab_id"],
        "reference": None,
        "reference_code": None,
    },
}


def norm(value):
    return "" if value is None else str(value).strip()


def norm_text(value):
    value = norm(value).casefold()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def pct(a, b):
    return round(100.0 * a / b, 3) if b else None


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def pg_rows(conn, query, params=()):
    with conn.cursor() as cursor:
        cursor.execute(query, params)
        columns = [c.name for c in cursor.description]
        return [
            dict(zip(columns, row))
            for row in cursor.fetchall()
        ]


def graph_rows(session, query, params=None):
    return [
        dict(record)
        for record in session.run(query, params or {})
    ]


def schema_info(conn):
    rows = pg_rows(
        conn,
        """
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position
        """,
    )

    result = defaultdict(dict)

    for row in rows:
        result[row["table_name"]][row["column_name"]] = row["data_type"]

    return dict(result)


def first_column(columns, candidates):
    return next((c for c in candidates if c in columns), None)


def fetch_table_columns(conn, table, columns):
    query = sql.SQL("SELECT {} FROM public.{}").format(
        sql.SQL(", ").join(
            sql.SQL("{}::text AS {}").format(
                sql.Identifier(c),
                sql.Identifier(c),
            )
            for c in columns
        ),
        sql.Identifier(table),
    )
    return pg_rows(conn, query)


def patient_check(original, scaled, graph):
    original_ids = [norm(x["patient_id"]) for x in original]
    graph_ids = [norm(x["patient_id"]) for x in graph]

    original_set = {x for x in original_ids if x}
    graph_set = {x for x in graph_ids if x}

    original_uuids = {
        norm(x["synthea_id"]).lower()
        for x in original
        if norm(x["synthea_id"])
    }

    scaled_uuids = {
        norm(x["synthea_id"]).lower()
        for x in scaled
        if norm(x["synthea_id"])
    }

    matches = original_set & graph_set
    uuid_matches = original_uuids & scaled_uuids

    complete = (
        original_set == graph_set
        and len(original_ids) == len(original_set)
        and len(graph_ids) == len(graph_set)
    )

    return {
        "status": "PASS" if complete else "REVIEW_REQUIRED",
        "original_patients": len(original),
        "scaled_patients": len(scaled),
        "graph_patients": len(graph),
        "graph_original_exact_matches": len(matches),
        "graph_original_match_pct": pct(
            len(matches), len(graph_set)
        ),
        "graph_original_exact_set_equal": original_set == graph_set,
        "graph_duplicate_ids": len(graph_ids) - len(graph_set),
        "original_duplicate_ids": len(original_ids) - len(original_set),
        "graph_missing_ids": sum(not x for x in graph_ids),
        "original_scaled_shared_synthea_uuids": len(uuid_matches),
        "original_scaled_overlap_pct": pct(
            len(uuid_matches), len(original_uuids)
        ),
        "scaled_population_separate": len(uuid_matches) == 0,
    }


def get_entities(session, label):
    return [
        row["properties"]
        for row in graph_rows(
            session,
            f"MATCH (n:`{label}`) "
            "RETURN properties(n) AS properties",
        )
    ]


def reference_data(conn, schema, tables, ids, names):
    for table in tables:
        fields = schema.get(table, {})
        key = first_column(fields, ids)
        name = first_column(fields, names)

        if not key or not name or key == name:
            continue

        rows = fetch_table_columns(conn, table, [key, name])
        values = defaultdict(set)

        for row in rows:
            identifier = norm(row[key])
            description = norm(row[name])

            if identifier and description:
                values[identifier].add(description)

        return {
            "table": table,
            "id_column": key,
            "name_column": name,
            "values": dict(values),
            "row_count": len(rows),
        }

    return None


def terminology_check(
    label,
    entities,
    reference,
    graph_key,
    graph_names,
    out,
):
    if not entities:
        return {
            "status": "NOT_TESTABLE",
            "reason": f"No {label} nodes",
        }

    properties = set()

    for entity in entities:
        properties.update(entity)

    if graph_key not in properties:
        return {
            "status": "NOT_TESTABLE",
            "reason": f"Missing graph identifier: {graph_key}",
            "graph_properties": sorted(properties),
        }

    if reference is None:
        return {
            "status": "NOT_TESTABLE",
            "reason": "No compatible PostgreSQL reference table",
            "graph_nodes": len(entities),
        }

    graph_name = first_column(properties, graph_names)

    counts = Counter()
    details = []

    identifiers = Counter(
        norm(entity.get(graph_key))
        for entity in entities
        if norm(entity.get(graph_key))
    )

    for entity in entities:
        identifier = norm(entity.get(graph_key))
        name = norm(entity.get(graph_name)) if graph_name else ""
        values = reference["values"].get(identifier, set())

        if not identifier:
            state = "MISSING_IDENTIFIER"
        elif not values:
            state = "NOT_IN_POSTGRES"
        elif len(values) > 1:
            state = "AMBIGUOUS_REFERENCE"
        elif not name:
            state = "IDENTIFIER_ONLY"
        elif norm_text(name) == norm_text(next(iter(values))):
            state = "IDENTIFIER_AND_TEXT_AGREE"
        else:
            state = "TERMINOLOGY_REVIEW_REQUIRED"

        counts[state] += 1

        details.append({
            "entity": label,
            "identifier": identifier,
            "status": state,
            "graph_term": name,
            "postgres_term": " | ".join(sorted(values))[:1000],
        })

    write_csv(
        out / f"{label.lower()}_terminology.csv",
        details,
        [
            "entity",
            "identifier",
            "status",
            "graph_term",
            "postgres_term",
        ],
    )

    complete = (
        counts["IDENTIFIER_AND_TEXT_AGREE"] == len(entities)
        and len(identifiers) == len(entities)
        and all(count == 1 for count in identifiers.values())
    )

    return {
        "status": "PASS" if complete else "REVIEW_REQUIRED",
        "graph_nodes": len(entities),
        "graph_unique_identifiers": len(identifiers),
        "graph_duplicate_identifiers": sum(
            count - 1
            for count in identifiers.values()
            if count > 1
        ),
        "postgres_reference": reference["table"],
        "postgres_identifier": reference["id_column"],
        "postgres_name_column": reference["name_column"],
        "counts": dict(counts),
    }


def lab_code_coverage(conn, schema, entities, out):
    if "loinc_code" not in schema.get("lab_result", {}):
        return {
            "status": "NOT_TESTABLE",
            "reason": "lab_result.loinc_code is missing",
        }

    rows = pg_rows(
        conn,
        """
        SELECT DISTINCT loinc_code::text AS code
        FROM public.lab_result
        WHERE loinc_code IS NOT NULL
        """,
    )

    sql_codes = {
        norm(row["code"])
        for row in rows
        if norm(row["code"])
    }

    graph_codes = {
        norm(entity.get("loinc_code"))
        for entity in entities
        if norm(entity.get("loinc_code"))
    }

    shared = graph_codes & sql_codes

    write_csv(
        out / "lab_code_coverage.csv",
        [
            {
                "loinc_code": code,
                "in_postgres_lab_results": code in sql_codes,
            }
            for code in sorted(graph_codes)
        ],
        ["loinc_code", "in_postgres_lab_results"],
    )

    return {
        "status": "COMPLETED",
        "validation_type": "IDENTIFIER_COVERAGE_ONLY",
        "graph_unique_loinc_codes": len(graph_codes),
        "postgres_unique_lab_result_codes": len(sql_codes),
        "shared_codes": len(shared),
        "graph_codes_missing_in_postgres": len(
            graph_codes - sql_codes
        ),
        "graph_code_coverage_pct": pct(
            len(shared), len(graph_codes)
        ),
        "note": (
            "This measures use of LOINC concepts in patient lab "
            "results, not terminology-reference agreement."
        ),
    }


def discover_clinical_source(schema, config):
    found = []

    for table in config["tables"]:
        fields = schema.get(table, {})

        if "patient_id" not in fields:
            continue

        for target in config["direct"]:
            if target in fields:
                found.append({
                    "table": table,
                    "target": target,
                    "mode": "DIRECT_CODE",
                })

        for target in config["foreign"]:
            if target in fields:
                found.append({
                    "table": table,
                    "target": target,
                    "mode": "FOREIGN_KEY",
                })

    return found


def load_reference_bridge(conn, schema, config, fk_column):
    table = config["reference"]

    if not table or table not in schema:
        return None

    fields = schema[table]
    code_column = config["reference_code"]

    if fk_column not in fields or code_column not in fields:
        return None

    rows = fetch_table_columns(
        conn,
        table,
        [fk_column, code_column],
    )

    bridge = defaultdict(set)

    for row in rows:
        key = norm(row[fk_column])
        code = norm(row[code_column])

        if key and code:
            bridge[key].add(code)

    return dict(bridge)


def load_sql_pairs(conn, schema, config, source):
    table = source["table"]
    target = source["target"]

    rows = fetch_table_columns(
        conn,
        table,
        ["patient_id", target],
    )

    raw = {
        (norm(row["patient_id"]), norm(row[target]))
        for row in rows
        if norm(row["patient_id"]) and norm(row[target])
    }

    if source["mode"] == "DIRECT_CODE":
        return {
            "pairs": raw,
            "raw_pairs": len(raw),
            "unmapped": 0,
            "ambiguous": 0,
            "mapping_method": "DIRECT_CODE",
        }

    bridge = load_reference_bridge(
        conn, schema, config, target
    )

    if bridge is None:
        return None

    mapped = set()
    unmapped = 0
    ambiguous = 0

    for patient_id, target_id in raw:
        codes = bridge.get(target_id, set())

        if len(codes) == 1:
            mapped.add(
                (patient_id, next(iter(codes)))
            )
        elif len(codes) > 1:
            ambiguous += 1
        else:
            unmapped += 1

    return {
        "pairs": mapped,
        "raw_pairs": len(raw),
        "unmapped": unmapped,
        "ambiguous": ambiguous,
        "mapping_method": (
            f"FOREIGN_KEY:{config['reference']}.{target}"
        ),
    }


def load_graph_pairs(session, config):
    label = config["label"]
    relationship = config["relationship"]
    key = config["graph_key"]

    rows = graph_rows(
        session,
        f"""
        MATCH (p:Patient)-[r:`{relationship}`]->(t:`{label}`)
        RETURN
            p.patient_id AS patient_id,
            t.`{key}` AS target_code
        """,
    )

    pairs = {
        (norm(row["patient_id"]), norm(row["target_code"]))
        for row in rows
        if norm(row["patient_id"]) and norm(row["target_code"])
    }

    missing = sum(
        not norm(row["patient_id"])
        or not norm(row["target_code"])
        for row in rows
    )

    return {
        "rows": len(rows),
        "pairs": pairs,
        "missing_endpoints": missing,
    }


def relationship_check(
    conn,
    schema,
    session,
    name,
    config,
    out,
):
    graph = load_graph_pairs(session, config)
    sources = discover_clinical_source(schema, config)

    if not sources:
        return {
            "status": "NOT_TESTABLE",
            "relationship": config["relationship"],
            "graph_relationship_rows": graph["rows"],
            "reason": "No compatible patient-level SQL source",
        }

    tested = []
    unresolved = []

    for source in sources:
        result = load_sql_pairs(
            conn, schema, config, source
        )

        if result is None:
            unresolved.append({
                **source,
                "reason": "Foreign-key reference not resolvable",
            })
            continue

        matches = graph["pairs"] & result["pairs"]

        tested.append({
            "sql_table": source["table"],
            "sql_target": source["target"],
            "mapping_method": result["mapping_method"],
            "raw_sql_pairs": result["raw_pairs"],
            "sql_unique_pairs": len(result["pairs"]),
            "graph_unique_pairs": len(graph["pairs"]),
            "verified_pairs": len(matches),
            "graph_unverified_pairs": len(
                graph["pairs"] - result["pairs"]
            ),
            "sql_pairs_missing_graph": len(
                result["pairs"] - graph["pairs"]
            ),
            "unmapped_sql_pairs": result["unmapped"],
            "ambiguous_sql_pairs": result["ambiguous"],
            "graph_agreement_pct": pct(
                len(matches), len(graph["pairs"])
            ),
            "_pairs": result["pairs"],
        })

    if not tested:
        return {
            "status": "NOT_TESTABLE",
            "relationship": config["relationship"],
            "reason": "No resolvable SQL-to-graph bridge",
            "unresolved_sources": unresolved,
        }

    tested.sort(
        key=lambda item: (
            item["graph_unverified_pairs"] == 0
            and item["sql_pairs_missing_graph"] == 0
            and item["unmapped_sql_pairs"] == 0
            and item["ambiguous_sql_pairs"] == 0,
            item["verified_pairs"],
            -item["graph_unverified_pairs"],
        ),
        reverse=True,
    )

    best = tested[0]
    best_pairs = best.pop("_pairs")

    for alternative in tested[1:]:
        alternative.pop("_pairs")

    graph_only = graph["pairs"] - best_pairs
    sql_only = best_pairs - graph["pairs"]

    write_csv(
        out / f"{name}_relationship_diagnostics.csv",
        [
            {"category": "GRAPH_ONLY", "count": len(graph_only)},
            {"category": "POSTGRES_ONLY", "count": len(sql_only)},
            {
                "category": "UNMAPPED_SQL_REFERENCE",
                "count": best["unmapped_sql_pairs"],
            },
            {
                "category": "AMBIGUOUS_SQL_REFERENCE",
                "count": best["ambiguous_sql_pairs"],
            },
        ],
        ["category", "count"],
    )

    complete = (
        not graph_only
        and not sql_only
        and best["unmapped_sql_pairs"] == 0
        and best["ambiguous_sql_pairs"] == 0
        and graph["missing_endpoints"] == 0
    )

    return {
        "status": "PASS" if complete else "REVIEW_REQUIRED",
        "relationship": config["relationship"],
        "graph_relationship_rows": graph["rows"],
        "graph_missing_endpoint_ids": graph["missing_endpoints"],
        **best,
        "alternative_sql_sources": tested[1:],
        "unresolved_sources": unresolved,
        "comparison_scope": "Distinct patient and clinical-code pairs",
    }


def validate_prescription_medications(conn, schema, session, out):
    required = {
        "prescription": {"prescription_id", "patient_id"},
        "prescription_item": {"prescription_id", "rxcui"},
    }

    for table, fields in required.items():
        missing = fields - set(schema.get(table, {}))

        if missing:
            return {
                "status": "NOT_TESTABLE",
                "reason": f"Missing {table} columns: {sorted(missing)}",
            }

    pg = pg_rows(
        conn,
        """
        SELECT
            p.patient_id::text AS patient_id,
            pi.rxcui::text AS rxcui
        FROM public.prescription AS p
        INNER JOIN public.prescription_item AS pi
            ON p.prescription_id = pi.prescription_id
        WHERE p.patient_id IS NOT NULL
          AND pi.rxcui IS NOT NULL
        """,
    )

    neo = graph_rows(
        session,
        """
        MATCH (p:Patient)-[:HAS_MEDICATION]->(d:Drug)
        RETURN
            p.patient_id AS patient_id,
            d.rxcui AS rxcui
        """,
    )

    pg_pairs = {
        (norm(row["patient_id"]), norm(row["rxcui"]))
        for row in pg
        if norm(row["patient_id"]) and norm(row["rxcui"])
    }

    neo_pairs = {
        (norm(row["patient_id"]), norm(row["rxcui"]))
        for row in neo
        if norm(row["patient_id"]) and norm(row["rxcui"])
    }

    verified = pg_pairs & neo_pairs
    graph_only = neo_pairs - pg_pairs
    postgres_only = pg_pairs - neo_pairs

    missing_pg = sum(
        not norm(row["patient_id"]) or not norm(row["rxcui"])
        for row in pg
    )

    missing_neo = sum(
        not norm(row["patient_id"]) or not norm(row["rxcui"])
        for row in neo
    )

    diagnostics = [
        {
            "category": "VERIFIED",
            "count": len(verified),
        },
        {
            "category": "GRAPH_ONLY",
            "count": len(graph_only),
        },
        {
            "category": "POSTGRES_ONLY",
            "count": len(postgres_only),
        },
        {
            "category": "MISSING_GRAPH_ENDPOINT",
            "count": missing_neo,
        },
        {
            "category": "MISSING_SQL_ENDPOINT",
            "count": missing_pg,
        },
        {
            "category": "DUPLICATE_GRAPH_PAIR_ROWS",
            "count": len(neo) - missing_neo - len(neo_pairs),
        },
        {
            "category": "REPEATED_SQL_PAIR_ROWS",
            "count": len(pg) - missing_pg - len(pg_pairs),
        },
    ]

    write_csv(
        out / "medications_relationship_diagnostics.csv",
        diagnostics,
        ["category", "count"],
    )

    complete = (
        pg_pairs == neo_pairs
        and missing_pg == 0
        and missing_neo == 0
        and len(neo_pairs) == len(neo)
    )

    return {
        "status": "PASS" if complete else "REVIEW_REQUIRED",
        "relationship": "HAS_MEDICATION",
        "sql_source": "prescription JOIN prescription_item",
        "join_key": "prescription_id",
        "clinical_identifier": "rxcui",
        "postgres_joined_rows": len(pg),
        "postgres_unique_pairs": len(pg_pairs),
        "graph_relationship_rows": len(neo),
        "graph_unique_pairs": len(neo_pairs),
        "verified_pairs": len(verified),
        "graph_unverified_pairs": len(graph_only),
        "postgres_pairs_missing_graph": len(postgres_only),
        "graph_missing_endpoint_ids": missing_neo,
        "postgres_missing_endpoint_ids": missing_pg,
        "graph_duplicate_pair_rows": (
            len(neo) - missing_neo - len(neo_pairs)
        ),
        "postgres_repeated_pair_rows": (
            len(pg) - missing_pg - len(pg_pairs)
        ),
        "graph_agreement_pct": pct(
            len(verified), len(neo_pairs)
        ),
        "postgres_agreement_pct": pct(
            len(verified), len(pg_pairs)
        ),
        "comparison_scope": "Distinct patient-RxCUI pairs",
    }


def run_check(report, name, function):
    LOG.info("Running %s", name)

    try:
        result = function()
        report["checks"][name] = result

        LOG.info(
            "%s: %s",
            name,
            json.dumps(result, default=str)[:2000],
        )

    except Exception as exc:
        LOG.exception("Check failed: %s", name)

        report["checks"][name] = {
            "status": "ERROR",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }

        report["errors"].append(name)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pg-original",
        default="adaptive_healthcare_db",
    )
    parser.add_argument(
        "--pg-scaled",
        default="adaptive_healthcare_scale_10k",
    )
    parser.add_argument("--pg-host", default="localhost")
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

    args = parser.parse_args()

    out = (
        ROOT
        / "results"
        / "phase10_8_complete_validation"
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )

    out.mkdir(parents=True, exist_ok=False)

    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s"
    )

    for handler in [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            out / "run.log",
            encoding="utf-8",
        ),
    ]:
        handler.setFormatter(formatter)
        LOG.addHandler(handler)

    pg_password = getpass.getpass("PostgreSQL password: ")
    neo_password = getpass.getpass("Neo4j password: ")

    report = {
        "phase": "10.8",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "original_database": args.pg_original,
        "scaled_database": args.pg_scaled,
        "neo4j_database": args.neo4j_database,
        "read_only": True,
        "checks": {},
        "errors": [],
        "limitations": [
            "The original PostgreSQL database is the graph source.",
            "The scaled database contains a separate patient population.",
            "Relationship checks compare patient-code pairs.",
            "Event timestamps and repeated event multiplicity are not "
            "fully validated by pair comparisons.",
            "Terminology agreement is not independent clinical validation.",
        ],
    }

    pg_config = {
        "host": args.pg_host,
        "port": args.pg_port,
        "user": args.pg_user,
        "password": pg_password,
        "connect_timeout": 15,
        "options": (
            "-c default_transaction_read_only=on "
            "-c statement_timeout=120000"
        ),
    }

    try:
        with (
            psycopg.connect(
                dbname=args.pg_original,
                **pg_config,
            ) as original,
            psycopg.connect(
                dbname=args.pg_scaled,
                **pg_config,
            ) as scaled,
            GraphDatabase.driver(
                args.neo4j_uri,
                auth=(args.neo4j_user, neo_password),
                connection_timeout=15,
            ) as driver,
        ):
            driver.verify_connectivity()

            with driver.session(
                database=args.neo4j_database,
                default_access_mode=READ_ACCESS,
            ) as neo:

                original_schema = schema_info(original)
                scaled_schema = schema_info(scaled)

                write_json(
                    out / "original_schema.json",
                    original_schema,
                )
                write_json(
                    out / "scaled_schema.json",
                    scaled_schema,
                )

                original_patients = pg_rows(
                    original,
                    """
                    SELECT
                        patient_id::text AS patient_id,
                        synthea_id::text AS synthea_id
                    FROM public.patient
                    """,
                )

                scaled_patients = pg_rows(
                    scaled,
                    """
                    SELECT
                        patient_id::text AS patient_id,
                        synthea_id::text AS synthea_id
                    FROM public.patient
                    """,
                )

                neo_patients = graph_rows(
                    neo,
                    """
                    MATCH (p:Patient)
                    RETURN p.patient_id AS patient_id
                    """,
                )

                run_check(
                    report,
                    "patient_identity",
                    lambda: patient_check(
                        original_patients,
                        scaled_patients,
                        neo_patients,
                    ),
                )

                terminology = [
                    (
                        "Disease",
                        ["condition"],
                        ["source_code"],
                        ["description", "name"],
                        "source_code",
                        ["name", "description"],
                    ),
                    (
                        "Drug",
                        ["drug_concept", "drug", "medication"],
                        ["rxcui", "rxnorm_code"],
                        ["name", "description"],
                        "rxcui",
                        ["name", "description"],
                    ),
                    (
                        "LabTest",
                        [
                            "loinc_concept",
                            "loinc",
                            "loinc_term",
                            "lab_test",
                            "labtest",
                            "loinc_reference",
                        ],
                        ["loinc_code", "loinc", "code"],
                        [
                            "long_common_name",
                            "name",
                            "description",
                            "long_name",
                        ],
                        "loinc_code",
                        ["long_common_name", "name"],
                    ),
                ]

                for (
                    label,
                    tables,
                    identifiers,
                    names,
                    graph_key,
                    graph_names,
                ) in terminology:

                    def check_terminology(
                        label=label,
                        tables=tables,
                        identifiers=identifiers,
                        names=names,
                        graph_key=graph_key,
                        graph_names=graph_names,
                    ):
                        reference = reference_data(
                            original,
                            original_schema,
                            tables,
                            identifiers,
                            names,
                        )

                        entities = get_entities(
                            neo,
                            label,
                        )

                        return terminology_check(
                            label,
                            entities,
                            reference,
                            graph_key,
                            graph_names,
                            out,
                        )

                    run_check(
                        report,
                        f"{label.lower()}_terminology",
                        check_terminology,
                    )

                run_check(
                    report,
                    "lab_code_coverage",
                    lambda: lab_code_coverage(
                        original,
                        original_schema,
                        get_entities(neo, "LabTest"),
                        out,
                    ),
                )

                for name, config in RELATIONSHIPS.items():
                    run_check(
                        report,
                        f"{name}_relationships",
                        lambda name=name, config=config:
                            relationship_check(
                                original,
                                original_schema,
                                neo,
                                name,
                                config,
                                out,
                            ),
                    )

                run_check(
                    report,
                    "medications_relationships",
                    lambda: validate_prescription_medications(
                        original,
                        original_schema,
                        neo,
                        out,
                    ),
                )

    except Exception as exc:
        LOG.exception("Database connection or setup failed")

        report["errors"].append({
            "type": type(exc).__name__,
            "message": str(exc),
        })

    finally:
        statuses = [
            item.get("status")
            for item in report["checks"].values()
        ]

        if report["errors"]:
            overall = "ERRORS_PRESENT"
        elif not statuses:
            overall = "NO_CHECKS_COMPLETED"
        elif "NOT_TESTABLE" in statuses:
            overall = "COMPLETED_WITH_UNVERIFIED_CHECKS"
        elif "REVIEW_REQUIRED" in statuses:
            overall = "COMPLETED_WITH_REVIEW_ITEMS"
        elif all(
            status in ("PASS", "COMPLETED")
            for status in statuses
        ):
            overall = "ALL_APPLICABLE_CHECKS_PASSED"
        else:
            overall = "COMPLETED_WITH_REVIEW_ITEMS"

        report["overall_status"] = overall

        write_json(
            out / "phase10_8_validation_report.json",
            report,
        )

        print("\n" + "=" * 75)
        print("PHASE 10.8 - COMPLETE VALIDATION")
        print("=" * 75)

        for name, result in report["checks"].items():
            print(f"\n{name.upper()}")
            print("-" * 50)

            for key, value in result.items():
                print(f"{key}: {value}")

        print("\nOverall:", overall)
        print("Errors:", len(report["errors"]))
        print("Output directory:", out)
        print("Database modifications: NONE")

    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
