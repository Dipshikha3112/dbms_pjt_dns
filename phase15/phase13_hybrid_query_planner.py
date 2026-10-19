
from __future__ import annotations

import argparse
import atexit
import csv
import importlib.util
import json
import os
import re
import sqlite3
import statistics
import sys
import time

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from threading import Lock


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "phase13"

PHASE12 = Path(
    os.getenv(
        "PHASE12_SCRIPT",
        str(ROOT / "notebooks" / "phase12_graph_rag_v7.py")
    )
)

SQLITE = Path(
    os.getenv(
        "LITERATURE_SQLITE",
        str(ROOT / "results" / "phase12_store_v3" / "literature.sqlite")
    )
)

PROFILE_FILE = OUT / "cost_profiles.json"

STRATEGIES = ("sql", "graph", "vector", "hybrid")

RISK_NOTE = (
    "Research prototype only. Graph associations are observational, "
    "not proof of causality or clinical recommendations."
)


BIOMEDICAL_TERMS = {
    "kidney": (
        "creatinine",
        "glomerular filtration",
        "egfr",
        "urea",
        "renal",
        "urine albumin",
        "albumin/creatinine",
        "microalbumin",
        "cystatin c",
    ),
    "renal": (
        "creatinine",
        "glomerular filtration",
        "egfr",
        "urea",
        "renal",
        "urine albumin",
        "albumin/creatinine",
        "microalbumin",
        "cystatin c",
    ),
    "cardiovascular": (
        "cholesterol",
        "ldl",
        "hdl",
        "triglyceride",
        "troponin",
        "natriuretic peptide",
    ),
    "heart": (
        "troponin",
        "natriuretic peptide",
        "cholesterol",
        "ldl",
        "hdl",
    ),
    "liver": (
        "alanine aminotransferase",
        "aspartate aminotransferase",
        "bilirubin",
        "alkaline phosphatase",
    ),
    "diabetes": (
        "glucose",
        "hemoglobin a1c",
        "hba1c",
        "glycated hemoglobin",
    ),
}

RANK_STOPWORDS = {
    "which", "what", "are", "the", "with", "and",
    "for", "from", "into", "that", "have",
    "associated", "related", "tests", "test",
    "labs", "lab", "disease", "hypertension",
    "diabetes", "medications", "medication",
    "drugs", "drug", "laboratory", "findings",
    "find", "show", "give", "list", "me",
    "kidney", "renal", "heart", "cardiovascular",
    "liver", "blood", "about", "these",
}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def json_default(value):
    if isinstance(value, (datetime, Path)):
        return str(value)

    if hasattr(value, "isoformat"):
        return value.isoformat()

    if hasattr(value, "item"):
        return value.item()

    return str(value)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            default=json_default
        ),
        encoding="utf-8"
    )


def percentile(values, p):
    if not values:
        return None

    values = sorted(values)
    position = (len(values) - 1) * p / 100

    low = int(position)
    high = min(low + 1, len(values) - 1)

    return (
        values[low]
        + (values[high] - values[low])
        * (position - low)
    )


def has(text, pattern):
    return bool(re.search(pattern, text, re.I))


@lru_cache(maxsize=1)
def load_phase12():
    if not PHASE12.is_file():
        raise FileNotFoundError(
            f"Phase 12 implementation missing: {PHASE12}"
        )

    spec = importlib.util.spec_from_file_location(
        "phase12_v8_engine",
        PHASE12
    )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    required = (
        "fts",
        "metadata",
        "vector_search",
        "graph_retrieve",
        "profile",
        "run_question",
    )

    missing = [
        name
        for name in required
        if not hasattr(module, name)
    ]

    if missing:
        raise RuntimeError(
            "Phase 12 V8 functions missing: "
            + ", ".join(missing)
        )

    return module


@dataclass
class Intent:
    sql: bool = False
    graph: bool = False
    vector: bool = False
    rules: list = None
    entities: list = None

    def __post_init__(self):
        self.rules = self.rules or []
        self.entities = self.entities or []

    def required(self):
        return {
            name
            for name in ("sql", "graph", "vector")
            if getattr(self, name)
        }


def classify(question):
    q = question.strip()

    if not q:
        raise ValueError("Question cannot be empty")

    intent = Intent()

    def hit(attribute, rule):
        setattr(intent, attribute, True)
        intent.rules.append(rule)

    if has(
        q,
        r"\b(how many|count|total number|average|sum|"
        r"minimum|maximum|group by|aggregate|number of|"
        r"list all|show all)\b"
    ):
        hit(
            "sql",
            "Structured count, filter or aggregation requested"
        )

    if has(
        q,
        r"\b(patient|patients|demographic|diagnos(?:is|es)|"
        r"prescription records|database rows|stored records)\b"
    ):
        hit(
            "sql",
            "Structured clinical database records requested"
        )

    if has(
        q,
        r"\b(related to|relationship|relationships|connected|"
        r"connections|neighbor|neighbour|knowledge graph|"
        r"graph path|mapped to|mesh|"
        r"associated (?:with|labs?|medications?)|linked to)\b"
    ):
        hit(
            "graph",
            "Explicit relationship or knowledge-graph traversal requested"
        )

    if has(
        q,
        r"\b(lab tests?|laboratory findings|associated biomarkers|"
        r"drug associations|medication associations)\b"
    ):
        hit(
            "graph",
            "Biomedical concept association requested"
        )

    if has(
        q,
        r"\b(research|literature|papers?|articles?|publications?|"
        r"evidence|studies|semantic|similar|explain|mechanism|"
        r"why|summarize|discuss|find documents)\b"
    ):
        hit(
            "vector",
            "Semantic literature evidence requested"
        )

    if (
        has(q, r"\b(which|what)\b")
        and has(q, r"\b(studies|research|literature)\b")
    ):
        hit("vector", "Literature discovery requested")

    if (
        intent.sql
        and has(q, r"\b(chunks?|documents?)\b")
        and has(q, r"\b(how many|count|total|number)\b")
    ):
        intent.vector = False
        intent.rules.append(
            "Literature metadata count is exact SQL"
        )

    if (
        has(q, r"\b(patient|patients)\b")
        and has(q, r"\b(relationships?|connections?|graph)\b")
    ):
        hit(
            "graph",
            "Cross-database patient relationship request"
        )

    if not intent.required():
        hit(
            "vector",
            "Default semantic literature retrieval"
        )

    if (
        intent.sql
        and intent.vector
        and not intent.graph
        and has(q, r"\b(associated|linked|connections?)\b")
    ):
        hit(
            "graph",
            "Combined structured and relationship evidence requested"
        )

    known_entities = (
        "hypertension",
        "cardiovascular",
        "diabetes",
        "cholesterol",
        "stroke",
        "heart failure",
        "atherosclerosis",
        "obesity",
        "kidney disease",
        "statins",
        "blood pressure",
    )

    intent.entities = [
        entity
        for entity in known_entities
        if entity in q.lower()
    ]

    return intent


def capabilities(strategy):
    mapping = {
        "sql": {"sql"},
        "graph": {"graph"},
        "vector": {"vector"},
        "hybrid": {"sql", "graph", "vector"},
    }

    return mapping[strategy]


class CostRegistry:

    def __init__(self, path=PROFILE_FILE):
        self.path = Path(path)

        if self.path.is_file():
            self.rows = json.loads(
                self.path.read_text(encoding="utf-8")
            )
        else:
            self.rows = []

    def record(self, strategy, intent, ms, ok, cold=False):
        self.rows.append({
            "strategy": strategy,
            "intent": intent,
            "latency_ms": round(ms, 3),
            "success": bool(ok),
            "cold": bool(cold),
            "timestamp": stamp(),
        })

    def save(self):
        save(self.path, self.rows)

    def estimate(self, strategy, intent):
        eligible = [
            row["latency_ms"]
            for row in self.rows
            if (
                row["strategy"] == strategy
                and row["intent"] == intent
                and row["success"]
                and not row["cold"]
            )
        ]

        if len(eligible) < 3:
            eligible = [
                row["latency_ms"]
                for row in self.rows
                if (
                    row["strategy"] == strategy
                    and row["success"]
                    and not row["cold"]
                )
            ]

        if len(eligible) >= 3:
            return round(statistics.median(eligible), 3)

        return None

    def summary(self):
        output = {}

        for strategy in STRATEGIES:
            values = [
                row["latency_ms"]
                for row in self.rows
                if (
                    row["strategy"] == strategy
                    and row["success"]
                    and not row["cold"]
                )
            ]

            total = sum(
                row["strategy"] == strategy
                for row in self.rows
            )

            output[strategy] = {
                "runs": total,
                "successful_warm_runs": len(values),
                "median_ms": (
                    round(statistics.median(values), 3)
                    if values else None
                ),
                "p95_ms": (
                    round(percentile(values, 95), 3)
                    if values else None
                ),
            }

        return output


def plan(question, registry, force=None):
    intent = classify(question)
    required = intent.required()

    feasible = [
        strategy
        for strategy in STRATEGIES
        if required.issubset(capabilities(strategy))
    ]

    if force:
        if force not in STRATEGIES:
            raise ValueError("Unknown strategy")

        if not required.issubset(capabilities(force)):
            raise ValueError(
                f"Forced {force} cannot satisfy {sorted(required)}"
            )

        selected = force

    else:
        selected = (
            "hybrid"
            if len(required) > 1
            else next(iter(required))
        )

    key = "+".join(sorted(required))

    estimates = {
        strategy: registry.estimate(strategy, key)
        for strategy in feasible
    }

    if (
        not force
        and len(required) == 1
        and all(
            estimates[strategy] is not None
            for strategy in feasible
        )
    ):
        selected = min(
            feasible,
            key=lambda strategy: estimates[strategy]
        )

    justification = (
        f"Detected {sorted(required)} capabilities. "
        f"Matched rules: {'; '.join(dict.fromkeys(intent.rules))}. "
        f"Selected {selected} because it satisfies the "
        "required capabilities. Quality constraints take precedence."
    )

    return {
        "question": question,
        "selected_strategy": selected,
        "required_capabilities": sorted(required),
        "intent": asdict(intent),
        "feasible_strategies": feasible,
        "estimated_latency_ms": estimates[selected],
        "candidate_costs_ms": estimates,
        "justification": justification,
    }


def sql_execute(q, limit=8):
    if has(q, r"\b(chunks?|documents?|literature)\b"):
        if has(q, r"\b(how many|count|total|number)\b"):
            with sqlite3.connect(str(SQLITE)) as conn:
                chunk_count = conn.execute(
                    "SELECT COUNT(*) FROM chunks"
                ).fetchone()[0]

                document_count = conn.execute(
                    "SELECT COUNT(*) FROM documents"
                ).fetchone()[0]

            return {
                "query_template": "literature_counts",
                "rows": [{
                    "chunk_count": chunk_count,
                    "document_count": document_count,
                }],
                "source": "SQLite literature store",
            }

        raise ValueError(
            "SQL supports literature counts; use vector for content"
        )

    if has(q, r"\b(how many|count|total|number)\b"):
        if has(q, r"\b(patient|patients)\b"):
            name = "patient"
            query = (
                "SELECT COUNT(*)::bigint AS patient_count "
                "FROM public.patient"
            )

        elif has(q, r"\b(diagnos(?:is|es)|conditions?)\b"):
            name = "diagnosis"
            query = (
                "SELECT COUNT(*)::bigint AS diagnosis_count "
                "FROM public.diagnosis"
            )

        elif has(q, r"\b(lab|laboratory)\b"):
            name = "lab_result"
            query = (
                "SELECT COUNT(*)::bigint AS lab_result_count "
                "FROM public.lab_result"
            )

        elif has(q, r"\b(prescriptions?)\b"):
            name = "prescription"
            query = (
                "SELECT COUNT(*)::bigint AS prescription_count "
                "FROM public.prescription"
            )

        else:
            raise ValueError(
                "No allowlisted SQL count template matches"
            )

    elif has(q, r"\b(patient|patients)\b"):
        raise ValueError(
            "Patient-level data requires an authorized workflow"
        )

    else:
        raise ValueError(
            "No allowlisted SQL template matches"
        )

    import psycopg
    from psycopg.rows import dict_row

    with psycopg.connect(
        host=os.getenv("PGHOST", "127.0.0.1"),
        port=int(os.getenv("PGPORT", "5432")),
        dbname=os.getenv(
            "PG_ORIGINAL_DATABASE",
            "adaptive_healthcare_db"
        ),
        user=os.getenv("PGUSER", "postgres"),
        password=os.getenv("PGPASSWORD") or None,
        options=(
            "-c default_transaction_read_only=on "
            "-c statement_timeout=15000"
        ),
        connect_timeout=5,
    ) as conn:

        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(query)
            rows = [
                dict(row)
                for row in cursor.fetchmany(limit)
            ]

    return {
        "query_template": name + "_count",
        "rows": rows,
        "source": "PostgreSQL original healthcare database",
    }


_DRIVER = None
_DRIVER_LOCK = Lock()


def get_neo4j_driver():
    global _DRIVER

    if _DRIVER is not None:
        return _DRIVER, 0.0

    with _DRIVER_LOCK:
        if _DRIVER is not None:
            return _DRIVER, 0.0

        from neo4j import GraphDatabase

        password = os.getenv("NEO4J_PASSWORD")

        if not password:
            raise RuntimeError(
                "Set NEO4J_PASSWORD environment variable"
            )

        start = time.perf_counter()

        driver = GraphDatabase.driver(
            os.getenv(
                "NEO4J_URI",
                "bolt://127.0.0.1:7687"
            ),
            auth=(
                os.getenv("NEO4J_USER", "neo4j"),
                password,
            ),
            connection_timeout=5,
            max_connection_pool_size=20,
        )

        try:
            driver.verify_connectivity()
        except Exception:
            driver.close()
            raise

        _DRIVER = driver

        elapsed = round(
            (time.perf_counter() - start) * 1000,
            3
        )

        return _DRIVER, elapsed


def close_neo4j_driver():
    global _DRIVER

    with _DRIVER_LOCK:
        if _DRIVER is not None:
            _DRIVER.close()
            _DRIVER = None


atexit.register(close_neo4j_driver)


GRAPH_DISEASES = (
    "diabetes mellitus",
    "coronary heart disease",
    "heart failure",
    "atherosclerosis",
    "hypertension",
    "diabetes",
    "stroke",
    "obesity",
)


def biomedical_relevance_score(question, entity_name):
    question_tokens = set(
        re.findall(r"[a-z]{3,}", question.lower())
    )

    name = str(entity_name or "").lower()

    matched_terms = []

    for topic, related_terms in BIOMEDICAL_TERMS.items():
        if topic in question_tokens:
            for term in related_terms:
                if term in name:
                    matched_terms.append(term)

    matched_terms = sorted(set(matched_terms))

    return 3 * len(matched_terms), matched_terms


def graph_execute(q, limit=8, ranking="optimized"):
    from neo4j import READ_ACCESS

    names = [
        name
        for name in GRAPH_DISEASES
        if re.search(
            r"\b" + re.escape(name) + r"\b",
            q,
            re.I
        )
    ]

    if not names:
        raise ValueError(
            "Graph query needs a supported disease name"
        )

    disease = names[0].lower()

    if disease == "diabetes":
        disease = "diabetes mellitus"

    if has(q, r"\b(drug|medication|prescription)\b"):
        relationship = "OBSERVED_WITH_MEDICATION"
        endpoint = "Drug"
    else:
        relationship = "ASSOCIATED_WITH"
        endpoint = "LabTest"

    cypher = f"""
    MATCH (d:Disease)-[r:{relationship}]->(x:{endpoint})
    WHERE toLower(d.name) CONTAINS $name

    RETURN
        d.name AS disease,
        type(r) AS relationship,
        x.name AS associated_entity,
        coalesce(x.loinc_code, x.rxcui) AS concept_code,
        properties(r) AS relationship_properties

    ORDER BY associated_entity
    LIMIT $candidate_limit
    """

    total_start = time.perf_counter()

    driver, connect_ms = get_neo4j_driver()

    retrieval_start = time.perf_counter()

    if ranking == "optimized":
        candidate_limit = min(
            1000,
            max(200, limit * 25)
        )
    else:
        candidate_limit = limit

    with driver.session(
        database=os.getenv(
            "NEO4J_DATABASE",
            "neo4j"
        ),
        default_access_mode=READ_ACCESS,
    ) as session:

        rows = [
            dict(record)
            for record in session.run(
                cypher,
                {
                    "name": disease,
                    "candidate_limit": candidate_limit,
                }
            )
        ]

    retrieval_ms = round(
        (time.perf_counter() - retrieval_start) * 1000,
        3
    )

    ranking_start = time.perf_counter()

    question_terms = {
        term
        for term in re.findall(
            r"[a-z]{3,}",
            q.lower()
        )
        if term not in RANK_STOPWORDS
    }

    for row in rows:
        entity_name = str(
            row.get("associated_entity") or ""
        )

        entity_tokens = set(
            re.findall(
                r"[a-z]{3,}",
                entity_name.lower()
            )
        )

        overlap = len(
            question_terms & entity_tokens
        )

        biomedical_score, matched_terms = (
            biomedical_relevance_score(
                q,
                entity_name
            )
        )

        row["query_term_overlap"] = overlap

        row["biomedical_relevance_score"] = (
            biomedical_score
        )

        row["matched_biomedical_terms"] = (
            matched_terms
        )

        row["ranking_score"] = (
            biomedical_score + overlap
        )

    if ranking == "optimized":
        rows.sort(
            key=lambda row: (
                -row["ranking_score"],
                -row["biomedical_relevance_score"],
                str(
                    row.get("associated_entity") or ""
                ),
            )
        )

    chosen = rows[:limit]

    ranking_ms = round(
        (time.perf_counter() - ranking_start) * 1000,
        3
    )

    total_ms = round(
        (time.perf_counter() - total_start) * 1000,
        3
    )

    return {
        "rows": chosen,
        "candidate_count": len(rows),
        "source": "Neo4j",
        "query_template": (
            "disease_" + relationship.lower()
        ),
        "ranking_method": (
            "biomedical_term_expansion_plus_query_overlap"
            if ranking == "optimized"
            else "alphabetical_baseline"
        ),
        "ranking_limitation": (
            "Vocabulary matching is a retrieval heuristic, "
            "not validated clinical relevance. "
            "Only stored candidate associations can be returned."
        ),
        "interpretation": (
            "Observational associations only, "
            "not clinical recommendations"
        ),
        "timings_ms": {
            "driver_initialization": connect_ms,
            "graph_retrieval": retrieval_ms,
            "ranking": ranking_ms,
            "total": total_ms,
        },
    }


def vector_execute(q, limit=8, engine=None):
    e = engine or load_phase12()

    start = time.perf_counter()

    rows, warnings = e.vector_search(
        q,
        max(limit * 3, limit)
    )

    retrieval_ms = round(
        (time.perf_counter() - start) * 1000,
        3
    )

    ids = [
        item[0]
        for item in rows[:limit]
    ]

    metadata = e.metadata(ids)

    results = []

    for chunk_id, rank, similarity, table in rows[:limit]:
        info = metadata.get(
            str(chunk_id),
            {}
        )

        results.append({
            "chunk_id": chunk_id,
            "rank": rank,
            "similarity": round(
                float(similarity),
                4
            ),
            "title": info.get("title"),
            "page": info.get("page"),
            "excerpt": str(
                info.get("body") or ""
            )[:600],
            "source": table,
        })

    return {
        "rows": results,
        "source": (
            "PostgreSQL pgvector + SQLite literature metadata"
        ),
        "timings_ms": {
            "vector_search_and_embedding": retrieval_ms,
            "total": round(
                (time.perf_counter() - start) * 1000,
                3
            ),
        },
        "warnings": warnings,
    }


def hybrid_execute(
    q,
    required,
    limit=8,
    engine=None
):
    outputs = {}
    errors = {}

    if "sql" in required:
        try:
            outputs["sql"] = sql_execute(
                q,
                limit
            )
        except Exception as exc:
            errors["sql"] = (
                f"{type(exc).__name__}: {exc}"
            )

    if "graph" in required:
        try:
            outputs["graph"] = graph_execute(
                q,
                limit
            )
        except Exception as exc:
            errors["graph"] = (
                f"{type(exc).__name__}: {exc}"
            )

    if "vector" in required:
        try:
            e = engine or load_phase12()

            docs, timings, warnings = e.fuse(
                q,
                "hybrid",
                n=limit
            )

            literature_rows = []

            for doc in docs:
                literature_rows.append({
                    "chunk_id": doc.get("chunk_id"),
                    "title": doc.get("title"),
                    "page": doc.get("page"),
                    "excerpt": str(
                        doc.get("text") or ""
                    )[:600],
                    "source": doc.get("source_path"),
                })

            outputs["literature"] = {
                "rows": literature_rows,
                "source": (
                    "Phase 12 V8 FTS5 + pgvector RRF"
                ),
                "timings": timings,
                "warnings": warnings,
            }

        except Exception as exc:
            errors["vector"] = (
                f"{type(exc).__name__}: {exc}"
            )

    return {
        "components": outputs,
        "component_errors": errors,
        "partial": bool(errors),
        "source": "Read-only multi-engine retrieval",
    }


def execute(
    query,
    registry=None,
    limit=8,
    force=None,
    engine=None,
    record=True
):
    registry = (
        registry
        if registry is not None
        else CostRegistry()
    )

    selected_plan = plan(
        query,
        registry,
        force
    )

    strategy = selected_plan["selected_strategy"]

    start = time.perf_counter()

    result = None
    error = None

    try:
        if strategy == "sql":
            result = sql_execute(
                query,
                limit
            )

        elif strategy == "graph":
            result = graph_execute(
                query,
                limit
            )

        elif strategy == "vector":
            result = vector_execute(
                query,
                limit,
                engine
            )

        else:
            result = hybrid_execute(
                query,
                set(
                    selected_plan["required_capabilities"]
                ),
                limit,
                engine
            )

        ok = not (
            strategy == "hybrid"
            and result.get("partial")
        )

    except Exception as exc:
        error = (
            f"{type(exc).__name__}: {exc}"
        )
        ok = False

    elapsed = round(
        (time.perf_counter() - start) * 1000,
        3
    )

    if record:
        intent_key = "+".join(
            selected_plan["required_capabilities"]
        )

        first_run = not any(
            row["strategy"] == strategy
            and row["intent"] == intent_key
            for row in registry.rows
        )

        registry.record(
            strategy,
            intent_key,
            elapsed,
            ok,
            cold=first_run
        )

        registry.save()

    return {
        "timestamp": stamp(),
        "status": (
            "success"
            if ok
            else "partial"
            if result
            else "error"
        ),
        "plan": selected_plan,
        "actual_latency_ms": elapsed,
        "results": result,
        "error": error,
        "research_warning": RISK_NOTE,
    }


CASES = [
    (
        "How many patients are in the database?",
        "sql"
    ),
    (
        "How many literature chunks are stored?",
        "sql"
    ),
    (
        "Which lab tests are associated with hypertension?",
        "graph"
    ),
    (
        "Which medications are associated with hypertension?",
        "graph"
    ),
    (
        "Find research discussing endothelial dysfunction.",
        "vector"
    ),
    (
        "Explain mechanisms linking hypertension and cardiovascular risk.",
        "vector"
    ),
    (
        "What laboratory findings are associated with hypertension, and what does research say?",
        "hybrid"
    ),
    (
        "What evidence links hypertension to associated lab tests?",
        "hybrid"
    ),
    (
        "How many patients are recorded, and what does research say about hypertension?",
        "hybrid"
    ),
]


def count_results(result):
    if not result:
        return 0

    if "rows" in result:
        return len(result["rows"])

    if "components" in result:
        return sum(
            len(component.get("rows", []))
            for component in result["components"].values()
        )

    return 0


def benchmark(
    registry,
    repeats=3,
    limit=5,
    engine=None
):
    results = []

    for index, (question, expected) in enumerate(
        CASES,
        1
    ):
        for repeat in range(repeats):
            response = execute(
                question,
                registry,
                limit,
                engine=engine
            )

            selected = response["plan"][
                "selected_strategy"
            ]

            results.append({
                "case": index,
                "question": question,
                "expected_strategy": expected,
                "selected_strategy": selected,
                "route_correct": selected == expected,
                "status": response["status"],
                "latency_ms": response[
                    "actual_latency_ms"
                ],
                "result_count": count_results(
                    response.get("results")
                ),
                "justification": response["plan"][
                    "justification"
                ],
                "error": response["error"],
            })

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    with (
        OUT / "benchmark_results.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0])
        )

        writer.writeheader()
        writer.writerows(results)

    routing_accuracy = sum(
        row["route_correct"]
        for row in results
    ) / len(results)

    success_rate = sum(
        row["status"] == "success"
        for row in results
    ) / len(results)

    summary = {
        "timestamp": stamp(),
        "cases": len(CASES),
        "repeats": repeats,
        "routing_accuracy_on_rule_test_set": round(
            routing_accuracy,
            4
        ),
        "execution_success_rate": round(
            success_rate,
            4
        ),
        "cost_profiles": registry.summary(),
        "notes": [
            "Routing expectations are developer-authored.",
            "Latency includes backend and embedding initialization.",
            "Only read-only bounded queries are executed.",
            "No patient-level records are returned.",
            "Biomedical graph ranking is a heuristic, not clinical validation.",
        ],
    }

    save(
        OUT / "benchmark_summary.json",
        summary
    )

    return summary


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Phase 13 biomedical hybrid query planner"
        )
    )

    parser.add_argument(
        "--question"
    )

    parser.add_argument(
        "--mode",
        choices=["auto", *STRATEGIES],
        default="auto"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=8
    )

    parser.add_argument(
        "--benchmark",
        action="store_true"
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=3
    )

    parser.add_argument(
        "--plan-only",
        action="store_true"
    )

    parser.add_argument(
        "--show-costs",
        action="store_true"
    )

    args = parser.parse_args()

    if not 1 <= args.limit <= 100:
        parser.error(
            "--limit must be between 1 and 100"
        )

    if not 1 <= args.repeats <= 20:
        parser.error(
            "--repeats must be between 1 and 20"
        )

    registry = CostRegistry()

    if args.show_costs:
        output = registry.summary()

    elif args.benchmark:
        output = benchmark(
            registry,
            args.repeats,
            args.limit
        )

    elif args.question:
        force = (
            None
            if args.mode == "auto"
            else args.mode
        )

        if args.plan_only:
            output = plan(
                args.question,
                registry,
                force
            )

        else:
            output = execute(
                args.question,
                registry,
                args.limit,
                force
            )

            save(
                OUT / "last_query.json",
                output
            )

    else:
        parser.error(
            "Provide --question, --benchmark or --show-costs"
        )

    print(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
            default=json_default
        )
    )


if __name__ == "__main__":
    main()
