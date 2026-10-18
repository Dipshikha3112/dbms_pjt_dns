
#!/usr/bin/env python3
"""
PHASE 12.3
Adaptive Healthcare Knowledge Database
Clinical Literature Retrieval + Schema-Aware Neo4j Evidence

Research prototype only. Not validated for clinical decisions.

Local writes:
    results/phase12_store_v3/
    results/phase12_runs_v3/

PostgreSQL and Neo4j:
    Read-only access only.

Install:
    pip install pymupdf numpy scikit-learn joblib "psycopg[binary]" neo4j

Python 3.10+
"""

from __future__ import annotations

import argparse
import csv
import getpass
import hashlib
import json
import logging
import os
import re
import sqlite3
import sys
import time
import uuid

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CORPUS = ROOT / "data" / "clinical_literature"

DEFAULT_STORE = ROOT / "results" / "phase12_store_v3"

DEFAULT_RESULTS = ROOT / "results" / "phase12_runs_v3"

LOG = logging.getLogger("phase12")

CATEGORIES = {
    "clinical_tests": "Clinical Laboratory Tests",
    "cvd": "Cardiovascular Disease",
    "diabetes": "Type 2 Diabetes",
    "drug": "Drug Interactions and Adverse Reactions",
    "hypertension": "Hypertension",
}

PATIENT_RE = re.compile(
    r"^[0-9a-fA-F]{8}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{12}$"
)

INJECTION_RE = re.compile(
    r"(?i)("
    r"ignore (?:all |previous |above )?instructions|"
    r"system prompt|developer message|"
    r"you are now|reveal (?:the )?prompt|"
    r"override safety|execute (?:this )?command"
    r")"
)

REFERENCE_RE = re.compile(
    r"(?im)^\s*(?:"
    r"\d+[\.\)]?\s+)?"
    r"(?:references|bibliography|literature cited)"
    r"\s*$"
)

SECTION_RE = re.compile(
    r"(?i)^\s*(?:"
    r"\d+(?:\.\d+)*\.?\s+)?"
    r"(abstract|introduction|background|methods|"
    r"materials and methods|results|discussion|"
    r"conclusion|conclusions|limitations|"
    r"clinical implications|summary)"
    r"\s*$"
)

NOISE_RE = re.compile(
    r"(?i)("
    r"\bemail\s*:|"
    r"\be-?mail\s*:|"
    r"\bcorresponding author\b|"
    r"\bcorrespondence\b|"
    r"\bdoi\s*:|"
    r"\bcrossref\b|"
    r"\bpubmed\b|"
    r"\bcopyright\b|"
    r"\ball rights reserved\b|"
    r"\bfigure\s+\d+|"
    r"\bfig\.\s*\d+|"
    r"\btable\s+\d+|"
    r"\bauthor manuscript\b|"
    r"\baccessed\s+(?:on\s+)?[A-Z][a-z]+|"
    r"\bkeywords?\s*:|"
    r"\breceived\s*:|"
    r"\baccepted\s*:|"
    r"\bpublished\s*:"
    r")"
)

STOPWORDS = {
    "what", "which", "when", "where", "why", "how",
    "does", "with", "from", "this", "that", "these",
    "those", "are", "were", "used", "using", "about",
    "involving", "clinical", "patients", "patient",
    "their", "between", "associated", "assess",
    "assessment", "risk", "risks", "effect", "effects",
    "drug", "drugs", "test", "tests", "disease",
    "diseases", "treatment", "management",
}


# ============================================================
# GENERAL UTILITIES
# ============================================================

def now():
    return datetime.now(timezone.utc).isoformat()


def sha(value):
    return hashlib.sha256(
        str(value).encode("utf-8")
    ).hexdigest()


def clean(value):
    return re.sub(
        r"\s+", " ", str(value)
    ).strip()


def save_json(path, obj):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    path.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
            default=str
        ),
        encoding="utf-8"
    )


def normalize_text(text):
    """
    Repair some PDF extraction artifacts.

    This cannot perfectly reconstruct every two-column PDF.
    """

    text = str(text)

    text = re.sub(
        r"(\w)-\s*\n\s*(\w)",
        r"\1\2",
        text
    )

    text = re.sub(
        r"(?<=\w)-\s+(?=[a-z])",
        "",
        text
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


def tokenize(text):
    return re.findall(
        r"[a-z][a-z0-9-]{2,}",
        str(text).lower()
    )


def query_terms(question):
    return {
        word
        for word in tokenize(question)
        if word not in STOPWORDS
    }


# ============================================================
# SQLITE STORE
# ============================================================

def open_store(store):
    store = Path(store)

    store.mkdir(
        parents=True,
        exist_ok=True
    )

    db = sqlite3.connect(
        store / "literature.sqlite"
    )

    db.execute(
        "PRAGMA foreign_keys=ON"
    )

    db.execute(
        "PRAGMA busy_timeout=5000"
    )

    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            source_path TEXT UNIQUE NOT NULL,
            filename TEXT NOT NULL,
            category TEXT NOT NULL,
            title TEXT NOT NULL,
            doi TEXT,
            pmcid TEXT,
            source_hash TEXT NOT NULL,
            pages INTEGER NOT NULL,
            ingested_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL
                REFERENCES documents(id)
                ON DELETE CASCADE,
            ordinal INTEGER NOT NULL,
            page INTEGER NOT NULL,
            section TEXT NOT NULL,
            body TEXT NOT NULL,
            body_hash TEXT NOT NULL,
            UNIQUE(document_id, ordinal)
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_doc
        ON chunks(document_id);

        CREATE TABLE IF NOT EXISTS audit (
            id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            event TEXT NOT NULL,
            details TEXT NOT NULL
        );
        """
    )

    db.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts
        USING fts5(
            id UNINDEXED,
            body,
            tokenize='unicode61'
        )
        """
    )

    db.commit()

    return db


def audit(db, event, data):
    db.execute(
        "INSERT INTO audit VALUES (?,?,?,?)",
        (
            str(uuid.uuid4()),
            now(),
            event,
            json.dumps(
                data,
                default=str
            )
        )
    )

    db.commit()


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_title(pdf, first_page_text, filename):
    metadata = pdf.metadata or {}

    candidate = clean(
        metadata.get("title") or ""
    )

    bad_metadata = re.compile(
        r"(?i)("
        r"^untitled|"
        r"^microsoft word|"
        r"^article$|"
        r"^full text$|"
        r"^pdf$|"
        r"^nihms\d+|"
        r"\.indd$|"
        r"^\d[\d\s._-]+$|"
        r"^[A-Z]{2,}\d+[_\d.-]*$"
        r")"
    )

    if (
        20 <= len(candidate) <= 260
        and not bad_metadata.search(candidate)
    ):
        return candidate, "pdf_metadata"

    candidates = []

    if len(pdf):
        blocks = pdf[0].get_text(
            "dict"
        ).get("blocks", [])

        for block in blocks:
            for line in block.get(
                "lines", []
            ):
                spans = line.get(
                    "spans", []
                )

                text = clean(
                    " ".join(
                        s.get("text", "")
                        for s in spans
                    )
                )

                if not 20 <= len(text) <= 220:
                    continue

                if NOISE_RE.search(text):
                    continue

                if re.match(
                    r"(?i)^(abstract|introduction|"
                    r"research article|open access|"
                    r"www\.|https?://|journal|"
                    r"volume|original article)",
                    text
                ):
                    continue

                size = max(
                    (
                        s.get("size", 0)
                        for s in spans
                    ),
                    default=0
                )

                y = min(
                    (
                        s.get(
                            "bbox",
                            [0, 999]
                        )[1]
                        for s in spans
                    ),
                    default=999
                )

                if y < 450:
                    candidates.append(
                        (size, -y, text)
                    )

    if candidates:
        candidates.sort(
            reverse=True
        )

        return (
            candidates[0][2],
            "first_page_heading"
        )

    lines = [
        clean(x)
        for x in first_page_text.splitlines()
        if len(clean(x)) >= 20
    ]

    if lines:
        return (
            lines[0][:220],
            "first_page_text"
        )

    return filename, "filename_fallback"


def extract_page_text(page):
    """
    Read text blocks in an approximate column-aware order.
    """

    blocks = page.get_text("blocks")

    width = page.rect.width

    candidates = []

    for block in blocks:
        if len(block) < 5:
            continue

        x0, y0, x1, y1, text = block[:5]

        if not isinstance(text, str):
            continue

        text = normalize_text(text)

        if not text:
            continue

        candidates.append(
            {
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "text": text
            }
        )

    left = []
    right = []
    wide = []

    midpoint = width / 2

    for block in candidates:
        block_width = (
            block["x1"] - block["x0"]
        )

        if block_width > width * 0.65:
            wide.append(block)

        elif block["x0"] < midpoint:
            left.append(block)

        else:
            right.append(block)

    if left and right:
        # Wide title/header blocks are placed first.
        # Column text is then read left-to-right.
        ordered = (
            sorted(
                wide,
                key=lambda b: b["y0"]
            )
            +
            sorted(
                left,
                key=lambda b: b["y0"]
            )
            +
            sorted(
                right,
                key=lambda b: b["y0"]
            )
        )

    else:
        ordered = sorted(
            candidates,
            key=lambda b: (
                b["y0"],
                b["x0"]
            )
        )

    return "\n".join(
        b["text"]
        for b in ordered
    )


def read_pdf(path):
    import fitz

    with fitz.open(path) as pdf:
        if pdf.needs_pass:
            raise ValueError(
                "Password-protected PDF"
            )

        pages = [
            extract_page_text(page)
            for page in pdf
        ]

        if not any(
            len(clean(text)) >= 80
            for text in pages
        ):
            raise ValueError(
                "No extractable text"
            )

        title, method = extract_title(
            pdf,
            pages[0] if pages else "",
            path.stem
        )

        full = "\n".join(
            pages[:3]
        )

        doi_match = re.search(
            r"(?i)\b10\.\d{4,9}/"
            r"[^\s<>\";]+",
            full
        )

        pmcid_match = re.search(
            r"\bPMC\d{5,10}\b",
            full,
            re.I
        )

        doi = (
            doi_match.group(0).rstrip(
                ".,)"
            )
            if doi_match
            else None
        )

        pmcid = (
            pmcid_match.group(0).upper()
            if pmcid_match
            else None
        )

        return (
            pages,
            title,
            method,
            doi,
            pmcid
        )


# ============================================================
# SECTION-AWARE CHUNKING
# ============================================================

def remove_noise_lines(text):
    lines = text.splitlines()

    output = []

    for line in lines:
        line = clean(line)

        if not line:
            output.append("")
            continue

        if NOISE_RE.search(line):
            continue

        if re.fullmatch(
            r"(?:\d+|page\s+\d+"
            r"(?:\s+of\s+\d+)?)",
            line,
            re.I
        ):
            continue

        if re.match(
            r"(?i)^k\s*e\s*y\s*w\s*o\s*r\s*d\s*s",
            line
        ):
            continue

        output.append(line)

    return "\n".join(output)


def find_reference_start(text):
    """
    Return the start of a clearly marked reference section.
    """

    match = REFERENCE_RE.search(text)

    if match:
        return match.start()

    return None


def clean_document_pages(pages):
    """
    Remove reference sections from the document.

    Once an explicit References/Bibliography heading
    is found, subsequent pages are excluded.
    """

    cleaned = []

    references_started = False

    for text in pages:
        if references_started:
            cleaned.append("")
            continue

        text = normalize_text(text)

        position = find_reference_start(text)

        if position is not None:
            text = text[:position]

            references_started = True

        cleaned.append(
            remove_noise_lines(text)
        )

    return cleaned


def detect_section(text):
    for line in text.splitlines():
        line = clean(line)

        match = SECTION_RE.match(line)

        if match:
            return match.group(1).lower()

    return "main_text"


def make_chunks(pages, max_words, overlap):
    """
    Preserve PDF page numbers.

    Each chunk belongs to exactly one page.
    """

    step = max_words - overlap

    for page_number, text in enumerate(
        pages,
        start=1
    ):
        if not clean(text):
            continue

        section = detect_section(text)

        words = clean(text).split()

        for start in range(
            0,
            len(words),
            step
        ):
            part = words[
                start:start + max_words
            ]

            if len(part) >= 12:
                yield (
                    page_number,
                    section,
                    " ".join(part)
                )

            if (
                start + max_words
                >= len(words)
            ):
                break


# ============================================================
# INGESTION
# ============================================================

def ingest(args, db):
    base = Path(
        args.corpus
    ).expanduser().resolve()

    if not base.is_dir():
        raise RuntimeError(
            f"Corpus not found: {base}"
        )

    files = sorted(
        p
        for p in base.rglob("*")
        if p.is_file()
        and p.suffix.lower() == ".pdf"
    )

    if not files:
        raise RuntimeError(
            "No PDFs found"
        )

    stats = Counter(
        files_found=len(files)
    )

    categories = Counter()

    failures = []

    seen_hashes = {
        row[0]: row[1]
        for row in db.execute(
            "SELECT source_hash,source_path "
            "FROM documents"
        )
    }

    for path in files:
        rel = path.relative_to(
            base
        ).as_posix()

        parts = path.relative_to(
            base
        ).parts

        category = (
            CATEGORIES.get(
                parts[0].lower(),
                parts[0]
            )
            if len(parts) > 1
            else "Uncategorized"
        )

        try:
            raw_pages, title, method, doi, pmcid = (
                read_pdf(path)
            )

            pages = clean_document_pages(
                raw_pages
            )

            source_hash = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()

            doc_id = sha(rel)

            previous = db.execute(
                "SELECT source_hash "
                "FROM documents WHERE id=?",
                (doc_id,)
            ).fetchone()

            if (
                previous
                and previous[0] == source_hash
            ):
                stats["unchanged"] += 1
                categories[category] += 1
                continue

            if (
                source_hash in seen_hashes
                and seen_hashes[source_hash] != rel
            ):
                stats["duplicates_skipped"] += 1
                continue

            chunks = list(
                make_chunks(
                    pages,
                    args.chunk_words,
                    args.overlap_words
                )
            )

            if not chunks:
                raise ValueError(
                    "No usable chunks"
                )

            with db:
                old_ids = [
                    row[0]
                    for row in db.execute(
                        "SELECT id FROM chunks "
                        "WHERE document_id=?",
                        (doc_id,)
                    )
                ]

                for cid in old_ids:
                    db.execute(
                        "DELETE FROM chunk_fts "
                        "WHERE id=?",
                        (cid,)
                    )

                db.execute(
                    "DELETE FROM documents "
                    "WHERE id=?",
                    (doc_id,)
                )

                db.execute(
                    """
                    INSERT INTO documents
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        doc_id,
                        rel,
                        path.name,
                        category,
                        title,
                        doi,
                        pmcid,
                        source_hash,
                        len(raw_pages),
                        now()
                    )
                )

                for ordinal, (
                    page,
                    section,
                    body
                ) in enumerate(chunks):

                    cid = sha(
                        doc_id + ":" + str(ordinal)
                    )

                    db.execute(
                        """
                        INSERT INTO chunks
                        VALUES (?,?,?,?,?,?,?)
                        """,
                        (
                            cid,
                            doc_id,
                            ordinal,
                            page,
                            section,
                            body,
                            sha(body)
                        )
                    )

                    db.execute(
                        """
                        INSERT INTO chunk_fts(id,body)
                        VALUES (?,?)
                        """,
                        (
                            cid,
                            body
                        )
                    )

            seen_hashes[source_hash] = rel

            categories[category] += 1

            stats["ingested"] += 1

            stats["new_chunks"] += len(
                chunks
            )

            LOG.info(
                "%s | %s | %s pages | %s chunks",
                rel,
                category,
                len(raw_pages),
                len(chunks)
            )

        except Exception as exc:
            stats["errors"] += 1

            failures.append(
                {
                    "file": rel,
                    "error": str(exc)
                }
            )

            LOG.error(
                "Failed %s: %s",
                rel,
                exc
            )

            if args.strict:
                raise

    summary = dict(stats)

    summary["categories_processed"] = dict(
        categories
    )

    summary["failures"] = failures

    summary["total_documents_in_store"] = (
        db.execute(
            "SELECT COUNT(*) FROM documents"
        ).fetchone()[0]
    )

    summary["total_chunks_in_store"] = (
        db.execute(
            "SELECT COUNT(*) FROM chunks"
        ).fetchone()[0]
    )

    save_json(
        Path(args.results)
        / "ingestion_report.json",
        summary
    )

    audit(
        db,
        "ingest",
        {
            "files": len(files),
            "errors": len(failures)
        }
    )

    return summary


# ============================================================
# TF-IDF INDEX
# ============================================================

def index_signature(db):
    rows = db.execute(
        """
        SELECT id,body_hash
        FROM chunks
        ORDER BY id
        """
    ).fetchall()

    return sha(
        "\n".join(
            a + ":" + b
            for a, b in rows
        )
    )


def build_index(args, db):
    rows = db.execute(
        """
        SELECT id,body
        FROM chunks
        ORDER BY id
        """
    ).fetchall()

    if not rows:
        raise RuntimeError(
            "Run ingest first"
        )

    ids, bodies = zip(*rows)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=args.max_features
    )

    matrix = vectorizer.fit_transform(
        bodies
    )

    artifact = {
        "ids": list(ids),
        "matrix": matrix,
        "vectorizer": vectorizer,
        "signature": index_signature(db),
        "created_at": now(),
        "version": "12.3"
    }

    target = (
        Path(args.store)
        / "retrieval.joblib"
    )

    joblib.dump(
        artifact,
        target,
        compress=3
    )

    audit(
        db,
        "index",
        {
            "chunks": len(ids),
            "features": matrix.shape[1]
        }
    )

    return {
        "chunks": len(ids),
        "features": matrix.shape[1],
        "path": str(target)
    }


def load_index(args, db):
    path = (
        Path(args.store)
        / "retrieval.joblib"
    )

    if not path.exists():
        raise RuntimeError(
            "Index missing. Run build-index."
        )

    # Only load an artifact created locally
    # by this trusted project.
    obj = joblib.load(path)

    if (
        obj["signature"]
        != index_signature(db)
    ):
        raise RuntimeError(
            "Index outdated. Run build-index."
        )

    return obj


# ============================================================
# QUESTION INTENT AND ENTITY DETECTION
# ============================================================

def classify_question(question):
    q = question.lower()

    if re.search(
        r"\b(interaction|interactions|"
        r"coadminister|co-administer|"
        r"contraindication|"
        r"drug.drug)\b",
        q
    ):
        return "DRUG_INTERACTION"

    if re.search(
        r"\b(egfr|creatinine|"
        r"laboratory|biomarker|"
        r"blood test|lab test|"
        r"glomerular filtration)\b",
        q
    ):
        return "LAB_ASSESSMENT"

    if re.search(
        r"\b(hypertension|diabetes|"
        r"cardiovascular|"
        r"kidney disease|"
        r"heart disease)\b",
        q
    ):
        return "DISEASE_EVIDENCE"

    return "GENERAL_RESEARCH"


def detect_topics(question):
    q = question.lower()

    topics = []

    aliases = {
        "apixaban": [
            "apixaban",
            "eliquis"
        ],
        "creatinine": [
            "creatinine"
        ],
        "egfr": [
            "egfr",
            "estimated glomerular filtration"
        ],
        "kidney": [
            "kidney",
            "renal",
            "ckd"
        ],
        "diabetes": [
            "diabetes",
            "hba1c",
            "glycemic"
        ],
        "hypertension": [
            "hypertension",
            "blood pressure"
        ],
        "troponin": [
            "troponin"
        ],
        "cholesterol": [
            "cholesterol",
            "lipid",
            "ldl"
        ]
    }

    for topic, variants in aliases.items():
        if any(
            variant in q
            for variant in variants
        ):
            topics.append(topic)

    return topics


def required_topics(question):
    """
    Identify named concepts that must remain relevant
    in the evidence.

    General terms like 'kidney' are not mandatory
    if the question explicitly names creatinine/eGFR.
    """

    topics = detect_topics(question)

    specific = [
        topic
        for topic in topics
        if topic not in {
            "kidney"
        }
    ]

    return specific or topics


def topic_present(topic, text):
    text = text.lower()

    aliases = {
        "apixaban": [
            "apixaban",
            "eliquis"
        ],
        "egfr": [
            "egfr",
            "estimated glomerular filtration"
        ],
        "creatinine": [
            "creatinine"
        ],
        "kidney": [
            "kidney",
            "renal",
            "ckd"
        ],
        "diabetes": [
            "diabetes",
            "hba1c"
        ],
        "hypertension": [
            "hypertension",
            "blood pressure"
        ],
        "troponin": [
            "troponin"
        ],
        "cholesterol": [
            "cholesterol",
            "lipid",
            "ldl"
        ]
    }

    return any(
        variant in text
        for variant in aliases.get(
            topic,
            [topic]
        )
    )


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(args, db, question):
    obj = load_index(
        args,
        db
    )

    question = clean(question)

    if not question:
        raise ValueError(
            "Empty question"
        )

    qvec = obj[
        "vectorizer"
    ].transform([question])

    if qvec.nnz:
        sims = (
            obj["matrix"]
            @ qvec.T
        ).toarray().ravel()

    else:
        sims = np.zeros(
            len(obj["ids"])
        )

    ranks = defaultdict(float)

    candidate_count = max(
        args.top_k * 12,
        80
    )

    for rank, idx in enumerate(
        np.argsort(-sims)[
            :candidate_count
        ]
    ):
        if sims[idx] > 0:
            cid = obj["ids"][idx]

            ranks[cid] += (
                1 / (60 + rank)
            )

    terms = re.findall(
        r"[A-Za-z0-9]+",
        question
    )

    fts_query = " OR ".join(
        '"' + term + '"'
        for term in terms[:16]
        if len(term) > 1
    )

    if fts_query:
        try:
            hits = db.execute(
                """
                SELECT id
                FROM chunk_fts
                WHERE chunk_fts MATCH ?
                ORDER BY bm25(chunk_fts)
                LIMIT ?
                """,
                (
                    fts_query,
                    candidate_count
                )
            ).fetchall()

            for rank, (cid,) in enumerate(
                hits
            ):
                ranks[cid] += (
                    1 / (60 + rank)
                )

        except sqlite3.OperationalError as exc:
            LOG.warning(
                "FTS skipped: %s",
                exc
            )

    topics = required_topics(
        question
    )

    candidates = []

    for cid, score in sorted(
        ranks.items(),
        key=lambda item: -item[1]
    ):
        row = db.execute(
            """
            SELECT
                c.id,
                c.page,
                c.section,
                c.body,
                d.id,
                d.title,
                d.category,
                d.filename,
                d.source_path,
                d.doi,
                d.pmcid
            FROM chunks c
            JOIN documents d
              ON d.id=c.document_id
            WHERE c.id=?
            """,
            (cid,)
        ).fetchone()

        if not row:
            continue

        keys = (
            "chunk_id",
            "page",
            "section",
            "text",
            "document_id",
            "title",
            "category",
            "filename",
            "source_path",
            "doi",
            "pmcid"
        )

        item = dict(
            zip(keys, row)
        )

        title_text = (
            item["title"]
            + " "
            + item["text"]
        ).lower()

        topic_hits = sum(
            topic_present(
                topic,
                title_text
            )
            for topic in topics
        )

        if topics and not topic_hits:
            continue

        item["topic_hits"] = topic_hits

        item["score"] = round(
            score
            * (1 + 0.25 * topic_hits),
            8
        )

        candidates.append(item)

    candidates.sort(
        key=lambda item: (
            -item["score"],
            -item["topic_hits"]
        )
    )

    return candidates[
        :args.top_k
    ]


# ============================================================
# SENTENCE QUALITY FILTERING
# ============================================================

def split_sentences(text):
    """
    Sentence segmentation for extracted research text.
    """

    text = clean(text)

    return re.split(
        r"(?<=[.!?])\s+(?=[A-Z0-9])",
        text
    )


def is_reference_like(sentence):
    sentence = clean(sentence)

    if NOISE_RE.search(sentence):
        return True

    if re.search(
        r"(?i)\b(?:"
        r"references|bibliography|"
        r"et al\.\s*\d{4}|"
        r"journal of|"
        r"doi\.org|"
        r"vol\.\s*\d+"
        r")\b",
        sentence
    ):
        return True

    if re.search(
        r"\b\d{4};\d+"
        r"(?:\(\d+\))?:"
        r"\d+",
        sentence
    ):
        return True

    if re.search(
        r"https?://|www\.",
        sentence
    ):
        return True

    if re.search(
        r"\S+@\S+\.\S+",
        sentence
    ):
        return True

    if sentence.count(";") >= 4:
        return True

    return False


def sentence_quality(sentence):
    sentence = clean(sentence)

    if not 55 <= len(sentence) <= 450:
        return False

    if INJECTION_RE.search(sentence):
        return False

    if is_reference_like(sentence):
        return False

    if re.search(
        r"(?i)^\s*(?:"
        r"figure|fig\.|table|"
        r"references|"
        r"abbreviations|"
        r"acknowledgements"
        r")\b",
        sentence
    ):
        return False

    if len(
        re.findall(
            r"\b[A-Z]{2,}\b",
            sentence
        )
    ) > 12:
        return False

    if len(
        re.findall(
            r"\b\d{4}\b",
            sentence
        )
    ) > 4:
        return False

    if len(sentence.split()) < 9:
        return False

    return True


def evidence_sentences(
    question,
    evidence,
    max_sentences
):
    terms = query_terms(
        question
    )

    topics = required_topics(
        question
    )

    candidates = []

    for idx, item in enumerate(
        evidence
    ):
        for sentence in split_sentences(
            item["text"]
        ):
            sentence = clean(
                sentence
            )

            if not sentence_quality(
                sentence
            ):
                continue

            words = set(
                tokenize(sentence)
            )

            overlap = len(
                terms & words
            )

            if overlap < min(
                2,
                len(terms)
            ):
                continue

            direct_topic_hits = sum(
                topic_present(
                    topic,
                    sentence
                )
                for topic in topics
            )

            if (
                topics
                and direct_topic_hits == 0
            ):
                continue

            score = (
                overlap * 2
                + direct_topic_hits * 4
                + item["score"] * 100
            )

            candidates.append(
                (
                    score,
                    idx,
                    sentence
                )
            )

    candidates.sort(
        reverse=True,
        key=lambda item: item[0]
    )

    selected = []

    seen = set()

    for score, idx, sentence in candidates:
        normalized = re.sub(
            r"\W+",
            "",
            sentence.lower()
        )

        key = sha(
            normalized
        )

        if key in seen:
            continue

        seen.add(key)

        selected.append(
            {
                "sentence": sentence,
                "evidence_index": idx,
                "score": score
            }
        )

        if len(selected) >= max_sentences:
            break

    return selected


# ============================================================
# POSTGRESQL READ-ONLY CONTEXT
# ============================================================

def pg_context(patient_id):
    if not patient_id:
        return {
            "status": "NOT_REQUESTED",
            "facts": []
        }

    dsn = os.getenv(
        "PHASE12_PG_DSN"
    )

    if not dsn:
        return {
            "status": "NOT_CONFIGURED",
            "facts": []
        }

    try:
        import psycopg

        with psycopg.connect(
            dsn,
            options=(
                "-c default_transaction_read_only=on "
                "-c statement_timeout=5000"
            ),
            connect_timeout=5
        ) as conn:

            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT EXISTS(
                        SELECT 1
                        FROM patient
                        WHERE patient_id::text=%s
                    )
                    """,
                    (patient_id,)
                )

                found = bool(
                    cur.fetchone()[0]
                )

        return {
            "status": (
                "FOUND"
                if found
                else "NOT_FOUND"
            ),
            "facts": (
                [{"type": "patient_exists"}]
                if found
                else []
            )
        }

    except Exception as exc:
        return {
            "status": "ERROR",
            "error": str(exc)[:160],
            "facts": []
        }


# ============================================================
# NEO4J ENTITY MATCHING
# ============================================================

def graph_terms(question):
    topics = required_topics(
        question
    )

    if topics:
        return topics

    return list(
        query_terms(question)
    )[:5]


def graph_entity_score(
    name,
    labels,
    terms,
    intent
):
    """
    Rank matched graph concepts.

    This is terminology relevance, not a clinical
    confidence or causal score.
    """

    name = clean(name)

    lower = name.lower()

    labels = set(
        labels or []
    )

    score = 0

    for term in terms:
        if lower == term:
            score += 100

        elif lower.startswith(
            term + " "
        ):
            score += 45

        elif re.search(
            rf"\b{re.escape(term)}\b",
            lower
        ):
            score += 20

    if "deprecated" in lower:
        score -= 150

    if intent == "DRUG_INTERACTION":
        if "Drug" not in labels:
            score -= 100

        if lower in terms:
            score += 80

        if re.search(
            r"\b(?:mg|tablet|capsule|"
            r"oral product|pill|"
            r"powder|suspension)\b",
            lower
        ):
            score -= 35

    if intent == "LAB_ASSESSMENT":
        if "LabTest" not in labels:
            score -= 100

        if "creatinine" in terms:
            if re.search(
                r"creatinine.*"
                r"(?:serum|plasma|ser/plas)",
                lower
            ):
                score += 90

            if re.search(
                r"(?:serum|plasma).*creatinine",
                lower
            ):
                score += 90

            if (
                lower.startswith("creatinine")
                and (
                    "blood" in lower
                    or "serum" in lower
                    or "plasma" in lower
                )
            ):
                score += 80

            if re.search(
                r"/creatinine|"
                r"creatinine.*\burine\b|"
                r"\bin urine\b|"
                r"\bdbs\b|"
                r"\bbody fluid\b",
                lower
            ):
                score -= 120

        if "egfr" in terms:
            if (
                "egfr" in lower
                or "glomerular filtration" in lower
            ):
                score += 90

    return score


def graph_entity_filter(
    name,
    labels,
    terms,
    intent
):
    name = clean(name)

    lower = name.lower()

    labels = set(
        labels or []
    )

    if not name:
        return False

    if "deprecated" in lower:
        return False

    if intent == "DRUG_INTERACTION":
        if "Drug" not in labels:
            return False

    if intent == "LAB_ASSESSMENT":
        if "LabTest" not in labels:
            return False

        if "creatinine" in terms:
            if re.search(
                r"/creatinine|"
                r"\bin urine\b|"
                r"\bdbs\b|"
                r"\bbody fluid\b",
                lower
            ):
                return False

    return True


def graph_relation_category(
    relation,
    source_labels,
    target_labels
):
    if relation == "RELATED_TO":
        if (
            "Drug" in source_labels
            and "Drug" in target_labels
        ):
            return "DRUG_TERMINOLOGY"

    if relation == "ASSOCIATED_WITH":
        return "CLINICAL_ASSOCIATION"

    if relation == "OBSERVED_WITH_MEDICATION":
        return "OBSERVED_ASSOCIATION"

    return "OTHER_RELATIONSHIP"


def neo4j_context(
    args,
    question,
    patient_id
):
    if not args.graph:
        return {
            "status": "DISABLED",
            "facts": []
        }

    password = os.getenv(
        "PHASE12_NEO4J_PASSWORD"
    )

    if (
        not password
        and args.graph_prompt_password
    ):
        password = getpass.getpass(
            "Existing Neo4j password (not stored): "
        )

    if not password:
        return {
            "status": "NOT_CONFIGURED",
            "facts": [],
            "hint": (
                "Set PHASE12_NEO4J_PASSWORD "
                "or use --graph-prompt-password"
            )
        }

    try:
        from neo4j import (
            GraphDatabase,
            READ_ACCESS
        )

        uri = os.getenv(
            "PHASE12_NEO4J_URI",
            "bolt://127.0.0.1:7687"
        )

        username = os.getenv(
            "PHASE12_NEO4J_USER",
            "neo4j"
        )

        database = os.getenv(
            "PHASE12_NEO4J_DATABASE",
            "neo4j"
        )

        intent = classify_question(
            question
        )

        terms = graph_terms(
            question
        )

        facts = []

        matched_entities = []

        with GraphDatabase.driver(
            uri,
            auth=(username, password),
            connection_timeout=5
        ) as driver:

            with driver.session(
                database=database,
                default_access_mode=READ_ACCESS
            ) as session:

                if patient_id:
                    query = """
                    MATCH (p:Patient {patient_id:$pid})
                          -[r]-(n)
                    WHERE n:Disease
                       OR n:Drug
                       OR n:LabTest
                    RETURN
                        labels(n) AS labels,
                        type(r) AS relation,
                        count(*) AS occurrences
                    ORDER BY occurrences DESC
                    LIMIT $lim
                    """

                    rows = session.run(
                        query,
                        pid=patient_id,
                        lim=args.graph_limit
                    ).data()

                    for row in rows:
                        facts.append(
                            {
                                "type": "patient_topology",
                                "labels": row["labels"],
                                "relation": row["relation"],
                                "count": row["occurrences"]
                            }
                        )

                    return {
                        "status": "OK",
                        "facts": facts,
                        "coverage": "PATIENT_TOPOLOGY_ONLY",
                        "provenance": "Neo4j read-only"
                    }

                # ------------------------------------------------
                # Stage 1: Identify relevant named entities
                # ------------------------------------------------

                query = """
                MATCH (n)
                WHERE (
                    n:Drug OR
                    n:Disease OR
                    n:LabTest
                )
                AND any(
                    term IN $terms
                    WHERE
                    toLower(coalesce(n.name,'')) = term
                    OR
                    toLower(coalesce(n.name,'')) STARTS WITH term + ' '
                    OR
                    toLower(coalesce(n.name,'')) CONTAINS term
                )
                RETURN
                    elementId(n) AS element_id,
                    labels(n) AS labels,
                    n.name AS name,
                    properties(n) AS properties
                LIMIT $candidate_limit
                """

                rows = session.run(
                    query,
                    terms=terms,
                    candidate_limit=300
                ).data()

                scored = []

                for row in rows:
                    name = row.get(
                        "name"
                    ) or ""

                    labels = row.get(
                        "labels"
                    ) or []

                    if not graph_entity_filter(
                        name,
                        labels,
                        terms,
                        intent
                    ):
                        continue

                    score = graph_entity_score(
                        name,
                        labels,
                        terms,
                        intent
                    )

                    if score <= 0:
                        continue

                    scored.append(
                        (
                            score,
                            row
                        )
                    )

                scored.sort(
                    key=lambda item: (
                        -item[0],
                        len(
                            item[1].get(
                                "name"
                            ) or ""
                        )
                    )
                )

                # Prevent a large number of formulation nodes
                # from crowding out the primary entity.

                selected = []

                seen_names = set()

                for score, row in scored:
                    name = row["name"]

                    if name.lower() in seen_names:
                        continue

                    seen_names.add(
                        name.lower()
                    )

                    selected.append(
                        row
                    )

                    if len(selected) >= min(
                        args.graph_limit,
                        12
                    ):
                        break

                for row in selected:
                    matched_entities.append(
                        {
                            "name": row["name"],
                            "labels": row["labels"],
                            "matched_terms": [
                                term
                                for term in terms
                                if term in row["name"].lower()
                            ]
                        }
                    )

                # ------------------------------------------------
                # Stage 2: Traverse relationships only from
                # relevant selected entities
                # ------------------------------------------------

                ids = [
                    row["element_id"]
                    for row in selected
                ]

                if ids:
                    relation_query = """
                    MATCH (n)-[r]-(m)
                    WHERE elementId(n) IN $ids
                      AND (
                        m:Drug OR
                        m:Disease OR
                        m:LabTest
                      )
                      AND type(r) IN [
                        'RELATED_TO',
                        'ASSOCIATED_WITH',
                        'OBSERVED_WITH_MEDICATION'
                      ]
                    RETURN
                        n.name AS source_name,
                        labels(n) AS source_labels,
                        type(r) AS relation,
                        m.name AS target_name,
                        labels(m) AS target_labels
                    LIMIT $lim
                    """

                    relationship_rows = session.run(
                        relation_query,
                        ids=ids,
                        lim=args.graph_limit * 20
                    ).data()

                    seen_relations = set()

                    for row in relationship_rows:
                        source = clean(
                            row.get(
                                "source_name"
                            ) or ""
                        )

                        target = clean(
                            row.get(
                                "target_name"
                            ) or ""
                        )

                        relation = row.get(
                            "relation"
                        )

                        if not (
                            source
                            and target
                            and relation
                        ):
                            continue

                        key = (
                            source,
                            relation,
                            target
                        )

                        if key in seen_relations:
                            continue

                        seen_relations.add(
                            key
                        )

                        category = graph_relation_category(
                            relation,
                            row["source_labels"],
                            row["target_labels"]
                        )

                        fact = {
                            "type": "entity_relationship",
                            "source_labels": row["source_labels"],
                            "source_name": source,
                            "relation": relation,
                            "target_labels": row["target_labels"],
                            "target_name": target,
                            "evidence_category": category
                        }

                        # Drug formulation relationships are
                        # terminology, not interaction risk.

                        if intent == "DRUG_INTERACTION":
                            if category == "DRUG_TERMINOLOGY":
                                fact["clinical_interaction_verified"] = False

                        facts.append(
                            fact
                        )

                        if len(facts) >= args.graph_limit:
                            break

                # If no relationships were found, retain
                # the best matching entity as a graph fact.

                if not facts:
                    for row in selected[:args.graph_limit]:
                        facts.append(
                            {
                                "type": "entity",
                                "source_labels": row["labels"],
                                "source_name": row["name"],
                                "relation": None,
                                "target_labels": None,
                                "target_name": "",
                                "evidence_category": "ENTITY_ONLY"
                            }
                        )

        matched_terms = sorted(
            {
                term
                for entity in matched_entities
                for term in entity["matched_terms"]
            }
        )

        return {
            "status": "OK",
            "facts": facts,
            "terms": terms,
            "matched_entities": matched_entities,
            "matched_terms": matched_terms,
            "coverage": (
                "MATCHED_ENTITIES"
                if matched_entities
                else "NO_MATCH"
            ),
            "provenance": (
                "Neo4j read-only; graph relationships "
                "are not automatically clinical evidence"
            )
        }

    except Exception as exc:
        return {
            "status": "ERROR",
            "error": str(exc)[:220],
            "facts": []
        }


# ============================================================
# EVIDENCE FUSION
# ============================================================

def format_graph_evidence(
    graph,
    intent
):
    clinical_lines = []

    terminology_lines = []

    entity_lines = []

    graph_citations = []

    for idx, fact in enumerate(
        graph.get(
            "facts",
            []
        )
    ):
        gid = f"G{idx+1}"

        citation = {
            "id": gid,
            **fact,
            "source": (
                "Neo4j clinical knowledge graph"
            )
        }

        graph_citations.append(
            citation
        )

        if fact["type"] == "entity":
            entity_lines.append(
                f"{fact['source_name']} [{gid}]"
            )

            continue

        if fact["type"] != "entity_relationship":
            continue

        line = (
            f"{fact['source_name']} "
            f"—[{fact['relation']}]— "
            f"{fact['target_name']} "
            f"[{gid}]"
        )

        category = fact.get(
            "evidence_category"
        )

        if category == "DRUG_TERMINOLOGY":
            terminology_lines.append(
                line
            )

        elif category in {
            "CLINICAL_ASSOCIATION",
            "OBSERVED_ASSOCIATION"
        }:
            clinical_lines.append(
                line
            )

        else:
            entity_lines.append(
                line
            )

    # For drug-interaction questions, association
    # edges are not treated as proven DDIs.

    if intent == "DRUG_INTERACTION":
        clinical_lines = []

    return {
        "clinical_lines": clinical_lines,
        "terminology_lines": terminology_lines,
        "entity_lines": entity_lines,
        "citations": graph_citations
    }


def evidence_status(
    literature_lines,
    graph,
    formatted_graph,
    intent
):
    has_literature = bool(
        literature_lines
    )

    has_entities = bool(
        graph.get(
            "matched_entities"
        )
    )

    has_clinical_edges = bool(
        formatted_graph[
            "clinical_lines"
        ]
    )

    if not has_literature:
        return "ABSTAIN_INSUFFICIENT_LITERATURE"

    if graph.get("status") != "OK":
        return "LITERATURE_SUPPORTED_GRAPH_UNAVAILABLE"

    if intent == "DRUG_INTERACTION":
        if has_entities:
            return (
                "LITERATURE_SUPPORTED_"
                "GRAPH_INTERACTION_UNVERIFIED"
            )

    if has_clinical_edges:
        return (
            "LITERATURE_AND_GRAPH_"
            "ASSOCIATIONS_RETRIEVED"
        )

    if has_entities:
        return (
            "LITERATURE_SUPPORTED_"
            "GRAPH_ENTITY_MATCHED"
        )

    return "LITERATURE_ONLY_GRAPH_NO_MATCH"


# ============================================================
# ANSWERING
# ============================================================

def ask(
    args,
    db,
    question,
    patient_id=None,
    do_audit=True
):
    started = time.perf_counter()

    if (
        patient_id
        and not PATIENT_RE.fullmatch(
            patient_id
        )
    ):
        raise ValueError(
            "Patient ID must be a UUID"
        )

    intent = classify_question(
        question
    )

    topics = detect_topics(
        question
    )

    required = required_topics(
        question
    )

    evidence = retrieve(
        args,
        db,
        question
    )

    selected_sentences = evidence_sentences(
        question,
        evidence,
        args.max_sentences
    )

    # Only cite evidence actually used in the answer.
    used_indices = sorted(
        {
            item["evidence_index"]
            for item in selected_sentences
        }
    )

    citation_id_map = {
        idx: f"E{number}"
        for number, idx in enumerate(
            used_indices,
            start=1
        )
    }

    citations = []

    for idx in used_indices:
        ev = evidence[idx]

        citations.append(
            {
                "id": citation_id_map[idx],
                "chunk_id": ev["chunk_id"],
                "document_id": ev["document_id"],
                "title": ev["title"],
                "category": ev["category"],
                "filename": ev["filename"],
                "source_path": ev["source_path"],
                "page": ev["page"],
                "section": ev["section"],
                "doi": ev["doi"],
                "pmcid": ev["pmcid"],
                "excerpt": ev["text"][:350]
            }
        )

    literature_lines = []

    for item in selected_sentences:
        eid = citation_id_map[
            item["evidence_index"]
        ]

        literature_lines.append(
            f"{item['sentence']} [{eid}]"
        )

    graph = neo4j_context(
        args,
        question,
        patient_id
    )

    pg = pg_context(
        patient_id
    )

    formatted_graph = format_graph_evidence(
        graph,
        intent
    )

    parts = []

    if literature_lines:
        parts.append(
            "LITERATURE EVIDENCE "
            "(source extracts, not clinical conclusions):\n"
            + "\n".join(
                literature_lines
            )
        )

    else:
        parts.append(
            "LITERATURE EVIDENCE:\n"
            "Insufficient directly relevant evidence."
        )

    if args.graph:
        if patient_id:
            parts.append(
                "GRAPH PATIENT CONTEXT:\n"
                "Only relationship counts are returned. "
                "No patient-specific clinical conclusion."
            )

        else:
            matched = graph.get(
                "matched_entities",
                []
            )

            if matched:
                names = [
                    item["name"]
                    for item in matched[:5]
                ]

                parts.append(
                    "MATCHED GRAPH ENTITIES:\n"
                    + "\n".join(names)
                )

            if formatted_graph[
                "clinical_lines"
            ]:
                parts.append(
                    "GRAPH ASSOCIATIONS "
                    "(not proof of causality):\n"
                    + "\n".join(
                        formatted_graph[
                            "clinical_lines"
                        ]
                    )
                )

            if formatted_graph[
                "terminology_lines"
            ]:
                parts.append(
                    "GRAPH TERMINOLOGY "
                    "(formulations and related concepts; "
                    "not drug-interaction evidence):\n"
                    + "\n".join(
                        formatted_graph[
                            "terminology_lines"
                        ]
                    )
                )

            if (
                intent == "DRUG_INTERACTION"
            ):
                parts.append(
                    "GRAPH DRUG-INTERACTION EVIDENCE:\n"
                    "No verified drug-drug interaction "
                    "relationship established by this query."
                )

            elif (
                not formatted_graph[
                    "clinical_lines"
                ]
                and not formatted_graph[
                    "terminology_lines"
                ]
            ):
                parts.append(
                    "GRAPH CLINICAL EVIDENCE:\n"
                    "No question-relevant clinical "
                    "relationship established."
                )

    if patient_id:
        parts.append(
            "Patient-specific diagnosis or treatment "
            "recommendations are withheld."
        )

    status = evidence_status(
        literature_lines,
        graph,
        formatted_graph,
        intent
    )

    result = {
        "question": question,
        "question_intent": intent,
        "status": status,
        "answer": "\n\n".join(
            parts
        ),
        "citations": citations,
        "graph_citations": formatted_graph[
            "citations"
        ],
        "retrieval_diagnostics": {
            "retrieved_chunks": len(
                evidence
            ),
            "accepted_sentences": len(
                literature_lines
            ),
            "detected_topics": topics,
            "required_topics": required,
            "retrieved_sources": list(
                dict.fromkeys(
                    item["source_path"]
                    for item in evidence
                )
            )
        },
        "postgresql": pg,
        "neo4j": graph,
        "latency_ms": round(
            (
                time.perf_counter()
                - started
            ) * 1000,
            2
        ),
        "note": (
            "Research-level evidence retrieval. "
            "Literature, graph terminology, and "
            "graph associations are distinguished. "
            "No automated clinical inference."
        ),
        "disclaimer": (
            "Research only; not validated for "
            "diagnosis, treatment, or clinical decisions."
        )
    }

    if do_audit:
        audit(
            db,
            "ask",
            {
                "question_hash": sha(
                    question
                ),
                "citation_count": len(
                    citations
                ),
                "graph_fact_count": len(
                    formatted_graph[
                        "citations"
                    ]
                ),
                "patient_context_requested": bool(
                    patient_id
                ),
                "status": status
            }
        )

    return result


# ============================================================
# DIAGNOSTICS
# ============================================================

def doctor(args, db):
    doc_count = db.execute(
        "SELECT COUNT(*) FROM documents"
    ).fetchone()[0]

    chunk_count = db.execute(
        "SELECT COUNT(*) FROM chunks"
    ).fetchone()[0]

    fts_count = db.execute(
        "SELECT COUNT(*) FROM chunk_fts"
    ).fetchone()[0]

    categories = dict(
        db.execute(
            """
            SELECT category,COUNT(*)
            FROM documents
            GROUP BY category
            """
        ).fetchall()
    )

    result = {
        "phase": "12.3",
        "documents": doc_count,
        "chunks": chunk_count,
        "fts_chunks": fts_count,
        "fts_consistent": (
            chunk_count == fts_count
        ),
        "categories": categories,
        "postgresql_configured": bool(
            os.getenv(
                "PHASE12_PG_DSN"
            )
        ),
        "neo4j_password_in_environment": bool(
            os.getenv(
                "PHASE12_NEO4J_PASSWORD"
            )
        ),
        "production_database_writes": False
    }

    index_path = (
        Path(args.store)
        / "retrieval.joblib"
    )

    result["index_exists"] = (
        index_path.exists()
    )

    if index_path.exists():
        try:
            load_index(
                args,
                db
            )

            result[
                "index_current"
            ] = True

        except Exception as exc:
            result[
                "index_current"
            ] = False

            result[
                "index_error"
            ] = str(exc)

    return result


# ============================================================
# DOCUMENT CATALOG
# ============================================================

def catalog(args, db):
    rows = db.execute(
        """
        SELECT
            d.category,
            d.title,
            d.filename,
            d.source_path,
            d.pages,
            d.doi,
            d.pmcid,
            COUNT(c.id) AS chunks
        FROM documents d
        LEFT JOIN chunks c
          ON d.id=c.document_id
        GROUP BY d.id
        ORDER BY d.category,d.title
        """
    ).fetchall()

    keys = [
        "category",
        "title",
        "filename",
        "source_path",
        "pages",
        "doi",
        "pmcid",
        "chunks"
    ]

    data = [
        dict(
            zip(keys, row)
        )
        for row in rows
    ]

    target = (
        Path(args.results)
        / "document_catalog.csv"
    )

    target.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with target.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=keys
        )

        writer.writeheader()

        writer.writerows(
            data
        )

    return {
        "documents": len(data),
        "catalog_csv": str(target)
    }


# ============================================================
# EVALUATION
# ============================================================

def evaluate(args, db):
    if not args.eval_file:
        raise ValueError(
            "Provide --eval-file"
        )

    cases = [
        json.loads(line)
        for line in Path(
            args.eval_file
        ).read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if not cases:
        raise ValueError(
            "Evaluation file is empty"
        )

    rows = []

    for case in cases:
        question = case[
            "question"
        ]

        response = ask(
            args,
            db,
            question,
            do_audit=False
        )

        got = {
            citation[
                "source_path"
            ]
            for citation in response[
                "citations"
            ]
        }

        expected = set(
            case.get(
                "relevant_sources",
                []
            )
        )

        hits = len(
            got & expected
        )

        rows.append(
            {
                "question_hash": sha(
                    question
                ),
                "expected_count": len(
                    expected
                ),
                "retrieved_count": len(
                    got
                ),
                "hits": hits,
                "recall_at_k": (
                    hits / len(expected)
                    if expected
                    else None
                ),
                "precision_at_k": (
                    hits / len(got)
                    if got and expected
                    else None
                ),
                "latency_ms": response[
                    "latency_ms"
                ],
                "status": response[
                    "status"
                ]
            }
        )

    labeled = [
        row
        for row in rows
        if row[
            "expected_count"
        ]
    ]

    summary = {
        "cases": len(rows),
        "labeled_cases": len(
            labeled
        ),
        "mean_recall_at_k": (
            float(
                np.mean(
                    [
                        row[
                            "recall_at_k"
                        ]
                        for row in labeled
                    ]
                )
            )
            if labeled
            else None
        ),
        "mean_precision_at_k": (
            float(
                np.mean(
                    [
                        row[
                            "precision_at_k"
                        ]
                        for row in labeled
                    ]
                )
            )
            if labeled
            else None
        ),
        "mean_latency_ms": float(
            np.mean(
                [
                    row[
                        "latency_ms"
                    ]
                    for row in rows
                ]
            )
        ),
        "note": (
            "Retrieval evaluation only. "
            "Not clinical accuracy validation."
        )
    }

    outdir = (
        Path(args.results)
        / (
            "evaluation_"
            + datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
        )
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True
    )

    save_json(
        outdir / "summary.json",
        summary
    )

    with (
        outdir / "cases.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    return {
        "output": str(outdir),
        **summary
    }


# ============================================================
# COMMAND LINE
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "command",
        choices=[
            "ingest",
            "build-index",
            "ask",
            "doctor",
            "catalog",
            "evaluate",
            "all"
        ]
    )

    parser.add_argument(
        "--corpus",
        default=str(
            DEFAULT_CORPUS
        )
    )

    parser.add_argument(
        "--store",
        default=str(
            DEFAULT_STORE
        )
    )

    parser.add_argument(
        "--results",
        default=str(
            DEFAULT_RESULTS
        )
    )

    parser.add_argument(
        "--question"
    )

    parser.add_argument(
        "--patient-id"
    )

    parser.add_argument(
        "--graph",
        action="store_true"
    )

    parser.add_argument(
        "--graph-prompt-password",
        action="store_true"
    )

    parser.add_argument(
        "--graph-limit",
        type=int,
        default=12
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=8
    )

    parser.add_argument(
        "--chunk-words",
        type=int,
        default=220
    )

    parser.add_argument(
        "--overlap-words",
        type=int,
        default=45
    )

    parser.add_argument(
        "--max-features",
        type=int,
        default=60000
    )

    parser.add_argument(
        "--max-sentences",
        type=int,
        default=5
    )

    parser.add_argument(
        "--eval-file"
    )

    parser.add_argument(
        "--strict",
        action="store_true"
    )

    args = parser.parse_args()

    if not (
        1 <= args.top_k <= 50
    ):
        parser.error(
            "--top-k must be 1..50"
        )

    if (
        args.chunk_words < 30
        or not (
            0 <= args.overlap_words
            < args.chunk_words
        )
    ):
        parser.error(
            "Invalid chunk settings"
        )

    if not (
        1 <= args.graph_limit <= 100
    ):
        parser.error(
            "--graph-limit must be 1..100"
        )

    if args.max_features < 100:
        parser.error(
            "--max-features must be >=100"
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s"
    )

    db = open_store(
        Path(args.store)
    )

    try:
        if args.command == "ingest":
            output = ingest(
                args,
                db
            )

        elif args.command == "build-index":
            output = build_index(
                args,
                db
            )

        elif args.command == "doctor":
            output = doctor(
                args,
                db
            )

        elif args.command == "catalog":
            output = catalog(
                args,
                db
            )

        elif args.command == "evaluate":
            output = evaluate(
                args,
                db
            )

        elif args.command == "ask":
            if not args.question:
                parser.error(
                    "ask requires --question"
                )

            output = ask(
                args,
                db,
                args.question,
                args.patient_id
            )

        else:
            if not args.question:
                parser.error(
                    "all requires --question"
                )

            output = {
                "ingest": ingest(
                    args,
                    db
                ),
                "index": build_index(
                    args,
                    db
                ),
                "catalog": catalog(
                    args,
                    db
                ),
                "answer": ask(
                    args,
                    db,
                    args.question,
                    args.patient_id
                ),
                "doctor": doctor(
                    args,
                    db
                )
            }

        print(
            json.dumps(
                output,
                indent=2,
                ensure_ascii=False,
                default=str
            )
        )

    finally:
        db.close()


if __name__ == "__main__":
    try:
        main()

    except (
        Exception,
        KeyboardInterrupt
    ) as exc:

        LOG.error(
            "%s: %s",
            type(exc).__name__,
            exc
        )

        sys.exit(1)
