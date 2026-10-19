
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STORE = ROOT / "results" / "phase12_store_v3"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_audit(store, corpus, output):
    database = store / "literature.sqlite"
    if not database.is_file():
        raise FileNotFoundError(database)

    db = sqlite3.connect(database)
    db.row_factory = sqlite3.Row

    report = {
        "phase": 11,
        "task": "Literature ingestion provenance audit",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "store": str(store),
        "corpus": str(corpus),
        "documents": [],
        "issues": [],
    }

    try:
        integrity = db.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]

        documents = db.execute("""
            SELECT id, source_path, filename, category,
                   title, doi, pmcid, source_hash,
                   pages, ingested_at
            FROM documents
            ORDER BY source_path
        """).fetchall()

        chunks = db.execute("""
            SELECT c.id, c.document_id, c.ordinal,
                   c.page, c.section, c.body, c.body_hash,
                   d.title, d.source_path, d.pages
            FROM chunks c
            LEFT JOIN documents d
              ON d.id = c.document_id
            ORDER BY c.document_id, c.ordinal
        """).fetchall()

        fts_ids = [
            row[0] for row in db.execute(
                "SELECT id FROM chunk_fts"
            )
        ]

        document_ids = {d["id"] for d in documents}
        chunk_ids = {c["id"] for c in chunks}
        issues = report["issues"]
        seen_hashes = {}
        verified_files = 0

        for doc in documents:
            item = dict(doc)
            source_file = corpus / doc["source_path"]
            item["local_file_exists"] = source_file.is_file()
            item["hash_verified"] = False

            if source_file.is_file():
                item["hash_verified"] = (
                    sha256_file(source_file)
                    == doc["source_hash"]
                )
                if item["hash_verified"]:
                    verified_files += 1
                else:
                    issues.append({
                        "type": "SOURCE_HASH_MISMATCH",
                        "document_id": doc["id"]
                    })
            else:
                issues.append({
                    "type": "SOURCE_FILE_MISSING",
                    "document_id": doc["id"],
                    "path": str(source_file)
                })

            if not doc["title"].strip():
                issues.append({
                    "type": "MISSING_TITLE",
                    "document_id": doc["id"]
                })

            if doc["pages"] <= 0:
                issues.append({
                    "type": "INVALID_PAGE_COUNT",
                    "document_id": doc["id"]
                })

            previous = seen_hashes.get(doc["source_hash"])
            if previous is not None:
                issues.append({
                    "type": "DUPLICATE_SOURCE_HASH",
                    "document_id": doc["id"],
                    "other_document_id": previous
                })
            else:
                seen_hashes[doc["source_hash"]] = doc["id"]

            item["citation_source"] = {
                "document_id": doc["id"],
                "title": doc["title"],
                "filename": doc["filename"],
                "source_path": doc["source_path"],
                "doi": doc["doi"],
                "pmcid": doc["pmcid"]
            }
            report["documents"].append(item)

        valid_citations = 0
        example_citations = []

        for chunk in chunks:
            if chunk["document_id"] not in document_ids:
                issues.append({
                    "type": "ORPHAN_CHUNK",
                    "chunk_id": chunk["id"]
                })
                continue

            if not (1 <= chunk["page"] <= chunk["pages"]):
                issues.append({
                    "type": "INVALID_PAGE_NUMBER",
                    "chunk_id": chunk["id"],
                    "page": chunk["page"]
                })
                continue

            if not chunk["body"].strip():
                issues.append({
                    "type": "EMPTY_CHUNK",
                    "chunk_id": chunk["id"]
                })
                continue

            actual_hash = hashlib.sha256(
                chunk["body"].encode("utf-8")
            ).hexdigest()

            if actual_hash != chunk["body_hash"]:
                issues.append({
                    "type": "CHUNK_HASH_MISMATCH",
                    "chunk_id": chunk["id"]
                })
                continue

            valid_citations += 1

            if len(example_citations) < 10:
                example_citations.append({
                    "chunk_id": chunk["id"],
                    "document_id": chunk["document_id"],
                    "title": chunk["title"],
                    "source_path": chunk["source_path"],
                    "page": chunk["page"],
                    "section": chunk["section"],
                    "citation": (
                        f'{chunk["title"]}, '
                        f'PDF page {chunk["page"]}'
                    ),
                    "preview": chunk["body"][:220]
                })

        fts_counts = Counter(fts_ids)
        missing_fts = sorted(chunk_ids - set(fts_ids))
        extra_fts = sorted(set(fts_ids) - chunk_ids)
        duplicated_fts = [
            cid for cid, count in fts_counts.items()
            if count > 1
        ]

        if missing_fts:
            issues.append({
                "type": "MISSING_FTS_ENTRIES",
                "count": len(missing_fts)
            })
        if extra_fts:
            issues.append({
                "type": "ORPHAN_FTS_ENTRIES",
                "count": len(extra_fts)
            })
        if duplicated_fts:
            issues.append({
                "type": "DUPLICATE_FTS_ENTRIES",
                "count": len(duplicated_fts)
            })

        report["example_citations"] = example_citations
        report["summary"] = {
            "sqlite_integrity": integrity,
            "documents": len(documents),
            "chunks": len(chunks),
            "fts_entries": len(fts_ids),
            "verified_source_files": verified_files,
            "valid_page_citations": valid_citations,
            "invalid_page_citations": (
                len(chunks) - valid_citations
            ),
            "issues_by_type": dict(Counter(
                item["type"] for item in issues
            )),
            "acceptance_pass": (
                integrity == "ok"
                and len(documents) > 0
                and len(chunks) > 0
                and verified_files == len(documents)
                and valid_citations == len(chunks)
                and len(fts_ids) == len(chunks)
                and not issues
            )
        }

    finally:
        db.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(json.dumps(report["summary"], indent=2))
    print(f"\nReport: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--store", type=Path, default=DEFAULT_STORE
    )
    parser.add_argument(
        "--corpus", type=Path,
        default=ROOT / "data" / "clinical_literature"
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results" / "phase11_audit"
        / "literature_audit.json"
    )
    args = parser.parse_args()

    run_audit(
        args.store.resolve(),
        args.corpus.resolve(),
        args.output.resolve()
    )
