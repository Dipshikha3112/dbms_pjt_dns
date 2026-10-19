
import argparse
import getpass
import os
import sqlite3
from pathlib import Path

import numpy as np
import psycopg
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
SQLITE_PATH = ROOT / "results" / "phase12_store_v3" / "literature.sqlite"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DIMENSION = 384


def connect_pg():
    password = os.getenv("PGPASSWORD")
    if password is None:
        password = getpass.getpass("PostgreSQL password: ")

    return psycopg.connect(
        host=os.getenv("PGHOST", "127.0.0.1"),
        port=int(os.getenv("PGPORT", "5433")),
        dbname=os.getenv(
            "PGDATABASE",
            "adaptive_healthcare_phase9_benchmark"
        ),
        user=os.getenv("PGUSER", "postgres"),
        password=password,
        connect_timeout=10,
    )


def load_chunks():
    if not SQLITE_PATH.is_file():
        raise FileNotFoundError(SQLITE_PATH)

    with sqlite3.connect(SQLITE_PATH) as db:
        rows = db.execute("""
            SELECT
                c.id,
                c.document_id,
                c.page,
                c.body,
                d.title,
                d.source_path
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE TRIM(c.body) <> ''
            ORDER BY c.id
        """).fetchall()

    return rows


def vector_literal(vector):
    values = np.asarray(vector, dtype=np.float32)

    if values.shape != (DIMENSION,):
        raise ValueError(f"Invalid embedding shape: {values.shape}")

    if not np.isfinite(values).all():
        raise ValueError("Embedding contains NaN or infinity")

    return "[" + ",".join(str(float(x)) for x in values) + "]"


def check_schema(conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT EXISTS (
                SELECT 1
                FROM pg_extension
                WHERE extname = 'vector'
            )
        """)
        if not cur.fetchone()[0]:
            raise RuntimeError("pgvector is not enabled")

        cur.execute("""
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = 'literature'
                  AND table_name = 'embeddings'
            )
        """)
        if not cur.fetchone()[0]:
            raise RuntimeError("literature.embeddings table is missing")


def index_chunks(batch_size):
    rows = load_chunks()
    print(f"SQLite literature chunks: {len(rows)}")

    if not rows:
        raise RuntimeError("No literature chunks found")

    model = SentenceTransformer(MODEL_NAME)

    if model.get_sentence_embedding_dimension() != DIMENSION:
        raise RuntimeError("Embedding model dimension mismatch")

    inserted = 0
    skipped = 0

    with connect_pg() as conn:
        check_schema(conn)

        with conn.cursor() as cur:
            cur.execute("""
                SELECT chunk_id
                FROM literature.embeddings
                WHERE embedding_model = %s
            """, (MODEL_NAME,))
            existing = {row[0] for row in cur.fetchall()}

        pending = [row for row in rows if row[0] not in existing]
        skipped = len(rows) - len(pending)

        print(f"Already indexed: {skipped}")
        print(f"Pending embeddings: {len(pending)}")

        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            texts = [row[3] for row in batch]

            vectors = model.encode(
                texts,
                batch_size=min(batch_size, 32),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            with conn.cursor() as cur:
                for row, vector in zip(batch, vectors):
                    chunk_id, document_id, page, body, title, path = row

                    cur.execute("""
                        INSERT INTO literature.embeddings (
                            chunk_id,
                            document_id,
                            pdf_page,
                            embedding_model,
                            embedding
                        )
                        VALUES (%s, %s, %s, %s, %s::vector)
                        ON CONFLICT (chunk_id)
                        DO UPDATE SET
                            document_id = EXCLUDED.document_id,
                            pdf_page = EXCLUDED.pdf_page,
                            embedding_model = EXCLUDED.embedding_model,
                            embedding = EXCLUDED.embedding,
                            indexed_at = NOW()
                    """, (
                        chunk_id,
                        document_id,
                        page,
                        MODEL_NAME,
                        vector_literal(vector),
                    ))

            conn.commit()
            inserted += len(batch)
            print(f"Indexed: {inserted}/{len(pending)}")

        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*)
                FROM literature.embeddings
                WHERE embedding_model = %s
            """, (MODEL_NAME,))
            total = cur.fetchone()[0]

    print("\nPHASE 11 EMBEDDING REPORT")
    print("SQLite chunks:", len(rows))
    print("Previously indexed:", skipped)
    print("Newly indexed:", inserted)
    print("PostgreSQL model embeddings:", total)
    print("Embedding model:", MODEL_NAME)

    if total < len(rows):
        raise RuntimeError("Embedding count is below the SQLite chunk count")

    print("Embedding count check: PASS")


def search(question, top_k):
    model = SentenceTransformer(MODEL_NAME)

    query_vector = model.encode(
        question,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    chunks = load_chunks()
    metadata = {
        row[0]: {
            "document_id": row[1],
            "page": row[2],
            "body": row[3],
            "title": row[4],
            "source_path": row[5],
        }
        for row in chunks
    }

    with connect_pg() as conn:
        check_schema(conn)

        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    chunk_id,
                    document_id,
                    pdf_page,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM literature.embeddings
                WHERE embedding_model = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (
                vector_literal(query_vector),
                MODEL_NAME,
                vector_literal(query_vector),
                top_k,
            ))

            results = cur.fetchall()

    print("\nQUESTION:", question)
    print("=" * 80)

    for rank, result in enumerate(results, 1):
        chunk_id, document_id, page, score = result

        if chunk_id not in metadata:
            raise RuntimeError(
                f"Orphan PostgreSQL embedding: {chunk_id}"
            )

        source = metadata[chunk_id]

        if (
            source["document_id"] != document_id
            or source["page"] != page
        ):
            raise RuntimeError(
                f"Provenance mismatch for chunk: {chunk_id}"
            )

        print(f"\nRESULT {rank}")
        print("Similarity:", round(float(score), 4))
        print("Title:", source["title"])
        print("PDF:", source["source_path"])
        print("Page:", page)
        print("Document ID:", document_id)
        print("Chunk ID:", chunk_id)
        print("Passage:", source["body"][:600])
        print("-" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["index", "search"],
        required=True
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--question", type=str)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if args.mode == "index":
        index_chunks(args.batch_size)
    else:
        if not args.question:
            parser.error("--question is required for search")
        search(args.question, args.top_k)


if __name__ == "__main__":
    main()
