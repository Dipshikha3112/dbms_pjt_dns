import os
import sqlite3
from pathlib import Path

import psycopg
from psycopg import sql
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
SQLITE_PATH = ROOT / "results" / "phase12_store_v3" / "literature.sqlite"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DB_NAME = "adaptive_healthcare_rag"
BATCH_SIZE = 32

if not SQLITE_PATH.is_file():
    raise FileNotFoundError(f"SQLite database not found: {SQLITE_PATH}")

if "PGPASSWORD" not in os.environ:
    raise RuntimeError("Set PGPASSWORD in PowerShell before running.")

print("SQLite:", SQLITE_PATH)
print("Loading embedding model:", MODEL_NAME)

model = SentenceTransformer(MODEL_NAME)

with sqlite3.connect(SQLITE_PATH) as sqlite_conn:
    sqlite_conn.row_factory = sqlite3.Row

    rows = sqlite_conn.execute("""
        SELECT
            c.id AS chunk_id,
            c.document_id,
            c.page,
            c.section,
            c.body,
            d.title,
            d.source_path
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE TRIM(c.body) <> ''
        ORDER BY c.document_id, c.ordinal
    """).fetchall()

print("Literature chunks found:", len(rows))

if not rows:
    raise RuntimeError("No literature chunks found in SQLite.")

with psycopg.connect(
    host=os.getenv("PGHOST", "127.0.0.1"),
    port=int(os.getenv("PGPORT", "5432")),
    dbname=DB_NAME,
    user=os.getenv("PGUSER", "postgres"),
    password=os.environ["PGPASSWORD"],
) as conn:

    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("CREATE SCHEMA IF NOT EXISTS literature")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS literature.embeddings (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                page INTEGER,
                section TEXT,
                body TEXT NOT NULL,
                title TEXT,
                source_path TEXT,
                embedding_model TEXT NOT NULL,
                embedding vector(384) NOT NULL,
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

    conn.commit()
    register_vector(conn)

    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]
        texts = [r["body"][:2000] for r in batch]

        vectors = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        records = []

        for row, vector in zip(batch, vectors):
            records.append((
                row["chunk_id"],
                row["document_id"],
                row["page"],
                row["section"],
                row["body"],
                row["title"],
                row["source_path"],
                MODEL_NAME,
                vector,
            ))

        with conn.cursor() as cur:
            cur.executemany("""
                INSERT INTO literature.embeddings (
                    chunk_id,
                    document_id,
                    page,
                    section,
                    body,
                    title,
                    source_path,
                    embedding_model,
                    embedding
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (chunk_id)
                DO UPDATE SET
                    document_id = EXCLUDED.document_id,
                    page = EXCLUDED.page,
                    section = EXCLUDED.section,
                    body = EXCLUDED.body,
                    title = EXCLUDED.title,
                    source_path = EXCLUDED.source_path,
                    embedding_model = EXCLUDED.embedding_model,
                    embedding = EXCLUDED.embedding,
                    updated_at = NOW()
            """, records)

        conn.commit()

        print(
            f"Imported {min(start + BATCH_SIZE, len(rows))}"
            f"/{len(rows)} chunks"
        )

    with conn.cursor() as cur:
        cur.execute("""
            CREATE INDEX IF NOT EXISTS literature_embeddings_hnsw_idx
            ON literature.embeddings
            USING hnsw (embedding vector_cosine_ops)
        """)

        cur.execute("ANALYZE literature.embeddings")

        cur.execute("""
            SELECT COUNT(*)
            FROM literature.embeddings
        """)
        count = cur.fetchone()[0]

        cur.execute("""
            SELECT chunk_id, title, page
            FROM literature.embeddings
            ORDER BY chunk_id
            LIMIT 3
        """)
        samples = cur.fetchall()

    conn.commit()

print("\nPGVECTOR IMPORT COMPLETE")
print("Database:", DB_NAME)
print("Table: literature.embeddings")
print("Stored chunks:", count)
print("Sample records:")

for sample in samples:
    print(sample)