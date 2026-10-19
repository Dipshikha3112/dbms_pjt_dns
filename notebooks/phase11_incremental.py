#!/usr/bin/env python3
"""Phase 11: incremental PDF ingestion, curated metadata and pgvector synchronization.
Run from the project root; see --help. Requires notebooks/12_full.py and existing
results/phase12_store_v3/literature.sqlite. Does not touch Phase 9 tables.
"""
import argparse, csv, getpass, hashlib, importlib.util, json, os, re, sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

def sha(s): return hashlib.sha256(s.encode('utf-8')).hexdigest()
def filehash(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()

def legacy():
    path=ROOT/'notebooks'/'12_full.py'
    if not path.exists(): raise RuntimeError(f'Missing {path}; copy your existing 12_full.py here')
    spec=importlib.util.spec_from_file_location('phase11_legacy_ingest',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def metadata_tables(db):
    db.executescript('''
    CREATE TABLE IF NOT EXISTS literature_metadata (
      document_id TEXT PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
      verified_title TEXT, publisher TEXT, publication_year INTEGER,
      source_url TEXT, doi TEXT, pmcid TEXT, verification_status TEXT NOT NULL DEFAULT 'unverified',
      metadata_updated_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_chunks_body_hash ON chunks(body_hash);
    ''')

def overrides(path):
    if not path.exists(): return {}
    data=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data,dict): raise ValueError('Metadata overrides must be a JSON object keyed by source_path')
    allowed={'title','publisher','publication_year','source_url','doi','pmcid','verification_status'}
    for key,item in data.items():
        if not isinstance(item,dict) or set(item)-allowed: raise ValueError(f'Invalid metadata fields for {key}')
        if item.get('verification_status','unverified') not in ('verified','unverified'):
            raise ValueError(f'Invalid verification_status for {key}')
        if item.get('source_url') and not re.match(r'^https?://',item['source_url']):
            raise ValueError(f'source_url must be http(s): {key}')
    return data

def apply_metadata(db,items,stats):
    rows=db.execute('SELECT id,source_path,title,doi,pmcid FROM documents').fetchall()
    known={r[1] for r in rows}
    unknown=set(items)-known
    if unknown: print('WARNING: metadata overrides have no matching PDF:',sorted(unknown))
    with db:
        for did,path,title,doi,pmcid in rows:
            o=items.get(path,{})
            if not o: continue
            status=o.get('verification_status','unverified')
            db.execute('''INSERT INTO literature_metadata
            (document_id,verified_title,publisher,publication_year,source_url,doi,pmcid,verification_status,metadata_updated_at)
            VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(document_id) DO UPDATE SET
            verified_title=excluded.verified_title,publisher=excluded.publisher,
            publication_year=excluded.publication_year,source_url=excluded.source_url,
            doi=excluded.doi,pmcid=excluded.pmcid,verification_status=excluded.verification_status,
            metadata_updated_at=excluded.metadata_updated_at''',
            (did,o.get('title'),o.get('publisher'),o.get('publication_year'),o.get('source_url'),
             o.get('doi',doi),o.get('pmcid',pmcid),status,now()))
            stats['metadata_overrides_applied']+=1

def ingest(db,corpus,chunk_words,overlap,prune,stats):
    if not corpus.is_dir():raise RuntimeError(f'Corpus directory not found: {corpus}')
    if not 0<=overlap<chunk_words:raise ValueError('Require 0 <= overlap < chunk_words')
    m=legacy()
    paths=sorted(p for p in corpus.rglob('*') if p.is_file() and p.suffix.lower()=='.pdf')
    if not paths:raise RuntimeError('No PDFs found: refusing to prune an empty corpus')
    existing={r[1]:r for r in db.execute('SELECT id,source_path,source_hash FROM documents')}
    seen_hashes={r[2]:r[1] for r in existing.values()}
    encountered=set()
    for p in paths:
        rel=p.relative_to(corpus).as_posix();encountered.add(rel)
        digest=filehash(p)
        old=existing.get(rel)
        if old and old[2]==digest:
            stats['unchanged_pdfs']+=1;continue
        if digest in seen_hashes and seen_hashes[digest]!=rel:
            stats['duplicate_pdfs_skipped']+=1;print('DUPLICATE:',rel,'matches',seen_hashes[digest]);continue
        try:
            pages,title,method,doi,pmcid=m.read_pdf(p)
            cleaned=m.clean_document_pages(pages)
            parts=p.relative_to(corpus).parts
            category=m.CATEGORIES.get(parts[0].lower(),parts[0]) if len(parts)>1 else 'Uncategorized'
            chunks=list(m.make_chunks(cleaned,chunk_words,overlap))
            if not chunks:raise ValueError('No usable chunks')
            did=sha(rel)
            with db:
                old_chunks=db.execute('SELECT id FROM chunks WHERE document_id=?',(did,)).fetchall()
                for (cid,) in old_chunks:db.execute('DELETE FROM chunk_fts WHERE id=?',(cid,))
                # Keep document row and curated metadata on PDF updates.
                db.execute('''INSERT INTO documents(id,source_path,filename,category,title,doi,pmcid,source_hash,pages,ingested_at)
                  VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
                  source_path=excluded.source_path,filename=excluded.filename,category=excluded.category,
                  title=excluded.title,doi=excluded.doi,pmcid=excluded.pmcid,source_hash=excluded.source_hash,
                  pages=excluded.pages,ingested_at=excluded.ingested_at''',
                  (did,rel,p.name,category,title,doi,pmcid,digest,len(pages),now()))
                db.execute('DELETE FROM chunks WHERE document_id=?',(did,))
                for i,(page,section,body) in enumerate(chunks):
                    cid=sha(did+':'+str(i))
                    db.execute('INSERT INTO chunks(id,document_id,ordinal,page,section,body,body_hash) VALUES(?,?,?,?,?,?,?)',
                               (cid,did,i,page,section,body,sha(body)))
                    db.execute('INSERT INTO chunk_fts(id,body) VALUES(?,?)',(cid,body))
            stats['updated_pdfs' if old else 'new_pdfs']+=1
            stats['chunks_written']+=len(chunks)
            if old:seen_hashes.pop(old[2],None)
            seen_hashes[digest]=rel
        except Exception as e:
            stats['pdf_failures']+=1
            print('ERROR processing',rel,repr(e))
    if prune:
        missing=set(existing)-encountered
        with db:
            for rel in sorted(missing):
                did=existing[rel][0]
                for (cid,) in db.execute('SELECT id FROM chunks WHERE document_id=?',(did,)):
                    db.execute('DELETE FROM chunk_fts WHERE id=?',(cid,))
                db.execute('DELETE FROM documents WHERE id=?',(did,))
                stats['removed_pdfs']+=1
    return len(paths)

def connect_pg():
    import psycopg
    password=os.environ.get('PGPASSWORD')
    if password is None:password=getpass.getpass('PostgreSQL password: ')
    return psycopg.connect(host=os.getenv('PGHOST','127.0.0.1'),port=int(os.getenv('PGPORT','5433')),
      dbname=os.getenv('PGDATABASE','adaptive_healthcare_phase9_benchmark'),
      user=os.getenv('PGUSER','postgres'),password=password,connect_timeout=10)

def vector_literal(v):
    import numpy as np
    a=np.asarray(v,dtype=np.float32)
    if a.shape!=(384,) or not np.isfinite(a).all():raise ValueError('Invalid vector')
    return '['+','.join(map(lambda x:str(float(x)),a))+']'

def sync_vectors(db,batch_size,stats):
    # Track hashes in PostgreSQL, since the existing embeddings table lacks a hash column.
    # A vector row is only considered current if both model and body_hash match.
    with connect_pg() as pg:
        with pg.cursor() as c:
            c.execute("SELECT 1 FROM pg_extension WHERE extname='vector'")
            if not c.fetchone():raise RuntimeError('pgvector extension missing')
            c.execute('''CREATE TABLE IF NOT EXISTS literature.embedding_sync (
                chunk_id TEXT PRIMARY KEY REFERENCES literature.embeddings(chunk_id) ON DELETE CASCADE,
                body_hash TEXT NOT NULL, embedding_model TEXT NOT NULL,
                synced_at TIMESTAMPTZ NOT NULL DEFAULT now())''')
        pg.commit()
        local=db.execute('SELECT id,document_id,page,body,body_hash FROM chunks ORDER BY id').fetchall()
        local_ids={r[0] for r in local}
        with pg.cursor() as c:
            c.execute('''SELECT e.chunk_id,e.embedding_model,s.body_hash,s.embedding_model
                         FROM literature.embeddings e LEFT JOIN literature.embedding_sync s ON s.chunk_id=e.chunk_id''')
            remote={r[0]:r[1:] for r in c.fetchall()}
        stale=[cid for cid in remote if cid not in local_ids]
        # A missing hash is not trusted, so legacy rows are recomputed once.
        pending=[r for r in local if r[0] not in remote or remote[r[0]]!=(MODEL,r[4],MODEL)]
        stats['embeddings_unchanged']=len(local)-len(pending)
        stats['embeddings_to_update']=sum(r[0] in remote for r in pending)
        stats['embeddings_to_insert']=sum(r[0] not in remote for r in pending)
        stats['embeddings_to_delete']=len(stale)
        if pending:
            from sentence_transformers import SentenceTransformer
            model=SentenceTransformer(MODEL)
            dimension=(model.get_embedding_dimension() if hasattr(model,'get_embedding_dimension')
                       else model.get_sentence_embedding_dimension())
            if dimension!=384:raise RuntimeError(f'Unexpected model dimension: {dimension}')
            for start in range(0,len(pending),batch_size):
                batch=pending[start:start+batch_size]
                vectors=model.encode([r[3] for r in batch],batch_size=min(32,batch_size),
                    normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
                with pg.cursor() as c:
                    for (cid,did,page,body,h),v in zip(batch,vectors):
                        c.execute('''INSERT INTO literature.embeddings
                          (chunk_id,document_id,pdf_page,embedding_model,embedding)
                          VALUES(%s,%s,%s,%s,%s::vector) ON CONFLICT(chunk_id) DO UPDATE SET
                          document_id=excluded.document_id,pdf_page=excluded.pdf_page,
                          embedding_model=excluded.embedding_model,embedding=excluded.embedding,indexed_at=now()''',
                          (cid,did,page,MODEL,vector_literal(v)))
                        c.execute('''INSERT INTO literature.embedding_sync(chunk_id,body_hash,embedding_model)
                          VALUES(%s,%s,%s) ON CONFLICT(chunk_id) DO UPDATE SET
                          body_hash=excluded.body_hash,embedding_model=excluded.embedding_model,synced_at=now()''',
                          (cid,h,MODEL))
                pg.commit()
                print(f'Embedded {min(start+batch_size,len(pending))}/{len(pending)}')
        if stale:
            with pg.cursor() as c:
                for cid in stale:c.execute('DELETE FROM literature.embeddings WHERE chunk_id=%s',(cid,))
            pg.commit()
        with pg.cursor() as c:
            c.execute('SELECT COUNT(*) FROM literature.embeddings WHERE embedding_model=%s',(MODEL,))
            stats['embeddings_final']=c.fetchone()[0]
        if stats['embeddings_final']!=len(local):raise RuntimeError('Embedding count mismatch')

def export_metadata(db,path,stats):
    rows=db.execute('''SELECT d.source_path,d.title,COALESCE(NULLIF(m.verified_title,''),d.title),
        d.doi,d.pmcid,m.publisher,m.publication_year,m.source_url,
        COALESCE(m.verification_status,'unverified'),d.pages
        FROM documents d LEFT JOIN literature_metadata m ON m.document_id=d.id ORDER BY d.source_path''').fetchall()
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['source_path','extracted_title','display_title','extracted_doi','extracted_pmcid',
                                   'publisher','publication_year','source_url','verification_status','pdf_pages'])
        w.writerows(rows)
    stats['documents']=len(rows)
    stats['verified_metadata']=sum(r[8]=='verified' for r in rows)
    stats['unverified_metadata']=len(rows)-stats['verified_metadata']

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--corpus',type=Path,default=ROOT/'data'/'clinical_literature')
    p.add_argument('--sqlite',type=Path,default=ROOT/'results'/'phase12_store_v3'/'literature.sqlite')
    p.add_argument('--overrides',type=Path,default=ROOT/'data'/'literature_metadata_overrides.json')
    p.add_argument('--chunk-words',type=int,default=220)
    p.add_argument('--overlap-words',type=int,default=45)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--prune-missing',action='store_true',help='Remove documents whose PDF is no longer in corpus')
    p.add_argument('--skip-ingest',action='store_true',help='Do not rescan PDFs; sync existing SQLite only')
    p.add_argument('--skip-pgvector',action='store_true',help='Metadata/SQLite only')
    args=p.parse_args()
    if not args.sqlite.exists():raise RuntimeError(f'SQLite store not found: {args.sqlite}')
    if args.batch_size<1:raise ValueError('batch-size must be positive')
    stats=Counter()
    with sqlite3.connect(args.sqlite) as db:
        db.execute('PRAGMA foreign_keys=ON')
        metadata_tables(db)
        if not args.skip_ingest:stats['pdfs_found']=ingest(db,args.corpus,args.chunk_words,args.overlap_words,args.prune_missing,stats)
        apply_metadata(db,overrides(args.overrides),stats)
        export_metadata(db,ROOT/'results'/'phase11_audit'/'metadata_review.csv',stats)
        stats['sqlite_chunks']=db.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]
        stats['fts_chunks']=db.execute('SELECT COUNT(*) FROM chunk_fts').fetchone()[0]
        if stats['sqlite_chunks']!=stats['fts_chunks']:raise RuntimeError('SQLite FTS count mismatch')
        if not args.skip_pgvector:sync_vectors(db,args.batch_size,stats)
    report=ROOT/'results'/'phase11_audit'/'incremental_sync_report.json'
    report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(json.dumps({'timestamp':now(),'model':MODEL,'stats':dict(stats)},indent=2),encoding='utf-8')
    print(json.dumps(dict(stats),indent=2));print('Metadata review:',report.parent/'metadata_review.csv');print('Report:',report)
    if stats['pdf_failures']:raise SystemExit('Some PDF ingestion operations failed; see errors above')
if __name__=='__main__':main()
