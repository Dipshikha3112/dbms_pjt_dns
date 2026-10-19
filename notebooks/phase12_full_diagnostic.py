"""Read-only Phase 12 diagnostic. Python 3.10+. No patient rows, passwords, or raw clinical notes exported.
Usage: python notebooks/phase12_full_diagnostic.py
Optional: --deep (additional query plans), --out results/phase12_diagnostic
"""
from __future__ import annotations
import argparse, collections, hashlib, importlib.metadata, json, os, platform, re, sqlite3, statistics, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_QUESTIONS = [
    'How does hypertension contribute to cardiovascular disease risk?',
    'How does LDL cholesterol contribute to atherosclerosis?',
    'What are the major risk factors for cardiovascular disease?',
    'How does diabetes influence cardiovascular disease risk?',
    'What is the association between chronic kidney disease and hypertension?'
]
SYNONYMS = {'hypertension':['hypertension','high blood pressure'], 'cardiovascular disease':['cardiovascular disease','heart disease','coronary heart disease'], 'atherosclerosis':['atherosclerosis'], 'diabetes':['diabetes mellitus','diabetes'], 'LDL cholesterol':['ldl cholesterol','low density lipoprotein']}
SENSITIVE_LABELS={'Patient'}

def safe_error(exc):
    s=f'{type(exc).__name__}: {exc}'
    s=re.sub(r'(?i)(password|token|secret|apikey|api_key)\s*[=:]\s*[^\s,;]+',r'\1=<redacted>',s)
    return s[:700]

def timed(fn):
    t=time.perf_counter()
    try:
        result=fn()
        return {'ok':True,'elapsed_ms':round((time.perf_counter()-t)*1000,2),'result':result}
    except Exception as e:return {'ok':False,'elapsed_ms':round((time.perf_counter()-t)*1000,2),'error':safe_error(e)}

def json_safe(v):
    if isinstance(v,(int,float,str,bool)) or v is None:return v
    if isinstance(v,(tuple,list,set)):return [json_safe(x) for x in v]
    if isinstance(v,dict):return {str(k):json_safe(x) for k,x in v.items()}
    return str(v)

def sqlite_diagnosis(path):
    out={'path':str(path),'exists':path.exists()}
    if not path.exists():return out
    con=sqlite3.connect(f'file:{path.as_posix()}?mode=ro',uri=True,timeout=10)
    con.row_factory=sqlite3.Row
    try:
        out['file_bytes']=path.stat().st_size
        out['sqlite_version']=sqlite3.sqlite_version
        out['tables']=[r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view') ORDER BY name")]
        out['schemas']={}
        for t in ('documents','chunks','chunk_fts','literature_metadata'):
            if t in out['tables']:
                out['schemas'][t]=[{'column':r['name'],'type':r['type']} for r in con.execute(f'PRAGMA table_info("{t}")')]
        out['integrity_check']=timed(lambda:con.execute('PRAGMA quick_check(1)').fetchone()[0])
        for t in ('documents','chunks'):
            if t in out['tables']:out[t+'_count']=con.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0]
        if 'chunks' in out['tables']:
            out['chunk_stats']=dict(con.execute('SELECT count(*) AS total, count(DISTINCT id) AS unique_ids, count(DISTINCT body_hash) AS unique_hashes, sum(CASE WHEN length(trim(coalesce(body,"")))<40 THEN 1 ELSE 0 END) AS short_chunks, round(avg(length(body)),1) AS mean_chars, max(length(body)) AS max_chars FROM chunks').fetchone())
            out['duplicate_hash_groups']=[dict(r) for r in con.execute('SELECT body_hash, count(*) AS n FROM chunks WHERE body_hash IS NOT NULL GROUP BY body_hash HAVING count(*)>1 ORDER BY n DESC LIMIT 12')]
            out['sections']=[dict(r) for r in con.execute('SELECT coalesce(section,"<null>") AS section,count(*) AS n FROM chunks GROUP BY section ORDER BY n DESC LIMIT 15')]
        if 'documents' in out['tables']:
            out['document_categories']=[dict(r) for r in con.execute('SELECT coalesce(category,"<null>") AS category,count(*) AS n FROM documents GROUP BY category ORDER BY n DESC LIMIT 20')]
        if 'chunk_fts' in out['tables']:
            out['fts_integrity']=timed(lambda:con.execute("INSERT INTO chunk_fts(chunk_fts) VALUES('integrity-check')").fetchall()) if False else {'skipped':'read-only connection; virtual-table integrity-check requires write statement'}
            out['fts_samples']={}
            for term in ('hypertension','atherosclerosis','diabetes','stroke'):
                def fts_test():
                    rows=con.execute('SELECT rowid,bm25(chunk_fts) AS score FROM chunk_fts WHERE chunk_fts MATCH ? ORDER BY score LIMIT 5',(term,)).fetchall()
                    return [{'rowid':r[0],'bm25':round(r[1],4)} for r in rows]
                out['fts_samples'][term]=timed(fts_test)
            out['fts_join_check']=timed(lambda:{'matching_ids':con.execute('SELECT count(*) FROM chunk_fts f JOIN chunks c ON c.id=CAST(f.rowid AS TEXT)').fetchone()[0], 'note':'0 can be normal if FTS rowid is not the chunk id; inspect FTS CREATE SQL'})
            out['fts_definition']=con.execute("SELECT sql FROM sqlite_master WHERE name='chunk_fts'").fetchone()[0]
    finally:con.close()
    return out

def pg_diagnosis(deep=False):
    import psycopg
    from psycopg import sql
    conn=psycopg.connect(host=os.getenv('PGHOST','127.0.0.1'),port=int(os.getenv('PGPORT','5432')),dbname=os.getenv('PGDATABASE','adaptive_healthcare_rag'),user=os.getenv('PGUSER','postgres'),password=os.getenv('PGPASSWORD'),connect_timeout=8,options='-c statement_timeout=15000')
    try:
        with conn.cursor() as c:
            c.execute('SELECT current_database(),current_user,version()');db,user,version=c.fetchone()
            out={'database':db,'user':user,'server_version':version[:140]}
            c.execute("SELECT extname,extversion FROM pg_extension WHERE extname='vector'");out['pgvector_extension']=c.fetchall()
            c.execute("SELECT table_schema,table_name,column_name,data_type,udt_name FROM information_schema.columns WHERE table_schema NOT IN ('pg_catalog','information_schema') AND column_name IN ('chunk_id','embedding','embedding_model','body','document_id') ORDER BY 1,2,3")
            out['candidate_columns']=[dict(zip(('schema','table','column','type','udt'),r)) for r in c.fetchall()]
            c.execute("SELECT indexname,indexdef FROM pg_indexes WHERE schemaname='literature' AND tablename='embeddings'");out['vector_indexes']=[{'name':a,'definition':b} for a,b in c.fetchall()]
            c.execute("SELECT to_regclass('literature.embeddings')");table=c.fetchone()[0]
            if table:
                c.execute('SELECT count(*),count(DISTINCT chunk_id),count(DISTINCT embedding_model) FROM literature.embeddings');a,b,d=c.fetchone()
                out['embeddings']={'rows':a,'unique_chunk_ids':b,'distinct_models':d}
                c.execute('SELECT embedding_model,count(*) FROM literature.embeddings GROUP BY 1 ORDER BY 2 DESC');out['embedding_models']=[{'model':a,'rows':b} for a,b in c.fetchall()]
                c.execute('SELECT vector_dims(embedding),count(*) FROM literature.embeddings GROUP BY 1');out['embedding_dimensions']=[{'dimensions':a,'rows':b} for a,b in c.fetchall()]
                c.execute('SELECT count(*) FROM literature.embeddings WHERE embedding IS NULL');out['null_embeddings']=c.fetchone()[0]
                c.execute('SELECT chunk_id FROM literature.embeddings ORDER BY chunk_id LIMIT 1200');out['_chunk_ids']=[r[0] for r in c.fetchall()]
                c.execute('SELECT embedding::text FROM literature.embeddings LIMIT 1');row=c.fetchone()
                if row:
                    probe_vector=row[0]
                    def vector_probe():
                        c.execute('SELECT chunk_id,round((embedding <=> %s::vector)::numeric,5) FROM literature.embeddings ORDER BY embedding <=> %s::vector LIMIT 5',(probe_vector,probe_vector))
                        return [{'chunk_id':str(a),'cosine_distance':float(b)} for a,b in c.fetchall()]
                    out['vector_self_search']=timed(vector_probe)
                if deep:
                    row=(probe_vector,) if row else None
                    if row:
                        vec=row[0]
                        def explain():
                            c.execute('EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) SELECT chunk_id FROM literature.embeddings ORDER BY embedding <=> %s::vector LIMIT 5',(vec,))
                            return c.fetchone()[0]
                        out['query_plan']=timed(explain)
            return out
    finally:conn.close()

def graph_diagnosis(deep=False):
    from neo4j import GraphDatabase
    driver=GraphDatabase.driver(os.getenv('NEO4J_URI','bolt://127.0.0.1:7687'),auth=(os.getenv('NEO4J_USER','neo4j'),os.getenv('NEO4J_PASSWORD')),connection_timeout=8)
    out={}
    try:
        driver.verify_connectivity()
        with driver.session(database=os.getenv('NEO4J_DATABASE','neo4j')) as s:
            def query(cypher,**params):return [dict(r) for r in s.run(cypher,**params)]
            out['labels']=timed(lambda:query('CALL db.labels() YIELD label RETURN label ORDER BY label'))
            out['relationship_types']=timed(lambda:query('CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType'))
            out['indexes']=timed(lambda:query('SHOW INDEXES YIELD name,type,labelsOrTypes,properties,state RETURN name,type,labelsOrTypes,properties,state'))
            out['constraints']=timed(lambda:query('SHOW CONSTRAINTS YIELD name,type,labelsOrTypes,properties RETURN name,type,labelsOrTypes,properties'))
            out['node_counts']=timed(lambda:query('MATCH (n) UNWIND labels(n) AS label RETURN label,count(*) AS n ORDER BY n DESC'))
            out['relation_counts']=timed(lambda:query('MATCH ()-[r]->() RETURN type(r) AS relation,count(*) AS n ORDER BY n DESC'))
            out['edge_endpoints']=timed(lambda:query('MATCH (a)-[r]->(b) RETURN labels(a) AS source_labels,type(r) AS relation,labels(b) AS target_labels,count(*) AS n ORDER BY n DESC LIMIT 65'))
            out['property_schema']=timed(lambda:query('CALL db.schema.nodeTypeProperties() YIELD nodeLabels,propertyName,propertyTypes RETURN nodeLabels,propertyName,propertyTypes LIMIT 120'))
            out['relationship_property_schema']=timed(lambda:query('CALL db.schema.relTypeProperties() YIELD relType,propertyName,propertyTypes RETURN relType,propertyName,propertyTypes LIMIT 120'))
            out['clinical_relations']=timed(lambda:query('MATCH (a)-[r:ASSOCIATED_WITH|RELATED_TO|ABOUT_MESH]->(b) WHERE NOT a:Patient AND NOT b:Patient RETURN labels(a) AS source_labels,type(r) AS relation,labels(b) AS target_labels,count(*) AS n ORDER BY n DESC LIMIT 35'))
            out['entity_search']={}
            for concept,aliases in SYNONYMS.items():
                # Use label-qualified equality predicates; do not scan all nodes with toLower on every property.
                def lookup():
                    results=[]
                    for label in ('Disease','MeSHConcept','MONDOConcept','ChEBIConcept'):
                        for prop in ('name','label','term','title','display_name'):
                            q=f'MATCH (n:`{label}`) WHERE n.`{prop}` IN $names RETURN labels(n) AS labels, elementId(n) AS eid, n.`{prop}` AS matched_name, keys(n) AS property_keys LIMIT 8'
                            results.extend(query(q,names=aliases+[x.title() for x in aliases]))
                            if len(results)>=12:break
                        if len(results)>=12:break
                    return results[:12]
                out['entity_search'][concept]=timed(lookup)
            # Probe actual labels and property keys without exporting node values.
            out['node_property_keys']={}
            for label in ('Disease','MeSHConcept','MONDOConcept','ChEBIConcept','Drug','LabTest'):
                out['node_property_keys'][label]=timed(lambda label=label:query(f'MATCH (n:`{label}`) RETURN keys(n) AS property_keys LIMIT 3'))
            out['hypertension_neighborhood']={}
            ids=[]
            for x in out['entity_search'].get('hypertension',{}).get('result',[]):
                ids.append(x['eid'])
            for eid in ids[:3]:
                def neighborhood():
                    return query('MATCH (a)-[r]-(b) WHERE elementId(a)=$eid AND NOT b:Patient RETURN type(r) AS relation,labels(b) AS neighbor_labels, keys(b) AS neighbor_property_keys, count(*) AS n ORDER BY n DESC LIMIT 25',eid=eid)
                out['hypertension_neighborhood'][eid]=timed(neighborhood)
            if deep:
                out['graph_plan']=timed(lambda:query('EXPLAIN MATCH (n:Disease) WHERE n.name=$name RETURN n LIMIT 8',name='Hypertension'))
    finally:driver.close()
    return out

def project_diagnosis():
    out={'python':sys.version.split()[0],'os':platform.platform(),'project_root':str(ROOT),'packages':{},'paths':{},'env_present':{}}
    for p in ('neo4j','psycopg','pgvector','sentence-transformers','transformers','torch','requests'):
        try:out['packages'][p]=importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:out['packages'][p]='not installed'
    for p in ('notebooks/phase12_graph_rag_v7.py','notebooks/phase12_graph_rag.py','benchmark/phase12_v7_50_questions.json','results/phase12_v7/graph_schema.json'):
        path=ROOT/p;out['paths'][p]={'exists':path.exists(),'bytes':path.stat().st_size if path.exists() else None}
    for k in ('PGPASSWORD','NEO4J_PASSWORD','NLI_MODEL','OLLAMA_MODEL','EMBEDDING_MODEL','NEO4J_URI','PGDATABASE'):
        out['env_present'][k]=bool(os.getenv(k))
    if (ROOT/'notebooks/phase12_graph_rag_v7.py').exists():
        p=ROOT/'notebooks/phase12_graph_rag_v7.py'
        source=p.read_text(encoding='utf-8',errors='replace')
        out['v7_static_checks']={
            'hardcoded_nonexistent_neo4j_labels':any(x in source for x in ('n:Condition','n:MedicalConcept')),
            'nli_model_configured':bool(os.getenv('NLI_MODEL')),
            'uses_rrf':'RRF_K' in source,
            'uses_neo4j':'neo4j' in source.lower(),
            'sha256':hashlib.sha256(p.read_bytes()).hexdigest()
        }
    return out

def recommendations(report):
    rec=[];g=report.get('neo4j',{}).get('result',{})
    labels={r['label'] for r in g.get('labels',{}).get('result',[])}
    if labels and not {'Condition','MedicalConcept'}&labels:rec.append('Remove nonexistent Condition/MedicalConcept labels from V7 Cypher; dynamically use actual labels and indexed properties.')
    if g.get('entity_search',{}).get('hypertension',{}).get('result')==[]:rec.append('Hypertension concept lookup failed; inspect Disease and MeSHConcept property keys and case normalization.')
    p=report.get('postgresql',{}).get('result',{})
    s=report.get('sqlite',{}).get('result',{})
    if 'embeddings' in p and 'chunks_count' in s and p['embeddings']['rows']!=s['chunks_count']:rec.append('SQLite and pgvector row counts differ; check missing or stale embeddings.')
    if not report['project']['env_present']['NLI_MODEL']:rec.append('NLI model not configured; claim verification must abstain rather than mark claims supported.')
    rec.append('Create human-reviewed relevance labels before interpreting Precision@5, Recall@5, MRR, or nDCG.')
    return rec

def main():
    ap=argparse.ArgumentParser(description='Comprehensive read-only Graph-RAG V7 diagnosis')
    ap.add_argument('--deep',action='store_true',help='Include PostgreSQL EXPLAIN ANALYZE (read-only SELECT)')
    ap.add_argument('--out',default=str(ROOT/'results/phase12_diagnostic'))
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    print('PHASE 12 FULL DIAGNOSTIC — read-only; no patient records exported',flush=True)
    report={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'project':project_diagnosis(),'sections':{}}
    jobs=[('sqlite',lambda:sqlite_diagnosis(Path(os.getenv('LITERATURE_SQLITE',str(ROOT/'results/phase12_store_v3/literature.sqlite'))))),('postgresql',lambda:pg_diagnosis(args.deep)),('neo4j',lambda:graph_diagnosis(args.deep))]
    for name,fn in jobs:
        print(f'\nChecking {name}...',flush=True)
        r=timed(fn);report[name]=r
        if r['ok']:
            print(f'  OK ({r["elapsed_ms"]} ms)',flush=True)
            if name=='sqlite':print('  chunks:',r['result'].get('chunks_count'))
            if name=='postgresql':print('  embeddings:',r['result'].get('embeddings'))
            if name=='neo4j':print('  labels:',[x['label'] for x in r['result'].get('labels',{}).get('result',[])])
        else:print('  ERROR:',r['error'],flush=True)
    try:
        pgids=set(map(str,report['postgresql']['result'].pop('_chunk_ids',[])))
        dbpath=Path(os.getenv('LITERATURE_SQLITE',str(ROOT/'results/phase12_store_v3/literature.sqlite')))
        with sqlite3.connect(f'file:{dbpath.as_posix()}?mode=ro',uri=True) as con:sqlids={str(x[0]) for x in con.execute('SELECT id FROM chunks')}
        report['cross_database']={'sqlite_chunk_ids':len(sqlids),'pgvector_chunk_ids_sampled':len(pgids),'overlap_sampled':len(sqlids&pgids),'sqlite_ids_missing_in_pg_sample':len(sqlids-pgids),'pg_ids_not_in_sqlite_sample':len(pgids-sqlids),'sample_limit':1200}
    except Exception as e:report['cross_database']={'error':safe_error(e)}
    report['recommendations']=recommendations(report)
    dest=out/'phase12_full_diagnostic.json';dest.write_text(json.dumps(json_safe(report),indent=2,ensure_ascii=False),encoding='utf-8')
    summary={'timestamp_utc':report['timestamp_utc'],'components':{k:{'ok':report[k]['ok'],'elapsed_ms':report[k]['elapsed_ms'],'error':report[k].get('error')} for k in ('sqlite','postgresql','neo4j')},'cross_database':report.get('cross_database'),'recommendations':report['recommendations'],'report_path':str(dest)}
    (out/'phase12_diagnostic_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('\n'+json.dumps(summary,indent=2),flush=True)
    print('\nFULL REPORT:',dest,flush=True)
    print('NOTE: Report contains aggregate schema and concept metadata; review before sharing publicly.')

if __name__=='__main__':main()
