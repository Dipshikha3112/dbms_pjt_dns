"""Phase 12 V8: single-file, auditable biomedical Graph-RAG research prototype.

Install: pip install numpy requests 'psycopg[binary]' neo4j sentence-transformers
Optional entailment: pip install transformers torch

Place in <project>/notebooks/phase12_graph_rag_v8.py.
NOT a clinical decision support system. NLI is an imperfect research classifier;
no automatic verification substitutes for medical expert review.
"""
from __future__ import annotations
import os
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(v,'2')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import argparse, concurrent.futures, csv, hashlib, html, json, math, re, sqlite3, statistics, threading, time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SQLITE = Path(os.getenv('LITERATURE_SQLITE',str(ROOT/'results/phase12_store_v3/literature.sqlite')))
OUT = Path(os.getenv('GRAPH_RAG_OUTPUT',str(ROOT/'results/phase12_v8')))
MODEL = os.getenv('EMBEDDING_MODEL','sentence-transformers/all-MiniLM-L6-v2')
OLLAMA = os.getenv('OLLAMA_MODEL','qwen2.5:7b')
NLI_MODEL = os.getenv('NLI_MODEL','')  # Set to a validated 3-label NLI model; blank => abstention on semantic verification
PMC_URL = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search'
RRF_K = 60
LOCAL = threading.local()
MODEL_LOCK = threading.Lock()
ENTITY_ALIASES = {
 'hypertension':['hypertension','high blood pressure','elevated blood pressure'],
 'cardiovascular disease':['cardiovascular disease','cvd','heart disease','coronary heart disease'],
 'atherosclerosis':['atherosclerosis','atherosclerotic disease','arterial plaque'],
 'ldl cholesterol':['ldl','low density lipoprotein','cholesterol','dyslipidemia'],
 'diabetes':['diabetes','diabetes mellitus','type 2 diabetes'],
 'smoking':['smoking','tobacco','cigarettes'],
 'obesity':['obesity','overweight'],
 'stroke':['stroke','cerebrovascular accident'],
 'arterial stiffness':['arterial stiffness','vascular stiffness'],
 'endothelial dysfunction':['endothelial dysfunction','endothelial injury'],
 'left ventricular hypertrophy':['left ventricular hypertrophy','lvh','cardiac remodeling'],
 'inflammation':['inflammation','inflammatory'],
 'kidney disease':['chronic kidney disease','ckd','renal disease'],
 'statins':['statin','statins'],
 'hba1c':['hba1c','glycated hemoglobin'],
 'heart failure':['heart failure','cardiac failure'],
 'plaque':['plaque','foam cells','fatty streak'],
}
PROFILES = {
 'hypertension_mechanism': ['arterial stiffness','endothelial dysfunction','left ventricular hypertrophy','inflammation','atherosclerosis','cardiovascular outcomes'],
 'cvd_risk': ['hypertension','smoking','diabetes','ldl cholesterol','obesity','physical inactivity','diet','age','family history'],
 'ldl_atherosclerosis': ['ldl cholesterol','plaque','inflammation','atherosclerosis'],
}
NEGATION = re.compile(r'\b(no|not|never|without|neither|does not|did not|is not|was not|fails to)\b',re.I)
STOP = {'what','which','when','where','does','with','that','this','from','into','about','the','and','for','are','how','can','may','risk','major','contribute','disease'}


def now(): return datetime.now(timezone.utc).isoformat()
def normalize(s): return ' '.join(str(s or '').split())
def toks(s): return set(re.findall(r'[a-z][a-z0-9-]{2,}',str(s).lower()))-STOP
def contains(text, phrase): return bool(re.search(r'(?<!\w)'+re.escape(phrase).replace(r'\ ',r'[\s-]+')+r'(?!\w)',str(text),re.I))
def concepts(text): return [k for k,vs in ENTITY_ALIASES.items() if any(contains(text,v) for v in vs)]
def profile(q):
    ql=q.lower()
    if ('hypertension' in ql or 'blood pressure' in ql) and any(s in ql for s in ('cardiovascular','heart','stroke','mechanism','contribute')): name='hypertension_mechanism'
    elif ('ldl' in ql or 'cholesterol' in ql) and 'atherosclero' in ql: name='ldl_atherosclerosis'
    elif 'risk factor' in ql and any(x in ql for x in ('cardiovascular','cvd','heart')): name='cvd_risk'
    else: name='general'
    return {'name':name,'entities':concepts(q),'subtopics':PROFILES.get(name,[]),'intent':'mechanism' if name.endswith('mechanism') or name=='ldl_atherosclerosis' else 'research_qa'}
def topic_matches(text, topics): return [x for x in topics if contains(text,x) or (x=='cardiovascular outcomes' and any(contains(text,y) for y in ('stroke','heart attack','heart failure','cardiovascular event'))) or (x=='diet' and 'diet' in text.lower())]
def write_json(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
def stage_timer(metrics,name):
    class Timer:
        def __enter__(self): self.start=time.perf_counter();return self
        def __exit__(self,*args): metrics[name]=round((time.perf_counter()-self.start)*1000,2)
    return Timer()

@lru_cache(maxsize=1)
def embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL)
@lru_cache(maxsize=512)
def embed_query(query):
    with MODEL_LOCK:
        vec=embedder().encode(query,normalize_embeddings=True)
    return '['+','.join(format(float(x),'.8g') for x in vec)+']'

def sqlite_db():
    if not SQLITE.is_file():raise FileNotFoundError(f'SQLite store missing: {SQLITE}')
    db=sqlite3.connect(str(SQLITE),timeout=15);db.row_factory=sqlite3.Row;return db

@lru_cache(maxsize=1)
def sqlite_columns():
    with sqlite_db() as db:
        return {t:{r[1] for r in db.execute('PRAGMA table_info("'+t+'")')} for t in ('chunks','documents')}

def metadata(ids=None):
    with sqlite_db() as db:
        c=sqlite_columns(); cc=c['chunks'];dc=c['documents']
        def col(alias,field,fallback='NULL'):return f'{alias}."{field}"' if field in (cc if alias=='c' else dc) else fallback
        sql=f'''SELECT {col('c','id')} AS chunk_id, {col('c','document_id')} AS document_id,
          {col('c','page')} AS page, {col('c','body')} AS body,
          {col('d','title')} AS title, {col('d','source_path')} AS source_path
          FROM chunks c LEFT JOIN documents d ON c.document_id=d.id'''
        if ids is not None:
            ids=list(ids)
            if not ids:return {}
            sql+=' WHERE CAST(c.id AS TEXT) IN ('+','.join('?'*len(ids))+')'
            rows=db.execute(sql,[str(x) for x in ids]).fetchall()
        else:rows=db.execute(sql).fetchall()
        return {str(r['chunk_id']):dict(r) for r in rows}

def fts(q,n=40):
    words=sorted(toks(q),key=len,reverse=True)[:12]
    if not words:return []
    expression=' OR '.join('"'+w.replace('"','')+'"' for w in words)
    with sqlite_db() as db:
        try:
            return [(str(r[0]),i+1) for i,r in enumerate(db.execute('SELECT id FROM chunk_fts WHERE chunk_fts MATCH ? ORDER BY bm25(chunk_fts) LIMIT ?', (expression,n)))]
        except sqlite3.Error:
            # Fallback when FTS virtual table uses a different external-content id
            return []

def pg_connection():
    import psycopg
    return psycopg.connect(host=os.getenv('PGHOST','127.0.0.1'),port=int(os.getenv('PGPORT','5432')),
       dbname=os.getenv('PGDATABASE','adaptive_healthcare_rag'),user=os.getenv('PGUSER','postgres'),
       password=os.environ['PGPASSWORD'],connect_timeout=5)

@lru_cache(maxsize=1)
def pg_tables():
    with pg_connection() as conn,conn.cursor() as cur:
        cur.execute('''SELECT table_schema,table_name,column_name FROM information_schema.columns
          WHERE table_schema NOT IN ('pg_catalog','information_schema')
          AND column_name IN ('chunk_id','embedding','embedding_model')''')
        groups=defaultdict(set)
        for schema,table,col in cur.fetchall():groups[(schema,table)].add(col)
        return [{'schema':s,'table':t,'model_filter':'embedding_model' in cols}
                for (s,t),cols in groups.items() if {'chunk_id','embedding'}<=cols]

def vector_search(q,n=40):
    from psycopg import sql
    vec=embed_query(q);results=[];warnings=[]
    for table in pg_tables():
        with pg_connection() as conn,conn.cursor() as cur:
            ident=sql.Identifier(table['schema'],table['table'])
            filter_sql=sql.SQL(' WHERE embedding_model=%s') if table['model_filter'] else sql.SQL('')
            statement=sql.SQL('SELECT chunk_id::text, embedding <=> %s::vector AS distance FROM {}{} ORDER BY embedding <=> %s::vector LIMIT %s').format(ident,filter_sql)
            args=[vec]+([MODEL] if table['model_filter'] else [])+[vec,n]
            try:
                cur.execute(statement,args)
                rows=cur.fetchall()
                for cid,dist in rows:
                    if dist is not None:results.append((str(cid),float(dist),f"{table['schema']}.{table['table']}"))
            except Exception as e:warnings.append(f"{table['schema']}.{table['table']}: {type(e).__name__}: {str(e)[:150]}")
    results.sort(key=lambda x:x[1]);seen=set();unique=[]
    for cid,dist,source in results:
        if cid not in seen:unique.append((cid,len(unique)+1,1-dist,source));seen.add(cid)
    return unique[:n],warnings

def fuse(q,mode,n=10,depth=40):
    """Concurrent independent retrieval; separate connection per worker."""
    metrics={};warnings=[];ranks=defaultdict(dict)
    def timed(name,fn):
        started=time.perf_counter()
        try:return fn(),round((time.perf_counter()-started)*1000,2),None
        except Exception as exc:return None,round((time.perf_counter()-started)*1000,2),f'{name}: {type(exc).__name__}: {str(exc)[:180]}'
    jobs={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        if mode in ('sqlite','hybrid','hybrid_graph'):
            jobs['sqlite_fts_ms']=pool.submit(timed,'SQLite',lambda:fts(q,depth))
        if mode in ('pgvector','hybrid','hybrid_graph'):
            jobs['pgvector_ms']=pool.submit(timed,'pgvector',lambda:vector_search(q,depth))
        for name,future in jobs.items():
            data,ms,error=future.result();metrics[name]=ms
            if error:warnings.append(error);continue
            if name=='sqlite_fts_ms':
                for cid,rank in data:ranks[cid]['fts']=rank
            else:
                results,errs=data;warnings+=errs
                for cid,rank,similarity,table in results:ranks[cid].update(vector=rank,similarity=similarity,table=table)
                if not results:warnings.append('pgvector returned no matching results')
    meta=metadata(ranks.keys());out=[]
    for cid,rr in ranks.items():
        if cid not in meta:continue
        m=meta[cid];body=normalize(m.get('body'))
        if len(body)<30:continue
        rrf=sum(1/(RRF_K+rr[k]) for k in ('fts','vector') if k in rr)
        lex=len(toks(q)&toks(body))/max(1,len(toks(q)))
        score=rrf+0.002*lex
        out.append({'source':'local_pdf','chunk_id':cid,'document_id':m.get('document_id'),
                    'title':m.get('title') or 'Untitled PDF','page':m.get('page'),
                    'source_path':m.get('source_path'),'text':body,'score':score,
                    'retrievers':[k for k in ('fts','vector') if k in rr],
                    'similarity':rr.get('similarity')})
    out.sort(key=lambda x:-x['score']);return out[:n],metrics,warnings

def neo_driver():
    from neo4j import GraphDatabase
    return GraphDatabase.driver(os.getenv('NEO4J_URI','bolt://127.0.0.1:7687'),
       auth=(os.getenv('NEO4J_USER','neo4j'),os.environ['NEO4J_PASSWORD']),connection_timeout=5)

def graph_schema():
    report={}
    with neo_driver() as driver:
        with driver.session(database=os.getenv('NEO4J_DATABASE','neo4j')) as session:
            report['labels']=[dict(r) for r in session.run('MATCH (n) UNWIND labels(n) AS label RETURN label,count(*) AS total ORDER BY total DESC LIMIT 40')]
            report['relationships']=[dict(r) for r in session.run('MATCH ()-[r]->() RETURN type(r) AS type,count(*) AS total ORDER BY total DESC LIMIT 60')]
            report['indexes']=[dict(r) for r in session.run('SHOW INDEXES YIELD name,type,labelsOrTypes,properties RETURN name,type,labelsOrTypes,properties LIMIT 80')]
    return report

def classify_relation(name):
    s=name.upper()
    if any(t in s for t in ('MAPPED','SYNONYM','SAME_AS','BROADER','NARROWER','ONTOLOGY','MESH','CROSSREF','EQUIVALENT')):return 'terminology'
    if any(t in s for t in ('CAUSE','RISK','ASSOCIAT','COMPLICAT','CONTRIBUT','INCREASE','DECREASE','PREDISPOS','MECHANISM','MANIFEST','AFFECT','LINKED')):return 'clinical_association'
    if any(t in s for t in ('DIAGNOS','OBSERV','MEASURE','HAS_CONDITION','HAS_DISEASE','HAS_RESULT')):return 'observation'
    return 'other'

def safe_node(n):
    # Keep only whitelisted non-person-identifying concept properties.
    props=dict(n);allowed=('name','label','term','title','display_name','description','definition','code','id','source','pmid','doi')
    return {'labels':list(n.labels),'properties':{k:normalize(props[k])[:170] for k in allowed if k in props and isinstance(props[k],(str,int,float))}}

def node_string(n):return ' '.join(str(x) for x in n['properties'].values())

def graph_retrieve(q,p,n=8,allow_weak=False):
    # Indexed disease-name seeds; only observational Disease to LabTest/Drug edges.
    # No patient-level data, terminology mappings, or causal assertions.
    # end note
    diag={'matched_seeds':0,'candidate_paths':0,'accepted_paths':0,'rejected':{},'graph_scope':'observational disease–lab/medication associations'}
    warnings=[];found=[]
    # Query by actual diagnosed schema: Disease.name and indexed Disease.normalized_name.
    aliases=[]
    for entity in p['entities']:
        aliases.extend(ENTITY_ALIASES.get(entity,[entity]))
    # Keep original query phrases so disease terms absent from alias list can be discovered.
    aliases.extend([x for x in re.findall(r'[a-z][a-z ]{3,40}',q.lower()) if len(x.split())<=3])
    aliases=list(dict.fromkeys(normalize(a).lower() for a in aliases if len(a)>=4))[:30]
    if not aliases:return [],diag,['No disease query terms available']
    try:
        with neo_driver() as driver, driver.session(database=os.getenv('NEO4J_DATABASE','neo4j')) as session:
            # Indexed exact normalized name, plus a small explicit synonym set; no graph-wide scan.
            seed_query='''UNWIND $aliases AS alias MATCH (d:Disease)
                WHERE d.normalized_name = alias OR d.name = alias
                RETURN DISTINCT elementId(d) AS eid, d.name AS name LIMIT 12'''
            seeds=[dict(r) for r in session.run(seed_query,aliases=aliases)]
            if not seeds:
                # Only 202 Disease nodes in this graph; bounded label scan is acceptable.
                seed_query='''MATCH (d:Disease)
                   WHERE any(a IN $aliases WHERE toLower(d.name)=a OR
                      (size(a)>=7 AND toLower(d.name) CONTAINS a))
                   RETURN DISTINCT elementId(d) AS eid,d.name AS name LIMIT 12'''
                seeds=[dict(r) for r in session.run(seed_query,aliases=aliases)]
            diag['matched_seeds']=len(seeds)
            if not seeds:return [],diag,['No Disease node matched; MeSH-only concepts are excluded from clinical association retrieval']
            # Endpoint and relation type constraints avoid the 553k Drug→Drug RELATED_TO edges.
            cypher='''MATCH (d:Disease) WHERE elementId(d)=$eid
                MATCH (d)-[r:ASSOCIATED_WITH|OBSERVED_WITH_MEDICATION]->(t)
                WHERE t:LabTest OR t:Drug
                RETURN d.name AS disease,labels(t) AS target_labels,t.name AS target,
                  type(r) AS relation,r.source AS source,r.provenance AS provenance,
                  r.relationship_semantics AS semantics,r.patient_count AS patient_count,
                  elementId(t) AS target_eid LIMIT 350'''
            for seed in seeds:
                for row in session.run(cypher,eid=seed['eid']):
                    x=dict(row);diag['candidate_paths']+=1
                    target=normalize(x.get('target'))
                    if not target:continue
                    query_terms=toks(q);target_terms=toks(target)
                    target_overlap=len(query_terms & target_terms)
                    # Topic matching is used for ranking only; no invented causal pathway.
                    topic_hits=topic_matches(target,p['subtopics'])
                    target_concepts=concepts(target)
                    # Broad questions may need representative observations; rank with overlap and patient_count.
                    score=2*target_overlap+3*len(topic_hits)+2*len(target_concepts)
                    count=x.get('patient_count')
                    if isinstance(count,(int,float)) and count>0:score+=min(math.log1p(count)/10,1)
                    if p['intent']=='mechanism' and not (target_overlap or topic_hits or target_concepts):
                        # An observed laboratory test is not a mechanistic explanation.
                        continue
                    x.update({'seed':seed['name'],'score':round(score,4),
                       'category':'observational_association',
                       'interpretation':'Dataset-level association/co-occurrence, NOT causal or treatment evidence',
                       'causal_support':False})
                    found.append(x)
        found.sort(key=lambda r:(-r['score'],-(r.get('patient_count') or 0),r['target']))
        seen=set();unique=[]
        for x in found:
            key=(x['seed'],x['relation'],x['target_eid'])
            if key not in seen:unique.append(x);seen.add(key)
        diag['accepted_paths']=min(n,len(unique))
        if not unique:warnings.append('Matched disease nodes but no question-relevant observational endpoints; this graph lacks curated mechanism edges')
        return unique[:n],diag,warnings
    except Exception as e:
        return [],diag,[f'Neo4j retrieval error: {type(e).__name__}: {str(e)[:220]}']

def pmc(q,n=5,cache_hours=168):
    import requests
    cache=OUT/'cache';cache.mkdir(parents=True,exist_ok=True)
    params={'query':'('+' AND '.join('"'+e+'"' for e in concepts(q)[:2])+') AND HAS_ABSTRACT:y' if concepts(q) else q+' AND HAS_ABSTRACT:y',
            'format':'json','resultType':'core','pageSize':max(8,n*2)}
    key=hashlib.sha256(json.dumps(params,sort_keys=True).encode()).hexdigest()[:20];path=cache/f'{key}.json'
    if path.exists() and time.time()-path.stat().st_mtime<cache_hours*3600:data=json.loads(path.read_text(encoding='utf-8'))
    else:
        r=requests.get(PMC_URL,params=params,timeout=9,headers={'User-Agent':'BiomedicalGraphRAGResearch/8.0'});r.raise_for_status()
        data=r.json();write_json(path,data)
    items=[]
    for r in data.get('resultList',{}).get('result',[]):
        abstract=normalize(html.unescape(re.sub(r'<[^>]*>',' ',r.get('abstractText') or '')))
        if len(abstract)<90:continue
        pmid=r.get('pmid');doi=r.get('doi');pmcid=r.get('pmcid')
        url=f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/' if pmid else f'https://doi.org/{doi}' if doi else f'https://europepmc.org/articles/{pmcid}'
        items.append({'source':'europe_pmc','chunk_id':str(pmid or doi or pmcid),'document_id':str(pmid or doi or pmcid),
           'title':normalize(r.get('title')),'page':None,'source_path':url,'text':abstract,
           'score':len(toks(q)&toks(abstract))/max(1,len(toks(q))), 'retrievers':['europe_pmc']})
    return sorted(items,key=lambda x:-x['score'])[:n]

def sentences(text):
    s=normalize(text)
    # Split without assuming biomedical abbreviations always form sentence boundaries.
    return [x for x in re.split(r'(?<=[.!?])\s+',s) if 40<=len(x)<=750]

def candidate_quotes(q,docs,topics,max_per_doc=4):
    evidence=[]
    for doc in docs:
        candidates=[]
        for sentence in sentences(doc['text']):
            overlap=len(toks(q)&toks(sentence))/max(1,len(toks(q)))
            matches=topic_matches(sentence,topics)
            if overlap<0.12 and not matches:continue
            if re.search(r'copyright|all rights reserved|corresponding author',sentence,re.I):continue
            if re.search(r'\b(recommend|should receive|prescribe|treatment goal|target a goal)\b',sentence,re.I):continue
            candidates.append({'quote':sentence,'topics':matches,'score':overlap+0.2*len(matches)})
        candidates.sort(key=lambda x:-x['score'])
        for c in candidates[:max_per_doc]:
            evidence.append({'source_id':None,'source':doc['source'],'chunk_id':doc['chunk_id'],
               'document_id':doc['document_id'],'title':doc['title'],'page':doc['page'],
               'source_path':doc['source_path'],'quote':c['quote'],'topics':c['topics'],
               'score':c['score']+doc.get('score',0)})
    evidence.sort(key=lambda x:-x['score']);seen=set();unique=[]
    for e in evidence:
        key=(e['document_id'],e['quote'])
        if key not in seen:unique.append(e);seen.add(key)
    for i,e in enumerate(unique,1):e['source_id']=f'E{i}'
    return unique

@lru_cache(maxsize=1)
def nli_pipeline():
    if not NLI_MODEL:return None
    from transformers import pipeline
    return pipeline('text-classification',model=NLI_MODEL,tokenizer=NLI_MODEL,device=-1)

def relation_check(claim,quote):
    """Conservative heuristic: lexical entities and negation only, not semantic entailment."""
    c=set(concepts(claim));e=set(concepts(quote))
    missing=sorted(c-e)
    if missing:return {'pass':False,'reason':'Claim entities absent from evidence: '+', '.join(missing)}
    if bool(NEGATION.search(claim)) != bool(NEGATION.search(quote)):
        return {'pass':False,'reason':'Potential negation mismatch'}
    # Avoid false assurance: common terms alone are not sufficient for clinical relation support.
    return {'pass':True,'reason':'Entity and negation checks passed; semantic support still unverified'}

def entailment(claim,quote):
    base=relation_check(claim,quote)
    if not base['pass']:
        return {'status':'REJECTED','method':'entity_negation','reason':base['reason']}
    if not NLI_MODEL:
        return {'status':'REVIEW_REQUIRED','method':'heuristic_only',
          'reason':'No NLI model configured; semantic entailment not established'}
    try:
        pipe=nli_pipeline()
        # The HF text-classification pipeline expects a text/text_pair dictionary.
        raw=pipe({'text':quote[:1800],'text_pair':claim[:500]},top_k=None,truncation=True)
        if isinstance(raw,list) and raw and isinstance(raw[0],list):raw=raw[0]
        if isinstance(raw,dict):raw=[raw]
        probs={str(item['label']).lower():float(item['score']) for item in raw}
        entail=[v for k,v in probs.items() if 'entail' in k]
        contradict=[v for k,v in probs.items() if 'contrad' in k]
        if not entail or not contradict:
            return {'status':'REVIEW_REQUIRED','method':'nli','reason':'Model lacks recognizable entailment/contradiction labels; inspect config.id2label','probabilities':probs}
        if max(contradict)>=0.80:
            return {'status':'REJECTED','method':'nli','reason':'Model predicts contradiction','probabilities':probs}
        if max(entail)>=float(os.getenv('NLI_ENTAIL_THRESHOLD','0.90')):
            return {'status':'MODEL_SUPPORTED','method':'nli','reason':'Model predicts textual entailment; independent biomedical review required','probabilities':probs}
        return {'status':'REVIEW_REQUIRED','method':'nli','reason':'Insufficient entailment confidence','probabilities':probs}
    except Exception as e:
        return {'status':'REVIEW_REQUIRED','method':'nli_error','reason':f'{type(e).__name__}: {str(e)[:180]}'}

def ollama_claims(q,evidence,max_claims=8):
    import requests
    short=[{'id':e['source_id'],'quote':e['quote'][:550]} for e in evidence[:16]]
    schema={'type':'object','properties':{'claims':{'type':'array','items':{'type':'object','properties':{
      'claim':{'type':'string'},'evidence_id':{'type':'string'}},'required':['claim','evidence_id']}}},'required':['claims']}
    prompt={'question':q,'evidence':short,'instructions':[
      'Produce up to '+str(max_claims)+' atomic factual claims, each with exactly one evidence_id.',
      'Each claim must be directly entailed by its exact cited quote.',
      'Do not add mechanisms, causality, statistics or qualifiers not in the quote.',
      'Return an empty claims list when the supplied evidence is insufficient.']}
    payload={'model':OLLAMA,'stream':False,'format':schema,'messages':[
      {'role':'system','content':'You are an evidence extraction assistant. Return valid JSON only.'},
      {'role':'user','content':json.dumps(prompt,ensure_ascii=False)}],
      'options':{'temperature':0,'num_ctx':4096,'num_predict':1000}}
    r=requests.post(os.getenv('OLLAMA_URL','http://127.0.0.1:11434/api/chat'),json=payload,timeout=90);r.raise_for_status()
    content=r.json().get('message',{}).get('content','');return json.loads(content).get('claims',[])

def extractive_claims(evidence,max_claims=8):
    # Extractive mode is provenance-checked only, NOT clinically entailed.
    return [{'claim':e['quote'],'evidence_id':e['source_id']} for e in evidence[:max_claims]]

def verify_claims(claims,evidence,extractive=False):
    lookup={e['source_id']:e for e in evidence};audited=[]
    for item in claims:
        claim=normalize(item.get('claim'));eid=str(item.get('evidence_id',''))
        if not claim or eid not in lookup:
            audited.append({'claim':claim,'evidence_id':eid,'status':'REJECTED','reason':'Invalid or missing citation'});continue
        quote=lookup[eid]['quote']
        if extractive and claim==quote:
            check={'status':'EXTRACTIVE_QUOTE','method':'exact_quote','reason':'Exact source quote; no clinical entailment claim'}
        else:check=entailment(claim,quote)
        audited.append({'claim':claim,'evidence_id':eid,'quote':quote,'source':lookup[eid]['source_path'],**check})
    return audited

def coverage(audited,p):
    # Still a topic-mention heuristic; clearly labeled and NOT clinical evidence coverage.
    covered=set()
    for c in audited:
        if c['status'] in ('MODEL_SUPPORTED','EXTRACTIVE_QUOTE'):
            covered.update(topic_matches(c['claim'],p['subtopics']))
    topics=set(p['subtopics']);return {'method':'topic_mention_heuristic','supported':sorted(covered),'missing':sorted(topics-covered),
       'ratio':round(len(covered)/len(topics),3) if topics else None}

def graph_boost(docs,paths):
    # Small observational endpoint bonus; no causality implied.
    if not paths:return docs
    terms=[p['target'] for p in paths if p.get('score',0)>0]
    if not terms:return docs
    out=[]
    for doc in docs:
        d=dict(doc)
        d['graph_boost']=min(0.0015,0.0003*sum(contains(d['text'],term) for term in terms))
        d['score']+=d['graph_boost'];out.append(d)
    return sorted(out,key=lambda d:-d['score'])

def diversify(docs,limit=8,max_per_document=2):
    # Select diverse sources, then fill remaining slots if corpus is narrow.
    chosen=[];counts=Counter();remaining=[]
    for d in docs:
        key=d.get('document_id') or d.get('source_path') or d.get('chunk_id')
        if counts[key]<max_per_document:
            chosen.append(d);counts[key]+=1
        else:remaining.append(d)
        if len(chosen)>=limit:break
    if len(chosen)<limit:
        seen={d['chunk_id'] for d in chosen}
        chosen.extend(d for d in remaining if d['chunk_id'] not in seen)
    return chosen[:limit]

def run_question(q,mode='hybrid_graph',top_k=8,no_llm=False,live=False,verify=True):
    start=time.perf_counter();timings={};warnings=[];p=profile(q);paths=[];gdiag={}
    # Parallel graph and local retrieval: graph paths cannot change which documents
    # are initially retrieved, only the final graph-aware reranking.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        local_future=pool.submit(fuse,q,'hybrid' if mode=='hybrid_graph' else mode,max(top_k*2,12))
        graph_future=pool.submit(graph_retrieve,q,p) if mode=='hybrid_graph' else None
        docs,subtimings,w=local_future.result();timings.update(subtimings);warnings+=w
        if graph_future:
            graph_start=time.perf_counter()
            paths,gdiag,w=graph_future.result();warnings+=w
            # Includes waiting for graph after local results; independent graph
            # timing is added separately by a future profiling revision.
            timings['neo4j_wait_ms']=round((time.perf_counter()-graph_start)*1000,2)
            docs=graph_boost(docs,paths)
    timings['local_retrieval_ms']=round(max(timings.get('sqlite_fts_ms',0),timings.get('pgvector_ms',0)),2)
    docs=diversify(docs,top_k,max_per_document=int(os.getenv("MAX_CHUNKS_PER_DOCUMENT","2")))
    if live:
        with stage_timer(timings,'europe_pmc_ms'):
            try:docs+=pmc(q,n=5)
            except Exception as e:warnings.append(f'Europe PMC: {type(e).__name__}: {str(e)[:130]}')
    with stage_timer(timings,'evidence_screen_ms'):
        evidence=candidate_quotes(q,docs,p['subtopics'])[:32]
    with stage_timer(timings,'generation_ms'):
        if no_llm:claims=extractive_claims(evidence);extractive=True
        else:
            try:claims=ollama_claims(q,evidence);extractive=False
            except Exception as e:
                warnings.append(f'Ollama failed; extractive fallback: {type(e).__name__}: {str(e)[:130]}')
                claims=extractive_claims(evidence);extractive=True
    with stage_timer(timings,'verification_ms'):
        audited=verify_claims(claims,evidence,extractive) if verify else [dict(c,status='NOT_VERIFIED') for c in claims]
    timings['total_ms']=round((time.perf_counter()-start)*1000,2)
    result={'question':q,'timestamp_utc':now(),'mode':mode,'question_profile':p,
       'retrieved_chunks':docs,'retrieved_chunk_count':len(docs),'graph_paths':paths,
       'graph_diagnostics':gdiag,'evidence':evidence,'claims':audited,'coverage':coverage(audited,p),
       'latency_ms':timings,'warnings':warnings,'limitations':[
       'Graph paths and observational relationships are not evidence of causation.',
       'Extractive quotes verify provenance, not medical relevance.',
       'NLI confidence does not establish clinical correctness.',
       'The current Neo4j graph contains observational disease–lab/drug edges, not curated causal mechanism edges.',
       'Topic coverage is heuristic; expert evidence labels are required.']}
    return result

def report(result):
    r=result;lines=['# Phase 12 V8 — Evidence-Grounded Biomedical Research','',
       '**Question:** '+r['question'],'','**Status:** Research prototype; not clinically validated.','',
       '## Model-supported claims (NOT clinically verified)','']
    for c in r['claims']:
        if c['status']=='MODEL_SUPPORTED':
            lines.append(f"- [{c['status']}] {c['claim']} **[{c['evidence_id']}]**")
    if not any(c['status']=='MODEL_SUPPORTED' for c in r['claims']):
        lines.append('No claim passed configured NLI entailment checks. Source excerpts require independent review.')
    review=[c for c in r['claims'] if c['status'] in ('REVIEW_REQUIRED','EXTRACTIVE_QUOTE')]
    if review:
        lines+=['','## Unverified excerpts and claims (not medical recommendations)','']
        for c in review:lines.append(f"- {c['claim']} — {c.get('reason','requires review')}")
    lines+=['','## Graph evidence','']
    if r['graph_paths']:
        for path in r['graph_paths']:
            lines.append('- '+str(path.get('seed'))+' —'+str(path.get('relation'))+'→ '+str(path.get('target'))+' *(observational association; not causal proof)*')
    else:lines.append('No question-relevant clinical graph path found.')
    lines+=['','## References','']
    used_ids={c.get('evidence_id') for c in r['claims']}
    for e in r['evidence']:
        if e['source_id'] not in used_ids:continue
        lines.append(f"- **[{e['source_id']}]** {e['title']} — {e['source_path']}"+(f" (page {e['page']})" if e['page'] else ''))
    lines+=['','## Performance','',f"Total latency: {r['latency_ms']['total_ms']/1000:.2f}s",'',
      '## Limitations','']
    lines+=['- '+x for x in r['limitations']]
    if r['warnings']:lines+=['','## Warnings','']+['- '+x for x in r['warnings']]
    return '\n'.join(lines)+'\n'

# Evaluation questions are DRAFTS. Gold evidence labels and reference answers are intentionally empty.
QUESTION_BANK={
 'risk_factors':[
 'What are the major modifiable cardiovascular risk factors?',
 'How does smoking affect cardiovascular disease risk?',
 'What cardiovascular risks are associated with diabetes mellitus?',
 'How does obesity influence cardiovascular risk?',
 'How is physical inactivity associated with heart disease?',
 'How does dietary sodium relate to blood pressure?',
 'What is the relationship between family history and coronary disease?',
 'How does older age affect cardiovascular risk?',
 'How does dyslipidemia affect cardiovascular outcomes?',
 'What is the role of hypertension in stroke risk?'],
 'mechanisms':[
 'How does hypertension contribute to cardiovascular disease risk?',
 'How does hypertension contribute to left ventricular hypertrophy?',
 'What mechanisms link elevated blood pressure to arterial stiffness?',
 'How does LDL cholesterol contribute to atherosclerosis?',
 'How are foam cells involved in plaque formation?',
 'How does endothelial dysfunction relate to atherosclerosis?',
 'What role does vascular inflammation play in atherosclerosis?',
 'How does plaque rupture contribute to myocardial infarction?',
 'How does diabetes affect vascular endothelial function?',
 'How does chronic kidney disease influence cardiovascular risk?'],
 'diagnostics':[
 'How is HbA1c used to assess long-term glycemic control?',
 'What is the role of lipid panels in cardiovascular risk assessment?',
 'How is hypertension diagnosed using blood pressure measurements?',
 'What is the clinical purpose of measuring LDL cholesterol?',
 'What does an elevated troponin level indicate?',
 'How are kidney function tests used in cardiovascular risk assessment?',
 'What is the role of electrocardiography in cardiac evaluation?',
 'What does echocardiography assess in heart failure?',
 'How is albuminuria used to evaluate kidney disease?',
 'How is arterial stiffness assessed in cardiovascular research?'],
 'medications':[
 'How do statins reduce LDL cholesterol?',
 'What evidence supports statins for cardiovascular prevention?',
 'How do ACE inhibitors lower blood pressure?',
 'What is the mechanism of action of beta blockers?',
 'How do calcium channel blockers affect blood pressure?',
 'What is the cardiovascular role of antiplatelet therapy?',
 'How do SGLT2 inhibitors affect heart failure outcomes?',
 'How do GLP-1 receptor agonists affect cardiovascular outcomes?'],
 'comorbidities':[
 'How are hypertension and chronic kidney disease related?',
 'What is the association between diabetes and coronary artery disease?',
 'How does obesity contribute to hypertension?',
 'How is heart failure associated with kidney dysfunction?',
 'How do diabetes and dyslipidemia interact in cardiovascular risk?',
 'What is the relationship between atrial fibrillation and stroke?',
 'How does sleep apnea relate to hypertension?'],
 'evidence_limits':[
 'Does an association between hypertension and a biomarker establish causality?',
 'Can an observational cohort study prove a medication prevents stroke?',
 'What limitations arise when cardiovascular risk estimates are applied to new populations?',
 'How can confounding affect an observed association between obesity and heart disease?',
 'Why might two studies report different effects of a cardiovascular treatment?']}

def generate_benchmark(path):
    rows=[];n=0
    for category,qs in QUESTION_BANK.items():
        for q in qs:
            n+=1;rows.append({'question_id':f'Q{n:03d}','category':category,'question':q,
                'expected_concepts':concepts(q),'expected_keywords':[],
                'reference_answer':None,'relevant_chunk_ids':[],
                'relevant_graph_paths':[],'review_status':'pending','split':'development' if n%10 in (0,1,2) else 'held_out'})
    write_json(path,rows);return {'path':str(path),'questions':len(rows),'note':'Draft questions only; gold evidence and answers must be manually reviewed.'}

def load_questions(path):
    data=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if isinstance(data,dict):data=data.get('questions',[])
    return [r for r in data if isinstance(r,dict) and r.get('question')]

def export_review(questions,path,mode='hybrid_graph',top_k=8):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    count=0
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f);writer.writerow(['question_id','question','mode','rank','chunk_id','title','source','page',
             'passage_excerpt','relevance_label','reviewer_id','notes'])
        for i,item in enumerate(questions,1):
            result=run_question(item['question'],mode=mode,top_k=top_k,no_llm=True,live=False)
            for rank,doc in enumerate(result['retrieved_chunks'],1):
                writer.writerow([item.get('question_id',f'Q{i:03d}'),item['question'],mode,rank,
                    doc['chunk_id'],doc['title'],doc['source_path'],doc['page'],doc['text'][:900],'','',''])
                count+=1
    return {'review_csv':str(path),'candidate_passages':count,'note':'Fill relevance_label with 0,1,2; optionally -1 for misleading.'}

def metrics_at_k(docs,gold,k=5):
    # Gold must be a human-reviewed set; do not manufacture it from retrieved results.
    if not gold:return {'precision_at_k':None,'recall_at_k':None,'mrr':None,'ndcg_at_k':None}
    gold=set(str(x) for x in gold);ids=[str(d['chunk_id']) for d in docs[:k]]
    hits=[int(x in gold) for x in ids]
    precision=sum(hits)/k;recall=len(set(ids)&gold)/len(gold)
    rr=next((1/i for i,v in enumerate(hits,1) if v),0.0)
    dcg=sum(v/math.log2(i+2) for i,v in enumerate(hits))
    ideal=sum(1/math.log2(i+2) for i in range(min(k,len(gold))))
    return {'precision_at_k':precision,'recall_at_k':recall,'mrr':rr,'ndcg_at_k':dcg/ideal if ideal else None}

def benchmark(questions,mode='hybrid_graph',top_k=8,no_llm=True,live=False,save=True):
    results=[]
    for i,item in enumerate(questions,1):
        r=run_question(item['question'],mode=mode,top_k=top_k,no_llm=no_llm,live=live)
        claims=r['claims'];accepted=[c for c in claims if c['status'] in ('MODEL_SUPPORTED','EXTRACTIVE_QUOTE')]
        gold=(item.get('relevant_chunk_ids') or []) if item.get('review_status') in ('reviewed','adjudicated') else []
        stats=metrics_at_k(r['retrieved_chunks'],gold,min(5,top_k))
        results.append({'question_id':item.get('question_id',f'Q{i:03d}'),'question':item['question'],
            'mode':mode,'latency_ms':r['latency_ms'],'graph_paths':len(r['graph_paths']),
            'n_retrieved':len(r['retrieved_chunks']),'n_claims':len(claims),'n_displayed':len(accepted),
            'n_review_required':sum(c['status']=='REVIEW_REQUIRED' for c in claims),
            'topic_coverage_heuristic':r['coverage']['ratio'],'retrieval_metrics':stats,'warnings':r['warnings']})
    def mean(values):
        x=[v for v in values if isinstance(v,(int,float)) and not isinstance(v,bool)]
        return round(statistics.mean(x),4) if x else None
    summary={'mode':mode,'questions':len(results),'timestamp_utc':now(),
      'mean_latency_ms':mean([r['latency_ms']['total_ms'] for r in results]),
      'p95_latency_ms':round(sorted([r['latency_ms']['total_ms'] for r in results])[min(len(results)-1,math.ceil(.95*len(results))-1)],2) if results else None,
      'mean_graph_paths':mean([r['graph_paths'] for r in results]),
      'mean_topic_coverage_heuristic':mean([r['topic_coverage_heuristic'] for r in results]),
      'mean_precision_at_5':mean([r['retrieval_metrics']['precision_at_k'] for r in results]),
      'mean_recall_at_5':mean([r['retrieval_metrics']['recall_at_k'] for r in results]),
      'mean_ndcg_at_5':mean([r['retrieval_metrics']['ndcg_at_k'] for r in results]),
      'mean_mrr':mean([r['retrieval_metrics']['mrr'] for r in results]),
      'human_gold_question_count':sum(bool(q.get('relevant_chunk_ids')) and q.get('review_status') in ('reviewed','adjudicated') for q in questions),
      'per_question':results,
      'note':'Retrieval metrics are null without human-reviewed relevant_chunk_ids. Topic coverage is NOT clinical accuracy.'}
    if save:write_json(OUT/'benchmark'/f'{mode}.json',summary)
    return summary

def ablation(questions,top_k=8,no_llm=True):
    out={}
    for mode in ('sqlite','pgvector','hybrid','hybrid_graph'):
        out[mode]=benchmark(questions,mode,top_k,no_llm,live=False)
    write_json(OUT/'ablation'/'ablation_summary.json',out)
    with (OUT/'ablation'/'ablation_table.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['mode','questions','precision@5','recall@5','nDCG@5','MRR','mean_latency_ms','p95_latency_ms','mean_graph_paths'])
        for m,s in out.items():writer.writerow([m,s['questions'],s['mean_precision_at_5'],s['mean_recall_at_5'],s['mean_ndcg_at_5'],s['mean_mrr'],s['mean_latency_ms'],s['p95_latency_ms'],s['mean_graph_paths']])
    return {'status':'completed','modes':{m:{k:v for k,v in s.items() if k!='per_question'} for m,s in out.items()},
            'files':[str(OUT/'ablation'/'ablation_summary.json'),str(OUT/'ablation'/'ablation_table.csv')]}

def diagnostic():
    result={'sqlite_path':str(SQLITE),'sqlite_exists':SQLITE.exists(),'timestamp_utc':now()}
    if SQLITE.exists():
        with sqlite_db() as db:
            result['sqlite_chunks']=db.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]
            result['fts_tables']=[r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE name LIKE '%fts%'")]
    try:
        result['pgvector_tables']=pg_tables()
        with pg_connection() as conn,conn.cursor() as cur:
            for t in result['pgvector_tables']:
                from psycopg import sql
                cur.execute(sql.SQL('SELECT count(*) FROM {}').format(sql.Identifier(t['schema'],t['table'])))
                t['row_count']=cur.fetchone()[0]
    except Exception as e:result['pgvector_error']=f'{type(e).__name__}: {str(e)[:180]}'
    try:
        with neo_driver() as d:
            with d.session(database=os.getenv('NEO4J_DATABASE','neo4j')) as s:
                result['neo4j_nodes']=s.run('MATCH (n) RETURN count(n) AS n').single()['n']
                result['neo4j_relationships']=s.run('MATCH ()-[r]->() RETURN count(r) AS n').single()['n']
                result['clinical_relation_counts']=[dict(r) for r in s.run('MATCH (:Disease)-[r:ASSOCIATED_WITH|OBSERVED_WITH_MEDICATION]->() RETURN type(r) AS relation,count(*) AS n')]
    except Exception as e:result['neo4j_error']=f'{type(e).__name__}: {str(e)[:180]}'
    write_json(OUT/'diagnostics.json',result);return result

def main():
    ap=argparse.ArgumentParser(description='Phase 12 Graph-RAG V8, single-file research prototype')
    ap.add_argument('--question');ap.add_argument('--mode',choices=['sqlite','pgvector','hybrid','hybrid_graph'],default='hybrid_graph')
    ap.add_argument('--top-k',type=int,default=8);ap.add_argument('--live-limit',type=int,default=5)
    ap.add_argument('--no-live',action='store_true');ap.add_argument('--live',action='store_true')
    ap.add_argument('--no-llm',action='store_true');ap.add_argument('--diagnose-graph',action='store_true')
    ap.add_argument('--graph-schema',action='store_true');ap.add_argument('--generate-benchmark',nargs='?',const=str(ROOT/'benchmark/phase12_v7_50_questions.json'))
    ap.add_argument('--export-review',type=Path);ap.add_argument('--external-benchmark',type=Path)
    ap.add_argument('--ablation',type=Path);ap.add_argument('--limit',type=int,default=0)
    args=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.diagnose_graph:output=diagnostic()
    elif args.graph_schema:
        output=graph_schema();write_json(OUT/'graph_schema.json',output)
    elif args.generate_benchmark:output=generate_benchmark(Path(args.generate_benchmark))
    elif args.export_review:
        if not args.external_benchmark:ap.error('--export-review requires --external-benchmark')
        qs=load_questions(args.external_benchmark)
        output=export_review(qs[:args.limit] if args.limit else qs,args.export_review,args.mode,args.top_k)
    elif args.ablation:
        qs=load_questions(args.ablation);output=ablation(qs[:args.limit] if args.limit else qs,args.top_k,args.no_llm)
    elif args.external_benchmark:
        qs=load_questions(args.external_benchmark)
        output=benchmark(qs[:args.limit] if args.limit else qs,args.mode,args.top_k,args.no_llm,args.live and not args.no_live)
        output={k:v for k,v in output.items() if k!='per_question'}
    else:
        if not args.question:ap.error('Supply --question or an evaluation action')
        output=run_question(args.question,args.mode,args.top_k,args.no_llm,args.live and not args.no_live)
        write_json(OUT/'last_query.json',output)
        md=report(output);(OUT/'last_query.md').write_text(md,encoding='utf-8');print(md)
        print('JSON:',OUT/'last_query.json');return
    print(json.dumps(output,indent=2,ensure_ascii=False,default=str))

if __name__=='__main__':main()
