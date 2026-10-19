"""Phase 12 Graph-RAG V6 — biomedical literature QA research prototype.

Not a medical device or clinical decision-support system. Do not use for
patient diagnosis, treatment, or triage. Quote verification is not clinical
fact verification. Install: pip install numpy requests psycopg[binary]
neo4j sentence-transformers (pgvector extension required for vector queries).
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(k,'1')
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse, csv, hashlib, html, json, math, re, sqlite3, time, traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import requests

ROOT=Path(__file__).resolve().parent.parent
STORE=Path(os.getenv('LITERATURE_SQLITE',str(ROOT/'results/phase12_store_v3/literature.sqlite')))
OUT=Path(os.getenv('GRAPH_RAG_OUTPUT',str(ROOT/'results/phase12_graph_rag')))
MODEL_NAME=os.getenv('EMBEDDING_MODEL','sentence-transformers/all-MiniLM-L6-v2')
OLLAMA_MODEL=os.getenv('OLLAMA_MODEL','qwen2.5:7b')
OLLAMA_URL=os.getenv('OLLAMA_URL','http://127.0.0.1:11434/api/chat')
PMC_URL='https://www.ebi.ac.uk/europepmc/webservices/rest/search'
MODEL=None
CONCEPTS={
 'hypertension':['hypertension','high blood pressure','elevated blood pressure','htn'],
 'cardiovascular disease':['cardiovascular disease','cardiovascular diseases','cvd','heart disease','coronary heart disease','coronary artery disease'],
 'stroke':['stroke','cerebrovascular accident'],
 'atherosclerosis':['atherosclerosis','atherosclerotic plaque','arterial plaque'],
 'diabetes':['diabetes','diabetes mellitus','type 2 diabetes','t2dm'],
 'dyslipidemia':['dyslipidemia','dyslipidaemia','cholesterol','ldl','hypercholesterolemia'],
 'smoking':['smoking','tobacco','cigarette'],
 'obesity':['obesity','overweight','body mass index'],
 'physical inactivity':['physical inactivity','sedentary','lack of exercise'],
 'unhealthy diet':['unhealthy diet','dietary','high salt intake'],
 'age':['older age','aging','ageing','age factors'],
 'family history':['family history','genetic predisposition'],
 'endothelial dysfunction':['endothelial dysfunction','vascular injury'],
 'arterial stiffness':['arterial stiffness','vascular stiffness'],
 'left ventricular hypertrophy':['left ventricular hypertrophy','cardiac workload','afterload'],
 'inflammation':['inflammation','inflammatory','macrophage'],
 'plaque formation':['plaque formation','foam cells','foam cell','fatty streak'],
 'plaque rupture':['plaque rupture','thrombosis','vulnerable plaque'],
 'chronic kidney disease':['chronic kidney disease','ckd','renal insufficiency'],
}
STOP={'what','which','where','when','how','are','the','and','for','with','from','into','about','major','these','those','their','does','have','been','that','this','can','may','between'}
TOPICS={
 'cvd_risk_factors':{
  'hypertension':['hypertension','blood pressure'], 'smoking':['smoking','tobacco'],
  'diabetes':['diabetes'], 'dyslipidemia':['cholesterol','ldl','dyslipidemia'],
  'obesity':['obesity','overweight'], 'physical inactivity':['physical inactivity','sedentary'],
  'unhealthy diet':['diet','dietary'], 'age':['age','aging','older'],
  'family history':['family history','genetic predisposition']},
 'hypertension_cvd_mechanism':{
  'endothelial dysfunction':['endothelial dysfunction','vascular injury'],
  'arterial stiffness':['arterial stiffness','vascular stiffness'],
  'left ventricular hypertrophy':['left ventricular hypertrophy','afterload','cardiac workload'],
  'atherosclerosis':['atherosclerosis','plaque'],
  'inflammation':['inflammation','inflammatory'],
  'cardiovascular outcomes':['stroke','heart failure','myocardial infarction','cardiovascular events','coronary']},
 'ldl_atherosclerosis_mechanism':{
  'ldl retention':['ldl retention','intimal retention','subendothelial'],
  'plaque formation':['plaque','foam cell','fatty streak'],
  'inflammation':['inflammation','macrophage'],
  'plaque rupture':['plaque rupture','thrombosis','vulnerable plaque']},
}

def norm(s): return ' '.join(str(s or '').split())
def terms(s): return set(re.findall(r'[a-z]{3,}',str(s or '').lower()))-STOP
def hit(s,t): return bool(re.search(r'(?<!\w)'+re.escape(t).replace(r'\ ',r'[\s-]+')+r'(?!\w)',str(s or ''),re.I))
def tags(s): return {k for k,vs in CONCEPTS.items() if any(hit(s,v) for v in vs)}
def entities(q): return [{'canonical':k,'matched_variant':next(v for v in vs if hit(q,v))} for k,vs in CONCEPTS.items() if any(hit(q,v) for v in vs)]
def profile(q):
 l=q.lower()
 if re.search(r'(?:ldl|cholesterol|low.density).*atherosclero|atherosclero.*(?:ldl|cholesterol)',l): key='ldl_atherosclerosis_mechanism'
 elif re.search(r'(?:hypertension|blood pressure).*(?:cardiovascular|heart|stroke|mechanism|contribute|risk|increase)',l): key='hypertension_cvd_mechanism'
 elif re.search(r'risk factors?|predispos',l) and re.search(r'cardiovascular|heart|cvd',l): key='cvd_risk_factors'
 else: key='general'
 return {'intent':key,'subtopics':TOPICS.get(key,{}),'targets':list(TOPICS.get(key,{}))}
def topic_hits(text,prof): return {k for k,vs in prof['subtopics'].items() if any(hit(text,v) for v in vs)}
def model():
 global MODEL
 if MODEL is None:
  from sentence_transformers import SentenceTransformer
  MODEL=SentenceTransformer(MODEL_NAME)
 return MODEL

def sqlite_metadata():
 if not STORE.exists(): raise FileNotFoundError(f'SQLite store missing: {STORE}')
 with sqlite3.connect(STORE) as db:
  db.row_factory=sqlite3.Row
  return {r['chunk_id']:dict(r) for r in db.execute('SELECT c.id chunk_id,c.document_id,c.page,c.body,d.title,d.source_path,d.pages FROM chunks c JOIN documents d ON d.id=c.document_id')}

def fts_search(q,n=50):
 query=' OR '.join('"'+w+'"' for w in sorted(terms(q))[:14]);
 if not query:return []
 with sqlite3.connect(STORE) as db:
  try:return [(x[0],i+1) for i,x in enumerate(db.execute('SELECT id FROM chunk_fts WHERE chunk_fts MATCH ? ORDER BY bm25(chunk_fts) LIMIT ?', (query,n)).fetchall())]
  except sqlite3.Error:return []

def pg_connect():
 import psycopg
 password=os.getenv('PGPASSWORD')
 if not password: raise RuntimeError('Set PGPASSWORD environment variable (non-interactive connection).')
 return psycopg.connect(host=os.getenv('PGHOST','127.0.0.1'),port=int(os.getenv('PGPORT','5433')),
  dbname=os.getenv('PGDATABASE','adaptive_healthcare_phase9_benchmark'),user=os.getenv('PGUSER','postgres'),password=password,connect_timeout=4)

def discover_pgvector():
 """Only choose a table whose chunk IDs can join the SQLite corpus."""
 with pg_connect() as conn,conn.cursor() as cur:
  cur.execute('''SELECT c.table_schema,c.table_name,c.column_name FROM information_schema.columns c
   WHERE c.table_schema NOT IN ('pg_catalog','information_schema')
   AND c.column_name IN ('embedding','chunk_id','embedding_model') ORDER BY 1,2''')
  groups=defaultdict(set)
  for s,t,c in cur.fetchall(): groups[(s,t)].add(c)
  return [{'schema':s,'table':t,'model_filter':'embedding_model' in cols} for (s,t),cols in groups.items() if {'embedding','chunk_id'}<=cols]

def pgvector_search(q,valid_ids,n=50):
 """Probe schemas, verify joins, skip missing/incompatible tables safely."""
 import psycopg
 tables=discover_pgvector()
 if not tables:return [],'No table with embedding and chunk_id columns discovered.'
 vector=model().encode(q,normalize_embeddings=True)
 vec='['+','.join(map(str,np.asarray(vector,dtype=np.float32)))+']'
 diagnostics=[]
 with pg_connect() as conn:
  for tbl in tables:
   schema,table=tbl['schema'],tbl['table']
   # identifiers from information_schema, quoted with psycopg.sql
   from psycopg import sql
   ident=sql.Identifier(schema,table)
   predicate=sql.SQL(' WHERE embedding_model = %s') if tbl['model_filter'] else sql.SQL('')
   stmt=sql.SQL('SELECT chunk_id::text, 1-(embedding <=> %s::vector) AS sim FROM {}{} ORDER BY embedding <=> %s::vector LIMIT %s').format(ident,predicate)
   args=[vec]+([MODEL_NAME] if tbl['model_filter'] else [])+[vec,max(n*3,100)]
   try:
    with conn.cursor() as cur:
     cur.execute('SAVEPOINT pg_probe')
     try:
      cur.execute(stmt,args); rows=cur.fetchall()
      cur.execute('RELEASE SAVEPOINT pg_probe')
     except Exception:
      cur.execute('ROLLBACK TO SAVEPOINT pg_probe');cur.execute('RELEASE SAVEPOINT pg_probe');raise
    joined=[(cid,float(sim),schema+'.'+table) for cid,sim in rows if cid in valid_ids and sim is not None]
    if joined:return joined[:n],None
    diagnostics.append(f'{schema}.{table}: no matching SQLite chunk IDs in top {len(rows)}')
   except Exception as e:diagnostics.append(f'{schema}.{table}: {type(e).__name__}: {str(e)[:100]}')
 return [],'; '.join(diagnostics)

def retrieve_local(q,n=5):
 meta=sqlite_metadata(); ranks=defaultdict(dict); warnings=[]
 for cid,r in fts_search(q,max(60,n*12)):
  if cid in meta:ranks[cid]['fts']=r
 try:
  vec,warning=pgvector_search(q,set(meta),max(60,n*12))
  if warning:warnings.append('pgvector: '+warning)
  for r,(cid,sim,table) in enumerate(vec,1):ranks[cid].update(vector=r,similarity=sim,table=table)
 except Exception as e:warnings.append(f'pgvector unavailable ({type(e).__name__}: {str(e)[:160]}); SQLite FTS5 fallback active')
 # If FTS misses synonyms, semantic embedding search over local corpus can be enabled explicitly.
 if not ranks and os.getenv('LOCAL_EMBEDDING_FALLBACK','0')=='1':
  ids=list(meta); vecs=model().encode([meta[k]['body'][:2000] for k in ids],normalize_embeddings=True,batch_size=64)
  qvec=model().encode(q,normalize_embeddings=True)
  for r,i in enumerate(np.argsort(-(vecs@qvec))[:max(60,n*12)],1):ranks[ids[int(i)]].update(vector=r,similarity=float(vecs[int(i)]@qvec))
 results=[]
 for cid,rank in ranks.items():
  x=meta[cid]; body=norm(x['body']);
  if not body:continue
  lexical=len(terms(q)&terms(body))/max(1,len(terms(q)))
  rrf=sum(1/(60+rank[k]) for k in ('fts','vector') if k in rank)
  score=0.65*max(0,rank.get('similarity',0))+0.2*lexical+0.15*min(1,30*rrf)
  results.append(dict(source='local_pdf',chunk_id=cid,document_id=x['document_id'],pdf_page=x['page'],pdf=x['source_path'],title=x['title'],text=body,rerank_score=round(score,5)))
 results.sort(key=lambda x:-x['rerank_score']);selected=[];seen=Counter()
 for x in results:
  if seen[x['document_id']]>=2:continue
  selected.append(x);seen[x['document_id']]+=1
  if len(selected)>=n:break
 return selected,warnings

def pmc_queries(q,ent,prof):
 names=[x['canonical'] for x in ent if x['canonical'] not in ('age','inflammation')]
 out=[]
 if len(names)>=2:out.append(' AND '.join('"'+s+'"' for s in names[:2]))
 if names:out.append('"'+names[0]+'" AND (risk OR mechanism OR cardiovascular)')
 if prof['intent']=='cvd_risk_factors':out.extend(['"cardiovascular diseases" AND "risk factors"','"cardiovascular risk" AND prevention'])
 elif prof['intent']=='hypertension_cvd_mechanism':out.extend(['hypertension AND (vascular OR cardiac OR stroke)','"blood pressure" AND "cardiovascular disease"'])
 elif prof['intent']=='ldl_atherosclerosis_mechanism':out.extend(['LDL AND atherosclerosis','"low density lipoprotein" AND plaque'])
 else:out.append(q)
 return list(dict.fromkeys(out))[:4]

def pmc_search(q,ent,prof,n=8):
 cache=OUT/'live_cache';cache.mkdir(parents=True,exist_ok=True)
 articles={};warnings=[]; start=time.monotonic()
 for query in pmc_queries(q,ent,prof):
  if len(articles)>=max(n*3,16) or time.monotonic()-start>45:break
  params={'query':f'({query}) AND HAS_ABSTRACT:y','format':'json','resultType':'core','pageSize':min(25,max(n*2,12))}
  key=hashlib.sha1(json.dumps(params,sort_keys=True).encode()).hexdigest()[:20];p=cache/f'{key}.json'
  try:
   if p.exists() and time.time()-p.stat().st_mtime<3600:data=json.loads(p.read_text(encoding='utf-8'))
   else:
    response=requests.get(PMC_URL,params=params,timeout=14,headers={'User-Agent':'ResearchGraphRAG/6.0'})
    response.raise_for_status();data=response.json();p.write_text(json.dumps(data),encoding='utf-8')
   for item in data.get('resultList',{}).get('result',[]):
    pmid,pmcid,doi=item.get('pmid'),item.get('pmcid'),item.get('doi')
    if not (pmid or pmcid or doi):continue
    abstract=norm(html.unescape(re.sub('<[^>]+>',' ',item.get('abstractText') or '')))
    if len(abstract)<100:continue
    key=pmid or pmcid or doi
    if key in articles:continue
    url=('https://pubmed.ncbi.nlm.nih.gov/'+str(pmid)+'/' if pmid else 'https://europepmc.org/articles/'+str(pmcid) if pmcid else 'https://doi.org/'+str(doi))
    articles[key]=dict(source='europe_pmc',title=norm(item.get('title')),text=abstract,pmid=pmid,pmcid=pmcid,doi=doi,url=url,year=item.get('pubYear'))
  except Exception as e:warnings.append(f'Europe PMC query {query[:50]}: {type(e).__name__}: {str(e)[:110]}')
 out=list(articles.values())
 if out:
  try:
   texts=[q]+[x['title']+'. '+x['text'][:2500] for x in out]
   vectors=model().encode(texts,normalize_embeddings=True)
   sims=vectors[1:]@vectors[0]
   for x,s in zip(out,sims):x['rerank_score']=round(0.8*float(s)+0.2*len(terms(q)&terms(x['title']+' '+x['text']))/max(1,len(terms(q))),5)
  except Exception as e:
   warnings.append(f'Semantic rerank unavailable: {type(e).__name__}')
   for x in out:x['rerank_score']=len(terms(q)&terms(x['title']+' '+x['text']))/max(1,len(terms(q)))
  out.sort(key=lambda x:-x['rerank_score'])
 return out[:max(n*2,16)],warnings

def graph_driver():
 from neo4j import GraphDatabase
 password=os.getenv('NEO4J_PASSWORD')
 if not password:raise RuntimeError('Set NEO4J_PASSWORD to enable graph retrieval')
 return GraphDatabase.driver(os.getenv('NEO4J_URI','bolt://127.0.0.1:7687'),auth=(os.getenv('NEO4J_USER','neo4j'),password),connection_timeout=4)

def graph_node(n):
 props={}
 for k,v in dict(n).items():
  if re.search(r'patient|person|name_full|address|email|phone|birth|mrn|ssn|uuid',k,re.I):continue
  if isinstance(v,(str,int,float)):props[k]=str(v)[:180]
  elif isinstance(v,list):props[k]=[str(z)[:90] for z in v[:4]]
 return {'labels':sorted(n.labels),'properties':props}
def node_text(n):return ' '.join(str(v) for v in n.get('properties',{}).values())
def graph_retrieve(q,ent,prof,limit=8):
 diag={'nodes_matched':0,'candidate_paths':0,'accepted_paths':0,'rejections':Counter(),'entities_extracted':ent}
 if not ent:return [],diag,['No recognized clinical entities for graph search']
 warnings=[];paths=[]
 # Require a graph connection between distinct question concepts whenever possible.
 try:
  with graph_driver() as driver:
   driver.verify_connectivity()
   with driver.session(database=os.getenv('NEO4J_DATABASE','neo4j')) as session:
    seeds=[]
    for e in ent[:5]:
     variants=list(dict.fromkeys(CONCEPTS.get(e['canonical'],[e['canonical']])+[e['canonical']]))
     records=session.run('''MATCH (n) WHERE (n:Disease OR n:MeSHConcept OR n:MONDOConcept)
       AND any(k IN ['name','label','term','title','description','synonym','display_name'] WHERE
        n[k] IS NOT NULL AND any(t IN $terms WHERE toLower(toString(n[k]))=t OR
        toLower(toString(n[k])) CONTAINS t)) RETURN n LIMIT 6''',terms=[v.lower() for v in variants])
     for r in records:seeds.append((e['canonical'],r['n']))
    diag['nodes_matched']=len(seeds)
    if not seeds:return [],diag,['No graph concept nodes matched question entities']
    id_concept={n.element_id:c for c,n in seeds};seed_ids=list(id_concept)
    # Bounded expansion: direct first; 2-hop only if both ends are question concepts.
    for depth in (1,2):
     query=('''MATCH p=(a)-[r]-(b) WHERE elementId(a) IN $ids AND
      type(r) IN $rels AND NOT b:Patient RETURN p LIMIT 180''' if depth==1 else
      '''MATCH p=(a)-[r1]-(mid)-[r2]-(b) WHERE elementId(a) IN $ids
      AND elementId(b) IN $ids AND a<>b AND NOT mid:Patient
      AND type(r1) IN $rels AND type(r2) IN $rels RETURN p LIMIT 120''')
     records=session.run(query,ids=seed_ids,rels=['ASSOCIATED_WITH','RELATED_TO','MAPPED_TO','BROADER_THAN','ABOUT_MESH'])
     for record in records:
      p=record['p'];ns=list(p.nodes);rs=list(p.relationships)
      diag['candidate_paths']+=1
      if any('Patient' in n.labels for n in ns):diag['rejections']['patient_node']+=1;continue
      if any(('LabTest' in n.labels or 'Drug' in n.labels or 'ChEBIConcept' in n.labels) for n in ns) and not re.search(r'lab|test|drug|medication|treat',q,re.I):
       diag['rejections']['lab_or_drug_unrequested']+=1;continue
      if any(r.type not in ('ASSOCIATED_WITH','RELATED_TO','MAPPED_TO') for r in rs):
       diag['rejections']['taxonomy_only_or_mapping_chain']+=1;continue
      endpoint=ns[-1];endpoint_text=node_text(graph_node(endpoint))
      endpoint_tags=tags(endpoint_text)
      relevant=bool(endpoint.element_id in id_concept and endpoint.element_id!=ns[0].element_id)
      relevant|=bool(endpoint_tags & set(prof['targets']))
      relevant|=bool(topic_hits(endpoint_text,prof))
      if not relevant:diag['rejections']['endpoint_irrelevant']+=1;continue
      # A RELATED_TO edge is an association, never automatically causal evidence.
      safe=[graph_node(n) for n in ns]
      score=(2 if endpoint.element_id in id_concept else 1)+len(tags(endpoint_text)&set(prof['targets']))-0.1*(depth-1)
      paths.append({'source':'neo4j','nodes':safe,'relationships':[{'type':r.type} for r in rs],
       'hops':depth,'seed_entity':id_concept.get(ns[0].element_id,''),'score':round(score,3),
       'interpretation':'Graph relation only; not evidence of causation'})
    paths.sort(key=lambda x:(-x['score'],x['hops']))
    unique=[];seen=set()
    for p in paths:
     sig=tuple(node_text(n) for n in p['nodes'])+tuple(r['type'] for r in p['relationships'])
     if sig not in seen:unique.append(p);seen.add(sig)
    paths=unique[:limit];diag['accepted_paths']=len(paths)
 except Exception as e:warnings.append(f'Graph retrieval unavailable: {type(e).__name__}: {str(e)[:200]}')
 diag['rejections']=dict(diag['rejections'])
 return paths,diag,warnings

def sentences(s):return [norm(x) for x in re.split(r'(?<=[.!?])\s+',norm(s)) if 40<=len(norm(x))<=650]
def screen(q,items,prof,n):
 candidates=[];excluded=0
 for item in items:
  quotes=[]
  for s in sentences(item['text']):
   sub=topic_hits(s,prof); overlap=len(terms(q)&terms(s))/max(1,len(terms(q)))
   title_overlap=len(terms(q)&terms(item['title']))/max(1,len(terms(q)))
   if prof['subtopics'] and not sub and overlap<0.24:excluded+=1;continue
   if not prof['subtopics'] and overlap<0.16 and title_overlap<0.25:excluded+=1;continue
   if re.search(r'copyright|corresponding author|all rights reserved',s,re.I):excluded+=1;continue
   # Lexical relevance, NOT a formal GRADE quality rating.
   score=0.5*overlap+0.35*min(1,len(sub))+0.15*max(0,item.get('rerank_score',0))
   quotes.append({'quote':s,'matched_subtopics':sorted(sub),'relevance_score':round(score,4)})
  quotes.sort(key=lambda x:-x['relevance_score'])
  if not quotes:continue
  x=dict(item);x['selected_quotes_detail']=quotes[:12];x['selected_quotes']=[z['quote'] for z in quotes[:12]]
  x['matched_subtopics']=sorted(set().union(*(set(z['matched_subtopics']) for z in quotes[:12])))
  x['evidence_score']=round(max(z['relevance_score'] for z in quotes)+0.1*len(x['matched_subtopics']),4)
  candidates.append(x)
 selected=[];covered=set();used=set()
 while candidates and len(selected)<n:
  candidates.sort(key=lambda x:(len(set(x['matched_subtopics'])-covered),x['evidence_score']),reverse=True)
  x=candidates.pop(0);identity=x.get('pmid') or x.get('doi') or x.get('document_id') or x['title']
  if identity in used:continue
  used.add(identity);selected.append(x);covered.update(x['matched_subtopics'])
 targets=prof['targets'];coverage={'required':targets,'supported':sorted(covered),'missing':[x for x in targets if x not in covered],
  'coverage_ratio':round(len(covered)/len(targets),3) if targets else None}
 return selected,coverage,excluded

def build_evidence(local,live,graph):
 out=[]
 for prefix,items in (('L',local),('P',live)):
  for i,x in enumerate(items,1):
   obj={k:v for k,v in x.items() if k!='text'}
   obj['id']=f'{prefix}{i}';obj['quotes']=x['selected_quotes'];obj['citation_valid']=True
   obj['retrieval_timestamp_utc']=datetime.now(timezone.utc).isoformat();out.append(obj)
 for i,p in enumerate(graph,1):out.append({'id':f'G{i}','source':'neo4j','path':p,'quotes':[],'citation_valid':False})
 return out

def allocate_quotes(evidence,prof,max_quotes=24):
 """Round-robin by required topic; no first-source truncation bias."""
 pool=[]
 for e in evidence:
  if e['source']=='neo4j':continue
  for q in e['selected_quotes_detail']:
   pool.append({'id':e['id'],'quote':q['quote'][:650],'subtopics':q['matched_subtopics'],'score':q['relevance_score']})
 selected=[];seen=set()
 for sub in prof['targets']:
  matches=sorted((x for x in pool if sub in x['subtopics']),key=lambda x:-x['score'])
  for x in matches[:2]:
   sig=(x['id'],x['quote'])
   if sig not in seen:selected.append(x);seen.add(sig)
   if len(selected)>=max_quotes:return selected
 for x in sorted(pool,key=lambda x:-x['score']):
  sig=(x['id'],x['quote'])
  if sig not in seen:selected.append(x);seen.add(sig)
  if len(selected)>=max_quotes:break
 return selected

SYNTH_SCHEMA={'type':'object','properties':{'sections':{'type':'array','items':{'type':'object','properties':{
 'subtopic':{'type':'string'},'heading':{'type':'string'},'claims':{'type':'array','items':{'type':'object','properties':{
 'claim':{'type':'string'},'citation_id':{'type':'string'},'quote_id':{'type':'integer'}},
 'required':['claim','citation_id','quote_id']}}},'required':['subtopic','heading','claims']}}},'required':['sections']}

def llm_synthesis(q,evidence,prof):
 allocated=allocate_quotes(evidence,prof)
 if not allocated:return {'status':'insufficient_evidence','sections':[],'answer_summary':''}
 quotes=[{'quote_id':i,'citation_id':x['id'],'text':x['quote'],'subtopics':x['subtopics']} for i,x in enumerate(allocated)]
 prompt={'question':q,'required_subtopics':prof['targets'],'quotes':quotes,
  'task':'Write 1-2 concise claims per supported subtopic. Each claim MUST be directly supported by its quote_id. Do not infer causality from association. Use only quote IDs and citation IDs supplied. Leave unsupported topics out.'}
 payload={'model':OLLAMA_MODEL,'stream':False,'format':SYNTH_SCHEMA,
  'options':{'temperature':0,'num_ctx':6144,'num_predict':1700},
  'messages':[{'role':'system','content':'Biomedical literature research assistant. Return only JSON. Claims must not exceed the meaning of cited quotations. No medical advice.'},
   {'role':'user','content':json.dumps(prompt,ensure_ascii=False)}]}
 response=requests.post(OLLAMA_URL,json=payload,timeout=int(os.getenv('OLLAMA_TIMEOUT','160')))
 response.raise_for_status();body=response.json()
 if body.get('done_reason')=='length':raise ValueError('Ollama response truncated (done_reason=length)')
 parsed=json.loads(body['message']['content']);return {'status':'synthesized','sections':parsed.get('sections',[]),'quote_registry':quotes,'answer_summary':''}

def extractive(q,evidence,prof):
 sections=[]
 for sub in prof['targets']:
  found=[]
  for e in evidence:
   if e['source']=='neo4j':continue
   for item in e['selected_quotes_detail']:
    if sub in item['matched_subtopics']:found.append({'claim':item['quote'],'citation_id':e['id'],'supporting_quote':item['quote']})
  if found:sections.append({'heading':sub.title(),'subtopic':sub,'claims':found[:2]})
 if not sections:
  for e in evidence:
   if e['source']!='neo4j' and e.get('selected_quotes_detail'):
    quote=e['selected_quotes_detail'][0]['quote']
    sections.append({'heading':'Retrieved evidence','subtopic':'general','claims':[{'claim':quote,'citation_id':e['id'],'supporting_quote':quote}]})
    if len(sections)>=3:break
 return {'status':'extractive_fallback','sections':sections,'answer_summary':''}

def audit(answer,evidence):
 """Strict citation+quote ID checks. Lexical overlap is NOT clinical entailment."""
 src={e['id']:e for e in evidence if e['source']!='neo4j'}
 registry={x['quote_id']:x for x in answer.get('quote_registry',[])}
 verified=[];dropped=0
 for section in answer.get('sections',[]):
  claims=[]
  for c in section.get('claims',[]):
   cid=c.get('citation_id');ref=src.get(cid)
   if ref is None:dropped+=1;continue
   if 'quote_id' in c:
    quote=registry.get(c['quote_id'])
    if not quote or quote['citation_id']!=cid:dropped+=1;continue
    text=quote['text']
    # Truncated quote must match prefix of an actual retrieved sentence.
    if not any(norm(z['quote']).startswith(norm(text)) for z in ref['selected_quotes_detail']):dropped+=1;continue
   else:
    text=c.get('supporting_quote','')
    if not any(norm(z['quote'])==norm(text) for z in ref['selected_quotes_detail']):dropped+=1;continue
   claim=norm(c.get('claim',''))
   if not claim:dropped+=1;continue
   # Do not claim semantic verification from word overlap.
   claims.append({'claim':claim,'citation_ids':[cid],'supporting_quotes':[text],'verification':'citation_and_quote_checked_only'})
  if claims:verified.append({'heading':norm(section.get('heading') or section.get('subtopic') or 'Findings'),
   'subtopic':section.get('subtopic','general'),'claims':claims,
   'explanation':' '.join(x['claim'] for x in claims)})
 return {'status':'citation_checked_not_clinically_validated' if verified else 'insufficient_evidence',
  'answer_summary':'','sections':verified,'claim_audit':{'accepted_claims':sum(len(x['claims']) for x in verified),
   'rejected_claims':dropped,'validation_scope':'Citation and quote existence only; semantic entailment and clinical truth not verified'}}

def answer_coverage(answer,prof):
 targets=prof['targets'];supported={s['subtopic'] for s in answer['sections'] if s.get('claims') and s.get('subtopic') in targets}
 return {'required':targets,'supported':sorted(supported),'missing':[t for t in targets if t not in supported],
  'coverage_ratio':round(len(supported)/len(targets),3) if targets else None}

def render(result):
 a=result['validated_answer'];ev={x['id']:x for x in result['evidence']}
 lines=['# Evidence-grounded biomedical research report','',f"**Question:** {result['question']}",'',
  '**Status:** Research prototype — not clinically validated, not for patient care.','',f"**Answer status:** {a['status']}",'']
 for section in a['sections']:
  lines+=['## '+section['heading'],'']
  for c in section['claims']:lines+=['- '+c['claim']+' **['+','.join(c['citation_ids'])+']**']
  lines+=['']
 if not a['sections']:lines+=['No sufficiently citation-checked answer available.','']
 for name,key in [('Evidence coverage (pre-generation)','evidence_coverage'),('Answer coverage (post-generation)','answer_coverage')]:
  x=result[key];lines+=['## '+name,'',f"Supported: {', '.join(x['supported']) or 'None'}",'',f"Missing: {', '.join(x['missing']) or 'None'}",'',f"Ratio: {x['coverage_ratio']}",'']
 lines+=['## Graph evidence','']
 for p in result['graph_paths']:
  parts=[node_text(n)[:85] for n in p['nodes']];rels=[r['type'] for r in p['relationships']]
  lines+=['- '+' '.join(parts[i]+' —'+rels[i]+'→ ' for i in range(len(rels)))+parts[-1]+' (association, not causation)']
 if not result['graph_paths']:lines+=['No question-relevant graph paths found.']
 lines+=['','## References','']
 cited={cid for s in a['sections'] for c in s['claims'] for cid in c['citation_ids']}
 for cid in sorted(cited):
  e=ev[cid];where=(e.get('url') if e['source']=='europe_pmc' else f"{e.get('pdf')} page {e.get('pdf_page')}")
  lines+=['- **['+cid+']** '+e['title']+' — '+str(where)]
 lines+=['','## Warnings and limitations','']
 for w in result['warnings']:lines+=['- '+w]
 lines+=['- Keyword and topic coverage are heuristic, not evidence of clinical correctness.',
  '- Quote matching verifies provenance, not whether a paraphrased claim is entailed.',
  '- Graph edges do not establish causation. Abstracts may omit study limitations.',
  '- Independent expert review and external validation are required before any clinical use.']
 return '\n'.join(lines)

def run_pipeline(question,top_k=5,live_limit=8,no_live=False,no_llm=False,mode='hybrid_verified'):
 start=time.perf_counter();timings={};warnings=[];prof=profile(question);ent=entities(question)
 t=time.perf_counter();graph=[];graph_diag={}
 if mode!='literature_only':graph,graph_diag,w=graph_retrieve(question,ent,prof);warnings+=w
 timings['graph_retrieval_seconds']=round(time.perf_counter()-t,3)
 t=time.perf_counter();local=[];live=[]
 if mode!='graph_only':
  try:local,w=retrieve_local(question,max(top_k*2,8));warnings+=w
  except Exception as e:warnings.append(f'Local retrieval failed: {type(e).__name__}: {str(e)[:220]}')
  if not no_live:
   try:live,w=pmc_search(question,ent,prof,live_limit);warnings+=w
   except Exception as e:warnings.append(f'Live retrieval failed: {type(e).__name__}: {str(e)[:220]}')
 timings['literature_retrieval_seconds']=round(time.perf_counter()-t,3)
 t=time.perf_counter();local,loc_cov,exc1=screen(question,local,prof,top_k);live,live_cov,exc2=screen(question,live,prof,live_limit)
 ev_supported=set(loc_cov['supported'])|set(live_cov['supported']);targets=prof['targets']
 ev_cov={'required':targets,'supported':sorted(ev_supported),'missing':[x for x in targets if x not in ev_supported],
  'coverage_ratio':round(len(ev_supported)/len(targets),3) if targets else None}
 evidence=build_evidence(local,live,graph);timings['screen_seconds']=round(time.perf_counter()-t,3)
 t=time.perf_counter()
 if no_llm or mode=='graph_only':answer=extractive(question,evidence,prof)
 else:
  try:answer=llm_synthesis(question,evidence,prof)
  except Exception as e:
   warnings.append(f'Ollama synthesis failed; extractive fallback: {type(e).__name__}: {str(e)[:180]}')
   answer=extractive(question,evidence,prof)
 timings['synthesis_seconds']=round(time.perf_counter()-t,3)
 t=time.perf_counter();answer=audit(answer,evidence);timings['verification_seconds']=round(time.perf_counter()-t,3)
 ans_cov=answer_coverage(answer,prof);timings['total_seconds']=round(time.perf_counter()-start,3)
 result={'question':question,'timestamp_utc':datetime.now(timezone.utc).isoformat(),'mode':('graph_plus_literature' if graph and (local or live) else 'literature_only' if local or live else 'graph_only' if graph else 'insufficient_evidence'),
  'ablation_mode':mode,'intent':prof['intent'],'entities':ent,'graph_paths':graph,'graph_path_count':len(graph),'graph_retrieval_report':graph_diag,
  'local_pdf_count':len(local),'live_article_count':len(live),'local_results':local,'live_results':live,'evidence':evidence,
  'validated_answer':answer,'evidence_coverage':ev_cov,'answer_coverage':ans_cov,
  'evidence_quality_audit':{'excluded_low_relevance_sentences':exc1+exc2,'screening_method':'lexical_heuristic_not_GRADE'},
  'validation_status':'RESEARCH_PROTOTYPE_NOT_CLINICALLY_VALIDATED','warnings':warnings,
  'latency_seconds':timings['total_seconds'],'latency_breakdown':timings}
 result['report']=render(result);return result

def save_result(result):
 OUT.mkdir(parents=True,exist_ok=True)
 (OUT/'last_query.json').write_text(json.dumps(result,indent=2,default=str,ensure_ascii=False),encoding='utf-8')
 (OUT/'last_query.md').write_text(result['report'],encoding='utf-8')
 print(result['report']);print('\nLATENCY:',result['latency_breakdown']);print('JSON:',OUT/'last_query.json')

def benchmark(path,no_llm=False,mode='hybrid_verified'):
 data=json.loads(Path(path).read_text(encoding='utf-8-sig'));rows=[]
 for item in data:
  if not item.get('question'):continue
  r=run_pipeline(item['question'],no_llm=no_llm,mode=mode)
  blob=' '.join(c['claim']+' '+' '.join(c['supporting_quotes']) for s in r['validated_answer']['sections'] for c in s['claims']).lower()
  expected=item.get('expected_keywords',[])
  hits=[x for x in expected if str(x).lower() in blob];recall=len(hits)/len(expected) if expected else None
  rows.append({'question':item['question'],'keyword_recall':round(recall,3) if recall is not None else None,
   'keyword_pass_at_0_5':recall>=0.5 if recall is not None else None,'expected_keywords':expected,'matched_keywords':hits,
   'evidence_coverage':r['evidence_coverage']['coverage_ratio'],'answer_coverage':r['answer_coverage']['coverage_ratio'],
   'graph_paths':r['graph_path_count'],'accepted_claims':r['validated_answer']['claim_audit']['accepted_claims'],
   'latency_seconds':r['latency_seconds'],'warnings':r['warnings']})
 valid=[x['keyword_recall'] for x in rows if x['keyword_recall'] is not None]
 summary={'status':'completed','mode':mode,'n_questions':len(rows),'mean_keyword_recall':round(float(np.mean(valid)),3) if valid else None,
  'keyword_pass_rate_at_0_5':round(sum(x['keyword_pass_at_0_5'] for x in rows if x['keyword_pass_at_0_5'] is not None)/len(valid),3) if valid else None,
  'mean_evidence_coverage':round(float(np.mean([x['evidence_coverage'] for x in rows if x['evidence_coverage'] is not None])),3) if any(x['evidence_coverage'] is not None for x in rows) else None,
  'mean_answer_coverage':round(float(np.mean([x['answer_coverage'] for x in rows if x['answer_coverage'] is not None])),3) if any(x['answer_coverage'] is not None for x in rows) else None,
  'mean_graph_paths':round(float(np.mean([x['graph_paths'] for x in rows])),2) if rows else None,
  'mean_latency_seconds':round(float(np.mean([x['latency_seconds'] for x in rows])),2) if rows else None,
  'per_question':rows,'note':'Keyword pass rate is NOT clinical accuracy; no clinical correctness metric is computed.'}
 folder=OUT/'external_benchmark';folder.mkdir(parents=True,exist_ok=True)
 (folder/f'external_benchmark_summary_{mode}.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
 return summary

def ablation(path,no_llm=False):
 results={m:benchmark(path,no_llm,m) for m in ['literature_only','graph_only','hybrid','hybrid_verified']}
 folder=OUT/'ablation';folder.mkdir(parents=True,exist_ok=True)
 (folder/'ablation_summary.json').write_text(json.dumps(results,indent=2),encoding='utf-8');return results

def clinician_template(path=None):
 folder=OUT/'clinician_eval';folder.mkdir(parents=True,exist_ok=True)
 questions=Path(path).read_text(encoding='utf-8-sig').splitlines() if path else []
 dest=folder/'clinician_eval_template.csv'
 with dest.open('w',newline='',encoding='utf-8-sig') as f:
  writer=csv.writer(f);writer.writerow(['question_id','question','answers_question','uses_evidence','clinical_concordance','safety_flag','annotator_id','annotator_role','annotation_notes'])
  for i,q in enumerate(questions,1):
   if q.strip():writer.writerow([f'Q{i:03d}',q.strip()]+['']*7)
 return {'status':'created','path':str(dest),'questions':len(questions)}

def clinician_eval(path):
 with Path(path).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
 groups=defaultdict(dict)
 for r in rows:
  if r.get('question_id') and r.get('annotator_id'):groups[r['question_id']][r['annotator_id']]=r
 eligible={k:v for k,v in groups.items() if len(v)>=2}
 if len(eligible)<3:return {'status':'insufficient_annotations','n_questions_with_multiple_raters':len(eligible),'note':'At least 3 multiply-rated questions required; ideally use blinded clinician raters.'}
 # Avoid invalid Fleiss kappa on unequal rater panels; report descriptive agreement.
 results={}
 for dim in ['answers_question','uses_evidence','clinical_concordance','safety_flag']:
  agreement=[]
  for anns in eligible.values():
   vals=[r.get(dim,'').strip().lower() for r in anns.values() if r.get(dim,'').strip()]
   if len(vals)>=2:agreement.append(len(set(vals))==1)
  results[dim]={'unanimous_agreement_rate':round(sum(agreement)/len(agreement),4) if agreement else None,'n_rated_questions':len(agreement)}
 return {'status':'descriptive_only','n_questions':len(eligible),'dimensions':results,'note':'No inter-rater reliability statistic claimed without balanced complete rater panels.'}

def diagnose():
 result={'sqlite_path':str(STORE),'sqlite_exists':STORE.exists()}
 if STORE.exists():
  with sqlite3.connect(STORE) as db:
   result['sqlite_tables']=[r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
   result['chunks']=db.execute('SELECT count(*) FROM chunks').fetchone()[0]
 try:result['pgvector_tables']=discover_pgvector()
 except Exception as e:result['pgvector_error']=f'{type(e).__name__}: {str(e)[:200]}'
 try:
  with graph_driver() as d:
   d.verify_connectivity()
   with d.session(database=os.getenv('NEO4J_DATABASE','neo4j')) as s:
    result['neo4j_nodes']=s.run('MATCH (n) RETURN count(n) AS n').single()['n']
    result['neo4j_relationships']=s.run('MATCH ()-[r]->() RETURN count(r) AS n').single()['n']
 except Exception as e:result['neo4j_error']=f'{type(e).__name__}: {str(e)[:200]}'
 OUT.mkdir(parents=True,exist_ok=True)
 (OUT/'graph_diagnostics.json').write_text(json.dumps(result,indent=2),encoding='utf-8');return result

def main():
 p=argparse.ArgumentParser(description='Phase 12 Graph-RAG V6 research prototype (NOT clinical software)')
 p.add_argument('--question');p.add_argument('--top-k',type=int,default=5);p.add_argument('--live-limit',type=int,default=8)
 p.add_argument('--no-live',action='store_true');p.add_argument('--no-llm',action='store_true');p.add_argument('--diagnose-graph',action='store_true')
 p.add_argument('--mode',choices=['literature_only','graph_only','hybrid','hybrid_verified'],default='hybrid_verified')
 p.add_argument('--external-benchmark',type=Path);p.add_argument('--ablation',type=Path)
 p.add_argument('--generate-eval-questions',action='store_true');p.add_argument('--eval-question-file',type=Path);p.add_argument('--clinician-eval',type=Path)
 a=p.parse_args()
 if a.diagnose_graph:out=diagnose()
 elif a.generate_eval_questions:out=clinician_template(a.eval_question_file)
 elif a.clinician_eval:out=clinician_eval(a.clinician_eval)
 elif a.external_benchmark:out=benchmark(a.external_benchmark,a.no_llm,a.mode)
 elif a.ablation:out=ablation(a.ablation,a.no_llm)
 else:
  if not a.question:p.error('--question required for QA mode')
  save_result(run_pipeline(a.question,max(1,min(a.top_k,20)),max(1,min(a.live_limit,20)),a.no_live,a.no_llm,a.mode));return
 print(json.dumps(out,indent=2,default=str))

if __name__=='__main__':main()
