"""CareScope Phase 15. Read-only research dashboard; not clinical decision support."""
from __future__ import annotations
import csv, importlib.util, json, os, re, statistics, sys, time, urllib.request, urllib.error
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from uuid import UUID
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STATIC = HERE / 'static'
for p in (ROOT / 'notebooks', HERE):
    if str(p) not in sys.path: sys.path.insert(0, str(p))

# Preserve the original Phase 14 endpoints while adding Phase 15 capabilities.
def load_legacy():
    candidates = [ROOT/'phase14_full.py', ROOT/'carescope_phase14_complete.py', HERE/'phase14_legacy.py']
    path = next((p for p in candidates if p.is_file()), None)
    if not path: raise RuntimeError('Phase 14 backend not found')
    spec = importlib.util.spec_from_file_location('carescope_phase14_backend', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

if not (ROOT/'notebooks/phase12_graph_rag_v7.py').is_file():
    os.environ.setdefault('PHASE12_SCRIPT', str(HERE/'phase12_graph_rag_v7.py'))
legacy = load_legacy()
app = legacy.APP
app.title = 'CareScope Phase 15 | Doctor & DBA Research Portals'
app.version = '15.0'
KEY = APIKeyHeader(name='X-API-Key', auto_error=False)
AUDIT = deque(maxlen=250)
AUDIT_LOCK = Lock()


def require_key(key=Depends(KEY)):
    expected = os.getenv('PHASE15_API_KEY') or os.getenv('PHASE14_API_KEY')
    if not expected: raise HTTPException(503, 'Set PHASE15_API_KEY before starting the portal')
    import secrets
    if not key or not secrets.compare_digest(key, expected): raise HTTPException(401, 'Invalid access key')
    return True


def log_event(kind, elapsed, ok=True):
    with AUDIT_LOCK:
        AUDIT.append({'type':kind,'latency_ms':round(elapsed*1000,2),'ok':ok,
                      'timestamp':datetime.now(timezone.utc).isoformat()})


def db(sql, args=()):
    return legacy.query(sql, args)


def check_pid(pid):
    try: return str(UUID(pid))
    except (ValueError, AttributeError): raise HTTPException(422, 'Invalid patient UUID')


@app.get('/portal', include_in_schema=False)
def portal(): return FileResponse(STATIC/'index.html')


@app.get('/phase15/app.js', include_in_schema=False)
def javascript(): return FileResponse(STATIC/'app.js', media_type='application/javascript')


@app.get('/phase15/styles.css', include_in_schema=False)
def stylesheet(): return FileResponse(STATIC/'styles.css', media_type='text/css')


@app.get('/api/v15/patients', dependencies=[Depends(require_key)])
def patients(search: str='', limit:int=Query(40,ge=1,le=100)):
    return legacy.patients(search=search, limit=limit)


@app.get('/api/v15/patient/{pid}', dependencies=[Depends(require_key)])
def patient(pid:str):
    pid=check_pid(pid)
    t=time.perf_counter()
    result=legacy.patient(pid)
    # The Phase 14 discovery helper does not join prescription_item to prescription.
    prescriptions=db('''SELECT p.prescription_id, p.start_time AS time, p.stop_time,
        p.reason_description, pi.rxcui AS code,
        COALESCE(dc.name, pi.description, pi.rxcui) AS name,
        pi.description AS details
        FROM prescription p JOIN prescription_item pi ON pi.prescription_id=p.prescription_id
        LEFT JOIN drug_concept dc ON dc.rxcui=pi.rxcui
        WHERE p.patient_id=%s::uuid ORDER BY p.start_time DESC LIMIT 500''',(pid,))
    allergies=db('''SELECT allergy_id, start_date, stop_date, code, system,
        description, allergy_type, category, reaction_description_1,
        severity_1, reaction_description_2, severity_2
        FROM patient_allergy WHERE patient_id=%s::uuid
        ORDER BY start_date DESC LIMIT 300''',(pid,))
    result['medications']=legacy.jsonable(prescriptions)
    result['medication_source']='prescription JOIN prescription_item LEFT JOIN drug_concept'
    result['allergies']=legacy.jsonable(allergies)
    result['summary']['medication_records']=len(prescriptions)
    result['summary']['allergy_records']=len(allergies)
    result['timeline']=[x for x in result['timeline'] if x.get('type')!='Medication record']
    result['timeline'] += [{'time':m['time'],'type':'Medication record','name':m['name'],
                             'detail':m['reason_description']} for m in prescriptions]
    result['timeline'] += [{'time':a['start_date'],'type':'Allergy record','name':a['description'],
                             'detail':a['reaction_description_1']} for a in allergies]
    result['timeline'].sort(key=lambda x:str(x.get('time') or ''), reverse=True)
    if not prescriptions: result['review_prompts'].append('No linked prescription items were found.')
    if not allergies: result['review_prompts'].append('No allergy records found. This does not establish no known allergies.')
    log_event('patient_360',time.perf_counter()-t)
    return result


@app.get('/api/v15/medication-review/{pid}', dependencies=[Depends(require_key)])
def medication_review(pid:str, condition:str=Query('',max_length=120)):
    """Evidence discovery only; never asserts indication, efficacy or patient suitability."""
    pid=check_pid(pid)
    legacy.patient_exists(pid)
    t=time.perf_counter()
    conditions=db('''SELECT DISTINCT COALESCE(c.description,c.source_code) AS name,
        c.source_code AS code FROM diagnosis d
        JOIN condition c ON c.condition_id=d.condition_id
        WHERE d.patient_id=%s::uuid ORDER BY name LIMIT 100''',(pid,))
    prescriptions=db('''SELECT DISTINCT pi.rxcui AS code,
        COALESCE(dc.name,pi.description,pi.rxcui) AS name,
        p.reason_description AS recorded_reason
        FROM prescription p JOIN prescription_item pi ON pi.prescription_id=p.prescription_id
        LEFT JOIN drug_concept dc ON dc.rxcui=pi.rxcui
        WHERE p.patient_id=%s::uuid LIMIT 100''',(pid,))
    allergies=db('''SELECT description, code, reaction_description_1, severity_1,
        reaction_description_2, severity_2 FROM patient_allergy
        WHERE patient_id=%s::uuid LIMIT 100''',(pid,))
    selected=condition.strip()
    if selected and not any(selected.lower() in str(c['name']).lower() for c in conditions):
        raise HTTPException(422,'Condition must match a recorded patient diagnosis')
    selected=selected or (str(conditions[0]['name']) if conditions else '')
    # The graph is observational; use a bounded lookup for research context only.
    graph=[]
    if selected:
        try:
            from neo4j import GraphDatabase
            with GraphDatabase.driver(os.getenv('NEO4J_URI','bolt://127.0.0.1:7687'),
                 auth=(os.getenv('NEO4J_USER','neo4j'),os.getenv('NEO4J_PASSWORD')),
                 connection_timeout=3) as driver:
                with driver.session(database=os.getenv('NEO4J_DATABASE','neo4j')) as session:
                    graph=session.run('''MATCH (d:Disease)-[r:OBSERVED_WITH_MEDICATION]-(m:Drug)
                      WHERE any(k IN keys(d) WHERE toLower(toString(d[k])) CONTAINS toLower($term))
                      RETURN properties(m) AS drug, properties(d) AS disease,
                      type(r) AS relationship LIMIT 20''',term=selected[:70]).data()
        except Exception:
            graph=[]
    def trim_props(v):
        return {str(k):str(x)[:160] for k,x in list((v or {}).items())[:7]}
    observations=[{'drug':trim_props(r.get('drug')),'disease':trim_props(r.get('disease')),
                   'relationship':r['relationship'], 'classification':'observational_only'} for r in graph]
    evidence=legacy.literature(selected+' medication treatment guideline' if selected else 'medication guideline',limit=6) if selected else {'status':'no_condition','evidence':[]}
    log_event('medication_evidence_review',time.perf_counter()-t)
    return legacy.jsonable({'status':'evidence_review_only','condition':selected,
        'diagnoses':conditions,'recorded_prescriptions':prescriptions,'recorded_allergies':allergies,
        'observed_graph_associations':observations,'literature':evidence,
        'verified_indications':[], 'patient_specific_recommendations':[],
        'safety_status':'not_assessed',
        'required_before_clinical_use':['Verified drug–condition indications and guideline provenance',
           'Validated contraindication and drug–drug interaction data',
           'Complete allergy reconciliation, medication status and relevant laboratory context',
           'Clinical expert validation and governance'],
        'interpretation':'Observed prescribing is not an indication, and keyword-matched literature is not verified clinical guidance. No patient-specific drug recommendation is produced.'})


@app.get('/api/v15/evidence', dependencies=[Depends(require_key)])
def evidence(q:str=Query(min_length=3,max_length=250)):
    t=time.perf_counter(); out=legacy.literature(q);log_event('literature',time.perf_counter()-t)
    return out


@app.get('/api/v15/graph', dependencies=[Depends(require_key)])
def graph(q:str=Query(min_length=2,max_length=100)):
    t=time.perf_counter(); out=legacy.graph(q);log_event('graph',time.perf_counter()-t)
    return out


class PlannerRequest(BaseModel):
    question:str=Field(min_length=3,max_length=500)
    limit:int=Field(default=6,ge=1,le=20)
    plan_only:bool=True


@app.post('/api/v15/planner', dependencies=[Depends(require_key)])
def planner(req:PlannerRequest):
    t=time.perf_counter()
    try:
        # Phase 13 planner is imported lazily because it can load heavyweight Phase 12 models.
        import phase13_hybrid_query_planner as engine
        registry=engine.CostRegistry()
        out=engine.plan(req.question,registry) if req.plan_only else engine.execute(req.question,registry,req.limit)
        log_event('planner_plan' if req.plan_only else 'planner_execute',time.perf_counter()-t)
        return {'mode':'plan_only' if req.plan_only else 'executed','result':out}
    except Exception as exc:
        log_event('planner',time.perf_counter()-t,False)
        raise HTTPException(503,f'Planner unavailable: {type(exc).__name__}: {str(exc)[:180]}')


def profile_path():
    for p in [ROOT/'results/phase13/cost_profiles.json', HERE/'cost_profiles.json']:
        if p.is_file(): return p
    return None


@app.get('/api/v15/strategy-analytics', dependencies=[Depends(require_key)])
def strategy_analytics():
    p=profile_path()
    if not p:return {'source':'unavailable','strategies':[]}
    rows=json.loads(p.read_text(encoding='utf-8'))
    if isinstance(rows,dict):
        rows=rows.get('observations',rows.get('profiles',[]))
    groups={}
    for r in rows if isinstance(rows,list) else []:
        if not isinstance(r,dict):continue
        strategy=str(r.get('strategy','unknown'))
        if not isinstance(r.get('latency_ms'),(float,int)):continue
        groups.setdefault(strategy,[]).append(float(r['latency_ms']))
    out=[]
    for k,vals in groups.items():
        v=sorted(vals)
        out.append({'strategy':k,'count':len(v),'median_ms':round(statistics.median(v),2),
                    'p95_ms':round(v[min(len(v)-1,int(.95*(len(v)-1)))],2),
                    'min_ms':round(v[0],2),'max_ms':round(v[-1],2)})
    return {'source':str(p),'source_type':'historical recorded samples','strategies':out}


def benchmark_path():
    for p in [ROOT/'results/phase9_benchmark_results.csv',ROOT/'results/phase9/phase9_benchmark_results.csv',HERE/'phase9_benchmark_results.csv']:
        if p.is_file():return p
    return None


@app.get('/api/v15/optimizer', dependencies=[Depends(require_key)])
def optimizer(limit:int=Query(100,ge=1,le=500)):
    p=benchmark_path()
    if not p:return {'status':'unavailable','rows':[]}
    with p.open(newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))[:limit]
    fields=['query_name','query_family','recommendation_type','target_table','target_columns',
            'baseline_latency_ms','candidate_latency_ms','latency_improvement_percent',
            'baseline_cost','candidate_cost','cost_improvement_percent','candidate_index_used','status','reason']
    return {'status':'ok','source':str(p),'source_type':'historical benchmark (not live)',
            'total_displayed':len(rows),'status_counts':dict(Counter(r.get('status','unknown') for r in rows)),
            'rows':[{k:r.get(k) for k in fields} for r in rows]}


@app.get('/api/v15/health', dependencies=[Depends(require_key)])
def health():
    t=time.perf_counter()
    state=legacy.health()
    state['phase15']={'status':'available','phase13_planner_file':(ROOT/'notebooks/phase13_hybrid_query_planner.py').is_file() or (HERE/'phase13_hybrid_query_planner.py').is_file(),
                      'optimizer_report':benchmark_path() is not None,'cost_profiles':profile_path() is not None,
                      'medication_indications':'not validated'}
    log_event('health',time.perf_counter()-t)
    return state


@app.get('/api/v15/activity', dependencies=[Depends(require_key)])
def activity():
    with AUDIT_LOCK: rows=list(AUDIT)
    values=[r['latency_ms'] for r in rows if r['ok']]
    return {'scope':'this application process since startup','total_requests':len(rows),
            'median_ms':round(statistics.median(values),2) if values else None,
            'recent':rows[-40:][::-1]}




# Ollama is optional, local-only, and receives literature snippets rather than patient records.
class EvidenceExplainRequest(BaseModel):
    question: str = Field(min_length=5, max_length=350)
    max_sources: int = Field(default=5, ge=1, le=8)


def ollama_json(url, payload=None, timeout=8):
    base=os.getenv('OLLAMA_URL','http://127.0.0.1:11434').rstrip('/')
    # Never allow arbitrary remote destinations from browser input.
    if not (base.startswith('http://127.0.0.1:') or base.startswith('http://localhost:')):
        raise ValueError('OLLAMA_URL must point to localhost')
    target=base+url
    data=None if payload is None else json.dumps(payload).encode('utf-8')
    request=urllib.request.Request(target,data=data,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=timeout) as response:
        return json.load(response)


@app.get('/api/v15/ollama-status',dependencies=[Depends(require_key)])
def ollama_status():
    model=os.getenv('OLLAMA_MODEL','qwen2.5:7b')
    try:
        models=ollama_json('/api/tags',timeout=3).get('models',[])
        names=[x.get('name','') for x in models]
        return {'online':True,'model':model,'installed':model in names,
                'models':names,'privacy':'Only retrieved literature excerpts are sent to Ollama.'}
    except Exception as exc:
        return {'online':False,'model':model,'installed':False,'models':[],
                'error':type(exc).__name__+': '+str(exc)[:120]}


@app.post('/api/v15/ai-evidence',dependencies=[Depends(require_key)])
def ai_evidence(req:EvidenceExplainRequest):
    started=time.perf_counter()
    # This is NOT a medication recommendation endpoint.
    # Do not accept or transmit patient identifiers or patient records.
    retrieved=legacy.literature(req.question,limit=req.max_sources)
    items=(retrieved.get('evidence') or [])[:req.max_sources]
    citations=[]
    for i,item in enumerate(items,1):
        excerpt=str(item.get('excerpt') or '')[:1500]
        if not excerpt:continue
        citations.append({'id':f'S{i}','title':str(item.get('title') or item.get('document_id') or 'Literature record')[:200],
                          'document_id':str(item.get('document_id') or '')[:100],
                          'source_table':str(item.get('source_table') or '')[:100],
                          'excerpt':excerpt})
    if not citations:
        return {'status':'no_evidence','answer':'No matching literature excerpts were retrieved. An evidence-grounded explanation cannot be generated.',
                'sources':[],'model':None,'limitations':['No retrieved supporting evidence']}
    prompt='\n'.join(f"[{c['id']}] {c['title']} (record {c['document_id']}): {c['excerpt']}" for c in citations)
    model=os.getenv('OLLAMA_MODEL','qwen2.5:7b')
    system=('You are a biomedical literature explanation assistant for an academic research prototype. '
            'Use ONLY the supplied excerpts. Explain in clear, nontechnical language. '
            'Cite every substantive claim with the provided source IDs such as [S1]. '
            'Never recommend a drug, dose, treatment choice or claim patient-specific suitability. '
            'Do not assume an observational association is a treatment indication. '
            'If evidence is weak or missing, state that clearly. '
            'Do not invent study findings, clinical guidelines, contraindications or citations. '
            'Use short sections: What the retrieved evidence says; What remains uncertain.')
    try:
        response=ollama_json('/api/chat',{'model':model,'stream':False,
            'options':{'temperature':0.1,'num_predict':650},
            'messages':[{'role':'system','content':system},
                        {'role':'user','content':'Question: '+req.question+'\n\nRetrieved evidence:\n'+prompt}]},
            timeout=int(os.getenv('OLLAMA_TIMEOUT','90')))
        answer=str((response.get('message') or {}).get('content') or '').strip()
        if not answer:raise ValueError('Ollama returned an empty response')
        # References displayed separately remain the source of truth, not generated claims.
        log_event('ollama_evidence',time.perf_counter()-started)
        return {'status':'generated_unverified','answer':answer,'sources':citations,
                'model':model,'limitations':['AI text is not clinically validated',
                'Source IDs refer to retrieved excerpts, not verified prescribing guidelines',
                'No patient data was sent to Ollama']}
    except Exception as exc:
        log_event('ollama_evidence',time.perf_counter()-started,False)
        return {'status':'ollama_unavailable','answer':'Ollama could not generate an explanation. The retrieved literature excerpts are shown below.',
                'sources':citations,'model':model,'error':type(exc).__name__+': '+str(exc)[:180],
                'limitations':['Generation unavailable; original excerpts remain accessible']}


if __name__=='__main__':
    import uvicorn
    uvicorn.run(app,host='127.0.0.1',port=int(os.getenv('PHASE15_PORT','8015')))
