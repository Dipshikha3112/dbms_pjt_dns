"""CareScope Phase 14: standalone research-only patient workspace.
Install: pip install fastapi uvicorn psycopg[binary] neo4j
Run: python carescope_phase14_complete.py
Test: python carescope_phase14_complete.py --self-test
"""
import os, sys, json, sqlite3, html, re, time, secrets, traceback, logging
from pathlib import Path
from datetime import date, datetime
from contextlib import contextmanager
from urllib.parse import quote
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.responses import HTMLResponse
from fastapi.security import APIKeyHeader
import psycopg
from psycopg.rows import dict_row
from psycopg import sql

APP = FastAPI(title="CareScope Phase 14", version="14.1", description="Research only; not clinical decision support")
KEY = APIKeyHeader(name="X-API-Key", auto_error=False)
DB = os.getenv("PHASE14_PATIENT_DB", "adaptive_healthcare_db")
SQL_LIMIT = 200


def auth(key=Depends(KEY)):
    expected = os.getenv("PHASE14_API_KEY")
    if not expected:
        raise HTTPException(503, "PHASE14_API_KEY not configured")
    if not key or not secrets.compare_digest(key, expected):
        raise HTTPException(401, "Invalid access key")

@contextmanager
def pg():
    with psycopg.connect(host=os.getenv("PGHOST", "127.0.0.1"), port=int(os.getenv("PGPORT", "5432")),
                         dbname=DB, user=os.getenv("PGUSER", "postgres"),
                         password=os.getenv("PGPASSWORD"), connect_timeout=5,
                         row_factory=dict_row, options="-c default_transaction_read_only=on") as conn:
        yield conn

def query(statement, args=()):
    try:
        with pg() as conn, conn.cursor() as cur:
            cur.execute(statement, args)
            return [dict(x) for x in cur.fetchall()]
    except psycopg.Error as exc:
        print("\n[PHASE14 POSTGRESQL ERROR]", type(exc).__name__, str(exc), file=sys.stderr, flush=True)
        print("[PHASE14 SQL]", str(statement)[:1800], file=sys.stderr, flush=True)
        traceback.print_exc()
        raise HTTPException(503, f"PostgreSQL query unavailable: {type(exc).__name__}: {str(exc)[:250]}") from exc

def columns(table):
    return {r['column_name'] for r in query("""SELECT column_name FROM information_schema.columns
        WHERE table_schema='public' AND table_name=%s""", (table,))}

def exists(table):
    return bool(query("SELECT to_regclass(%s) AS tbl", ("public." + table,))[0]['tbl'])

def jsonable(x):
    if isinstance(x, (date, datetime)): return x.isoformat()
    if isinstance(x, dict): return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, list): return [jsonable(v) for v in x]
    return x

def patient_exists(pid):
    rows = query("SELECT patient_id::text AS patient_id FROM patient WHERE patient_id=%s::uuid", (pid,))
    if not rows: raise HTTPException(404, "Patient not found")

def uuid_ok(pid):
    if not re.fullmatch(r'[0-9a-fA-F]{8}-[0-9a-fA-F-]{27,36}', pid):
        raise HTTPException(422, "Invalid patient identifier")
    return pid

@APP.get('/api/v14/health', dependencies=[Depends(auth)])
def health():
    out = {}
    try:
        out['postgresql'] = {'status': 'available', 'patients': query('SELECT count(*) AS n FROM patient')[0]['n']}
    except Exception as e: out['postgresql'] = {'status': 'unavailable', 'reason': str(e)}
    try:
        from neo4j import GraphDatabase
        with GraphDatabase.driver(os.getenv('NEO4J_URI', 'bolt://127.0.0.1:7687'),
                                  auth=(os.getenv('NEO4J_USER', 'neo4j'), os.getenv('NEO4J_PASSWORD')),
                                  connection_timeout=4) as driver:
            driver.verify_connectivity()
        out['neo4j'] = {'status': 'available'}
    except Exception as e: out['neo4j'] = {'status': 'unavailable', 'reason': type(e).__name__}
    store = find_literature()
    out['literature'] = {'status': 'available' if store else 'unavailable', 'path': str(store) if store else None}
    out['overall'] = 'available' if all(v['status']=='available' for k,v in out.items() if isinstance(v,dict)) else 'degraded'
    return out

@APP.get('/api/v14/patients', dependencies=[Depends(auth)])
def patients(search: str = '', limit: int = Query(30,ge=1,le=100)):
    # Never select personal names unless needed; the research dashboard displays pseudonymous IDs.
    if search:
        return jsonable(query('SELECT patient_id::text AS patient_id FROM patient WHERE patient_id::text ILIKE %s ORDER BY patient_id LIMIT %s', ('%'+search+'%', limit)))
    return jsonable(query('SELECT patient_id::text AS patient_id FROM patient ORDER BY patient_id LIMIT %s', (limit,)))

@APP.get('/api/v14/patient/{pid}', dependencies=[Depends(auth)])
def patient(pid: str):
    uuid_ok(pid)
    patient_exists(pid)
    info = query('SELECT patient_id::text, gender, birth_date FROM patient WHERE patient_id=%s::uuid', (pid,))[0]
    diagnoses = query('''SELECT d.diagnosis_id, c.description AS name, c.source_code AS code,
      d.start_date, d.stop_date, d.encounter_id::text AS encounter_id
      FROM diagnosis d LEFT JOIN condition c ON c.condition_id=d.condition_id
      WHERE d.patient_id=%s::uuid ORDER BY d.start_date DESC LIMIT 500''', (pid,))
    lab = []
    if exists('lab_result'):
        lc = columns('loinc_concept') if exists('loinc_concept') else set()
        name_col = next((x for x in ['long_common_name','long_name','display_name','description','name','short_name'] if x in lc), None)
        if name_col:
            try:
                lab = query(sql.SQL('''SELECT l.lab_result_id, l.result_time AS time, l.loinc_code AS code,
                COALESCE(c.{name}, l.loinc_code) AS name, l.result_value AS value,
                l.result_numeric AS numeric_value, l.unit, l.reference_range, l.status
                FROM lab_result l LEFT JOIN loinc_concept c ON l.loinc_code=c.loinc_code
                WHERE l.patient_id=%s::uuid ORDER BY l.result_time DESC LIMIT 500''').format(name=sql.Identifier(name_col)), (pid,))
            except HTTPException:
                print('[PHASE14] LOINC enrichment unavailable; using raw lab codes', file=sys.stderr, flush=True)
                name_col = None
        if not name_col:
            lab = query('''SELECT lab_result_id, result_time AS time, loinc_code AS code,
                loinc_code AS name, result_value AS value, result_numeric AS numeric_value,
                unit, reference_range, status FROM lab_result WHERE patient_id=%s::uuid
                ORDER BY result_time DESC LIMIT 500''', (pid,))
    observations = []
    if exists('observation'):
        observations = query('''SELECT observation_id, observation_time AS time, loinc_code AS code,
          description AS name, value_text AS value, value_numeric AS numeric_value, unit
          FROM observation WHERE patient_id=%s::uuid ORDER BY observation_time DESC LIMIT 500''', (pid,))
    encounters = query('''SELECT e.encounter_id::text AS encounter_id, e.start_time AS time, e.end_time,
      e.encounter_class, COALESCE(e.description, et.description, 'Encounter') AS name,
      e.reason_description FROM encounter e LEFT JOIN encounter_type et
      ON e.encounter_type_id=et.encounter_type_id
      WHERE e.patient_id=%s::uuid ORDER BY e.start_time DESC LIMIT 500''', (pid,))
    meds, med_source = medications(pid)
    timeline=[]
    for d in diagnoses:
        timeline.append({'time':d['start_date'],'type':'Diagnosis recorded','name':d['name'] or d['code'] or 'Unknown condition', 'detail':d['code']})
    for l in lab:
        timeline.append({'time':l['time'],'type':'Laboratory','name':l['name'], 'detail':str(l['value'] if l['value'] is not None else l['numeric_value'])+' '+str(l['unit'] or '')})
    for e in encounters:
        timeline.append({'time':e['time'],'type':'Encounter','name':e['name'], 'detail':e['reason_description']})
    for m in meds:
        timeline.append({'time':m.get('time'),'type':'Medication record','name':m.get('name') or m.get('code') or 'Medication', 'detail':m.get('details')})
    timeline.sort(key=lambda x: str(x['time'] or ''), reverse=True)
    distinct = len({(d['name'],d['code']) for d in diagnoses})
    prompts=[]
    if not lab: prompts.append('No laboratory results found in the lab_result table; verify source coverage.')
    if not meds: prompts.append('No medication history was resolved from a patient-linked medication table.')
    if not encounters: prompts.append('No encounters available in the selected data source.')
    if any(not d['name'] for d in diagnoses): prompts.append('Some diagnosis codes lack descriptions in the condition terminology table.')
    if any(not l.get('unit') for l in lab if l.get('numeric_value') is not None): prompts.append('Some numeric laboratory results have no recorded units.')
    return jsonable({'patient':info, 'summary':{'diagnosis_records':len(diagnoses),'distinct_diagnoses':distinct,
        'laboratory_records':len(lab),'observation_records':len(observations), 'medication_records':len(meds),
        'encounter_records':len(encounters)}, 'diagnoses':diagnoses,'laboratory':lab,
        'observations':observations,'medications':meds,'medication_source':med_source,
        'encounters':encounters,'timeline':timeline[:800], 'review_prompts':prompts,
        'warnings':['Research only; records may be synthetic.','A missing stop date does not prove an active diagnosis.',
                    'No automated diagnosis or treatment advice.','PostgreSQL patient identity is not assumed to match Neo4j identity.']})

def medications(pid):
    """Discover actual patient-linked medication records; do not mistake terminology for prescriptions."""
    candidates=query('''SELECT table_name FROM information_schema.columns WHERE table_schema='public'
        GROUP BY table_name HAVING bool_or(column_name='patient_id')
        AND (table_name ILIKE '%%med%%' OR table_name ILIKE '%%prescrip%%' OR table_name ILIKE '%%drug%%')
        ORDER BY table_name''')
    for x in candidates:
        table=x['table_name']; cols=columns(table)
        if 'patient_id' not in cols: continue
        date_col=next((c for c in ('start_date','start_time','date','prescribed_at','dispensed_at','medication_time') if c in cols),None)
        code_col=next((c for c in ('rxcui','drug_rxcui','medication_code','drug_code','code') if c in cols),None)
        name_col=next((c for c in ('description','medication_name','drug_name','name') if c in cols),None)
        fields=[sql.SQL('t.*')]
        join=sql.SQL('')
        if code_col and exists('drug_concept'):
            join=sql.SQL(' LEFT JOIN drug_concept dc ON t.{code}::text=dc.rxcui').format(code=sql.Identifier(code_col))
            fields.append(sql.SQL('dc.name AS resolved_drug_name'))
        order=sql.SQL(' ORDER BY t.{} DESC').format(sql.Identifier(date_col)) if date_col else sql.SQL('')
        stmt=sql.SQL('SELECT {} FROM {} t {} WHERE t.patient_id=%s::uuid {} LIMIT 500').format(
            sql.SQL(', ').join(fields),sql.Identifier(table),join,order)
        try:
            rows=query(stmt,(pid,))
        except HTTPException as exc:
            print(f"[PHASE14] Skipping optional medication source {table}: {exc.detail}", file=sys.stderr, flush=True)
            continue
        if rows:
            out=[]
            for r in rows:
                name=r.get('resolved_drug_name') or (r.get(name_col) if name_col else None)
                out.append({'name':name or (str(r.get(code_col)) if code_col else 'Unresolved medication'),
                            'code':str(r.get(code_col)) if code_col and r.get(code_col) is not None else None,
                            'time':r.get(date_col) if date_col else None,
                            'details':'; '.join(f'{k}: {v}' for k,v in r.items() if k not in ('patient_id','resolved_drug_name') and v is not None)[:350]})
            return out,table
    return [],'No populated patient-linked medication table discovered'

def find_literature():
    choices=[]
    env=os.getenv('PHASE14_LITERATURE_SQLITE')
    if env: choices.append(Path(env))
    choices += [Path('results/phase12_store_v3/literature.sqlite'),Path('results/phase12_store_v3/literature.db')]
    return next((p for p in choices if p.is_file()),None)

def literature(question, limit=6):
    """Read-only FTS/LIKE retrieval. Reports provenance; never fabricates an LLM answer."""
    path=find_literature()
    if not path: return {'status':'unavailable','reason':'Phase 12 literature SQLite not found','evidence':[]}
    try:
        con=sqlite3.connect(f'file:{quote(str(path.resolve()))}?mode=ro',uri=True,timeout=4)
        con.row_factory=sqlite3.Row
        tables={r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        candidates=[t for t in ('chunks','literature_chunks','document_chunks','passages','papers','documents') if t in tables]
        if not candidates:
            return {'status':'unavailable','reason':'No recognized literature table; inspect Phase 12 schema','evidence':[]}
        tokens=[t.lower() for t in re.findall(r'[A-Za-z][A-Za-z0-9-]{2,}',question) if t.lower() not in
                {'what','which','where','when','about','with','from','that','this','does','the','and','for','are','associated','patient'}][:8]
        if not tokens: return {'status':'no_query_terms','evidence':[]}
        results=[]
        for table in candidates:
            names={r[1] for r in con.execute(f'PRAGMA table_info("{table}")')}
            content=next((c for c in ('chunk_text','text','content','passage','body','abstract') if c in names),None)
            if not content: continue
            title=next((c for c in ('title','document_title','source','filename','paper_title') if c in names),None)
            docid=next((c for c in ('document_id','doc_id','id','chunk_id') if c in names),None)
            where=' OR '.join(f'LOWER("{content}") LIKE ?' for _ in tokens)
            stmt=f'SELECT "{content}" AS snippet, '+(f'"{title}"' if title else "NULL")+' AS title, '+(f'"{docid}"' if docid else 'NULL')+f' AS document_id FROM "{table}" WHERE {where} LIMIT 150'
            rows=con.execute(stmt,tuple('%'+t+'%' for t in tokens)).fetchall()
            for r in rows:
                body=str(r['snippet'] or '')
                score=sum(t in body.lower() for t in tokens)
                if score: results.append({'source_table':table,'title':r['title'],'document_id':r['document_id'],
                                         'excerpt':body[:800], 'matched_terms':score})
        con.close()
        results.sort(key=lambda x:x['matched_terms'],reverse=True)
        return {'status':'ok' if results else 'no_matches','retrieval':'SQLite keyword matching, not validated semantic RAG',
                'evidence':results[:limit], 'disclaimer':'These are source excerpts, not a generated clinical conclusion.'}
    except Exception as exc:
        return {'status':'unavailable','reason':f'SQLite retrieval error: {type(exc).__name__}: {str(exc)[:180]}','evidence':[]}

@APP.get('/api/v14/evidence', dependencies=[Depends(auth)])
def evidence(q: str=Query(min_length=3,max_length=250)):
    return literature(q)

@APP.get('/api/v14/graph', dependencies=[Depends(auth)])
def graph(q: str=Query(min_length=2,max_length=100)):
    try:
        from neo4j import GraphDatabase
        with GraphDatabase.driver(os.getenv('NEO4J_URI','bolt://127.0.0.1:7687'),
                                  auth=(os.getenv('NEO4J_USER','neo4j'),os.getenv('NEO4J_PASSWORD')),
                                  connection_timeout=4) as driver:
            with driver.session(database=os.getenv('NEO4J_DATABASE','neo4j')) as session:
                # Schema-independent observational neighborhood; no PostgreSQL patient joins.
                rows=session.run('''MATCH (n)-[r]-(m) WHERE any(k IN keys(n) WHERE
                    toLower(toString(n[k])) CONTAINS toLower($term))
                    RETURN labels(n) AS source_labels, properties(n) AS source,
                    type(r) AS relationship, labels(m) AS target_labels,
                    properties(m) AS target LIMIT 20''',term=q).data()
        def shorten(v):
            if isinstance(v,dict): return {k:str(x)[:150] for k,x in list(v.items())[:6]}
            return v
        return {'status':'ok','matches':[{'source_labels':r['source_labels'],'source':shorten(r['source']),
                 'relationship':r['relationship'],'target_labels':r['target_labels'],
                 'target':shorten(r['target'])} for r in rows],
                'warning':'Observational graph associations do not establish causality or patient-specific risk.'}
    except Exception as exc:
        return {'status':'unavailable','reason':f'{type(exc).__name__}: {str(exc)[:160]}','matches':[]}

PAGE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CareScope | Phase 14</title><style>
:root{font-family:Inter,Segoe UI,Arial,sans-serif;color:#26354c;background:#f5f8fd}*{box-sizing:border-box}body{margin:0}
header{background:#fff;border-bottom:1px solid #e1e8f1;padding:18px 5%;display:flex;justify-content:space-between;align-items:center}
h1{margin:0;color:#176c9b;font-size:24px}small,.muted{color:#718199}.wrap{max-width:1300px;margin:22px auto;padding:0 18px}
.card{background:#fff;border:1px solid #e3eaf3;border-radius:14px;padding:20px;box-shadow:0 4px 16px #2038540a;margin-bottom:16px}
.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}
.stat{padding:18px;background:#f1f7fd;border-radius:12px}.stat b{font-size:29px;color:#1d6996;display:block}
input{padding:11px;border:1px solid #cbd8e8;border-radius:9px;min-width:220px;flex:1}
button{border:0;background:#1976a7;color:white;padding:11px 16px;border-radius:9px;cursor:pointer}button:hover{background:#125b82}
button.secondary{background:#e7f1f8;color:#176c9b}nav button{margin:3px}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:10px;border-bottom:1px solid #edf1f6;text-align:left;vertical-align:top}th{background:#f3f8fc;color:#4a617c;position:sticky;top:0}
.scroll{max-height:450px;overflow:auto}.pill{display:inline-block;background:#e8f3fb;color:#286b96;border-radius:14px;padding:4px 9px;font-size:12px}
.warning{background:#fff8e7;border:1px solid #f2dfaa;border-radius:10px;padding:12px;color:#805c23;margin:10px 0}
.result{padding:12px;border-bottom:1px solid #e7edf5}.result:hover{background:#f5f9fd}.click{cursor:pointer}
@media(max-width:850px){.grid{grid-template-columns:repeat(2,1fr)}}
</style></head><body><header><div><h1>✚ CareScope</h1><small>Integrated biomedical research workspace · Phase 14</small></div><span class="pill">RESEARCH ONLY · NOT FOR CLINICAL USE</span></header>
<main class="wrap"><div class="card"><div class="row"><label>Access key <input type="password" id="key" placeholder="Enter PHASE14_API_KEY" autocomplete="off"></label><button onclick="loadPatients()">Connect & load patients</button><span id="status" class="muted">Not connected</span></div></div>
<div class="card"><h2>Patient workspace</h2><div class="row"><input id="search" placeholder="Search patient ID" onkeydown="if(event.key==='Enter')loadPatients()"><button onclick="loadPatients()">Search</button><select id="patientList" onchange="loadPatient(this.value)" style="padding:11px;border:1px solid #cbd8e8;border-radius:9px;max-width:100%"><option value="">Choose a patient</option></select></div><p id="identity" class="muted">Select a patient to view recorded history.</p>
<div class="grid"><div class="stat">Diagnoses<b id="d">—</b></div><div class="stat">Lab tests<b id="l">—</b></div><div class="stat">Medications<b id="m">—</b></div><div class="stat">Encounters<b id="e">—</b></div></div></div>
<div class="card"><h2>Patient history</h2><nav id="tabs"><button class="secondary" onclick="tab('timeline')">Timeline</button><button class="secondary" onclick="tab('diagnoses')">Diagnoses</button><button class="secondary" onclick="tab('laboratory')">Laboratory</button><button class="secondary" onclick="tab('medications')">Medications</button><button class="secondary" onclick="tab('encounters')">Encounters</button><button class="secondary" onclick="tab('observations')">Observations</button></nav><div id="history" class="scroll muted">No patient selected</div></div>
<div class="card"><h2>Record review prompts</h2><div id="prompts" class="muted">Source-quality checks will appear here.</div></div>
<div class="card"><h2>Research assistant · literature evidence</h2><p class="muted">Keyword retrieval from the Phase 12 SQLite store. Results are quoted excerpts with provenance, not medical advice or an LLM-generated diagnosis.</p><div class="row"><input id="question" placeholder="e.g. hypertension kidney laboratory tests"><button onclick="findEvidence()">Find evidence</button></div><div id="evidence" class="scroll"></div></div>
<div class="card"><h2>Biomedical graph explorer</h2><p class="muted">Observational associations only. PostgreSQL and Neo4j patient identities are not automatically joined.</p><div class="row"><input id="graphTerm" placeholder="e.g. hypertension"><button onclick="findGraph()">Explore associations</button></div><div id="graph" class="scroll"></div></div>
<div class="card"><h2>Integration status</h2><button onclick="checkHealth()">Check services</button><pre id="health" style="white-space:pre-wrap"></pre></div>
<p class="muted">Read-only research prototype. Use synthetic or appropriately de-identified data only.</p></main>
<script>
let data=null;
const $=id=>document.getElementById(id);
const esc=x=>String(x===null||x===undefined?'—':x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function api(path){let r=await fetch(path,{headers:{'X-API-Key':$('key').value}}),j=await r.json();if(!r.ok)throw Error(j.detail||r.status);return j}
function err(e){$('status').textContent='Error: '+e.message}
async function loadPatients(){try{let a=await api('/api/v14/patients?limit=100&search='+encodeURIComponent($('search').value));$('patientList').innerHTML='<option value="">Choose a patient</option>'+a.map(p=>'<option value="'+esc(p.patient_id)+'">'+esc(p.patient_id)+'</option>').join('');$('status').textContent=a.length+' patient IDs loaded';if(a.length)loadPatient(a[0].patient_id)}catch(e){err(e)}}
async function loadPatient(id){if(!id)return;try{data=await api('/api/v14/patient/'+encodeURIComponent(id));$('patientList').value=id;let p=data.patient,s=data.summary;$('identity').textContent='Patient '+p.patient_id+' · Sex recorded: '+(p.gender||'unknown')+' · Birth date: '+(p.birth_date||'unknown');$('d').textContent=s.diagnosis_records+' ('+s.distinct_diagnoses+' unique)';$('l').textContent=s.laboratory_records;$('m').textContent=s.medication_records;$('e').textContent=s.encounter_records;$('prompts').innerHTML=(data.review_prompts.length?data.review_prompts:['No data-quality prompts generated. This does not imply medical clearance.']).map(x=>'<div class="warning">'+esc(x)+'</div>').join('');tab('timeline');$('status').textContent='Patient record loaded'}catch(e){err(e)}}
const fields={timeline:['time','type','name','detail'],diagnoses:['name','code','start_date','stop_date'],laboratory:['time','name','value','numeric_value','unit','reference_range','status'],medications:['time','name','code','details'],encounters:['time','name','encounter_class','reason_description'],observations:['time','name','value','numeric_value','unit']};
function tab(t){if(!data)return;let a=data[t]||[],cols=fields[t];$('history').innerHTML=a.length?'<table><thead><tr>'+cols.map(c=>'<th>'+esc(c.replaceAll('_',' '))+'</th>').join('')+'</tr></thead><tbody>'+a.map(r=>'<tr>'+cols.map(c=>'<td>'+esc(r[c])+'</td>').join('')+'</tr>').join('')+'</tbody></table>':'<p>No records found in this category.</p>'}
async function findEvidence(){try{let x=await api('/api/v14/evidence?q='+encodeURIComponent($('question').value));$('evidence').innerHTML='<p>Status: '+esc(x.status)+' · '+esc(x.retrieval||x.reason||'')+'</p>'+x.evidence.map(r=>'<div class="result"><b>'+esc(r.title||r.document_id||r.source_table)+'</b><p>'+esc(r.excerpt)+'</p><small>Table: '+esc(r.source_table)+' · Matched terms: '+esc(r.matched_terms)+'</small></div>').join('')}catch(e){err(e)}}
async function findGraph(){try{let x=await api('/api/v14/graph?q='+encodeURIComponent($('graphTerm').value));$('graph').innerHTML='<p>'+esc(x.status)+' · '+esc(x.warning||x.reason||'')+'</p>'+x.matches.map(r=>'<div class="result"><b>'+esc(r.source_labels.join(', '))+' → '+esc(r.relationship)+' → '+esc(r.target_labels.join(', '))+'</b><p>'+esc(JSON.stringify(r.source))+' → '+esc(JSON.stringify(r.target))+'</p></div>').join('')}catch(e){err(e)}}
async function checkHealth(){try{$('health').textContent=JSON.stringify(await api('/api/v14/health'),null,2)}catch(e){err(e)}}
</script></body></html>'''

@APP.get('/', response_class=HTMLResponse)
def home(): return PAGE

def self_test():
    """End-to-end live smoke test; never modifies the databases."""
    from fastapi.testclient import TestClient
    c=TestClient(APP)
    key=os.getenv('PHASE14_API_KEY')
    if not key:
        print('FAIL: Set PHASE14_API_KEY first'); return 1
    h={'X-API-Key':key}
    checks=[]
    def check(name,func):
        try: func();checks.append((name,'PASS'))
        except Exception as e: checks.append((name,'FAIL: '+str(e)[:160]))
    check('Unauthorized request rejected',lambda: assert_status(c.get('/api/v14/patients'),401))
    check('HTML dashboard',lambda: assert_status(c.get('/'),200))
    check('Health endpoint',lambda: assert_status(c.get('/api/v14/health',headers=h),200))
    def patient_workflow():
        r=c.get('/api/v14/patients?limit=1',headers=h);assert_status(r,200)
        ids=r.json();assert ids,'No patients found'
        r=c.get('/api/v14/patient/'+ids[0]['patient_id'],headers=h);assert_status(r,200)
        j=r.json();assert j['summary']['diagnosis_records']==len(j['diagnoses'])
        assert 'timeline' in j and 'review_prompts' in j
    check('Patient search + history + timeline',patient_workflow)
    def literature_workflow():
        r=c.get('/api/v14/evidence?q=hypertension',headers=h);assert_status(r,200)
        assert r.json()['status']=='ok',r.json().get('reason') or r.json()['status']
    check('Literature retrieval (actual evidence)',literature_workflow)
    def graph_workflow():
        r=c.get('/api/v14/graph?q=hypertension',headers=h);assert_status(r,200)
        assert r.json()['status']=='ok',r.json().get('reason')
        assert r.json()['matches'],'No graph relationships returned for test term'
    check('Neo4j graph traversal (actual matches)',graph_workflow)
    for name,result in checks:print(f'{result:65} {name}')
    n=sum(not x.startswith('PASS') for _,x in checks)
    print(f'\n{len(checks)-n}/{len(checks)} checks passed; '+('ALL PASS' if not n else 'NOT FULLY VERIFIED'))
    return int(bool(n))

def assert_status(response,expected):
    assert response.status_code==expected,f'Expected HTTP {expected}, got {response.status_code}: {response.text[:200]}'

if __name__=='__main__':
    if '--self-test' in sys.argv:sys.exit(self_test())
    import uvicorn
    uvicorn.run(APP,host='127.0.0.1',port=int(os.getenv('PHASE14_PORT','8001')))
