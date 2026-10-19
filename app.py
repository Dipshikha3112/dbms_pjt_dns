"""Phase 14 research-only FastAPI backend and clinician review workspace.
Run from project root: uvicorn app:app --host 127.0.0.1 --port 8000
Patient endpoints require PHASE14_API_KEY and read-only database credentials.
"""
from __future__ import annotations
import csv
import importlib.util
import json
import os
import re
import sqlite3
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Security
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
NOTEBOOKS = ROOT / 'notebooks'
STATIC = ROOT / 'static'
AUDIT = ROOT / 'results/phase8_phase9_v7_audit/20261018_130202_315654'
SQLITE_STORE = ROOT / 'results/phase12_store_v3/literature.sqlite'
PROFILE = ROOT / 'results/phase13/cost_profiles.json'
PATIENT_DB = os.getenv('PHASE14_PATIENT_DB', os.getenv('PG_ORIGINAL_DATABASE', 'adaptive_healthcare_db'))
PATIENT_SCHEMA = os.getenv('PHASE14_PATIENT_SCHEMA', 'public')
MONITOR_DB = os.getenv('PHASE14_MONITOR_DB', 'adaptive_healthcare_phase9_benchmark')
API_KEY = os.getenv('PHASE14_API_KEY', '')
KEY_HEADER = APIKeyHeader(name='X-API-Key', auto_error=False)


def protect(key: str | None = Security(KEY_HEADER)):
    # No silent unauthenticated patient data exposure, including in demo mode.
    import secrets
    if not API_KEY:
        raise HTTPException(503, 'Set PHASE14_API_KEY before using protected endpoints')
    if not key or not secrets.compare_digest(key, API_KEY):
        raise HTTPException(401, 'Valid X-API-Key required')
    return True


def now():
    return datetime.now(timezone.utc).isoformat()


def jsonable(value):
    if isinstance(value, (datetime, Path)):
        return str(value)
    if hasattr(value, 'isoformat'):
        return value.isoformat()
    if hasattr(value, 'item'):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def pg(dbname=PATIENT_DB):
    try:
        import psycopg
        from psycopg.rows import dict_row
        return psycopg.connect(host=os.getenv('PGHOST', '127.0.0.1'),
            port=int(os.getenv('PGPORT', '5432')), dbname=dbname,
            user=os.getenv('PGUSER', 'postgres'), password=os.getenv('PGPASSWORD') or None,
            row_factory=dict_row, connect_timeout=4,
            options='-c default_transaction_read_only=on -c statement_timeout=10000')
    except Exception as e:
        raise HTTPException(503, f'PostgreSQL connection unavailable: {type(e).__name__}')


def schema_map(conn):
    """Discover supported names; never interpolate client-controlled SQL identifiers."""
    with conn.cursor() as cur:
        cur.execute('''SELECT table_name, column_name FROM information_schema.columns
            WHERE table_schema=%s ORDER BY table_name, ordinal_position''', (PATIENT_SCHEMA,))
        mapping = {}
        for r in cur.fetchall():
            mapping.setdefault(r['table_name'].lower(), set()).add(r['column_name'].lower())
        return mapping


def table_match(mapping, options):
    return next((x for x in options if x in mapping), None)


def col_match(columns, options):
    return next((x for x in options if x in columns), None)


def patient_config(mapping):
    t = table_match(mapping, ['patient', 'patients'])
    if not t:
        raise HTTPException(503, 'Patient table not found; configure schema mapping')
    cols = mapping[t]
    pid = col_match(cols, ['patient_id', 'id', 'patient_uuid', 'person_id'])
    if not pid:
        raise HTTPException(503, 'Patient ID column not found')
    return t, pid, cols


def identifier(s):
    from psycopg import sql
    return sql.Identifier(s)


def fetch_rows(conn, table, conditions, params, limit=50, order=None):
    from psycopg import sql
    q = sql.SQL('SELECT * FROM {}.{} WHERE ').format(identifier(PATIENT_SCHEMA), identifier(table))
    q += sql.SQL(' AND ').join(conditions)
    if order:
        q += sql.SQL(' ORDER BY {} DESC NULLS LAST').format(identifier(order))
    q += sql.SQL(' LIMIT %s')
    with conn.cursor() as cur:
        cur.execute(q, (*params, limit))
        return [jsonable(dict(r)) for r in cur.fetchall()]


def safe_patient(row, columns):
    # Patient display is intentionally minimal. No arbitrary column dump.
    permitted = ['patient_id', 'id', 'patient_uuid', 'person_id', 'age', 'sex', 'gender',
                 'birth_date', 'date_of_birth', 'name', 'full_name']
    return {k: jsonable(row[k]) for k in permitted if k in columns and k in row}


def find_patient(conn, mapping, pid):
    from psycopg import sql
    t, key, cols = patient_config(mapping)
    with conn.cursor() as cur:
        # Compare as text to support UUID and integer identifiers without unsafe casts.
        cur.execute(sql.SQL('SELECT * FROM {}.{} WHERE {}::text=%s LIMIT 1').format(
            identifier(PATIENT_SCHEMA), identifier(t), identifier(key)), (pid,))
        row = cur.fetchone()
    if row is None:
        raise HTTPException(404, 'Patient not found')
    return safe_patient(dict(row), cols)


def get_history(conn, mapping, pid, limit=80):
    """History queries use only verified FK columns; never guess a patient join."""
    sections = {
        'diagnoses': ['diagnosis', 'diagnoses', 'condition', 'conditions'],
        'lab_results': ['lab_result', 'lab_results', 'laboratory_result', 'laboratory_results'],
        'prescriptions': ['prescription', 'prescriptions', 'medication_order', 'medication_orders'],
        'encounters': ['encounter', 'encounters', 'visit', 'visits'],
    }
    output = {}
    availability = {}
    for section, options in sections.items():
        t = table_match(mapping, options)
        if not t:
            output[section] = []
            availability[section] = 'table_unavailable'
            continue
        cols = mapping[t]
        fk = col_match(cols, ['patient_id', 'patient_uuid', 'person_id'])
        if not fk:
            output[section] = []
            availability[section] = 'patient_key_unmapped'
            continue
        from psycopg import sql
        condition = sql.SQL('{}::text=%s').format(identifier(fk))
        date_col = col_match(cols, ['recorded_at', 'result_date', 'diagnosis_date', 'start_date',
                                    'encounter_date', 'visit_date', 'created_at', 'date'])
        output[section] = fetch_rows(conn, t, [condition], [pid], limit, date_col)
        availability[section] = 'available'
    return output, availability


def review_flags(history, availability):
    """Data-quality prompts only; never infer diagnoses, risk, or treatment."""
    flags = []
    for label, field in [('Diagnoses', 'diagnoses'), ('Laboratory results', 'lab_results'),
                         ('Prescriptions', 'prescriptions'), ('Encounters', 'encounters')]:
        status = availability[field]
        if status != 'available':
            flags.append({'type':'data_availability','title':f'{label} source unavailable',
                          'detail':f'Source status: {status}. Verify database mapping.',
                          'priority':'information','evidence':'schema inspection'})
        elif not history[field]:
            flags.append({'type':'record_review','title':f'No {label.lower()} returned',
                          'detail':'No records were returned for this patient in the configured source. This does not prove the information is absent.',
                          'priority':'information','evidence':'patient query'})
    return flags


@lru_cache(maxsize=1)
def planner_module():
    path = NOTEBOOKS / 'phase13_hybrid_query_planner.py'
    if not path.is_file():
        raise HTTPException(503, 'Phase 13 planner file missing')
    spec = importlib.util.spec_from_file_location('phase13_backend', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def neo_driver():
    password = os.getenv('NEO4J_PASSWORD')
    if not password:
        raise HTTPException(503, 'NEO4J_PASSWORD is not configured')
    from neo4j import GraphDatabase
    return GraphDatabase.driver(os.getenv('NEO4J_URI','bolt://127.0.0.1:7687'),
         auth=(os.getenv('NEO4J_USER','neo4j'),password), connection_timeout=5)


class PlannerRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    limit: int = Field(default=8, ge=1, le=30)
    plan_only: bool = False
    mode: Literal['auto','sql','graph','vector','hybrid'] = 'auto'

class GraphRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    limit: int = Field(default=8, ge=1, le=30)

class RAGRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    mode: Literal['vector','hybrid'] = 'hybrid'


app = FastAPI(title='Healthcare Research — Doctor Review Workspace', version='14.0.0',
    description='Research-only patient review, knowledge graph, planner, RAG and optimizer APIs. Not validated for clinical decision-making.')

@app.get('/', include_in_schema=False)
def homepage():
    p = STATIC / 'index.html'
    if not p.is_file():
        raise HTTPException(503, 'Dashboard HTML missing')
    return FileResponse(p)

@app.get('/api/v1/system/health', tags=['System'])
def health():
    status = {'timestamp':now(), 'service':'running', 'dependencies':{}}
    status['dependencies']['planner_file'] = (NOTEBOOKS / 'phase13_hybrid_query_planner.py').is_file()
    status['dependencies']['literature_sqlite'] = SQLITE_STORE.is_file()
    status['dependencies']['optimizer_audit'] = AUDIT.is_dir()
    try:
        with pg() as conn:
            with conn.cursor() as cur:
                cur.execute('SELECT 1')
                status['dependencies']['patient_postgresql'] = cur.fetchone() is not None
    except Exception:
        status['dependencies']['patient_postgresql'] = False
    try:
        with neo_driver() as driver:
            driver.verify_connectivity()
            status['dependencies']['neo4j'] = True
    except Exception:
        status['dependencies']['neo4j'] = False
    status['overall'] = 'ready' if all(status['dependencies'].values()) else 'degraded'
    return status

@app.get('/api/v1/system/capabilities', tags=['System'])
def capabilities():
    return {'phase14':'integrated research prototype', 'clinical_validation':False,
      'patient_identity_bridge':'unverified between PostgreSQL and Neo4j',
      'suggestion_scope':'data-quality review only, not diagnoses or treatment',
      'components':{'phase9_optimizer':AUDIT.is_dir(), 'phase12_literature':SQLITE_STORE.is_file(),
                    'phase13_planner':(NOTEBOOKS/'phase13_hybrid_query_planner.py').is_file()}}

@app.get('/api/v1/patients/search', tags=['Patients'], dependencies=[Depends(protect)])
def patient_search(q: str = Query('', max_length=100), limit: int = Query(20, ge=1, le=50),
                   offset: int = Query(0, ge=0, le=100000)):
    from psycopg import sql
    with pg() as conn:
        mapping = schema_map(conn)
        t, pid, cols = patient_config(mapping)
        searchable = [x for x in ('name','full_name','patient_id','patient_uuid','id','person_id') if x in cols]
        if q:
            terms = [sql.SQL('{}::text ILIKE %s').format(identifier(x)) for x in searchable]
            if not terms:
                raise HTTPException(503,'No searchable patient fields')
            where = sql.SQL(' WHERE ') + sql.SQL(' OR ').join(terms)
            params = ['%'+q+'%'] * len(terms)
        else:
            where = sql.SQL('')
            params = []
        stmt = sql.SQL('SELECT * FROM {}.{}').format(identifier(PATIENT_SCHEMA),identifier(t))
        stmt += where + sql.SQL(' ORDER BY {}::text LIMIT %s OFFSET %s').format(identifier(pid))
        with conn.cursor() as cur:
            cur.execute(stmt, (*params, limit, offset))
            rows = [safe_patient(dict(r), cols) for r in cur.fetchall()]
        return {'patients':rows, 'count':len(rows), 'limit':limit, 'offset':offset,
                'patient_id_field':pid, 'source':'PostgreSQL', 'research_only':True}

@app.get('/api/v1/patients/{patient_id}', tags=['Patients'], dependencies=[Depends(protect)])
def patient_profile(patient_id: str):
    with pg() as conn:
        return {'patient':find_patient(conn,schema_map(conn),patient_id),'source':'PostgreSQL'}

@app.get('/api/v1/patients/{patient_id}/history', tags=['Patients'], dependencies=[Depends(protect)])
def patient_history(patient_id: str, limit: int=Query(80,ge=1,le=200)):
    with pg() as conn:
        mapping=schema_map(conn)
        patient=find_patient(conn,mapping,patient_id)
        history, availability=get_history(conn,mapping,patient_id,limit)
        return {'patient':patient,'history':history,'availability':availability,
                'review_flags':review_flags(history,availability),
                'clinical_decision_support':False,
                'note':'Recorded data only; no diagnosis or treatment is inferred.'}

@app.get('/api/v1/graph/patients', tags=['Graph'], dependencies=[Depends(protect)])
def graph_patients(limit: int=Query(20,ge=1,le=100)):
    with neo_driver() as driver:
        with driver.session(database=os.getenv('NEO4J_DATABASE','neo4j')) as s:
            rows=s.run('MATCH (p:Patient) RETURN p.patient_id AS patient_id LIMIT $n',n=limit).data()
    return {'patients':[r['patient_id'] for r in rows if r.get('patient_id')],
            'identity_bridge_verified':False}

@app.get('/api/v1/graph/patient/{patient_id}', tags=['Graph'], dependencies=[Depends(protect)])
def graph_patient(patient_id: str, limit: int=Query(50,ge=1,le=100)):
    with neo_driver() as driver:
        with driver.session(database=os.getenv('NEO4J_DATABASE','neo4j')) as s:
            rows=s.run('''MATCH (p:Patient {patient_id:$pid})-[r]-(n)
              RETURN type(r) AS relationship, labels(n) AS labels,
              coalesce(n.name,n.description,n.loinc_code,n.rxcui) AS label
              LIMIT $n''',pid=patient_id,n=limit).data()
    return {'patient_id':patient_id,'relationships':rows,'identity_bridge_verified':False,
            'warning':'Graph patient identifiers have not been verified against PostgreSQL identifiers.'}

@app.post('/api/v1/graph/explore', tags=['Graph'], dependencies=[Depends(protect)])
def graph_explore(req:GraphRequest):
    try:
        return jsonable(planner_module().graph_execute(req.question,req.limit))
    except ValueError as e:
        raise HTTPException(422,str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(503,f'Graph unavailable: {type(e).__name__}')

@app.post('/api/v1/planner/execute', tags=['Planner'], dependencies=[Depends(protect)])
def planner_execute(req:PlannerRequest):
    p=planner_module()
    registry=p.CostRegistry()
    force=None if req.mode=='auto' else req.mode
    try:
        result=p.plan(req.question,registry,force) if req.plan_only else p.execute(
            req.question,registry,req.limit,force)
        return jsonable(result)
    except ValueError as e:
        raise HTTPException(422,str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(503,f'Planner unavailable: {type(e).__name__}')

@app.get('/api/v1/planner/costs', tags=['Planner'], dependencies=[Depends(protect)])
def planner_costs():
    return planner_module().CostRegistry().summary()

@app.post('/api/v1/rag/retrieve', tags=['Evidence'], dependencies=[Depends(protect)])
def rag_retrieve(req:RAGRequest):
    p=planner_module()
    try:
        if req.mode=='vector':
            result=p.vector_execute(req.question,req.top_k)
        else:
            result=p.hybrid_execute(req.question,{'vector'},req.top_k)
        return {'question':req.question,'mode':req.mode,'retrieval':jsonable(result),
                'answer_generated':False,'clinical_validation':False}
    except Exception as e:
        raise HTTPException(503,f'Evidence retrieval unavailable: {type(e).__name__}: {str(e)[:180]}')

@app.post('/api/v1/rag/answer', tags=['Evidence'], dependencies=[Depends(protect)])
def rag_answer(req:RAGRequest):
    # Avoid inventing LLM answers. This endpoint is explicitly retrieval-only until verified generator is integrated.
    result=rag_retrieve(req)
    result['answer']=None
    result['generation_status']='not_configured'
    result['message']='Retrieved passages only; no clinically validated answer generator configured.'
    return result

@app.get('/api/v1/optimizer/summary', tags=['Optimizer'], dependencies=[Depends(protect)])
def optimizer_summary():
    path=AUDIT/'portfolio_comparison.csv'
    if not path.is_file():
        raise HTTPException(503,'Phase 9 portfolio_comparison.csv missing')
    with path.open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    return {'source':str(path.relative_to(ROOT)),'rows':len(rows),
            'samples':rows[:20], 'interpretation':'Historical optimizer audit, not live clinical recommendations'}

@app.get('/api/v1/optimizer/recommendations', tags=['Optimizer'], dependencies=[Depends(protect)])
def optimizer_recommendations():
    costs=planner_module().CostRegistry().summary()
    suggestions=[]
    for name,profile in costs.items():
        median=profile.get('median_ms')
        p95=profile.get('p95_ms')
        if median is None:
            suggestions.append({'component':name,'action':'Collect more successful warm runs',
                                'reason':'No usable median measured','kind':'measurement'})
        elif p95 is not None and p95>median*3:
            suggestions.append({'component':name,'action':'Profile high-latency requests and connection/model initialization',
                                'reason':f'Median {median} ms, P95 {p95} ms','kind':'investigation'})
        if name=='vector' and median is not None and median>1000:
            suggestions.append({'component':'vector','action':'Cache the Phase 12 embedding model and measure warm-query latency',
                                'reason':f'Measured median {median} ms','kind':'investigation'})
    return {'recommendations':suggestions,'cost_profiles':costs,
            'scope':'Database and retrieval performance only; no medical recommendations'}

# Backward-compatible read-only aliases for selected existing routes.
@app.get('/api/optimizer/summary', include_in_schema=False, dependencies=[Depends(protect)])
def legacy_optimizer():
    return optimizer_summary()

@app.get('/api/system/capabilities', include_in_schema=False)
def legacy_capabilities():
    return capabilities()
