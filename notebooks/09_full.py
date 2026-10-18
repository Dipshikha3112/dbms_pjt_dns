#!/usr/bin/env python3
"""Phase 9: conservative PostgreSQL index advisor and isolated ABBA evaluator.

No production DDL. Defaults to offline mode. Live mode connects to production
READ ONLY for EXPLAIN (without ANALYZE) and catalog inspection. DDL and
EXPLAIN ANALYZE are allowed ONLY in an explicitly named isolated database.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
LOG = logging.getLogger('phase9')


def safe_json(v):
    if isinstance(v, dict): return {str(k): safe_json(x) for k,x in v.items()}
    if isinstance(v, (list,tuple,set)): return [safe_json(x) for x in v]
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v) if math.isfinite(float(v)) else None
    if isinstance(v, (np.bool_,)): return bool(v)
    if isinstance(v, (Path,)): return str(v)
    if isinstance(v, (datetime,)): return v.isoformat()
    if isinstance(v, float) and not math.isfinite(v): return None
    return v


def write_json(path, data):
    path.write_text(json.dumps(safe_json(data), indent=2, ensure_ascii=False), encoding='utf-8')


def pick(df, names):
    cols = {str(c).lower().strip(): c for c in df.columns}
    return next((cols[n] for n in names if n in cols), None)


def norm_sql(s):
    s = re.sub(r'/\*.*?\*/|--[^\n]*', ' ', str(s), flags=re.S)
    s = re.sub(r"'(?:''|[^'])*'", '?', s)
    s = re.sub(r'\b\d+(?:\.\d+)?\b', '?', s)
    return re.sub(r'\s+', ' ', s).strip().lower()


def hashid(s): return hashlib.sha256(str(s).encode()).hexdigest()[:16]


def qident(s):
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', str(s)):
        raise ValueError('Unsafe SQL identifier: ' + str(s))
    return '"' + str(s).replace('"', '""') + '"'


def read_csv_safe(p):
    try: return pd.read_csv(p, low_memory=False)
    except Exception as e:
        LOG.warning('Skipping unreadable CSV %s: %s', p, e)
        return pd.DataFrame()


def workload_rows(path):
    if not path.exists(): raise FileNotFoundError(f'Missing workload: {path}')
    df = read_csv_safe(path)
    sqlc = pick(df, ['sql','query','sql_text','query_text'])
    if not sqlc: raise ValueError(f'Workload CSV has no SQL column: {list(df.columns)}')
    labelc = pick(df, ['workload_type','role_label','query_family'])
    rows=[]
    for i,r in df.iterrows():
        sql = str(r[sqlc]) if pd.notna(r[sqlc]) else ''
        if not sql.strip(): continue
        if not re.match(r'^\s*(select|with)\b', sql, flags=re.I): continue
        if re.search(r'\b(insert|update|delete|truncate|alter|drop|create|grant|revoke|copy|call|execute)\b', norm_sql(sql)):
            continue
        rows.append({'sql':sql,'fingerprint':hashid(norm_sql(sql)),
                     'role':str(r[labelc]).lower() if labelc else 'unknown', 'position':int(i)})
    if not rows: raise ValueError('No eligible read-only SELECT queries in workload')
    return pd.DataFrame(rows)


def monitoring_priorities(path):
    if not path.exists(): return pd.DataFrame(columns=['fingerprint','observed_calls','mean_ms','p95_ms','total_ms'])
    df=read_csv_safe(path)
    sqlc=pick(df,['sql','query','sql_text','query_text'])
    latc=pick(df,['execution_time_ms','latency_ms','duration_ms','elapsed_ms'])
    if not sqlc or not latc: return pd.DataFrame(columns=['fingerprint','observed_calls','mean_ms','p95_ms','total_ms'])
    tmp=pd.DataFrame({'fingerprint':df[sqlc].fillna('').map(lambda s:hashid(norm_sql(s))),
                      'latency_ms':pd.to_numeric(df[latc],errors='coerce')})
    tmp=tmp[tmp.latency_ms.ge(0)&np.isfinite(tmp.latency_ms)]
    if tmp.empty: return pd.DataFrame(columns=['fingerprint','observed_calls','mean_ms','p95_ms','total_ms'])
    return tmp.groupby('fingerprint').latency_ms.agg(observed_calls='size',mean_ms='mean',p95_ms=lambda x:x.quantile(.95),total_ms='sum').reset_index()


def find_evidence_files(root, out):
    # Evidence is catalogued, not silently treated as comparable timing data.
    found=[]
    if not root.exists(): return found
    for p in root.rglob('*'):
        if not p.is_file() or p.suffix.lower() not in ('.csv','.json'): continue
        if out in p.parents: continue
        name=str(p.relative_to(root)).lower()
        if ('phase9' in name or 'phase_9' in name or 'index' in name or 'abba' in name) and p.stat().st_size < 100_000_000:
            found.append({'path':str(p),'bytes':p.stat().st_size,'type':p.suffix.lower()})
    return found[:500]


def existing_evidence(files):
    """Import only explicit decisions. Unknown CSV layouts are not reinterpreted."""
    entries=[]
    for item in files:
        p=Path(item['path'])
        if p.suffix!='.csv': continue
        df=read_csv_safe(p)
        if df.empty: continue
        dc=pick(df,['decision','final_decision','recommendation_decision','verdict'])
        if not dc: continue
        idc=pick(df,['candidate_id','index_id','hypothesis_id','recommendation_id','id'])
        for _,r in df.iterrows():
            d=str(r[dc]).strip().upper()
            if d in ('ACCEPT','ACCEPTED','REJECT','REJECTED','DEFER','NEEDS_MORE_EVIDENCE'):
                entries.append({'source':str(p),'candidate_id':str(r[idc]) if idc else None,'historical_decision':d})
    return entries


def sql_tables(sql):
    # Conservative parser: only unquoted simple names. Live catalog verifies them.
    pairs=re.findall(r'\b(?:FROM|JOIN)\s+((?:[A-Za-z_]\w*\.)?[A-Za-z_]\w*)(?:\s+(?:AS\s+)?([A-Za-z_]\w*))?',sql,re.I)
    stop={'where','join','left','right','inner','outer','on','group','order','having','limit','offset','cross','full','union','using'}
    out=[]
    for table,alias in pairs:
        if alias.lower() in stop: alias=''
        out.append((table.lower(),(alias or table.split('.')[-1]).lower()))
    return out


def candidate_hypotheses(workload, priorities, max_candidates):
    # Candidate extraction is deliberately conservative; no blind DDL from regex.
    byfp=workload.groupby('fingerprint').agg(sql=('sql','first'),role=('role','first'),occurrences=('sql','size')).reset_index()
    byfp=byfp.merge(priorities,on='fingerprint',how='left')
    byfp['priority']=byfp['total_ms'].fillna(0)
    byfp=byfp.sort_values(['priority','occurrences'],ascending=False)
    out={}
    for _,row in byfp.iterrows():
        sql=row['sql']; tables=sql_tables(sql)
        if not tables: continue
        for table,alias in tables:
            # Only alias.column predicates. Unqualified names are ambiguous.
            for col in re.findall(r'\b'+re.escape(alias)+r'\.([A-Za-z_]\w*)\s*(?:=|<|>|<=|>=|\bIN\s*\(|\bBETWEEN\b)',sql,re.I):
                if col.lower() in ('select','from','where'): continue
                key=(table,col.lower())
                if key not in out:
                    out[key]={'candidate_id':'idx_'+hashid('|'.join(key))[:12], 'table':table,
                              'column':col.lower(),'fingerprints':[], 'roles':set(),
                              'observed_total_ms':0.0,'query_count':0,'sample_sql':sql,
                              'source':'conservative_SQL_predicate_hypothesis'}
                x=out[key]
                if row['fingerprint'] not in x['fingerprints']:
                    x['fingerprints'].append(row['fingerprint'])
                    x['observed_total_ms']+=float(row['total_ms']) if pd.notna(row['total_ms']) else 0.0
                    x['query_count']+=int(row['occurrences'])
                x['roles'].add(row['role'])
    candidates=list(out.values())
    for x in candidates: x['roles']=sorted(x['roles'])
    candidates.sort(key=lambda x:(x['observed_total_ms'],x['query_count']),reverse=True)
    return candidates[:max_candidates], byfp


def connect_pg(db, user, host, port, password, readonly=True):
    import psycopg
    if readonly:
        return psycopg.connect(dbname=db,user=user,host=host,port=port,password=password,
            options='-c default_transaction_read_only=on -c statement_timeout=120000',autocommit=True)
    return psycopg.connect(dbname=db,user=user,host=host,port=port,password=password,autocommit=True)


def catalog(conn):
    sql="""SELECT n.nspname,t.relname,a.attname,
      COALESCE(pg_relation_size(t.oid),0) AS table_bytes,
      COALESCE(t.reltuples,0)::bigint AS estimated_rows,
      EXISTS (SELECT 1 FROM pg_index i WHERE i.indrelid=t.oid AND i.indisvalid
        AND i.indkey[0]=a.attnum) AS leading_index,
      EXISTS (SELECT 1 FROM pg_constraint k WHERE k.conrelid=t.oid AND k.contype='p'
        AND a.attnum=ANY(k.conkey)) AS primary_key_column
      FROM pg_class t JOIN pg_namespace n ON n.oid=t.relnamespace
      JOIN pg_attribute a ON a.attrelid=t.oid
      WHERE t.relkind IN ('r','p') AND n.nspname NOT IN ('pg_catalog','information_schema')
      AND n.nspname NOT LIKE 'pg_toast%' AND a.attnum>0 AND NOT a.attisdropped"""
    with conn.cursor() as cur:
        cur.execute(sql)
        cols=[d.name for d in cur.description]
        return pd.DataFrame(cur.fetchall(),columns=cols)


def resolve_candidates(candidates, cat, min_table_mb):
    eligible=[]
    for c in candidates:
        table=c['table'].split('.')[-1]
        schema=c['table'].split('.')[0] if '.' in c['table'] else None
        rows=cat[(cat.relname.str.lower()==table.lower())&(cat.attname.str.lower()==c['column'])]
        if schema: rows=rows[rows.nspname.str.lower()==schema.lower()]
        if len(rows)!=1:
            c.update({'safety_status':'DEFER','safety_reason':'missing_or_ambiguous_catalog_relation'});continue
        r=rows.iloc[0]
        c.update({'schema':str(r.nspname),'table':str(r.relname),'column':str(r.attname),
                  'table_bytes':int(r.table_bytes),'estimated_rows':int(r.estimated_rows)})
        if bool(r.leading_index) or bool(r.primary_key_column):
            c.update({'safety_status':'REJECT','safety_reason':'leading_or_primary_key_index_exists'});continue
        if r.table_bytes < min_table_mb*1024*1024:
            c.update({'safety_status':'DEFER','safety_reason':'table_below_minimum_size'});continue
        c.update({'safety_status':'ELIGIBLE','safety_reason':'catalog_verified'})
        eligible.append(c)
    return eligible


def explain_readonly(conn, sql):
    if not re.match(r'^\s*(select|with)\b',sql,re.I): raise ValueError('Only SELECT or WITH allowed')
    with conn.cursor() as cur:
        cur.execute('EXPLAIN (FORMAT JSON, COSTS TRUE) '+sql)
        data=cur.fetchone()[0]
    if isinstance(data,str): data=json.loads(data)
    plan=data[0]['Plan']
    return {'total_cost':float(plan.get('Total Cost',float('nan'))),
            'plan_rows':int(plan.get('Plan Rows',0)),
            'node_type':str(plan.get('Node Type','unknown')),
            'plan':plan}


def safe_database_identity(conn, expected):
    with conn.cursor() as cur:
        cur.execute('SELECT current_database()')
        actual=cur.fetchone()[0]
    if actual!=expected: raise RuntimeError(f'Wrong database: {actual} != {expected}')
    return actual


def isolated_abba(conn, candidate, sql, repetitions, timeout_ms):
    """Use transaction-scoped DDL; rollback every experiment, even on failure.

    Each A/B measurement runs on the same isolated database, never production.
    B creates an index inside its own transaction and rolls it back afterward.
    PostgreSQL DDL rollback guarantees cleanup. No persistent indexes remain.
    """
    qualified=qident(candidate['schema'])+'.'+qident(candidate['table'])
    index_name=qident('p9_tmp_'+hashid(candidate['candidate_id'])[:12])
    create=f'CREATE INDEX {index_name} ON {qualified} ({qident(candidate["column"])})'
    observations=[]
    for sequence,arm in enumerate(['A','B','B','A']):
        for rep in range(repetitions):
            # Manual transaction always rolled back. Never commit DDL.
            try:
                with conn.cursor() as cur:
                    cur.execute('BEGIN')
                    cur.execute("SELECT set_config('statement_timeout', %s, true)", (str(timeout_ms),))
                    cur.execute("SELECT set_config('lock_timeout', %s, true)", (str(min(timeout_ms,5000)),))
                    if arm=='B':
                        cur.execute(create)
                        cur.execute('SELECT pg_relation_size(%s::regclass)',(f'{candidate["schema"]}.p9_tmp_{hashid(candidate["candidate_id"])[:12]}',))
                        index_bytes=int(cur.fetchone()[0])
                    else:
                        index_bytes=None
                    cur.execute('EXPLAIN (ANALYZE TRUE, BUFFERS TRUE, FORMAT JSON) '+sql)
                    result=cur.fetchone()[0]
                    if isinstance(result,str): result=json.loads(result)
                    p=result[0]
                    observations.append({'candidate_id':candidate['candidate_id'],'arm':arm,
                       'sequence':sequence,'repeat':rep,'execution_ms':float(p.get('Execution Time',np.nan)),
                       'planning_ms':float(p.get('Planning Time',np.nan)),
                       'total_cost':float(p['Plan'].get('Total Cost',np.nan)),
                       'node_type':str(p['Plan'].get('Node Type','unknown')),
                       'index_bytes':index_bytes,'success':True,'error':''})
            except Exception as e:
                observations.append({'candidate_id':candidate['candidate_id'],'arm':arm,
                    'sequence':sequence,'repeat':rep,'execution_ms':np.nan,'planning_ms':np.nan,
                    'total_cost':np.nan,'node_type':'','index_bytes':None,'success':False,'error':str(e)[:400]})
            finally:
                try:
                    with conn.cursor() as cur: cur.execute('ROLLBACK')
                except Exception:
                    try: conn.rollback()
                    except Exception: pass
    return observations


def bootstrap_ci(a,b,seed=42,n=2000):
    # Percentage speedup = 100*(1 - mean(B)/mean(A)); bootstrap uncertainty.
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    if len(a)<2 or len(b)<2 or np.mean(a)<=0: return (None,None)
    rng=np.random.default_rng(seed)
    ia=rng.integers(0,len(a),(n,len(a)))
    ib=rng.integers(0,len(b),(n,len(b)))
    am=np.mean(a[ia],axis=1);bm=np.mean(b[ib],axis=1)
    samples=100*(1-bm/np.maximum(am,1e-12))
    return tuple(np.quantile(samples,[.025,.975]).tolist())


def decision_for(candidate, observations, min_samples, min_speedup, max_regression, max_index_mb):
    """Fail closed: ACCEPT is an isolated recommendation, never deployment."""
    result={'candidate_id':candidate['candidate_id'],'table':candidate['table'],
        'column':candidate['column'],'safety_status':candidate.get('safety_status','UNVERIFIED'),
        'decision':'DEFER','reason':'insufficient_evidence', 'speedup_pct':None,
        'ci95_lower_pct':None,'ci95_upper_pct':None,'regression_fraction':None,
        'a_samples':0,'b_samples':0,'estimated_index_mb':None,
        'observed_total_ms':candidate.get('observed_total_ms',0)}
    if candidate.get('safety_status')=='REJECT':
        result.update(decision='REJECT',reason=candidate.get('safety_reason','catalog_reject'));return result
    if candidate.get('safety_status')!='ELIGIBLE':
        result['reason']=candidate.get('safety_reason','catalog_not_verified');return result
    obs=[o for o in observations if o['candidate_id']==candidate['candidate_id'] and o['success']]
    a=[o['execution_ms'] for o in obs if o['arm']=='A' and math.isfinite(o['execution_ms'])]
    b=[o['execution_ms'] for o in obs if o['arm']=='B' and math.isfinite(o['execution_ms'])]
    result['a_samples']=len(a);result['b_samples']=len(b)
    if len(a)<min_samples or len(b)<min_samples:
        result['reason']='not_enough_successful_abba_samples';return result
    ma=np.mean(a);mb=np.mean(b)
    if ma<=0:
        result['reason']='nonpositive_baseline_latency';return result
    speedup=100*(1-mb/ma)
    lo,hi=bootstrap_ci(a,b)
    # Unpaired regression fraction: conservative proxy, not per-query SLA proof.
    threshold=np.median(a)*(1+max_regression/100)
    reg=float(np.mean(np.asarray(b)>threshold))
    sizes=[o.get('index_bytes') for o in obs if o['arm']=='B' and o.get('index_bytes') is not None]
    measured_size_mb=max(sizes)/1048576 if sizes else None
    result.update(speedup_pct=float(speedup),ci95_lower_pct=lo,ci95_upper_pct=hi,
                  regression_fraction=reg,estimated_index_mb=measured_size_mb)
    if speedup<0 and hi is not None and hi<0:
        result.update(decision='REJECT',reason='consistent_measured_regression')
    elif lo is not None and lo>=min_speedup and reg<=.05 and measured_size_mb is not None and measured_size_mb<=max_index_mb:
        result.update(decision='ACCEPT',reason='isolated_abba_passed_guardrails')
    else:
        result.update(decision='DEFER',reason='uncertain_benefit_or_regression_risk')
    if measured_size_mb is not None and measured_size_mb>max_index_mb:
        result.update(decision='REJECT',reason='measured_index_exceeds_size_budget')
    return result


def charts(out, decisions, obs):
    if decisions:
        counts=Counter(x['decision'] for x in decisions)
        fig,ax=plt.subplots(figsize=(7,4))
        ax.bar(list(counts),list(counts.values()))
        ax.set_title('Phase 9 safety decisions');ax.set_ylabel('Candidates')
        fig.tight_layout();fig.savefig(out/'decision_counts.png',dpi=160);plt.close(fig)
    valid=[x for x in decisions if x['speedup_pct'] is not None]
    if valid:
        fig,ax=plt.subplots(figsize=(max(7,len(valid)*.55),4))
        ax.bar([x['candidate_id'] for x in valid],[x['speedup_pct'] for x in valid])
        ax.axhline(0,linewidth=1);ax.tick_params(axis='x',rotation=70)
        ax.set_ylabel('Measured speedup (%)');ax.set_title('Isolated A/B comparisons')
        fig.tight_layout();fig.savefig(out/'speedup_by_candidate.png',dpi=160);plt.close(fig)
    if obs:
        df=pd.DataFrame(obs)
        if 'success' in df and df.success.any():
            good=df[df.success].copy()
            if not good.empty:
                fig,ax=plt.subplots(figsize=(7,4))
                data=[good[good.arm==a].execution_ms.dropna() for a in ('A','B')]
                if all(len(x)>0 for x in data):
                    ax.boxplot(data,tick_labels=['No temporary index','Temporary index'])
                    ax.set_ylabel('Execution time (ms)');ax.set_title('Isolated benchmark distribution')
                    fig.tight_layout();fig.savefig(out/'abba_distribution.png',dpi=160)
                plt.close(fig)


def report(out, summary, decisions):
    lines=['# Phase 9 — Safety-Constrained Index Optimization','',
       f"**Run status:** {summary['status']}",f"**Mode:** {summary['mode']}",'',
       '## Safety boundaries','',
       '- Production connection, when enabled, is read-only.','- Production EXPLAIN never uses ANALYZE.',
       '- Candidate SQL is validated against the PostgreSQL catalog before any benchmark.',
       '- DDL and EXPLAIN ANALYZE are restricted to the explicitly named isolated database.',
       '- Every isolated trial runs inside a transaction that is rolled back.',
       '- No index is deployed by this program.','',
       '## Decision summary','',
       f"- Candidates: {summary['candidates']}",
       f"- Decisions: {summary['decision_counts']}",
       f"- Isolated observations: {summary['observations']}",'',
       '## Research interpretation','',
       'ACCEPT means an isolated experiment passed guardrails; it is not permission for production deployment.',
       'DEFER is not a failed implementation; it means the evidence is insufficient for safe deployment.',
       'The ABBA experiment is sensitive to caching, workload order, table statistics, and index build effects.',
       'Bootstrap intervals are descriptive and may understate correlated-run uncertainty.',
       'Regex-derived single-column hypotheses are not an exhaustive index design search.',
       'EXPLAIN costs are planner estimates, not measured speedups.','',
       '## Candidate decisions','',
       '| Candidate | Relation | Decision | Reason | Speedup (%) |','|---|---|---|---|---:|']
    for d in decisions:
        speed='n/a' if d['speedup_pct'] is None else f"{d['speedup_pct']:.2f}"
        lines.append(f"| {d['candidate_id']} | {d['table']}.{d['column']} | {d['decision']} | {d['reason']} | {speed} |")
    (out/'research_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=['offline','read-only','isolated-benchmark'],default='offline')
    ap.add_argument('--workload',default=str(RESULTS/'phase6_workloads'/'combined_workload.csv'))
    ap.add_argument('--monitoring',default=str(RESULTS/'phase7_monitoring'/'phase7_query_monitoring.csv'))
    ap.add_argument('--results-root',default=str(RESULTS))
    ap.add_argument('--max-candidates',type=int,default=20)
    ap.add_argument('--min-table-mb',type=float,default=1.0)
    ap.add_argument('--min-samples',type=int,default=6)
    ap.add_argument('--repetitions',type=int,default=4)
    ap.add_argument('--min-speedup-pct',type=float,default=10.0)
    ap.add_argument('--max-regression-pct',type=float,default=10.0)
    ap.add_argument('--max-index-mb',type=float,default=256.0)
    ap.add_argument('--timeout-ms',type=int,default=20000)
    ap.add_argument('--host',default='127.0.0.1')
    ap.add_argument('--port',type=int,default=5432)
    ap.add_argument('--user',default='postgres')
    ap.add_argument('--production-db',default='adaptive_healthcare_db')
    ap.add_argument('--isolated-db',default='adaptive_healthcare_phase9_benchmark')
    ap.add_argument('--confirm-isolated',default='',help='Must equal isolated database name to enable isolated DDL')
    ap.add_argument('--password-env',default='PGPASSWORD')
    args=ap.parse_args()
    if args.max_candidates<1 or args.repetitions<1 or args.min_samples<2 or args.timeout_ms<1000:
        ap.error('Invalid candidate/sample/repetition/timeout limits')
    if args.mode=='isolated-benchmark':
        if args.confirm_isolated!=args.isolated_db:
            ap.error('Isolated mode requires --confirm-isolated EXACT_DATABASE_NAME')
        if args.isolated_db==args.production_db or not re.search(r'(benchmark|isolated|sandbox)',args.isolated_db,re.I):
            ap.error('Isolated DB must differ from production and include benchmark/isolated/sandbox in its name')
    out=Path(args.results_root)/'phase9_complete'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False)
    LOG.setLevel(logging.INFO);LOG.handlers.clear()
    fmt=logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    for h in (logging.StreamHandler(sys.stdout),logging.FileHandler(out/'phase9.log',encoding='utf-8')):
        h.setFormatter(fmt);LOG.addHandler(h)
    summary={'status':'RUNNING','mode':args.mode,'output':str(out),'production_ddl':'FORBIDDEN',
       'isolated_ddl':'TRANSACTION_ROLLBACK_ONLY','errors':[],'configuration':{k:v for k,v in vars(args).items() if k!='password-env'}}
    observations=[];decisions=[];candidates=[]
    try:
        LOG.info('Phase 9 complete pipeline; mode=%s',args.mode)
        work=workload_rows(Path(args.workload));priorities=monitoring_priorities(Path(args.monitoring))
        candidates,ranking=candidate_hypotheses(work,priorities,args.max_candidates)
        ranking.drop(columns=['sql'],errors='ignore').to_csv(out/'workload_priorities.csv',index=False)
        files=find_evidence_files(Path(args.results_root),out)
        write_json(out/'prior_phase9_evidence_catalog.json',files)
        write_json(out/'historical_decisions.json',existing_evidence(files))
        summary['workload_queries']=len(work);summary['workload_fingerprints']=work.fingerprint.nunique()
        summary['monitoring_fingerprints']=len(priorities)
        if args.mode!='offline':
            password=os.environ.get(args.password_env)
            if not password: raise RuntimeError(f'Set environment variable {args.password_env} before live mode')
            LOG.info('Inspecting production catalog and read-only plans')
            with connect_pg(args.production_db,args.user,args.host,args.port,password,readonly=True) as conn:
                safe_database_identity(conn,args.production_db)
                cat=catalog(conn);cat.to_csv(out/'production_catalog.csv',index=False)
                eligible=resolve_candidates(candidates,cat,args.min_table_mb)
                plans=[]
                for c in eligible:
                    try:
                        p=explain_readonly(conn,c['sample_sql'])
                        plans.append({'candidate_id':c['candidate_id'],'status':'EXPLAIN_OK',
                            'total_cost':p['total_cost'],'node_type':p['node_type'],'plan_rows':p['plan_rows']})
                        write_json(out/f"plan_{c['candidate_id']}.json",p['plan'])
                    except Exception as e:
                        plans.append({'candidate_id':c['candidate_id'],'status':'EXPLAIN_FAILED','error':str(e)[:400]})
                        c.update(safety_status='DEFER',safety_reason='production_explain_failed')
                pd.DataFrame(plans).to_csv(out/'readonly_explain_results.csv',index=False)
            if args.mode=='isolated-benchmark':
                LOG.info('Connecting to explicitly confirmed isolated benchmark DB')
                with connect_pg(args.isolated_db,args.user,args.host,args.port,password,readonly=False) as conn:
                    safe_database_identity(conn,args.isolated_db)
                    isolated_cat=catalog(conn)
                    # Check isolated relation and column independently; never assume matching schemas.
                    for c in eligible:
                        if c.get('safety_status')!='ELIGIBLE': continue
                        match=isolated_cat[(isolated_cat.nspname==c['schema'])&
                            (isolated_cat.relname==c['table'])&(isolated_cat.attname==c['column'])]
                        if len(match)!=1:
                            c.update(safety_status='DEFER',safety_reason='isolated_catalog_mismatch');continue
                        if bool(match.iloc[0].leading_index):
                            c.update(safety_status='REJECT',safety_reason='isolated_index_already_exists');continue
                        LOG.info('ABBA isolated: %s %s.%s',c['candidate_id'],c['table'],c['column'])
                        observations.extend(isolated_abba(conn,c,c['sample_sql'],args.repetitions,args.timeout_ms))
        else:
            for c in candidates: c.update(safety_status='UNVERIFIED',safety_reason='offline_no_catalog_validation')
        for c in candidates:
            decisions.append(decision_for(c,observations,args.min_samples,args.min_speedup_pct,
                                          args.max_regression_pct,args.max_index_mb))
        pd.DataFrame([{**c,'roles':','.join(c['roles']),'fingerprints':','.join(c['fingerprints']),
                       'sample_sql':'[stored separately in candidate_samples.json]'} for c in candidates]).to_csv(
                           out/'index_hypotheses.csv',index=False)
        write_json(out/'candidate_samples.json',[{'candidate_id':c['candidate_id'],'sql':c['sample_sql']} for c in candidates])
        pd.DataFrame(observations).to_csv(out/'isolated_abba_observations.csv',index=False)
        pd.DataFrame(decisions).to_csv(out/'safety_decisions.csv',index=False)
        write_json(out/'safety_decisions.json',decisions)
        charts(out,decisions,observations)
        summary.update(status='COMPLETED_WITH_LIMITATIONS',candidates=len(candidates),
                       observations=len(observations),decision_counts=dict(Counter(x['decision'] for x in decisions)))
    except Exception as e:
        LOG.exception('Phase 9 failed safely')
        summary.update(status='FAILED_SAFE',errors=[str(e)])
    finally:
        summary.setdefault('candidates',len(candidates));summary.setdefault('observations',len(observations))
        summary.setdefault('decision_counts',dict(Counter(x['decision'] for x in decisions)))
        write_json(out/'summary.json',summary)
        report(out,summary,decisions)
        LOG.info('FINAL: %s; %s; %s',summary['status'],summary['decision_counts'],out)
        LOG.info('Production modifications: NONE')
    return 0 if not summary['errors'] else 1

if __name__=='__main__':
    sys.exit(main())
