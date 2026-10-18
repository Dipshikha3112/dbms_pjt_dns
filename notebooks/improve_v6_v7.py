"""V7.5 — Fixes shared-session HypoPG contamination between Evaluator instances.

Two Evaluator instances (disc/val/tst) share one PostgreSQL connection, but
HypoPG state is session-scoped. V7.4 attempted incremental index management via
self._last_portfolio, which is per-instance and therefore does not reflect what
another Evaluator has done to the session. That can leak unintended hypothetical
indexes into an EXPLAIN.

V7.5 fix: _apply_portfolio() ALWAYS issues hypopg_reset() and then recreates the
requested portfolio. This is slower but correct. Search / caching / validation
logic is unchanged.

Run: python improve_v6_v7.py --seeds 5 --mode both --budget 16 --beam-width 12 \
     --discovery-catalog --composite-max-cols 3 --budget-sweep --include-fk \
     --pair-top-k 30
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import psycopg
import sqlglot
from sqlglot import exp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == 'notebooks' else Path.cwd()
DB = 'adaptive_healthcare_phase9_benchmark'
DEFAULT_WORKLOAD = ROOT / 'results/phase9_research_optimizer/20261017_185042_5b6ae098/workload_validation.csv'
DEFAULT_CANDIDATES = ROOT / 'results/phase9_m1_v6/20261018_051918_197465/v6_candidates.csv'
OUTPUT = ROOT / 'results/phase8_phase9_v7_audit'
IDENT = re.compile(r'^[a-zA-Z_][a-zA-Z_0-9]*$')
EPS = 1e-9

CODE_VERSION = '7.5.0'


# ---------- IO ----------
def save(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, pd.DataFrame):
        rows.to_csv(path, index=False)
    else:
        pd.DataFrame(list(rows)).to_csv(path, index=False)


def jdump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str, allow_nan=False), encoding='utf-8')


def h(text):
    return hashlib.sha256(str(text).encode()).hexdigest()[:20]


# ---------- SQL parsing ----------
def sql_ast(text):
    tree = sqlglot.parse_one(str(text).strip().rstrip(';'), read='postgres')
    if not isinstance(tree, (exp.Select, exp.Union, exp.Subquery, exp.Paren)):
        raise ValueError('Expected read-only SELECT')
    if any(isinstance(n, (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop, exp.Command)) for n in tree.walk()):
        raise ValueError('DML/DDL not permitted')
    return tree


def normalized(text):
    tree = sql_ast(text).copy()
    for lit in list(tree.find_all(exp.Literal)):
        lit.replace(exp.Literal.string('__parameter__'))
    return tree.sql(dialect='postgres', normalize=True)


def extract_tables(text):
    tree = sql_ast(text)
    ctes = {x.alias.lower() for x in tree.find_all(exp.CTE) if x.alias}
    return sorted({x.name.lower() for x in tree.find_all(exp.Table) if x.name and x.name.lower() not in ctes})


def extract_columns(text):
    tree = sql_ast(text)
    aliases = {}
    for t in tree.find_all(exp.Table):
        if t.name:
            aliases[t.alias_or_name.lower()] = t.name.lower()
    cols = defaultdict(set)
    unqualified = set()
    for c in tree.find_all(exp.Column):
        name = c.name.lower()
        qual = (c.table or '').lower()
        if qual and qual in aliases:
            cols[aliases[qual]].add(name)
        elif not qual:
            unqualified.add(name)
    return cols, unqualified


def load_workload(path):
    df = pd.read_csv(path, low_memory=False)
    field = next((c for c in ('sql', 'query_text', 'sql_text') if c in df), None)
    if not field:
        raise RuntimeError('No SQL column found')
    df['sql'] = df[field].astype(str).str.strip().str.rstrip(';')
    accepted = []
    for row in df.to_dict('records'):
        try:
            row['template'] = h(normalized(row['sql']))
            row['tables'] = extract_tables(row['sql'])
            cols, loose = extract_columns(row['sql'])
            row['columns_by_table'] = {k: sorted(v) for k, v in cols.items()}
            row['unqualified_columns'] = sorted(loose)
            row['query_id'] = h(row['sql'])
            row['family'] = str(row.get('query_family') or row.get('query_name') or 'unknown')
            accepted.append(row)
        except Exception:
            pass
    frame = pd.DataFrame(accepted).drop_duplicates('query_id').reset_index(drop=True)
    if len(frame) < 10:
        raise RuntimeError('Too few valid SELECT queries')
    return frame


def load_monitor(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='phase7_monitoring' AND table_name='query_log'")
        cols = {r[0] for r in cur.fetchall()}
        if not cols:
            return pd.DataFrame(), None, None
        fp = next((x for x in ('fingerprint', 'query_fingerprint') if x in cols), None)
        family = next((x for x in ('query_family', 'query_name') if x in cols), None)
        status = next((x for x in ('status',) if x in cols), None)
        if not fp:
            return pd.DataFrame(), None, None
        fields = [fp] + ([family] if family else [])
        where = " WHERE status='ok'" if status else ''
        cur.execute('SELECT ' + ', '.join('"'+x+'"' for x in fields) + ' FROM phase7_monitoring.query_log' + where)
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=fields), fp, family


def assign_frequency(workload, logs, fp_field, family_field):
    workload = workload.copy()
    if len(logs) > 0 and family_field and 'query_family' in workload:
        log_counts = logs.groupby(family_field).size().to_dict()
        families = workload[['template', 'family']].drop_duplicates()
        template_counts = families.groupby('family')['template'].nunique().to_dict()
        freq = {r.template: float(log_counts.get(r.family, 0)) / max(1, template_counts[r.family])
                for r in families.itertuples()}
        if sum(freq.values()) > 0:
            workload['template_frequency'] = workload.template.map(freq).fillna(0).astype(float)
            if workload.template_frequency.sum() <= 0:
                workload['template_frequency'] = 1.0
                return workload, 'uniform_fallback'
            return workload, 'monitoring_family_counts'
    workload['template_frequency'] = 1.0
    return workload, 'instance_count'


def partition_weights(frame):
    counts = frame.groupby('template')['query_id'].transform('count').astype(float)
    total = counts.sum()
    if total <= 0:
        return np.ones(len(frame), dtype=float) / max(1, len(frame))
    return (counts / total).to_numpy()


def split(df, seed, mode):
    rng = np.random.default_rng(seed)
    if mode == 'within_template':
        parts = [[], [], []]
        for _, group in df.groupby('template', sort=True):
            ix = list(rng.permutation(group.index.to_numpy()))
            n = len(ix)
            if n >= 3:
                for k, idx in enumerate(ix):
                    parts[k % 3].append(int(idx)) if k < 3 else parts[0 if k % 5 < 3 else (1 if k % 5 == 3 else 2)].append(int(idx))
            else:
                for k, idx in enumerate(ix):
                    parts[k % 3].append(int(idx))
        if any(len(x) == 0 for x in parts):
            raise RuntimeError('Within-template partition empty')
        return [df.loc[sorted(p)].copy().reset_index(drop=True) for p in parts]
    groups = list(sorted(df.template.unique()))
    rng.shuffle(groups)
    family_by_template = df.groupby('template')['family'].first().to_dict()
    by_family = defaultdict(list)
    for g in groups:
        by_family[family_by_template[g]].append(g)
    bins = [set(), set(), set()]
    for fam, vals in sorted(by_family.items()):
        for j, template in enumerate(vals):
            bins[j % 3].add(template)
    if any(not b for b in bins):
        bins = [set(groups[i::3]) for i in range(3)]
    return [df[df.template.isin(b)].copy().reset_index(drop=True) for b in bins]


def db_schema(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT table_name,column_name,data_type FROM information_schema.columns WHERE table_schema='public'")
        columns = defaultdict(dict)
        for table, col, typ in cur.fetchall():
            columns[table][col] = typ
        cur.execute("""SELECT tc.table_name,kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name=kcu.constraint_name AND tc.table_schema=kcu.table_schema
            WHERE tc.table_schema='public' AND tc.constraint_type='FOREIGN KEY'""")
        fk = {(a, b) for a, b in cur.fetchall()}
    return columns, fk, []


def parse_candidate(value):
    value = str(value).strip()
    m = re.fullmatch(r'([A-Za-z_][A-Za-z_0-9]*)\s*\(([^()]*)\)', value)
    if not m:
        raise ValueError(f'Cannot parse: {value}')
    return m.group(1), [c.strip() for c in m.group(2).split(',')]


def index_def(table, columns):
    if not IDENT.fullmatch(table):
        raise ValueError(table)
    tokens = []
    for col in columns:
        m = re.fullmatch(r'([A-Za-z_][A-Za-z_0-9]*)(?:\s+(ASC|DESC))?', col, re.I)
        if not m:
            raise ValueError(col)
        tokens.append('"'+m.group(1)+'"' + (' '+m.group(2).upper() if m.group(2) else ''))
    return f'CREATE INDEX ON public."{table}" ({", ".join(tokens)})'


def load_candidates(path, schema, fks, discovery, include_fk):
    frame = pd.read_csv(path)
    if 'candidate' not in frame and 'create_sql' not in frame:
        raise RuntimeError('Expected V6 candidate or create_sql column')
    base = []
    for record in frame.to_dict('records'):
        if 'candidate' in record and pd.notna(record['candidate']):
            table, cols = parse_candidate(record['candidate'])
            definition = index_def(table, cols)
        else:
            definition = str(record['create_sql'])
            m = re.fullmatch(r'CREATE INDEX ON public\."?([A-Za-z_][A-Za-z_0-9]*)"?\s*\((.*)\)', definition, re.I)
            if not m:
                continue
            table, cols = m.group(1), [c.strip().replace('"','') for c in m.group(2).split(',')]
        if table not in schema or any(c.split()[0].replace('"','') not in schema[table] for c in cols):
            continue
        base.append({'definition': definition, 'table': table,
                     'columns': tuple(c.split()[0].replace('"','') for c in cols), 'group': 'v6_original'})
    extra = []
    if include_fk:
        keys = set(fks)
        for text in discovery.sql:
            tree = sql_ast(text)
            aliases = {t.alias_or_name.lower():t.name.lower() for t in tree.find_all(exp.Table) if t.name}
            for eq in tree.find_all(exp.EQ):
                if isinstance(eq.left, exp.Column) and isinstance(eq.right, exp.Column):
                    for c in (eq.left, eq.right):
                        t = aliases.get((c.table or '').lower())
                        if t and c.name in schema.get(t, {}):
                            keys.add((t, c.name))
        for table, col in sorted(keys):
            if table in schema and col in schema[table]:
                extra.append({'definition':index_def(table, [col]), 'table':table,
                              'columns':(col,), 'group':'fk_join_schema_discovery'})
    unique = {}
    for candidate in base + extra:
        unique.setdefault(candidate['definition'], candidate)
    return list(unique.values())


def build_catalog_from_discovery(discovery_df, schema, max_cols_per_composite=3):
    def table_of_column(col, alias_map):
        qual = (col.table or '').lower()
        return alias_map.get(qual, qual) if qual else None

    generated = {}
    for text in discovery_df.sql:
        try:
            tree = sql_ast(text)
        except Exception:
            continue
        alias_map = {}
        for t in tree.find_all(exp.Table):
            if t.name:
                alias_map[t.alias_or_name.lower()] = t.name.lower()
                alias_map.setdefault(t.name.lower(), t.name.lower())

        per_table = defaultdict(set)
        for node in list(tree.find_all(exp.Where)) + list(tree.find_all(exp.Join)) + list(tree.find_all(exp.Group)) + list(tree.find_all(exp.Order)):
            for col in node.find_all(exp.Column):
                t = table_of_column(col, alias_map)
                if t and t in schema and col.name in schema[t]:
                    per_table[t].add(col.name)

        for t, cols in per_table.items():
            cols = sorted(cols)
            for c in cols:
                definition = index_def(t, [c])
                generated.setdefault(definition, {'definition': definition, 'table': t, 'columns': (c,), 'group': 'discovery_generated'})
            for a, b in itertools.combinations(cols, 2):
                definition = index_def(t, [a, b])
                generated.setdefault(definition, {'definition': definition, 'table': t, 'columns': (a, b), 'group': 'discovery_generated_pair'})
            if max_cols_per_composite >= 3:
                for triple in itertools.combinations(cols, 3):
                    definition = index_def(t, list(triple))
                    generated.setdefault(definition, {'definition': definition, 'table': t, 'columns': tuple(triple), 'group': 'discovery_generated_triple'})
    return list(generated.values())


def merge_catalogs(base, extra):
    seen = set()
    out = []
    for c in base + extra:
        if c['definition'] in seen:
            continue
        seen.add(c['definition'])
        out.append(c)
    return out


def connection(args):
    if args.db != DB or args.host not in ('127.0.0.1','localhost') or args.port != 5433:
        raise RuntimeError('Only local benchmark DB on port 5433 allowed')
    pw = os.environ.get('PGPASSWORD')
    if not pw:
        import getpass
        pw = getpass.getpass('Benchmark PostgreSQL password: ')
    conn = psycopg.connect(host=args.host, port=args.port, dbname=args.db, user=args.user,
                           password=pw, autocommit=True, connect_timeout=15)
    with conn.cursor() as cur:
        cur.execute('SELECT current_database()')
        if cur.fetchone()[0] != DB:
            raise RuntimeError('Wrong database')
        cur.execute("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname='hypopg')")
        if not cur.fetchone()[0]:
            raise RuntimeError('HypoPG extension not installed')
        cur.execute("SELECT set_config('statement_timeout',%s,false)", (str(args.timeout_ms),))
        cur.execute("SELECT set_config('default_transaction_read_only','on',false)")
    return conn, None


# ---------- Evaluator (FIXED V7.5: always-reset HypoPG) ----------
class Evaluator:
    """Planner-cost evaluator over a fixed partition.

    V7.5 correctness fix:
      Multiple Evaluator instances typically share one connection (one session).
      HypoPG state is session-scoped. Previously we tried to track which indexes
      were already installed per-instance and skip re-creation on the assumption
      that the session still held exactly that set. That assumption is false when
      another Evaluator has run EXPLAINs in between.

      Now _apply_portfolio() ALWAYS resets HypoPG and recreates the requested
      portfolio. We still keep self._last_portfolio for observability, but it is
      no longer used to decide whether a reset is safe.

    This is slower but guaranteed correct. Search logic and caching are unchanged.
    """

    def __init__(self, conn, frame, candidates, timeout_ms, label='eval'):
        self.conn = conn
        self.frame = frame.reset_index(drop=True)
        self.candidates = candidates
        self.weights = partition_weights(self.frame)
        self.cache = {}
        self.timeout_ms = timeout_ms
        self.evaluations = 0
        self.label = label
        self._batch_sql = '; '.join('EXPLAIN (FORMAT JSON) ' + q for q in self.frame.sql)
        self._n_expected = len(self.frame.sql)
        self._last_portfolio = ()          # observability only (V7.5)
        self._reset_count = 0              # observability only
        self.baseline = self.costs(())
        self._baseline_dot = float(np.dot(self.weights, self.baseline))

    def _apply_portfolio(self, cur, portfolio):
        """V7.5: unconditional reset + recreate. Correct across shared sessions."""
        cur.execute('SELECT hypopg_reset()')
        self._reset_count += 1
        for i in sorted(set(portfolio)):
            cur.execute('SELECT * FROM hypopg_create_index(%s)', (self.candidates[i]['definition'],))
        self._last_portfolio = tuple(sorted(set(portfolio)))

    def costs(self, portfolio):
        key = tuple(sorted(portfolio))
        if key in self.cache:
            return self.cache[key]
        with self.conn.cursor() as cur:
            self._apply_portfolio(cur, key)
            cur.execute(self._batch_sql)
            costs = []
            while True:
                if cur.description is not None:
                    row = cur.fetchone()
                    if row:
                        doc = row[0]
                        if isinstance(doc, str):
                            doc = json.loads(doc)
                        costs.append(float(doc[0]['Plan']['Total Cost']))
                try:
                    if not cur.nextset():
                        break
                except Exception:
                    break
            if len(costs) != self._n_expected:
                # Fallback: re-apply and EXPLAIN one-by-one (still correct).
                costs = []
                self._apply_portfolio(cur, key)
                for q in self.frame.sql:
                    cur.execute('EXPLAIN (FORMAT JSON) ' + q)
                    doc = cur.fetchone()[0]
                    if isinstance(doc, str):
                        doc = json.loads(doc)
                    costs.append(float(doc[0]['Plan']['Total Cost']))
        result = np.asarray(costs, dtype=float)
        self.cache[key] = result
        self.evaluations += 1
        return result

    def gain(self, portfolio):
        new = self.costs(portfolio)
        return 100.0 * (1.0 - float(np.dot(self.weights, new)) / self._baseline_dot) if self._baseline_dot > EPS else 0.0

    def unweighted(self, portfolio):
        new = self.costs(portfolio)
        s = float(self.baseline.sum())
        return 100.0 * (1.0 - float(new.sum()) / s) if s > EPS else 0.0


# ---------- Search ----------
def coverage_rows(evaluator, candidates, mode, seed, group):
    frame = evaluator.frame
    rows = []
    all_cost = evaluator._baseline_dot
    for family, subset in frame.groupby('family'):
        ix = subset.index.to_numpy()
        tables = sorted({t for arr in subset.tables for t in arr})
        qualified = defaultdict(set)
        loose = set()
        for row in subset.itertuples():
            for t, cs in row.columns_by_table.items():
                qualified[t].update(cs)
            loose.update(row.unqualified_columns)
        touching = [i for i, c in enumerate(candidates) if c['table'] in tables]
        matching = [i for i in touching if any(col in qualified.get(candidates[i]['table'], set()) or col in loose for col in candidates[i]['columns'])]
        cost = float(np.dot(evaluator.weights[ix], evaluator.baseline[ix]))
        rows.append({'mode':mode,'seed':seed,'group':group,'family':family,'queries':len(subset),
                     'templates':subset.template.nunique(),'tables_touched':'|'.join(tables),
                     'candidate_table_matches':len(touching),'candidate_column_matches':len(matching),
                     'candidate_ids_table':'|'.join(map(str,touching)),
                     'candidate_ids_column':'|'.join(map(str,matching)),
                     'baseline_cost_share_pct':100*cost/all_cost if all_cost>EPS else 0,
                     'baseline_weighted_cost':cost})
    return rows


def singleton_pairs(evaluator, n, top_k=30):
    singles = {i: evaluator.gain((i,)) for i in range(n)}
    top = sorted(singles.items(), key=lambda kv: -kv[1])[:top_k]
    top_idx = [i for i, _ in top]
    pairs = []
    for i, j in itertools.combinations(top_idx, 2):
        g = evaluator.gain((i, j))
        pairs.append({'i': i, 'j': j, 'gain': g, 'single_i': singles[i], 'single_j': singles[j],
                      'synergy_pp': g - singles[i] - singles[j]})
    return singles, pairs, top_idx


def greedy(evaluator, n, budget, restrict=None):
    pool = list(restrict) if restrict is not None else list(range(n))
    selected = ()
    prev = evaluator.gain(selected)
    while len(selected) < budget:
        options = [(evaluator.gain(tuple(sorted(selected + (i,)))), i) for i in pool if i not in selected]
        if not options:
            break
        value, i = max(options, key=lambda x: (x[0], -x[1]))
        if value <= prev + EPS:
            break
        selected = tuple(sorted(selected + (i,)))
        prev = value
    return selected


def beam(evaluator, n, budget, width, pairs, restrict=None, patience=2, min_delta=0.01):
    pool = list(restrict) if restrict is not None else list(range(n))
    if not pool or budget <= 0:
        return ()
    current = [()]
    best_portfolio = ()
    best_gain = evaluator.gain(())
    stagnant = 0
    max_depth = min(len(pool), budget)
    for depth in range(1, max_depth + 1):
        possible = {tuple(sorted(p + (i,))) for p in current for i in pool if i not in p}
        if depth == 2 and pairs:
            for pr in pairs:
                if pr['i'] in pool and pr['j'] in pool:
                    possible.add(tuple(sorted((pr['i'], pr['j']))))
        if not possible:
            break
        scored = sorted(((evaluator.gain(p), p) for p in possible), key=lambda x: (-x[0], x[1]))
        current = [p for _, p in scored[:width]]
        dg, dp = scored[0]
        imp = dg - best_gain
        if imp > 1e-9:
            best_gain, best_portfolio = dg, dp
        stagnant = stagnant + 1 if imp < min_delta else 0
        print(f'    beam depth {depth}/{max_depth}: best={dg:.3f}% Δ={imp:+.4f}pp (eval {len(possible)}, cache={len(evaluator.cache)})', flush=True)
        if patience is not None and stagnant >= patience:
            print(f'    Early stop at depth {depth}', flush=True)
            break
    return best_portfolio


def swap(evaluator, n, initial, restrict=None, max_rounds=2):
    current = tuple(sorted(set(initial)))
    current_gain = evaluator.gain(current)
    pool = sorted(set(restrict)) if restrict is not None else list(range(n))
    for r in range(max_rounds):
        best_p, best_g, evaluated = current, current_gain, 0
        for i in current:
            rem = set(current) - {i}
            for j in pool:
                if j in current:
                    continue
                cand = tuple(sorted(rem | {j}))
                g = evaluator.gain(cand)
                evaluated += 1
                if g > best_g + 1e-9:
                    best_g, best_p = g, cand
        imp = best_g - current_gain
        print(f'    swap round {r+1}/{max_rounds}: gain={best_g:.4f}% Δ={imp:+.4f}pp eval={evaluated}', flush=True)
        if best_p == current:
            break
        current, current_gain = best_p, best_g
    return current


def choose_validation(methods, evaluator, candidates):
    scored = []
    for name, p in methods.items():
        g = evaluator.gain(p)
        sizes = [candidates[i].get('size_bytes') for i in p]
        size = sum(sizes) if all(x is not None for x in sizes) else None
        scored.append({'method': name, 'portfolio': p, 'gain': g, 'indexes': len(p), 'size_bytes': size})
    mx = max(r['gain'] for r in scored)
    tied = [r['method'] for r in scored if abs(r['gain'] - mx) <= 1e-9]
    ranked = sorted(scored, key=lambda r: (-round(r['gain'], 9), r['indexes'],
                                            r['size_bytes'] if r['size_bytes'] is not None else float('inf'), r['method']))
    return ranked[0], sorted(tied), scored


# ---------- One (mode, seed) run ----------
def run_one(conn, df, base, extra, args, out, mode, seed, all_coverage, all_results, all_synergy, all_splits, all_budget_sweep):
    parts = split(df, seed, mode)
    names = ('discovery', 'validation', 'test')
    for label, part in zip(names, parts):
        for row in part.itertuples():
            all_splits.append({'mode': mode, 'seed': seed, 'partition': label, 'query_id': row.query_id,
                               'template': row.template, 'family': row.family})
    print(f'\n[{mode} seed={seed}] ' + ', '.join(f'{n}={len(p)}' for n, p in zip(names, parts)), flush=True)

    for use_fk in ([False, True] if args.fk_ablation else [args.include_fk]):
        group = 'v6_plus_fk_join' if use_fk else 'v6_only'
        candidates = load_candidates(args.candidates, base, extra, parts[0], use_fk)
        if args.discovery_catalog:
            gen = build_catalog_from_discovery(parts[0], base, max_cols_per_composite=args.composite_max_cols)
            before = len(candidates)
            candidates = merge_catalogs(candidates, gen)
            print(f'  Discovery catalog: {before} V6/FK -> {len(candidates)} ({len(gen)} generated)', flush=True)
        if args.candidate_cap and len(candidates) > args.candidate_cap:
            candidates = candidates[:args.candidate_cap]

        with conn.cursor() as cur:
            cur.execute('SELECT hypopg_reset()')
            for c in candidates:
                cur.execute('SELECT * FROM hypopg_create_index(%s)', (c['definition'],))
                c['oid'] = int(cur.fetchone()[0])
                c['size_bytes'] = None
            cur.execute('SELECT hypopg_reset()')

        save(out / f'candidate_catalog_{mode}_{seed}_{group}.csv', [{'id': i, **c} for i, c in enumerate(candidates)])

        # V7.5: label evaluators for observability. All share `conn`; the fix
        # ensures each costs() call resets + recreates so contamination is gone.
        disc = Evaluator(conn, parts[0], candidates, args.timeout_ms, label='discovery')
        val = Evaluator(conn, parts[1], candidates, args.timeout_ms, label='validation')
        tst = Evaluator(conn, parts[2], candidates, args.timeout_ms, label='test')
        for part_name, ev in zip(names, (disc, val, tst)):
            all_coverage.extend(coverage_rows(ev, candidates, mode, seed, group + '_' + part_name))

        cov_df = pd.DataFrame([r for r in all_coverage
                               if r['mode'] == mode and r['seed'] == seed and r['group'].startswith(group + '_')])
        if not cov_df.empty:
            ceilings = {}
            for part_name in ('discovery', 'validation', 'test'):
                sub = cov_df[cov_df['group'] == group + '_' + part_name]
                if sub.empty: continue
                unc = float(sub.loc[sub['candidate_column_matches'] == 0, 'baseline_cost_share_pct'].sum())
                ceilings[part_name] = (unc, 100 - unc)
            for pn, (unc, ceil) in ceilings.items():
                print(f'  [{pn}] uncovered: {unc:.2f}%  => ceiling ~{ceil:.2f}%', flush=True)
            bind = 'test' if mode == 'unseen_template' else 'discovery'
            if bind in ceilings:
                print(f'  === Binding ceiling ({bind}): {ceilings[bind][1]:.2f}% ===', flush=True)

        print(f'  Candidates: {len(candidates)} | discovery baseline: {np.dot(disc.weights, disc.baseline):.2f}', flush=True)
        singles, pairs, top_idx = singleton_pairs(disc, len(candidates), top_k=args.pair_top_k)
        for pr in pairs:
            all_synergy.append({'mode': mode, 'seed': seed, 'group': group, **pr})
        print(f'  Singletons: {len(singles)} | Pairs: {len(pairs)} | synergy>0.5pp: {sum(p["synergy_pp"]>0.5 for p in pairs)}', flush=True)

        beam_pool = top_idx[:50] if len(top_idx) > 50 else (top_idx if top_idx else None)
        methods = {'no_new_indexes': (), 'greedy': greedy(disc, len(candidates), args.budget, restrict=beam_pool)}
        methods['beam'] = beam(disc, len(candidates), args.budget, args.beam_width, pairs, restrict=beam_pool)
        methods['beam_swap'] = swap(disc, len(candidates), methods['beam'], restrict=beam_pool, max_rounds=2)

        if args.budget_sweep:
            sweep = sorted({b for b in [1, 2, 3, 5, 8, 12, 16] if b <= args.budget} | {args.budget})
            for b in sweep:
                pb = beam(disc, len(candidates), b, args.beam_width, pairs, restrict=beam_pool)
                ps = swap(disc, len(candidates), pb, restrict=beam_pool, max_rounds=2)
                all_budget_sweep.append({
                    'mode': mode, 'seed': seed, 'group': group, 'budget': b,
                    'beam_test_gain': tst.gain(pb),
                    'beam_swap_test_gain': tst.gain(ps),
                    'beam_valid_gain': val.gain(pb),
                    'beam_swap_valid_gain': val.gain(ps),
                    'portfolio': '|'.join(map(str, pb)),
                })
            print(f'  Budget sweep: {sweep}', flush=True)

        winner, ties, _ = choose_validation({k: v for k, v in methods.items() if k != 'no_new_indexes'}, val, candidates)
        print(f'  Validation winner: {winner["method"]} | ties: {ties}', flush=True)
        for name, p in methods.items():
            all_results.append({'mode': mode, 'seed': seed, 'group': group, 'method': name, 'selected': name == winner['method'],
                                'validation_tied': '|'.join(ties), 'portfolio': '|'.join(map(str, p)), 'indexes': len(p),
                                'discovery_weighted_gain_pct': disc.gain(p), 'validation_weighted_gain_pct': val.gain(p),
                                'test_weighted_gain_pct': tst.gain(p), 'test_unweighted_gain_pct': tst.unweighted(p),
                                'candidate_count': len(candidates),
                                'disc_resets': disc._reset_count,
                                'val_resets': val._reset_count,
                                'tst_resets': tst._reset_count})
        print(f'  Selected held-out gain: {tst.gain(winner["portfolio"]):.6f}%', flush=True)

    save(out / f'coverage_audit_{mode}_{seed}.csv', all_coverage)
    save(out / f'portfolio_comparison_{mode}_{seed}.csv', all_results)
    save(out / f'pair_synergy_{mode}_{seed}.csv', all_synergy)
    save(out / f'split_manifest_{mode}_{seed}.csv', all_splits)
    if all_budget_sweep:
        save(out / f'budget_sweep_{mode}_{seed}.csv', all_budget_sweep)


def run_one_worker(args_dict):
    args = SimpleNamespace(**args_dict)
    conn, _ = connection(args)
    try:
        out = Path(args_dict['out_path'])
        frame = load_workload(args.workload)
        logs, fp, family = load_monitor(conn)
        frame, _ = assign_frequency(frame, logs, fp, family)
        schema, fks, _ = db_schema(conn)
        lc, lr, ls, lm, lb = [], [], [], [], []
        run_one(conn, frame, schema, fks, args, out, args_dict['mode'], args_dict['seed'], lc, lr, ls, lm, lb)
        return f"{args_dict['mode']}_{args_dict['seed']}", 'ok', len(lr)
    except Exception as e:
        import traceback
        print(f'WORKER FAILED {args_dict["mode"]}/{args_dict["seed"]}:\n{traceback.format_exc()}', flush=True)
        return f"{args_dict['mode']}_{args_dict['seed']}", f'error: {e}', 0
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--db', default=DB)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=5433)
    ap.add_argument('--user', default='postgres')
    ap.add_argument('--workload', type=Path, default=DEFAULT_WORKLOAD)
    ap.add_argument('--candidates', type=Path, default=DEFAULT_CANDIDATES)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--start-seed', type=int, default=43)
    ap.add_argument('--mode', choices=('both', 'within_template', 'unseen_template'), default='both')
    ap.add_argument('--budget', type=int, default=16)
    ap.add_argument('--beam-width', type=int, default=12)
    ap.add_argument('--include-fk', action='store_true')
    ap.add_argument('--fk-ablation', action='store_true')
    ap.add_argument('--discovery-catalog', action='store_true')
    ap.add_argument('--composite-max-cols', type=int, default=3)
    ap.add_argument('--budget-sweep', action='store_true')
    ap.add_argument('--pair-top-k', type=int, default=30)
    ap.add_argument('--candidate-cap', type=int, default=0)
    ap.add_argument('--strict-scale', action='store_true')
    ap.add_argument('--timeout-ms', type=int, default=30000)
    ap.add_argument('--audit-only', action='store_true')
    ap.add_argument('--workers', type=int, default=0)
    args = ap.parse_args()
    if args.budget < 1 or args.seeds < 1:
        ap.error('budget and seeds must be >=1')
    if not args.workload.exists() or not args.candidates.exists():
        ap.error('Workload or V6 candidate CSV missing')

    out = OUTPUT / datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    print(f'Results: {out}  (code_version={CODE_VERSION})', flush=True)

    if args.audit_only:
        frame = load_workload(args.workload)
        conn, _ = connection(args)
        try:
            logs, fp, family = load_monitor(conn)
            frame, prov = assign_frequency(frame, logs, fp, family)
            schema, fks, _ = db_schema(conn)
            modes = ('within_template', 'unseen_template') if args.mode == 'both' else (args.mode,)
            cov = []
            for mode in modes:
                for seed in range(args.start_seed, args.start_seed + args.seeds):
                    parts = split(frame, seed, mode)
                    cands = load_candidates(args.candidates, schema, fks, parts[0], args.include_fk)
                    if args.discovery_catalog:
                        cands = merge_catalogs(cands, build_catalog_from_discovery(parts[0], schema, args.composite_max_cols))
                    for label, part in zip(('discovery', 'validation', 'test'), parts):
                        ev = Evaluator(conn, part, cands, args.timeout_ms, label=label)
                        cov.extend(coverage_rows(ev, cands, mode, seed, label))
        finally:
            conn.close()
        save(out / 'coverage_audit.csv', cov)
        jdump(out / 'metadata.json', {'args': vars(args), 'freq': prov, 'audit_only': True,
                                      'code_version': CODE_VERSION})
        print('Done:', out, flush=True)
        return

    from multiprocessing import Pool
    modes = ('within_template', 'unseen_template') if args.mode == 'both' else (args.mode,)
    jobs = []
    for mode in modes:
        for seed in range(args.start_seed, args.start_seed + args.seeds):
            d = vars(args).copy()
            d['mode'] = mode
            d['seed'] = seed
            d['out_path'] = str(out)
            jobs.append(d)

    workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 2) - 1)
    workers = min(workers, len(jobs))
    print(f'Running {len(jobs)} jobs with {workers} workers (CPU: {os.cpu_count()})', flush=True)

    with Pool(processes=workers) as pool:
        results = pool.map(run_one_worker, jobs)

    for tag, status, n in results:
        print(f'  {tag}: {status} ({n} results)', flush=True)

    for prefix, name in [('coverage_audit','coverage_audit'),('portfolio_comparison','portfolio_comparison'),
                          ('pair_synergy','pair_synergy'),('split_manifest','split_manifest'),('budget_sweep','budget_sweep')]:
        frames = []
        for tag, status, _ in results:
            if not status.startswith('ok'): continue
            p = out / f'{prefix}_{tag}.csv'
            if p.exists(): frames.append(pd.read_csv(p))
        if frames:
            save(out / f'{name}.csv', pd.concat(frames, ignore_index=True))

    if (out / 'portfolio_comparison.csv').exists():
        df_all = pd.read_csv(out / 'portfolio_comparison.csv')
        summ = df_all.groupby(['mode','group','method'], as_index=False).agg(
            seeds=('seed','nunique'),
            mean_test_gain=('test_weighted_gain_pct','mean'),
            std_test_gain=('test_weighted_gain_pct','std'),
            nonzero_test_seeds=('test_weighted_gain_pct', lambda x: int((abs(x)>EPS).sum())))
        summ['std_test_gain'] = summ.std_test_gain.fillna(0)
        save(out / 'method_summary.csv', summ)
        print('\nRESULT SUMMARY\n', summ.to_string(index=False), flush=True)

    if (out / 'budget_sweep.csv').exists():
        bs = pd.read_csv(out / 'budget_sweep.csv')
        if not bs.empty:
            bsum = bs.groupby(['mode','group','budget'], as_index=False).agg(
                seeds=('seed','nunique'),
                beam_test_mean=('beam_test_gain','mean'),
                beam_test_std=('beam_test_gain','std'),
                beam_swap_test_mean=('beam_swap_test_gain','mean'),
                beam_swap_test_std=('beam_swap_test_gain','std'),
                beam_valid_mean=('beam_valid_gain','mean'),
                beam_swap_valid_mean=('beam_swap_valid_gain','mean'))
            save(out / 'budget_sweep_summary.csv', bsum)
            for mode in bsum['mode'].unique():
                sub = bsum[bsum['mode'] == mode]
                plt.figure(figsize=(10,5))
                for grp in sub['group'].unique():
                    g = sub[sub['group']==grp].sort_values('budget')
                    plt.errorbar(g['budget'], g['beam_swap_test_mean'], yerr=g['beam_swap_test_std'].fillna(0),
                                 marker='o', capsize=4, label=f'{grp} (beam_swap)')
                plt.xlabel('Index budget'); plt.ylabel('Held-out weighted gain (%)')
                plt.title(f'Gain vs Budget — {mode}'); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
                plt.savefig(out / f'budget_sweep_{mode}.png', dpi=170); plt.close()

    jdump(out / 'metadata.json', {
        'code_version': CODE_VERSION,
        'args': vars(args),
        'physical_indexes_created': False,
        'planner_cost_not_latency': True,
        'weighting': 'instance_count',
        'hypopg_reset_policy': 'always (V7.5 correctness fix)',
        'limitations': [
            'HypoPG cost is not runtime latency.',
            'At 1,163 patients, planner preferences may not transfer to production scale.',
            'Unseen-template realized gain is bounded by query-shape overlap in discovery.',
        ]})
    print('Done:', out, flush=True)


if __name__ == '__main__':
    main()