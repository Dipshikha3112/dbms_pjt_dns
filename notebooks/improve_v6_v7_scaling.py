"""improve_v6_v7_scaling.py — Publication-ready experiment suite.

Features:
- Synthetic scale-up to 100K+ rows in schema bench_100k (UUID-safe, timeout-safe)
- Discovery-catalog + V6-only + HypoPG-advisor/naive-baseline comparison
- 5 seeds × 2 modes × 2 scales, 8-core parallel
- Statistical tests (paired t-test, Wilcoxon, bootstrap CI, Cohen's d)
- 3 figures + full manifest

Usage:
    python improve_v6_v7_scaling.py --seeds 5 --workers 8
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import psycopg
import sqlglot
from sqlglot import exp
from scipy import stats as scipy_stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == 'notebooks' else Path.cwd()
DB = 'adaptive_healthcare_phase9_benchmark'
SCHEMA_100K = 'bench_100k'
SCHEMA_SRC = 'public'
DEFAULT_WORKLOAD = ROOT / 'results/phase9_research_optimizer/20261017_185042_5b6ae098/workload_validation.csv'
DEFAULT_CANDIDATES = ROOT / 'results/phase9_m1_v6/20261018_051918_197465/v6_candidates.csv'
OUTPUT = ROOT / 'results/phase8_phase9_v7_final'
IDENT = re.compile(r'^[a-zA-Z_][a-zA-Z_0-9]*$')
EPS = 1e-9


# =============================================================================
# Helpers
# =============================================================================
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
    unq = set()
    for c in tree.find_all(exp.Column):
        if (c.table or '').lower() in aliases:
            cols[aliases[c.table.lower()]].add(c.name.lower())
        elif not c.table:
            unq.add(c.name.lower())
    return cols, unq


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
        raise RuntimeError('Too few valid queries')
    return frame


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


def parse_candidate(value):
    m = re.fullmatch(r'([A-Za-z_][A-Za-z_0-9]*)\s*\(([^()]*)\)', str(value).strip())
    if not m:
        raise ValueError(f'Cannot parse: {value}')
    return m.group(1), [c.strip() for c in m.group(2).split(',')]


def index_def(table, columns, schema='public'):
    tokens = []
    for col in columns:
        m = re.fullmatch(r'([A-Za-z_][A-Za-z_0-9]*)(?:\s+(ASC|DESC))?', col, re.I)
        if not m:
            raise ValueError(col)
        tokens.append('"'+m.group(1)+'"' + (' '+m.group(2).upper() if m.group(2) else ''))
    return f'CREATE INDEX ON {schema}."{table}" ({", ".join(tokens)})'


def connection(args):
    if args.db != DB or args.host not in ('127.0.0.1', 'localhost') or args.port != 5433:
        raise RuntimeError('Only local benchmark DB on port 5433 allowed')
    pw = os.environ.get('PGPASSWORD')
    if not pw:
        import getpass
        pw = getpass.getpass('Benchmark PostgreSQL password: ')
    conn = psycopg.connect(host=args.host, port=args.port, dbname=args.db, user=args.user,
                           password=pw, autocommit=True, connect_timeout=15)
    with conn.cursor() as cur:
        cur.execute("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname='hypopg')")
        if not cur.fetchone()[0]:
            raise RuntimeError('HypoPG extension not installed')
        # NOTE: statement_timeout is NOT set globally anymore.
        # It is enforced per-query inside Evaluator, and disabled for scale-up.
    return conn, None


# =============================================================================
# STEP 1: Scale-up (session-level SET, dedicated connection)
# =============================================================================
def build_scale_schema(conn, scale_factor=100, target=100_000):
    """Create bench_100k schema. Caller MUST pass a connection with
    `SET statement_timeout = 0` already applied (session-scoped).
    """
    print(f'\n=== Step 1: Building {SCHEMA_100K} schema ({scale_factor}x) ===', flush=True)

    # Verify the session timeout is 0. If not, set it now.
    with conn.cursor() as cur:
        cur.execute("SET statement_timeout = 0")
        cur.execute("SHOW statement_timeout")
        st = cur.fetchone()[0]
        print(f'  Statement timeout: {st} (expect 0 or 0ms)', flush=True)
        if st not in ('0', '0ms'):
            raise RuntimeError(f'Cannot disable statement_timeout; current value: {st}')

    # Discover PK columns and types
    with conn.cursor() as cur:
        cur.execute("""SELECT tc.table_name, kcu.column_name, c.data_type, c.udt_name
                       FROM information_schema.table_constraints tc
                       JOIN information_schema.key_column_usage kcu
                         ON tc.constraint_name = kcu.constraint_name
                        AND tc.table_schema = kcu.table_schema
                       JOIN information_schema.columns c
                         ON c.table_schema = tc.table_schema
                        AND c.table_name = tc.table_name
                        AND c.column_name = kcu.column_name
                       WHERE tc.table_schema = %s
                         AND tc.constraint_type = 'PRIMARY KEY'
                       ORDER BY tc.table_name, kcu.ordinal_position""", (SCHEMA_SRC,))
        pks = defaultdict(list)
        for tname, cname, dtype, udt in cur.fetchall():
            pks[tname].append((cname, dtype, udt))

    with conn.cursor() as cur:
        cur.execute("SELECT schema_name FROM information_schema.schemata WHERE schema_name=%s", (SCHEMA_100K,))
        if cur.fetchone():
            print(f'  Schema {SCHEMA_100K} exists; dropping', flush=True)
            cur.execute(f'DROP SCHEMA {SCHEMA_100K} CASCADE')
        cur.execute(f'CREATE SCHEMA {SCHEMA_100K}')

        cur.execute("""SELECT table_name FROM information_schema.tables
                       WHERE table_schema=%s AND table_type='BASE TABLE'
                       ORDER BY table_name""", (SCHEMA_SRC,))
        tables = [r[0] for r in cur.fetchall()]
        print(f'  Found {len(tables)} source tables', flush=True)

        for table in tables:
            cur.execute(f'CREATE TABLE {SCHEMA_100K}."{table}" '
                        f'(LIKE {SCHEMA_SRC}."{table}" INCLUDING DEFAULTS INCLUDING STORAGE)')

            cur.execute("""SELECT column_name, data_type, udt_name
                           FROM information_schema.columns
                           WHERE table_schema=%s AND table_name=%s
                           ORDER BY ordinal_position""", (SCHEMA_SRC, table))
            col_info = cur.fetchall()
            all_cols = [c[0] for c in col_info]
            col_list = ', '.join(f'"{c}"' for c in all_cols)

            cur.execute(f'SELECT COUNT(*) FROM {SCHEMA_SRC}."{table}"')
            src_count = cur.fetchone()[0]
            if src_count == 0:
                print(f'    {table}: empty, skipping', flush=True)
                continue

            cur.execute(f'INSERT INTO {SCHEMA_100K}."{table}" ({col_list}) '
                        f'SELECT {col_list} FROM {SCHEMA_SRC}."{table}"')

            copies = scale_factor - 1
            if copies > 0:
                pk_cols = {c[0]: (c[1], c[2]) for c in pks.get(table, [])}
                select_exprs = []
                for c in all_cols:
                    if c in pk_cols:
                        dtype, udt = pk_cols[c]
                        if udt == 'uuid':
                            select_exprs.append(
                                f'(md5(random()::text || clock_timestamp()::text || '
                                f't."{c}"::text || gs.n::text)::uuid)')
                        elif udt in ('int2', 'int4', 'int8'):
                            select_exprs.append(
                                f'(t."{c}" + (SELECT COALESCE(MAX("{c}"), 0) FROM {SCHEMA_SRC}."{table}") * gs.n)')
                        elif dtype == 'numeric':
                            select_exprs.append(
                                f'(t."{c}" + (SELECT COALESCE(MAX("{c}"), 0) FROM {SCHEMA_SRC}."{table}") * gs.n)')
                        elif udt in ('text', 'varchar', 'bpchar'):
                            select_exprs.append(f'(t."{c}" || \'_copy\' || gs.n::text)')
                        else:
                            select_exprs.append(
                                f'(md5(random()::text || clock_timestamp()::text)::uuid)')
                    else:
                        select_exprs.append(f't."{c}"')

                print(f'    {table}: inserting {src_count * copies} rows...', flush=True)
                cur.execute(f'''INSERT INTO {SCHEMA_100K}."{table}" ({col_list})
                                SELECT {", ".join(select_exprs)}
                                FROM {SCHEMA_SRC}."{table}" t,
                                     generate_series(1, {copies}) AS gs(n)''')

            cur.execute(f'SELECT COUNT(*) FROM {SCHEMA_100K}."{table}"')
            dst_count = cur.fetchone()[0]
            print(f'    {table}: {src_count} -> {dst_count}', flush=True)

        for table in tables:
            try:
                cur.execute(f'ANALYZE {SCHEMA_100K}."{table}"')
            except Exception:
                pass

        cur.execute(f"SELECT COUNT(*) FROM {SCHEMA_100K}.patient")
        pat = cur.fetchone()[0]
        print(f'  Patient count: {pat} (target >= {target})', flush=True)
        return pat


# =============================================================================
# STEP 2: Schema + SQL rewriting
# =============================================================================
def db_schema(conn, schema=SCHEMA_SRC):
    with conn.cursor() as cur:
        cur.execute("SELECT table_name,column_name FROM information_schema.columns WHERE table_schema=%s", (schema,))
        columns = defaultdict(dict)
        for t, c in cur.fetchall():
            columns[t][c] = True
        cur.execute(f"""SELECT tc.table_name,kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name=kcu.constraint_name AND tc.table_schema=kcu.table_schema
            WHERE tc.table_schema=%s AND tc.constraint_type='FOREIGN KEY'""", (schema,))
        fk = {(a, b) for a, b in cur.fetchall()}
    return columns, fk


def rewrite_sql_schema(text, schema):
    if schema == SCHEMA_SRC:
        return text
    tree = sqlglot.parse_one(text, read='postgres')
    for t in tree.find_all(exp.Table):
        if t.db is None or t.db.lower() == 'public':
            t.set('db', exp.Identifier(this=schema, quoted=True))
    return tree.sql(dialect='postgres')


# =============================================================================
# STEP 3: Candidate generators
# =============================================================================
def load_candidates_v6(path, schema, fks, discovery_df, include_fk, schema_name='public'):
    frame = pd.read_csv(path)
    base = []
    for record in frame.to_dict('records'):
        if 'candidate' in record and pd.notna(record['candidate']):
            table, cols = parse_candidate(record['candidate'])
        else:
            definition = str(record['create_sql'])
            m = re.fullmatch(r'CREATE INDEX ON public\."?([A-Za-z_][A-Za-z_0-9]*)"?\s*\((.*)\)', definition, re.I)
            if not m:
                continue
            table = m.group(1)
            cols = [c.strip().replace('"','') for c in m.group(2).split(',')]
        if table not in schema:
            continue
        col_names = [c.split()[0].replace('"','') for c in cols]
        if any(cn not in schema[table] for cn in col_names):
            continue
        base.append({'definition': index_def(table, cols, schema_name), 'table': table,
                     'columns': tuple(col_names), 'group': 'v6_original'})
    extra = []
    if include_fk:
        keys = set(fks)
        for text in discovery_df.sql:
            try:
                tree = sql_ast(text)
            except Exception:
                continue
            aliases = {t.alias_or_name.lower(): t.name.lower() for t in tree.find_all(exp.Table) if t.name}
            for eq in tree.find_all(exp.EQ):
                if isinstance(eq.left, exp.Column) and isinstance(eq.right, exp.Column):
                    for c in (eq.left, eq.right):
                        t = aliases.get((c.table or '').lower())
                        if t and c.name in schema.get(t, {}):
                            keys.add((t, c.name))
        for table, col in sorted(keys):
            if table in schema and col in schema[table]:
                extra.append({'definition': index_def(table, [col], schema_name), 'table': table,
                              'columns': (col,), 'group': 'fk_join_schema_discovery'})
    unique = {}
    for c in base + extra:
        unique.setdefault(c['definition'], c)
    return list(unique.values())


def build_catalog_from_discovery(discovery_df, schema, max_cols_per_composite=2, schema_name='public'):
    def table_of(col, alias_map):
        q = (col.table or '').lower()
        return alias_map.get(q, q) if q else None

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
                t = table_of(col, alias_map)
                if t and t in schema and col.name in schema[t]:
                    per_table[t].add(col.name)
        for t, cols in per_table.items():
            cols = sorted(cols)
            for c in cols:
                d = index_def(t, [c], schema_name)
                generated.setdefault(d, {'definition': d, 'table': t, 'columns': (c,), 'group': 'discovery_generated'})
            for a, b in itertools.combinations(cols, 2):
                d = index_def(t, [a, b], schema_name)
                generated.setdefault(d, {'definition': d, 'table': t, 'columns': (a, b), 'group': 'discovery_generated_pair'})
            if max_cols_per_composite >= 3:
                for triple in itertools.combinations(cols, 3):
                    d = index_def(t, list(triple), schema_name)
                    generated.setdefault(d, {'definition': d, 'table': t, 'columns': tuple(triple), 'group': 'discovery_generated_triple'})
    return list(generated.values())


# =============================================================================
# STEP 4: HypoPG advisor / naive baseline
# =============================================================================
def hypopg_advisor_candidates(conn, discovery_df, schema, schema_name='public'):
    print(f'\n  Analyzing {len(discovery_df)} discovery queries for baseline...', flush=True)
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_proc WHERE proname = 'hypopg_analyze_query'")
        has_advisor = cur.fetchone() is not None

    if has_advisor:
        print('  HypoPG advisor available: using hypopg_analyze_query()', flush=True)
        advisors = {}
        with conn.cursor() as cur:
            cur.execute('SELECT hypopg_reset()')
            for text in discovery_df.sql:
                try:
                    cur.execute('SELECT hypopg_analyze_query(%s)', (text,))
                    cur.execute("""SELECT indexdef FROM hypopg_list_indexes()
                                   WHERE indexdef LIKE 'CREATE INDEX%'""")
                    for (defn,) in cur.fetchall():
                        defn2 = re.sub(r'\bON public\.', f'ON {schema_name}.', defn)
                        m = re.search(
                            r'ON\s+\S*?\.?"?([A-Za-z_][A-Za-z_0-9]*)"?\s+USING\s+\w+\s*\(([^)]+)\)',
                            defn2, re.I)
                        if not m:
                            continue
                        table = m.group(1).lower()
                        cols = tuple(c.strip().strip('"').split()[0].lower()
                                     for c in m.group(2).split(','))
                        if table not in schema:
                            continue
                        advisors.setdefault(defn2, {
                            'definition': defn2, 'table': table,
                            'columns': cols, 'group': 'hypopg_advisor'})
                    cur.execute('SELECT hypopg_reset()')
                except Exception:
                    cur.execute('SELECT hypopg_reset()')
        print(f'  HypoPG advisor: {len(advisors)} candidates', flush=True)
        return list(advisors.values())

    # Fallback: naive structural baseline
    print('  HypoPG advisor NOT available. Falling back to NAIVE BASELINE:', flush=True)
    print('  single-column indexes on every (table, column) in discovery predicates.', flush=True)
    naive = {}
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
        for node in list(tree.find_all(exp.Where)) + list(tree.find_all(exp.Join)):
            for col in node.find_all(exp.Column):
                q = (col.table or '').lower()
                tbl = alias_map.get(q, q) if q else None
                if tbl and tbl in schema and col.name in schema[tbl]:
                    per_table[tbl].add(col.name)
        for tbl, cols in per_table.items():
            for c in sorted(cols):
                d = index_def(tbl, [c], schema_name)
                naive.setdefault(d, {'definition': d, 'table': tbl,
                                     'columns': (c,), 'group': 'naive_structural_baseline'})
    print(f'  Naive baseline: {len(naive)} candidates', flush=True)
    return list(naive.values())


# =============================================================================
# STEP 5: Evaluator (batched EXPLAIN, timeout per-query)
# =============================================================================
class Evaluator:
    def __init__(self, conn, frame, candidates, timeout_ms, schema_name='public'):
        self.conn = conn
        self.frame = frame.reset_index(drop=True)
        self.candidates = candidates
        self.weights = partition_weights(self.frame)
        self.cache = {}
        self.timeout_ms = timeout_ms
        self.evaluations = 0
        sqls = [rewrite_sql_schema(q, schema_name) for q in self.frame.sql]
        self._sqls = sqls
        self._batch_sql = '; '.join('EXPLAIN (FORMAT JSON) ' + q for q in sqls)
        self._n_expected = len(sqls)
        self._last_portfolio = ()
        self.baseline = self.costs(())
        self._baseline_dot = float(np.dot(self.weights, self.baseline))

    def _apply_portfolio(self, cur, portfolio):
        new_set = set(portfolio)
        last_set = set(self._last_portfolio)
        if not last_set.issubset(new_set):
            cur.execute('SELECT hypopg_reset()')
            for i in sorted(new_set):
                cur.execute('SELECT * FROM hypopg_create_index(%s)', (self.candidates[i]['definition'],))
        else:
            for i in sorted(new_set - last_set):
                cur.execute('SELECT * FROM hypopg_create_index(%s)', (self.candidates[i]['definition'],))
        self._last_portfolio = tuple(sorted(new_set))

    def costs(self, portfolio):
        key = tuple(sorted(portfolio))
        if key in self.cache:
            return self.cache[key]
        with self.conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout = %s", (self.timeout_ms,))
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
                costs = []
                cur.execute('SELECT hypopg_reset()')
                for i in key:
                    cur.execute('SELECT * FROM hypopg_create_index(%s)', (self.candidates[i]['definition'],))
                for q in self._sqls:
                    cur.execute('EXPLAIN (FORMAT JSON) ' + q)
                    doc = cur.fetchone()[0]
                    if isinstance(doc, str):
                        doc = json.loads(doc)
                    costs.append(float(doc[0]['Plan']['Total Cost']))
                self._last_portfolio = tuple(sorted(key))
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


# =============================================================================
# STEP 6: Search + coverage
# =============================================================================
def coverage_rows(ev, candidates, mode, seed, group):
    frame = ev.frame
    rows = []
    all_cost = ev._baseline_dot
    for family, subset in frame.groupby('family'):
        ix = subset.index.to_numpy()
        tables = sorted({t for arr in subset.tables for t in arr})
        qual = defaultdict(set)
        loose = set()
        for row in subset.itertuples():
            for t, cs in row.columns_by_table.items():
                qual[t].update(cs)
            loose.update(row.unqualified_columns)
        touching = [i for i, c in enumerate(candidates) if c['table'] in tables]
        matching = [i for i in touching if any(col in qual.get(candidates[i]['table'], set()) or col in loose for col in candidates[i]['columns'])]
        cost = float(np.dot(ev.weights[ix], ev.baseline[ix]))
        rows.append({'mode': mode, 'seed': seed, 'group': group, 'family': family, 'queries': len(subset),
                     'templates': subset.template.nunique(), 'tables_touched': '|'.join(tables),
                     'candidate_table_matches': len(touching), 'candidate_column_matches': len(matching),
                     'baseline_cost_share_pct': 100*cost/all_cost if all_cost > EPS else 0,
                     'baseline_weighted_cost': cost})
    return rows


def greedy(ev, n, budget, restrict=None):
    pool = list(restrict) if restrict is not None else list(range(n))
    selected = ()
    prev = ev.gain(selected)
    while len(selected) < budget:
        opts = [(ev.gain(tuple(sorted(selected + (i,)))), i) for i in pool if i not in selected]
        if not opts:
            break
        v, i = max(opts, key=lambda x: (x[0], -x[1]))
        if v <= prev + EPS:
            break
        selected = tuple(sorted(selected + (i,)))
        prev = v
    return selected


def beam(ev, n, budget, width, pairs, restrict=None, patience=2, min_delta=0.01):
    pool = list(restrict) if restrict is not None else list(range(n))
    if not pool or budget <= 0:
        return ()
    current = [()]
    best_p, best_g = (), ev.gain(())
    stagnant = 0
    for depth in range(1, min(len(pool), budget) + 1):
        possible = {tuple(sorted(p + (i,))) for p in current for i in pool if i not in p}
        if depth == 2 and pairs:
            for pr in pairs:
                if pr['i'] in pool and pr['j'] in pool:
                    possible.add(tuple(sorted((pr['i'], pr['j']))))
        if not possible:
            break
        scored = sorted(((ev.gain(p), p) for p in possible), key=lambda x: (-x[0], x[1]))
        current = [p for _, p in scored[:width]]
        dg, dp = scored[0]
        imp = dg - best_g
        if imp > 1e-9:
            best_g, best_p = dg, dp
        stagnant = stagnant + 1 if imp < min_delta else 0
        if patience is not None and stagnant >= patience:
            break
    return best_p


def singleton_pairs(ev, n, top_k=30):
    singles = {i: ev.gain((i,)) for i in range(n)}
    top = sorted(singles.items(), key=lambda kv: -kv[1])[:top_k]
    top_idx = [i for i, _ in top]
    pairs = []
    for i, j in itertools.combinations(top_idx, 2):
        g = ev.gain((i, j))
        pairs.append({'i': i, 'j': j, 'gain': g, 'single_i': singles[i], 'single_j': singles[j],
                      'synergy_pp': g - singles[i] - singles[j]})
    return singles, pairs, top_idx


def choose_validation(methods, ev, candidates):
    scored = []
    for name, p in methods.items():
        if name == 'no_new_indexes':
            continue
        g = ev.gain(p)
        scored.append({'method': name, 'portfolio': p, 'gain': g, 'indexes': len(p)})
    mx = max(r['gain'] for r in scored)
    tied = [r['method'] for r in scored if abs(r['gain'] - mx) <= 1e-9]
    ranked = sorted(scored, key=lambda r: (-round(r['gain'], 9), r['indexes'], r['method']))
    return ranked[0], sorted(tied)


# =============================================================================
# STEP 7: One (mode, seed) run
# =============================================================================
def run_seed_mode(conn, df, args, mode, seed, schema_name, schema_cols, fks, scale_label):
    parts = split(df, seed, mode)
    names = ('discovery', 'validation', 'test')
    print(f'\n[{scale_label} | {mode} seed={seed}] ' +
          ', '.join(f'{n}={len(p)}' for n, p in zip(names, parts)), flush=True)

    disc_df = parts[0]
    v6 = load_candidates_v6(args.candidates, schema_cols, fks, disc_df, include_fk=True, schema_name=schema_name)
    disc_cat = build_catalog_from_discovery(disc_df, schema_cols, max_cols_per_composite=2, schema_name=schema_name)
    merged = {}
    for c in v6 + disc_cat:
        merged.setdefault(c['definition'], c)
    discovery_catalog = list(merged.values())
    advisors = hypopg_advisor_candidates(conn, disc_df, schema_cols, schema_name)
    print(f'  Catalogs: V6={len(v6)}, Discovery={len(discovery_catalog)}, Baseline={len(advisors)}', flush=True)

    results = []
    baseline_label = 'hypopg_advisor' if advisors and advisors[0]['group'] == 'hypopg_advisor' else 'naive_structural_baseline'
    for label, cands in [('v6_only', v6), ('discovery', discovery_catalog), (baseline_label, advisors)]:
        if len(cands) == 0:
            print(f'  SKIP {label}: 0 candidates', flush=True)
            continue
        try:
            disc_ev = Evaluator(conn, parts[0], cands, args.timeout_ms, schema_name)
            val_ev = Evaluator(conn, parts[1], cands, args.timeout_ms, schema_name)
            tst_ev = Evaluator(conn, parts[2], cands, args.timeout_ms, schema_name)
        except Exception as e:
            print(f'  FAIL {label} evaluators: {e}', flush=True)
            continue
        cov = coverage_rows(disc_ev, cands, mode, seed, label + '_discovery')
        unc = sum(r['baseline_cost_share_pct'] for r in cov if r['candidate_column_matches'] == 0)
        ceiling = 100 - unc
        print(f'  [{label}] cands={len(cands)}, discovery ceiling~{ceiling:.2f}%', flush=True)

        try:
            _, pairs, top_idx = singleton_pairs(disc_ev, len(cands), top_k=args.pair_top_k)
            pool = top_idx[:50] if len(top_idx) > 50 else (top_idx if top_idx else None)
            p_greedy = greedy(disc_ev, len(cands), args.budget, restrict=pool)
            p_beam = beam(disc_ev, len(cands), args.budget, args.beam_width, pairs, restrict=pool)
        except Exception as e:
            print(f'  FAIL {label} search: {e}', flush=True)
            continue

        methods = {'no_new_indexes': (), 'greedy': p_greedy, 'beam': p_beam}
        winner, ties = choose_validation(methods, val_ev, cands)
        for mname, p in methods.items():
            results.append({
                'scale': scale_label, 'schema_name': schema_name, 'mode': mode, 'seed': seed,
                'catalog': label, 'method': mname, 'selected': mname == winner['method'],
                'indexes': len(p), 'portfolio': '|'.join(map(str, p)),
                'candidate_count': len(cands), 'discovery_ceiling_pct': ceiling,
                'discovery_weighted_gain_pct': disc_ev.gain(p),
                'validation_weighted_gain_pct': val_ev.gain(p),
                'test_weighted_gain_pct': tst_ev.gain(p),
                'test_unweighted_gain_pct': tst_ev.unweighted(p),
            })
        print(f'  Selected: {winner["method"]} | test gain={results[-1]["test_weighted_gain_pct"]:.4f}%', flush=True)
    return results


# =============================================================================
# STEP 8: Parallel worker
# =============================================================================
def worker(job):
    args = SimpleNamespace(**job['args'])
    conn, _ = connection(args)
    try:
        df = load_workload(args.workload)
        schema_cols, fks = db_schema(conn, job['schema_name'])
        res = run_seed_mode(conn, df, args, job['mode'], job['seed'],
                            job['schema_name'], schema_cols, fks, job['scale_label'])
        return job['tag'], 'ok', res
    except Exception as e:
        import traceback
        print(f'WORKER FAIL {job["tag"]}:\n{traceback.format_exc()}', flush=True)
        return job['tag'], f'error: {e}', []
    finally:
        conn.close()


# =============================================================================
# STEP 9: Statistics + Figures
# =============================================================================
def statistical_tests(results_df, out_dir):
    rows = []
    for mode in ['within_template', 'unseen_template']:
        for scale in ['base', '100k']:
            sel = results_df[(results_df['mode'] == mode) & (results_df['scale'] == scale) &
                             (results_df['catalog'] == 'discovery') & (results_df['method'] == 'beam')]
            if len(sel) < 2:
                continue
            vals = sel['test_weighted_gain_pct'].values
            n = len(vals)
            t_stat, t_p = scipy_stats.ttest_1samp(vals, 0.0)
            w_stat, w_p = scipy_stats.wilcoxon(vals) if np.std(vals) > 1e-12 else (0.0, 1.0)
            rng = np.random.default_rng(42)
            boot = [np.mean(rng.choice(vals, size=n, replace=True)) for _ in range(10000)]
            ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
            rows.append({'test': f'{scale}_{mode}_discovery_beam_gt_0',
                         'n': n, 'mean': float(np.mean(vals)), 'std': float(np.std(vals, ddof=1)),
                         't_stat': float(t_stat), 't_p': float(t_p),
                         'wilcoxon_stat': float(w_stat), 'wilcoxon_p': float(w_p),
                         'ci_95_lo': float(ci_lo), 'ci_95_hi': float(ci_hi)})

    for scale in ['base', '100k']:
        for mode in ['within_template', 'unseen_template']:
            for baseline in ['v6_only', 'hypopg_advisor', 'naive_structural_baseline']:
                disc = results_df[(results_df['mode'] == mode) & (results_df['scale'] == scale) &
                                  (results_df['catalog'] == 'discovery') & (results_df['method'] == 'beam')]
                base = results_df[(results_df['mode'] == mode) & (results_df['scale'] == scale) &
                                  (results_df['catalog'] == baseline) & (results_df['method'] == 'beam')]
                if len(disc) > 1 and len(base) == len(disc):
                    a = disc['test_weighted_gain_pct'].values
                    b = base['test_weighted_gain_pct'].values
                    diff = a - b
                    t_stat, t_p = scipy_stats.ttest_rel(a, b)
                    w_stat, w_p = scipy_stats.wilcoxon(a, b) if np.std(diff) > 1e-12 else (0.0, 1.0)
                    cohens_d = float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 1e-12 else 0.0
                    rows.append({'test': f'{scale}_{mode}_discovery_vs_{baseline}',
                                 'n': len(a), 'mean_disc': float(np.mean(a)), 'mean_base': float(np.mean(b)),
                                 'mean_diff': float(np.mean(diff)), 't_stat': float(t_stat), 't_p': float(t_p),
                                 'wilcoxon_stat': float(w_stat), 'wilcoxon_p': float(w_p),
                                 'cohens_d': cohens_d})
    df_out = pd.DataFrame(rows)
    save(out_dir / 'statistical_tests.csv', df_out)
    return df_out


def make_figures(results_df, out_dir):
    try:
        sel = results_df[results_df['selected']]
        disc_rows = results_df[(results_df['catalog'] == 'discovery')]

        # Fig 1: ceiling vs realized (base + 100k)
        fig, ax = plt.subplots(figsize=(12, 6))
        labels, values, errs, colors = [], [], [], []
        for scale in ['base', '100k']:
            scale_sel = sel[sel['scale'] == scale]
            if scale_sel.empty:
                continue
            for mode, color in [('within_template', '#2a6'), ('unseen_template', '#c33')]:
                d = scale_sel[scale_sel['mode'] == mode]['test_weighted_gain_pct']
                if len(d) == 0:
                    continue
                labels.append(f'{scale}\n{mode}')
                values.append(d.mean())
                errs.append(d.std() if len(d) > 1 else 0)
                colors.append(color)
        ax.bar(labels, values, yerr=errs, color=colors, capsize=8, edgecolor='black')
        for i, v in enumerate(values):
            ax.text(i, v + 2, f'{v:.2f}%', ha='center', fontweight='bold')
        ax.set_ylabel('Test weighted gain (%)')
        ax.set_title('Realized gain by scale and mode (discovery catalog, beam)')
        ax.set_ylim(0, 110)
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(out_dir / 'fig1_ceiling_realized.png', dpi=170)
        plt.close()

        # Fig 2: per-seed
        fig, ax = plt.subplots(figsize=(10, 6))
        positions = []
        pos = 0
        for scale in ['base', '100k']:
            for mode, color in [('within_template', '#2a6'), ('unseen_template', '#c33')]:
                d = sel[(sel['scale'] == scale) & (sel['mode'] == mode)]['test_weighted_gain_pct'].values
                if len(d) == 0:
                    continue
                ax.scatter([pos]*len(d), d, s=80, color=color, alpha=0.7, edgecolors='black')
                ax.errorbar(pos, d.mean(), yerr=d.std() if len(d) > 1 else 0, fmt='_', color='black',
                            capsize=20, markersize=30, markeredgewidth=3)
                positions.append((pos, f'{scale}\n{mode}'))
                pos += 1
        ax.set_xticks([p for p, _ in positions])
        ax.set_xticklabels([l for _, l in positions], rotation=15)
        ax.set_ylabel('Test weighted gain (%)')
        ax.set_title('Per-seed realized gain')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(-5, 100)
        plt.tight_layout()
        plt.savefig(out_dir / 'fig2_per_seed.png', dpi=170)
        plt.close()

        # Fig 3: catalog comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        catalogs = results_df['catalog'].unique()
        scales = ['base', '100k']
        modes = ['within_template', 'unseen_template']
        colors_cat = plt.cm.tab10(np.linspace(0, 1, len(catalogs)))
        w = 0.8 / max(1, len(catalogs))
        for i, cat in enumerate(catalogs):
            vals, errs, labels = [], [], []
            for scale in scales:
                for mode in modes:
                    sub = results_df[(results_df['scale'] == scale) & (results_df['mode'] == mode) &
                                     (results_df['catalog'] == cat) & (results_df['method'] == 'beam')]['test_weighted_gain_pct']
                    vals.append(sub.mean() if len(sub) else 0)
                    errs.append(sub.std() if len(sub) > 1 else 0)
                    labels.append(f'{scale}\n{mode}')
            x = np.arange(len(labels)) + (i - len(catalogs)/2 + 0.5) * w
            ax.bar(x, vals, w, label=cat, color=colors_cat[i], capsize=3, yerr=errs, edgecolor='black')
        ax.set_xticks(np.arange(len(scales)*len(modes)))
        ax.set_xticklabels([f'{s}\n{m}' for s in scales for m in modes], rotation=15)
        ax.set_ylabel('Test weighted gain (%)')
        ax.set_title('Catalog comparison (beam)')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / 'fig3_catalog_comparison.png', dpi=170)
        plt.close()
        print(f'  Figures written to {out_dir}', flush=True)
    except Exception as e:
        print(f'  Figure generation warning: {e}', flush=True)


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default=DB)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=5433)
    ap.add_argument('--user', default='postgres')
    ap.add_argument('--workload', type=Path, default=DEFAULT_WORKLOAD)
    ap.add_argument('--candidates', type=Path, default=DEFAULT_CANDIDATES)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--start-seed', type=int, default=43)
    ap.add_argument('--budget', type=int, default=16)
    ap.add_argument('--beam-width', type=int, default=12)
    ap.add_argument('--pair-top-k', type=int, default=30)
    ap.add_argument('--timeout-ms', type=int, default=120000, help='Per-query timeout for EXPLAIN (ms)')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--scale-factor', type=int, default=100)
    ap.add_argument('--target-rows', type=int, default=100_000)
    ap.add_argument('--skip-scale-build', action='store_true')
    ap.add_argument('--base-only', action='store_true')
    args = ap.parse_args()

    out = OUTPUT / datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=True)
    print(f'Results: {out}', flush=True)
    print(f'Config: seeds={args.seeds}, workers={args.workers}, budget={args.budget}, '
          f'beam_width={args.beam_width}, scale_factor={args.scale_factor}', flush=True)

    # ---- Dedicated scale-up connection: session-level timeout=0 ----
    pw = os.environ.get('PGPASSWORD')
    if not pw:
        import getpass
        pw = getpass.getpass('Benchmark PostgreSQL password: ')
    scale_conn = psycopg.connect(host=args.host, port=args.port, dbname=args.db,
                                 user=args.user, password=pw, autocommit=True, connect_timeout=15)
    with scale_conn.cursor() as cur:
        cur.execute("SET statement_timeout = 0")
        cur.execute("SHOW statement_timeout")
        print(f'  Scale-up connection statement_timeout: {cur.fetchone()[0]}', flush=True)

    conn, _ = connection(args)
    try:
        # ---- Step 1: Scale-up ----
        pat_count = None
        need_scaled = not args.base_only

        if need_scaled:
            rebuild = not args.skip_scale_build
            if not rebuild:
                try:
                    with scale_conn.cursor() as cur:
                        cur.execute(f"SELECT COUNT(*) FROM {SCHEMA_100K}.patient")
                        existing = cur.fetchone()[0]
                    if existing < args.target_rows:
                        print(f'  [skip-scale-build] {SCHEMA_100K}.patient has only '
                              f'{existing} rows (< {args.target_rows}); rebuilding', flush=True)
                        rebuild = True
                    else:
                        pat_count = existing
                        print(f'  [skip-scale-build] Reusing {SCHEMA_100K}: {existing} patients', flush=True)
                except Exception as e:
                    print(f'  [skip-scale-build] {SCHEMA_100K}.patient not accessible '
                          f'({e}); rebuilding', flush=True)
                    rebuild = True

            if rebuild:
                pat_count = build_scale_schema(scale_conn, scale_factor=args.scale_factor,
                                               target=args.target_rows)

            if pat_count is None or pat_count < args.target_rows:
                raise RuntimeError(
                    f'Scaled schema has {pat_count} patients, below target {args.target_rows}. '
                    f'Increase --scale-factor (currently {args.scale_factor}).')
            print(f'  Scale-up OK: {pat_count} patients in {SCHEMA_100K}', flush=True)
        else:
            print('  --base-only: skipping scale-up', flush=True)

        # ---- Step 2: Parallel jobs ----
        from multiprocessing import Pool
        scales = [(SCHEMA_SRC, 'base')]
        if need_scaled:
            scales.append((SCHEMA_100K, '100k'))

        jobs = []
        for schema_name, scale_label in scales:
            for mode in ('within_template', 'unseen_template'):
                for seed in range(args.start_seed, args.start_seed + args.seeds):
                    d = vars(args).copy()
                    jobs.append({
                        'tag': f'{scale_label}_{mode}_{seed}',
                        'args': d,
                        'schema_name': schema_name,
                        'scale_label': scale_label,
                        'mode': mode,
                        'seed': seed,
                    })

        workers = min(args.workers, len(jobs))
        print(f'\n=== Step 2: Running {len(jobs)} jobs with {workers} workers ===', flush=True)
        all_results = []
        failed = []
        with Pool(processes=workers) as pool:
            for tag, status, res in pool.imap_unordered(worker, jobs):
                marker = 'OK' if status == 'ok' else 'FAIL'
                print(f'  [{marker}] {tag}: {status} ({len(res)} rows)', flush=True)
                if status == 'ok':
                    all_results.extend(res)
                else:
                    failed.append((tag, status))

        if not all_results:
            raise RuntimeError(f'All workers failed: {failed}')

        df_results = pd.DataFrame(all_results)
        save(out / 'all_results.csv', df_results)

        # ---- Step 3: Statistics ----
        print('\n=== Step 3: Statistical tests ===', flush=True)
        try:
            stats_df = statistical_tests(df_results, out)
            if not stats_df.empty:
                print(stats_df.to_string(index=False), flush=True)
        except Exception as e:
            print(f'  WARNING: stats failed: {e}', flush=True)

        # ---- Step 4: Figures ----
        print('\n=== Step 4: Figures ===', flush=True)
        make_figures(df_results, out)

        # ---- Step 5: Summary ----
        summary = df_results[df_results['selected']].groupby(
            ['scale', 'mode', 'catalog'], as_index=False).agg(
            seeds=('seed', 'nunique'),
            mean_test_gain=('test_weighted_gain_pct', 'mean'),
            std_test_gain=('test_weighted_gain_pct', 'std'),
            mean_validation_gain=('validation_weighted_gain_pct', 'mean'),
            mean_discovery_ceiling=('discovery_ceiling_pct', 'mean'),
            mean_indexes=('indexes', 'mean'),
        )
        summary['std_test_gain'] = summary['std_test_gain'].fillna(0)
        save(out / 'summary.csv', summary)
        print('\n=== SUMMARY ===', flush=True)
        print(summary.to_string(index=False), flush=True)

        # ---- Step 6: Manifest ----
        jdump(out / 'manifest.json', {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'args': vars(args),
            'patient_count_100k': pat_count,
            'scaling_method': 'INSERT ... SELECT ... FROM generate_series with UUID-safe PK generation',
            'statement_timeout': '0 (session-level) for scale-up; per-query for Evaluator',
            'weighting': 'instance_count',
            'workers_used': workers,
            'jobs_run': len(jobs),
            'jobs_failed': len(failed),
            'budget_sweep': 'SKIPPED',
        })
        print(f'\nDone: {out}', flush=True)

    except Exception as e:
        import traceback
        print(f'\nFATAL ERROR:\n{traceback.format_exc()}', flush=True)
        raise
    finally:
        conn.close()
        try:
            scale_conn.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()