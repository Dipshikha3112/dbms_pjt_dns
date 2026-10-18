"""
app.py — Unified research interface for the Adaptive Healthcare Knowledge Database.

Serves one web page (/), one JSON API (/api/*), and Swagger UI (/docs).

Endpoints cover only subsystems with verified, existing results on disk:
  - Phase 9  : V7.5 optimizer audit (66.97% within-template, 0.008% unseen-template)
  - Phase 10 : Neo4j biomedical knowledge graph (551,615 nodes, 804,488 rels)
  - Phase 7  : Postgres query monitoring log
  - Phase 12 : Literature evidence retrieval (requires ingested PDFs)

Not covered (and correctly labeled in the UI):
  - Phase 11 literature ingestion pipeline      NOT IMPLEMENTED
  - Phase 13 hybrid SQL/graph/vector planner    NOT IMPLEMENTED
  - Phase 14 full backend                       this file serves as a subset
  - Phase 15 dashboard                          this HTML is a subset
  - Phase 16 formal evaluation                  PARTIAL via Phase 9 V7.5 audit
  - Phase 17 report                             NOT IMPLEMENTED

Run:
    uvicorn app:app --reload --port 8000
Then open:
    http://127.0.0.1:8000/       (interface)
    http://127.0.0.1:8000/docs   (Swagger UI)
"""
from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

ROOT = Path(__file__).resolve().parent

# --- Phase 9 V7.5 audit: the clean, verified run ---
PHASE9_RUN = ROOT / "results/phase8_phase9_v7_audit/20261018_130202_315654"

# --- Phase 12 SQLite store ---
PHASE12_STORE = ROOT / "results/phase12_store_v3"
PHASE12_SCRIPT = ROOT / "notebooks/12_full.py"

# --- Phase 7 monitoring DB ---
MONITOR_DB = "adaptive_healthcare_phase9_benchmark"
MONITOR_HOST = "127.0.0.1"
MONITOR_PORT = 5433
MONITOR_USER = "postgres"

# --- Phase 10 Neo4j ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


# ============================================================
# FASTAPI
# ============================================================
app = FastAPI(
    title="Adaptive Healthcare Knowledge Database",
    description=__doc__,
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HELPERS
# ============================================================
def _require(path: Path, what: str) -> Path:
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"{what} not found at {path}. Run the corresponding phase first.",
        )
    return path


def _pg_connect():
    import psycopg
    from psycopg.rows import dict_row
    pw = os.getenv("PGPASSWORD")
    if not pw:
        raise HTTPException(503, "PGPASSWORD environment variable not set.")
    return psycopg.connect(
        host=MONITOR_HOST, port=MONITOR_PORT, dbname=MONITOR_DB,
        user=MONITOR_USER, password=pw,
        row_factory=dict_row, connect_timeout=10,
    )


def _neo4j_driver():
    from neo4j import GraphDatabase
    pw = os.getenv("NEO4J_PASSWORD")
    if not pw:
        raise HTTPException(503, "NEO4J_PASSWORD environment variable not set.")
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, pw))


def _load_phase12_module():
    if not PHASE12_SCRIPT.exists():
        raise HTTPException(503, f"Phase 12 script not found at {PHASE12_SCRIPT}")
    spec = importlib.util.spec_from_file_location("p12", PHASE12_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["p12"] = module
    spec.loader.exec_module(module)
    return module


# ============================================================
# ROOT + /docs
# ============================================================
@app.get("/", response_class=HTMLResponse)
def index():
    return INTERFACE_HTML


# ============================================================
# PHASE 9 — Optimizer audit (V7.5)
# ============================================================
@app.get("/api/optimizer/summary")
def optimizer_summary():
    run = _require(PHASE9_RUN, "Phase 9 V7.5 run")
    comparison = pd.read_csv(run / "portfolio_comparison.csv")
    method_summary = (
        comparison.groupby(["mode", "method"])["test_weighted_gain_pct"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    method_summary["std"] = method_summary["std"].fillna(0.0)
    return {
        "run_directory": str(run.relative_to(ROOT)),
        "code_version": "7.5.0",
        "method_summary": method_summary.to_dict(orient="records"),
    }


@app.get("/api/optimizer/budget-sweep")
def optimizer_budget_sweep():
    run = _require(PHASE9_RUN, "Phase 9 V7.5 run")
    sweep = pd.read_csv(run / "budget_sweep.csv")
    agg = (
        sweep.groupby(["mode", "budget"])[["beam_test_gain", "beam_valid_gain"]]
        .mean()
        .reset_index()
    )
    return {"budget_sweep": agg.to_dict(orient="records")}


@app.get("/api/optimizer/reset-counters")
def optimizer_reset_counters():
    run = _require(PHASE9_RUN, "Phase 9 V7.5 run")
    comp = pd.read_csv(run / "portfolio_comparison.csv")
    cols = [c for c in ("disc_resets", "val_resets", "tst_resets") if c in comp]
    return {
        "counters_present": cols,
        "min_by_column": {c: int(comp[c].min()) for c in cols},
        "max_by_column": {c: int(comp[c].max()) for c in cols},
        "all_positive": all((comp[c] > 0).all() for c in cols),
    }


# ============================================================
# PHASE 10 — Knowledge graph
# ============================================================
@app.get("/api/graph/stats")
def graph_stats():
    driver = _neo4j_driver()
    try:
        with driver.session(database=NEO4J_DATABASE) as s:
            nodes = s.run(
                "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS count "
                "ORDER BY count DESC"
            ).data()
            rels = s.run(
                "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count "
                "ORDER BY count DESC"
            ).data()
        return {
            "nodes": nodes,
            "relationships": rels,
            "total_nodes": sum(n["count"] for n in nodes),
            "total_relationships": sum(r["count"] for r in rels),
        }
    finally:
        driver.close()


@app.get("/api/graph/patient/{patient_id}")
def graph_patient(patient_id: str, limit: int = Query(100, ge=1, le=500)):
    driver = _neo4j_driver()
    try:
        with driver.session(database=NEO4J_DATABASE) as s:
            exists = s.run(
                "MATCH (p:Patient {patient_id: $pid}) RETURN count(p) AS c",
                pid=patient_id,
            ).single()["c"]
            if not exists:
                raise HTTPException(404, f"Patient {patient_id} not found in graph.")
            rows = s.run(
                """
                MATCH (p:Patient {patient_id: $pid})-[r]-(n)
                RETURN type(r) AS relationship,
                       labels(n)[0] AS neighbor_type,
                       coalesce(n.name, n.description, n.loinc_code, n.rxcui) AS neighbor_label
                LIMIT $limit
                """,
                pid=patient_id, limit=limit,
            ).data()
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            key = f"{row['relationship']} → {row['neighbor_type']}"
            grouped.setdefault(key, []).append(
                {"label": row["neighbor_label"]}
            )
        return {"patient_id": patient_id, "groups": grouped}
    finally:
        driver.close()


@app.get("/api/graph/patients")
def graph_patients(limit: int = Query(20, ge=1, le=200)):
    driver = _neo4j_driver()
    try:
        with driver.session(database=NEO4J_DATABASE) as s:
            rows = s.run(
                "MATCH (p:Patient) RETURN p.patient_id AS patient_id LIMIT $limit",
                limit=limit,
            ).data()
        return {"patients": [r["patient_id"] for r in rows if r["patient_id"]]}
    finally:
        driver.close()


# ============================================================
# PHASE 7 — Monitoring
# ============================================================
@app.get("/api/monitoring/summary")
def monitoring_summary():
    with _pg_connect() as c, c.cursor() as cur:
        cur.execute(
            """
            SELECT count(*) AS total_executions,
                   count(*) FILTER (WHERE status='ok') AS successful,
                   avg(execution_ms) AS mean_execution_ms,
                   sum(seq_scans) AS seq_scans,
                   sum(index_scans) AS index_scans,
                   sum(shared_read) AS shared_read_blocks
            FROM phase7_monitoring.query_log
            """
        )
        return cur.fetchone()


@app.get("/api/monitoring/recent")
def monitoring_recent(limit: int = Query(20, ge=1, le=200)):
    with _pg_connect() as c, c.cursor() as cur:
        cur.execute(
            """
            SELECT collected_at, query_family, query_name, status,
                   execution_ms, planning_ms, seq_scans, index_scans
            FROM phase7_monitoring.query_log
            ORDER BY id DESC
            LIMIT %s
            """,
            (limit,),
        )
        return {"rows": cur.fetchall()}


@app.get("/api/monitoring/top-fingerprints")
def monitoring_top(limit: int = Query(10, ge=1, le=50)):
    with _pg_connect() as c, c.cursor() as cur:
        cur.execute(
            """
            SELECT fingerprint,
                   min(query_family) AS query_family,
                   count(*) AS executions,
                   avg(execution_ms) AS mean_ms
            FROM phase7_monitoring.query_log
            GROUP BY fingerprint
            ORDER BY executions DESC
            LIMIT %s
            """,
            (limit,),
        )
        return {"fingerprints": cur.fetchall()}


# ============================================================
# PHASE 12 — Literature evidence retrieval
# ============================================================
@app.get("/api/retrieval/ask")
def retrieval_ask(q: str, patient_id: str | None = None):
    if not PHASE12_STORE.exists():
        raise HTTPException(
            503,
            f"Phase 12 store not found at {PHASE12_STORE}. "
            f"Run: python {PHASE12_SCRIPT.name} ingest "
            f"--corpus <pdfs> && python {PHASE12_SCRIPT.name} build-index",
        )
    module = _load_phase12_module()
    try:
        args = _phase12_args(q, patient_id)
        db = module.open_store(PHASE12_STORE)
        try:
            return module.ask(args, db, q, patient_id)
        finally:
            db.close()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            500,
            f"Retrieval failed: {exc}\n{traceback.format_exc()[:1000]}",
        )


def _phase12_args(q: str, patient_id: str | None):
    from types import SimpleNamespace
    return SimpleNamespace(
        command="ask",
        corpus="",
        store=str(PHASE12_STORE),
        results=str(ROOT / "results/phase12_runs_v3"),
        question=q,
        patient_id=patient_id,
        graph=True,
        graph_prompt_password=False,
        graph_limit=12,
        top_k=8,
        chunk_words=220,
        overlap_words=45,
        max_features=60000,
        max_sentences=5,
        eval_file=None,
        strict=False,
    )


@app.get("/api/retrieval/doctor")
def retrieval_doctor():
    module = _load_phase12_module()
    db = module.open_store(PHASE12_STORE)
    try:
        args = _phase12_args("", None)
        return module.doctor(args, db)
    finally:
        db.close()


# ============================================================
# HONEST CAPABILITY MAP
# ============================================================
@app.get("/api/system/capabilities")
def capabilities():
    def exists(p: Path) -> bool:
        return p.exists()

    return {
        "phases": {
            "9_optimizer": {
                "status": "verified",
                "evidence": "66.97% ± 3.07% within-template; 0.008% unseen; monotone budget sweep; reset counters > 0",
                "source": str(PHASE9_RUN.relative_to(ROOT)),
                "present": exists(PHASE9_RUN),
            },
            "10_biomedical_kg": {
                "status": "built_and_audited",
                "evidence": "551,615 nodes / 804,488 rels; patient bridge unresolved; 11_5_3.py audit",
                "source": "Neo4j",
                "present": True,
            },
            "7_monitoring": {
                "status": "verified",
                "evidence": "persistent query_log + resource monitor",
                "source": f"postgres://{MONITOR_DB}",
                "present": True,
            },
            "12_literature": {
                "status": "prototype",
                "evidence": "local PDFs only; SQLite FTS5 + TF-IDF; no clinical validation",
                "source": str(PHASE12_STORE.relative_to(ROOT)),
                "present": exists(PHASE12_STORE),
            },
            "11_literature_pipeline": {"status": "not_implemented", "present": False},
            "13_hybrid_planner": {"status": "not_implemented", "present": False},
            "14_full_backend": {"status": "subset_only_see_app.py", "present": True},
            "15_dashboard": {"status": "subset_only_see_app.py", "present": True},
            "16_formal_evaluation": {"status": "partial_via_phase9", "present": True},
            "17_report": {"status": "not_implemented", "present": False},
        },
        "honest_notes": [
            "Nothing in this interface modifies the database.",
            "Phase 9 numbers are planner-cost, not latency.",
            "Phase 10 patient identity bridge is unresolved.",
            "Phase 12 is not clinically validated.",
        ],
    }


# ============================================================
# THE INTERFACE (single HTML page, no build step)
# ============================================================
INTERFACE_HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Adaptive Healthcare DB — Research Interface</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  body { font-family: system-ui, -apple-system, sans-serif; }
  .tab-active { background: #1e40af; color: white; }
  pre { white-space: pre-wrap; word-break: break-word; }
</style>
</head>
<body class="bg-slate-50 text-slate-800">

<header class="bg-white border-b border-slate-200 shadow-sm">
  <div class="max-w-7xl mx-auto px-6 py-4">
    <h1 class="text-2xl font-bold text-slate-900">
      Adaptive Healthcare Knowledge Database
    </h1>
    <p class="text-sm text-slate-600 mt-1">
      Research interface — subsystems with verified results on disk
    </p>
    <p class="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded px-2 py-1 mt-2 inline-block">
      Research prototype only. Not for clinical use. Database modifications: NONE.
    </p>
  </div>
</header>

<nav class="bg-white border-b border-slate-200">
  <div class="max-w-7xl mx-auto px-6 flex gap-1 overflow-x-auto">
    <button onclick="showTab('capabilities')" class="tab px-4 py-3 text-sm font-medium rounded-t hover:bg-slate-100" data-tab="capabilities">Capabilities</button>
    <button onclick="showTab('optimizer')"    class="tab px-4 py-3 text-sm font-medium rounded-t hover:bg-slate-100" data-tab="optimizer">Phase 9 — Optimizer</button>
    <button onclick="showTab('graph')"        class="tab px-4 py-3 text-sm font-medium rounded-t hover:bg-slate-100" data-tab="graph">Phase 10 — Knowledge Graph</button>
    <button onclick="showTab('monitoring')"   class="tab px-4 py-3 text-sm font-medium rounded-t hover:bg-slate-100" data-tab="monitoring">Phase 7 — Monitoring</button>
    <button onclick="showTab('retrieval')"    class="tab px-4 py-3 text-sm font-medium rounded-t hover:bg-slate-100" data-tab="retrieval">Phase 12 — Evidence</button>
  </div>
</nav>

<main class="max-w-7xl mx-auto px-6 py-6">

<!-- CAPABILITIES -->
<section id="tab-capabilities" class="tab-content hidden">
  <h2 class="text-xl font-semibold mb-4">What this interface honestly covers</h2>
  <div id="capabilities-body" class="text-sm text-slate-700">Loading…</div>
</section>

<!-- OPTIMIZER -->
<section id="tab-optimizer" class="tab-content hidden">
  <h2 class="text-xl font-semibold mb-2">Phase 9 — AI-guided index optimizer</h2>
  <p class="text-sm text-slate-600 mb-4">
    Source: <code>results/phase8_phase9_v7_audit/20261018_130202_315654</code>
    (V7.5, code version 7.5.0). Verified run: reset counters &gt; 0, monotone budget sweep.
  </p>

  <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
    <div class="bg-white border border-slate-200 rounded p-4">
      <div class="text-xs text-slate-500">Within-template held-out gain</div>
      <div class="text-2xl font-bold text-blue-700" id="opt-within">—</div>
      <div class="text-xs text-slate-500 mt-1">5 seeds, budget 16</div>
    </div>
    <div class="bg-white border border-slate-200 rounded p-4">
      <div class="text-xs text-slate-500">Unseen-template held-out gain</div>
      <div class="text-2xl font-bold text-red-700" id="opt-unseen">—</div>
      <div class="text-xs text-slate-500 mt-1">5 seeds, budget 16</div>
    </div>
    <div class="bg-white border border-slate-200 rounded p-4">
      <div class="text-xs text-slate-500">HypoPG reset counters</div>
      <div class="text-2xl font-bold text-green-700" id="opt-resets">—</div>
      <div class="text-xs text-slate-500 mt-1">all &gt; 0 = contamination-free</div>
    </div>
  </div>

  <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
    <div class="bg-white border border-slate-200 rounded p-4">
      <h3 class="font-semibold mb-2">Method comparison</h3>
      <canvas id="chart-method" height="220"></canvas>
    </div>
    <div class="bg-white border border-slate-200 rounded p-4">
      <h3 class="font-semibold mb-2">Gain vs budget (monotone sweep)</h3>
      <canvas id="chart-sweep" height="220"></canvas>
    </div>
  </div>

  <div class="mt-6 bg-amber-50 border border-amber-200 rounded p-4 text-sm">
    <strong>Honest interpretation.</strong>
    Within-template: strong (66.97% ± 3.07%). Unseen-template: essentially zero (0.008%).
    Beam search overfits to query shape on unfamiliar templates. Both numbers come from the same
    verified V7.5 run, with the shared-session HypoPG contamination fix applied.
  </div>
</section>

<!-- GRAPH -->
<section id="tab-graph" class="tab-content hidden">
  <h2 class="text-xl font-semibold mb-2">Phase 10 — Biomedical knowledge graph</h2>
  <p class="text-sm text-slate-600 mb-4">
    Neo4j read-only. Patient identity bridge is unresolved — this tab shows what the
    graph contains, not a resolved clinical cohort.
  </p>

  <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
    <div class="bg-white border border-slate-200 rounded p-4">
      <h3 class="font-semibold mb-2">Node counts</h3>
      <div id="graph-nodes">Loading…</div>
    </div>
    <div class="bg-white border border-slate-200 rounded p-4">
      <h3 class="font-semibold mb-2">Relationship counts</h3>
      <div id="graph-rels">Loading…</div>
    </div>
  </div>

  <div class="bg-white border border-slate-200 rounded p-4">
    <h3 class="font-semibold mb-2">Patient subgraph</h3>
    <div class="flex gap-2 mb-3">
      <input id="graph-pid" placeholder="patient UUID" class="border border-slate-300 rounded px-3 py-2 text-sm flex-1">
      <button onclick="loadPatient()" class="bg-blue-700 text-white rounded px-4 py-2 text-sm hover:bg-blue-800">Load</button>
    </div>
    <div id="graph-patient" class="text-sm text-slate-700">Enter a patient UUID to inspect.</div>
  </div>
</section>

<!-- MONITORING -->
<section id="tab-monitoring" class="tab-content hidden">
  <h2 class="text-xl font-semibold mb-2">Phase 7 — Live query monitoring</h2>
  <p class="text-sm text-slate-600 mb-4">
    Postgres <code>phase7_monitoring.query_log</code>. Populated by <code>07_update.py</code>.
  </p>

  <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
    <div class="bg-white border border-slate-200 rounded p-4">
      <div class="text-xs text-slate-500">Total executions</div>
      <div class="text-2xl font-bold" id="mon-total">—</div>
    </div>
    <div class="bg-white border border-slate-200 rounded p-4">
      <div class="text-xs text-slate-500">Mean execution (ms)</div>
      <div class="text-2xl font-bold" id="mon-mean">—</div>
    </div>
    <div class="bg-white border border-slate-200 rounded p-4">
      <div class="text-xs text-slate-500">Sequential scans</div>
      <div class="text-2xl font-bold" id="mon-seq">—</div>
    </div>
    <div class="bg-white border border-slate-200 rounded p-4">
      <div class="text-xs text-slate-500">Index scans</div>
      <div class="text-2xl font-bold" id="mon-idx">—</div>
    </div>
  </div>

  <div class="bg-white border border-slate-200 rounded p-4 mb-6">
    <h3 class="font-semibold mb-2">Recent executions</h3>
    <div class="overflow-x-auto"><table id="mon-recent" class="w-full text-sm"><tbody></tbody></table></div>
  </div>

  <div class="bg-white border border-slate-200 rounded p-4">
    <h3 class="font-semibold mb-2">Top fingerprints</h3>
    <div class="overflow-x-auto"><table id="mon-top" class="w-full text-sm"><tbody></tbody></table></div>
  </div>
</section>

<!-- RETRIEVAL -->
<section id="tab-retrieval" class="tab-content hidden">
  <h2 class="text-xl font-semibold mb-2">Phase 12 — Literature evidence retrieval</h2>
  <p class="text-sm text-slate-600 mb-4">
    SQLite FTS5 + TF-IDF hybrid retrieval over local PDFs, joined to Neo4j for context.
    This is a retrieval prototype, not a clinical decision-support system.
  </p>

  <div class="bg-white border border-slate-200 rounded p-4 mb-4">
    <div class="flex gap-2 mb-3">
      <input id="ret-q" placeholder="e.g. What laboratory monitoring is recommended for apixaban with reduced renal function?" class="border border-slate-300 rounded px-3 py-2 text-sm flex-1">
      <button onclick="askRetrieval()" class="bg-blue-700 text-white rounded px-4 py-2 text-sm hover:bg-blue-800">Ask</button>
    </div>
    <div id="ret-status" class="text-xs text-slate-500">Answer will appear below.</div>
  </div>

  <div id="ret-answer" class="bg-white border border-slate-200 rounded p-4 mb-4 hidden">
    <h3 class="font-semibold mb-2">Answer</h3>
    <pre id="ret-answer-body" class="text-sm"></pre>
  </div>

  <div id="ret-cites" class="bg-white border border-slate-200 rounded p-4 hidden">
    <h3 class="font-semibold mb-2">Citations</h3>
    <div id="ret-cites-body" class="space-y-3"></div>
  </div>
</section>

</main>

<footer class="max-w-7xl mx-auto px-6 py-8 text-xs text-slate-500">
  Phase 9 data provenance:
  <code>results/phase8_phase9_v7_audit/20261018_130202_315654</code>
  produced by <code>improve_v6_v7.py</code> V7.5, code version 7.5.0.
  &middot;
  <a href="/docs" class="text-blue-700 underline">Swagger UI</a>
  &middot;
  <a href="/api/system/capabilities" class="text-blue-700 underline">Capability map (JSON)</a>
</footer>

<script>
const tabs = document.querySelectorAll('.tab');
const contents = document.querySelectorAll('.tab-content');
const loaded = {};

function showTab(name) {
  tabs.forEach(t => t.classList.toggle('tab-active', t.dataset.tab === name));
  contents.forEach(c => c.classList.toggle('hidden', c.id !== 'tab-' + name));
  if (!loaded[name]) {
    loaded[name] = true;
    ({capabilities: loadCapabilities, optimizer: loadOptimizer,
      graph: loadGraph, monitoring: loadMonitoring,
      retrieval: () => {}})[name]?.();
  }
}

async function getJson(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

// --- Capabilities ---
async function loadCapabilities() {
  try {
    const data = await getJson('/api/system/capabilities');
    const rows = Object.entries(data.phases).map(([name, v]) => `
      <tr class="border-b border-slate-100">
        <td class="py-2 pr-4 font-mono text-xs">${name}</td>
        <td class="py-2 pr-4">
          <span class="px-2 py-1 rounded text-xs ${badgeClass(v.status)}">${v.status}</span>
        </td>
        <td class="py-2 text-xs">${v.evidence ?? ''}</td>
      </tr>`).join('');
    document.getElementById('capabilities-body').innerHTML = `
      <table class="w-full">
        <thead><tr class="border-b border-slate-200 text-left">
          <th class="py-2 pr-4">Phase</th><th class="py-2 pr-4">Status</th><th class="py-2">Evidence</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <div class="mt-4 text-xs text-slate-500">
        ${data.honest_notes.map(n => '• ' + n).join('<br>')}
      </div>`;
  } catch (e) {
    document.getElementById('capabilities-body').innerHTML =
      `<div class="text-red-700">${e.message}</div>`;
  }
}
function badgeClass(s) {
  if (s.includes('verified')) return 'bg-green-100 text-green-800';
  if (s.includes('built'))     return 'bg-blue-100 text-blue-800';
  if (s.includes('prototype')) return 'bg-amber-100 text-amber-800';
  if (s.includes('not_'))      return 'bg-slate-200 text-slate-700';
  return 'bg-slate-100 text-slate-700';
}

// --- Optimizer ---
let chartMethod, chartSweep;
async function loadOptimizer() {
  try {
    const [summary, sweep, resets] = await Promise.all([
      getJson('/api/optimizer/summary'),
      getJson('/api/optimizer/budget-sweep'),
      getJson('/api/optimizer/reset-counters'),
    ]);

    const within = summary.method_summary.filter(r => r.mode === 'within_template');
    const unseen = summary.method_summary.filter(r => r.mode === 'unseen_template');
    const withinBeam = within.find(r => r.method === 'beam');
    const unseenBeam = unseen.find(r => r.method === 'beam');

    document.getElementById('opt-within').textContent =
      withinBeam ? withinBeam.mean.toFixed(2) + '%' : '—';
    document.getElementById('opt-unseen').textContent =
      unseenBeam ? unseenBeam.mean.toFixed(4) + '%' : '—';
    document.getElementById('opt-resets').textContent =
      resets.all_positive ? '✓ all > 0' : '✗ see detail';

    const labels = summary.method_summary.map(r => r.mode.replace('_', ' ') + ' / ' + r.method);
    const values = summary.method_summary.map(r => r.mean);

    if (chartMethod) chartMethod.destroy();
    chartMethod = new Chart(document.getElementById('chart-method'), {
      type: 'bar',
      data: { labels, datasets: [{
        label: 'Held-out weighted gain (%)',
        data: values,
        backgroundColor: labels.map(l => l.includes('unseen') ? '#ef4444' : '#2563eb'),
      }]},
      options: { responsive: true, plugins: { legend: { display: false } } },
    });

    const sw = sweep.budget_sweep;
    const withinSweep = sw.filter(r => r.mode === 'within_template');
    const unseenSweep = sw.filter(r => r.mode === 'unseen_template');

    if (chartSweep) chartSweep.destroy();
    chartSweep = new Chart(document.getElementById('chart-sweep'), {
      type: 'line',
      data: {
        labels: withinSweep.map(r => r.budget),
        datasets: [
          { label: 'Within-template', data: withinSweep.map(r => r.beam_test_gain),
            borderColor: '#2563eb', backgroundColor: 'rgba(37,99,235,0.1)', tension: 0.3 },
          { label: 'Unseen-template', data: unseenSweep.map(r => r.beam_test_gain),
            borderColor: '#ef4444', backgroundColor: 'rgba(239,68,68,0.1)', tension: 0.3 },
        ],
      },
      options: { responsive: true, scales: { x: { title: { display: true, text: 'Index budget' } },
                                          y: { title: { display: true, text: 'Held-out gain (%)' } } } },
    });
  } catch (e) {
    document.getElementById('opt-within').textContent = 'err';
    console.error(e);
  }
}

// --- Graph ---
async function loadGraph() {
  try {
    const stats = await getJson('/api/graph/stats');
    document.getElementById('graph-nodes').innerHTML =
      renderCounts(stats.nodes, 'label');
    document.getElementById('graph-rels').innerHTML =
      renderCounts(stats.relationships, 'type');

    const patients = await getJson('/api/graph/patients?limit=1');
    if (patients.patients.length) {
      document.getElementById('graph-pid').value = patients.patients[0];
    }
  } catch (e) {
    document.getElementById('graph-nodes').textContent = e.message;
  }
}
function renderCounts(rows, key) {
  if (!rows.length) return '<em>No data</em>';
  return rows.map(r => `
    <div class="flex justify-between border-b border-slate-100 py-1">
      <span class="font-mono text-xs">${r[key]}</span>
      <span class="font-semibold">${r.count.toLocaleString()}</span>
    </div>`).join('');
}
async function loadPatient() {
  const pid = document.getElementById('graph-pid').value.trim();
  if (!pid) return;
  document.getElementById('graph-patient').textContent = 'Loading…';
  try {
    const data = await getJson('/api/graph/patient/' + encodeURIComponent(pid));
    const groups = Object.entries(data.groups);
    if (!groups.length) {
      document.getElementById('graph-patient').textContent = 'No relationships found.';
      return;
    }
    document.getElementById('graph-patient').innerHTML = groups.map(([k, items]) => `
      <div class="mb-3">
        <div class="font-semibold text-sm">${k} (${items.length})</div>
        <ul class="list-disc list-inside text-xs text-slate-600">
          ${items.map(i => `<li>${i.label ?? '(no label)'}</li>`).join('')}
        </ul>
      </div>`).join('');
  } catch (e) {
    document.getElementById('graph-patient').innerHTML =
      `<div class="text-red-700">${e.message}</div>`;
  }
}

// --- Monitoring ---
async function loadMonitoring() {
  try {
    const s = await getJson('/api/monitoring/summary');
    document.getElementById('mon-total').textContent = (s.total_executions || 0).toLocaleString();
    document.getElementById('mon-mean').textContent = s.mean_execution_ms ? s.mean_execution_ms.toFixed(2) : '—';
    document.getElementById('mon-seq').textContent = (s.seq_scans || 0).toLocaleString();
    document.getElementById('mon-idx').textContent = (s.index_scans || 0).toLocaleString();

    const recent = await getJson('/api/monitoring/recent?limit=10');
    document.getElementById('mon-recent').innerHTML = renderTable(recent.rows);

    const top = await getJson('/api/monitoring/top-fingerprints?limit=10');
    document.getElementById('mon-top').innerHTML = renderTable(top.fingerprints);
  } catch (e) {
    document.getElementById('mon-total').textContent = 'err';
    console.error(e);
  }
}
function renderTable(rows) {
  if (!rows.length) return '<tr><td>No rows</td></tr>';
  const cols = Object.keys(rows[0]);
  const head = '<tr class="border-b border-slate-200 text-left">' +
               cols.map(c => `<th class="py-2 px-2 text-xs">${c}</th>`).join('') + '</tr>';
  const body = rows.map(r =>
    '<tr class="border-b border-slate-100">' +
    cols.map(c => `<td class="py-1 px-2 text-xs">${r[c] ?? ''}</td>`).join('') +
    '</tr>').join('');
  return head + body;
}

// --- Retrieval ---
async function askRetrieval() {
  const q = document.getElementById('ret-q').value.trim();
  if (!q) return;
  document.getElementById('ret-status').textContent = 'Retrieving…';
  document.getElementById('ret-answer').classList.add('hidden');
  document.getElementById('ret-cites').classList.add('hidden');
  try {
    const data = await getJson('/api/retrieval/ask?q=' + encodeURIComponent(q));
    document.getElementById('ret-status').textContent =
      'Status: ' + (data.status || 'unknown');
    document.getElementById('ret-answer-body').textContent = data.answer || '(no answer)';
    document.getElementById('ret-answer').classList.remove('hidden');

    const cites = data.citations || [];
    if (cites.length) {
      document.getElementById('ret-cites-body').innerHTML = cites.map(c => `
        <div class="border border-slate-200 rounded p-3">
          <div class="text-xs text-slate-500">${c.id} · ${c.category} · page ${c.page ?? '?'}</div>
          <div class="font-semibold text-sm mt-1">${escapeHtml(c.title || c.filename || 'Untitled')}</div>
          <div class="text-xs text-slate-600 mt-1">${escapeHtml((c.excerpt || '').slice(0, 300))}…</div>
          <div class="text-xs text-slate-400 mt-1 font-mono">${escapeHtml(c.source_path || '')}</div>
        </div>`).join('');
      document.getElementById('ret-cites').classList.remove('hidden');
    }
  } catch (e) {
    document.getElementById('ret-status').innerHTML =
      `<span class="text-red-700">${escapeHtml(e.message)}</span>`;
  }
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, m => ({
    '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
  }[m]));
}

// --- initial ---
showTab('capabilities');
</script>

</body>
</html>
"""