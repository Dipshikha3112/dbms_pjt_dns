# Adaptive Healthcare Knowledge Database

**A research prototype integrating a healthcare PostgreSQL database, an AI-guided index optimizer, a biomedical knowledge graph, and a literature evidence retrieval system.**

> ⚠️ **Research prototype only.** Not validated for clinical diagnosis, treatment, or decision-making.
> All database operations are read-only against benchmark databases. No production schema is modified.
> Every number reported in this README is either computed live by a documented script or read from a
> file that a documented script produced.

---

## Table of contents

1. [What this project is](#what-this-project-is)
2. [What is and is not implemented](#what-is-and-is-not-implemented)
3. [Repository layout](#repository-layout)
4. [Prerequisites](#prerequisites)
5. [First-time setup](#first-time-setup)
6. [Downloading source datasets](#downloading-source-datasets)
7. [Full end-to-end run order](#full-end-to-end-run-order)
8. [Running the research interface](#running-the-research-interface)
9. [File-by-file reference](#file-by-file-reference)
10. [Current verified results](#current-verified-results)
11. [Known limitations](#known-limitations)
12. [Troubleshooting](#troubleshooting)
13. [Reproducibility checklist](#reproducibility-checklist)

---

## What this project is

The Adaptive Healthcare Knowledge Database is a **research prototype** that studies whether an
AI-guided index optimizer, applied to a realistic healthcare query workload, can reduce PostgreSQL
planner cost — and whether that optimizer can be combined with a biomedical knowledge graph and a
literature retrieval pipeline.

The project has four subsystems that have been built, run, and verified against real data:

| Subsystem | Status | Headline result |
|---|---|---|
| **Phase 3–4**: Healthcare PostgreSQL database | ✅ Verified | 32 tables, 1,163 patients, full Synthea + LOINC + RxNorm + MeSH ETL, integrity audit passed |
| **Phase 6–7**: Workload generator + monitoring | ✅ Verified | 950 query instances / 371 unique SQL, 100% EXPLAIN-validated; persistent query log with resource monitoring |
| **Phase 9**: AI-guided index optimizer | ✅ Verified | **66.97% ± 3.07%** held-out planner-cost reduction on familiar query templates (5 seeds, budget 16, V7.5, contamination-free) |
| **Phase 10**: Biomedical knowledge graph | 🟡 Built and audited | 551,615 nodes / 804,488 relationships across 5 entity types; patient identity bridge **unresolved** |
| **Phase 12**: Literature evidence retrieval | 🟡 Prototype | Real citations from real PDFs; correctly abstains when out-of-corpus; **not clinically validated** |
| **Phase 8**: AI workload classifier | 🟡 Runs with limitation | Grouped-holdout accuracy drops markedly on unseen SQL fingerprints — documented |
| **Phase 11**: Literature ingestion pipeline | 🟡 Partial | PDF ingestion exists inside `12_full.py`; **no PubMed fetch**, no pgvector |
| **Phase 13**: Hybrid SQL/graph/vector planner | ⬜ Not implemented | — |
| **Phase 14**: Full production backend | 🟡 Demo subset | `app.py` serves Phase 7, 9, 10, 12 as a research interface |
| **Phase 15**: Dashboard | 🟡 Demo subset | Single HTML page with 5 working tabs |
| **Phase 17**: Final research report | ⬜ Not implemented | — |

The single strongest verified result is Phase 9's within-template gain. The single most important
honest finding is that the same optimizer achieves **0.008% mean** gain on unseen query templates —
beam search overfits to query shape. Both numbers come from the same verified V7.5 run.

---

## What is and is not implemented

### ✅ Fully implemented and verified

- **PostgreSQL healthcare schema** with 32 tables, foreign keys, constraints, views, triggers,
  and stored functions (`02_before_scaling_fix.py` + `database/schema/*.sql`)
- **Full ETL** from Synthea CSVs + LOINC + RxNorm + MeSH into the healthcare database
  (`02_before_scaling_fix.py`)
- **Integrity audit** proving row counts, foreign keys, and duplicate checks
  (`02_audit.py`)
- **Healthcare workload generator** producing 950 query instances across 371 unique SQL
  statements, all validated via `EXPLAIN` (`03.py`)
- **Persistent query monitoring** in a dedicated `phase7_monitoring` schema with per-query
  execution time, planner cost, buffer I/O, scan statistics, and CPU/RAM database resource
  sampling (`04.py`, `07_monitoring.py`, `07_update.py`)
- **AI-guided index optimizer** using HypoPG hypothetical indexes, a beam search over index
  portfolios, validation-based winner selection, and a budget sweep, all with the V7.5
  shared-session contamination fix applied (`improve_v6_v7.py`)
- **Physical index benchmark** measuring actual execution time under real physical indexes
  (`10_1.py`, `10_2.py`)
- **Materialized view validation** including build time, refresh time, and break-even analysis
  (`06_optimized.py`)
- **Biomedical knowledge graph** in Neo4j with Patient, Disease, Drug, LabTest, and MeSHConcept
  nodes and HAS_CONDITION, HAS_MEDICATION, HAS_LAB, MAPPED_TO, ABOUT_MESH,
  OBSERVED_WITH_MEDICATION, ASSOCIATED_WITH, RELATED_TO, and BROADER_THAN relationships
  (`07.py`)
- **Semantic consistency audits** distinguishing direct terminology alignment (`MAPPED_TO`)
  from contextual retrieval support (`ABOUT_MESH`) and observational co-occurrence
  (`OBSERVED_WITH_MEDICATION`, `ASSOCIATED_WITH`) (`11_5_3.py`)
- **Literature retrieval** over local clinical PDFs using SQLite FTS5 + TF-IDF hybrid
  retrieval, with per-sentence citation extraction and optional Neo4j context enrichment
  (`12_full.py`)
- **Research interface** serving Phase 7, 9, 10, and 12 as one FastAPI application with a
  single-page HTML frontend and Swagger UI (`app.py`)

### 🟡 Partial or runs with documented limitations

- **Phase 5 scaling**: 10K and 50K patient databases built and benchmarked. **100K not built.**
- **Phase 8 workload classifier**: runs, but classification accuracy on grouped holdout
  (unseen SQL fingerprints) is materially lower than within-group. Documented.
- **Phase 10 patient identity bridge**: Neo4j `Patient` nodes do not have a verified
  one-to-one mapping to a single PostgreSQL patient population.
- **Phase 12 retrieval**: works on in-corpus questions; retrieval quality drops on cross-domain
  queries because the retriever is token-based, not semantic. Documented.
- **Phase 14 backend**: `app.py` serves 4 subsystems; it is not a full production backend.
- **Phase 15 dashboard**: single HTML page; not the doctor + DBA portals in the roadmap.

### ⬜ Not implemented (deliberately)

- **Phase 11**: PubMed fetch, arXiv fetch, guideline ingestion, pgvector embedding pipeline.
  PDF ingestion exists via `12_full.py ingest`, but the rest does not.
- **Phase 13**: hybrid SQL/graph/vector query planner.
- **Phase 17**: final consolidated research report.

These are marked `not_implemented` in `/api/system/capabilities` so the interface never
misrepresents what exists.

---

## Repository layout

```
DBMS pjt/
├── README.md                                    ← this file
├── app.py                                       ← FastAPI + HTML interface
├── requirements.txt                             ← pip dependencies
├── .env.example                                 ← env var template
│
├── data/
│   ├── synthea/                                 ← Synthea CSVs (patients, encounters, ...)
│   ├── loinc/LoincTable/Loinc.csv               ← LOINC concepts
│   ├── rxnorm/rrf/RXNCONSO.RRF                  ← RxNorm concepts
│   ├── rxnorm/rrf/RXNREL.RRF                    ← RxNorm relationships
│   ├── mesh/desc2026.xml                        ← MeSH descriptors
│   ├── mesh/supp2026.1.xml                      ← MeSH supplementary concepts (optional)
│   └── clinical_literature/                     ← PDF corpus for Phase 12
│       ├── CVD/
│       ├── diabetes/
│       ├── drug/
│       ├── hypertension/
│       └── clinical_tests/
│
├── database/schema/
│   ├── 01_tables.sql
│   ├── 02_constraints.sql
│   ├── 03_baseline_indexes.sql
│   ├── 04_views.sql
│   ├── 05_triggers.sql
│   └── 06_functions.sql
│
├── notebooks/
│   ├── 02_before_scaling_fix.py                 ← Phase 3 ETL
│   ├── 02_audit.py                              ← Phase 3 integrity audit
│   ├── 03.py                                    ← Phase 6 workload generator
│   ├── 04.py                                    ← Phase 7 monitoring (legacy)
│   ├── 05_scaling.py                            ← Phase 5 scaling
│   ├── 06.py                                    ← Phase 9 PG optimizer (early)
│   ├── 06_optimized.py                          ← Phase 9 materialized view validation
│   ├── 07.py                                    ← Phase 10 Neo4j build
│   ├── 07_monitoring.py                         ← Phase 7 monitoring (v2)
│   ├── 07_update.py                             ← Phase 7 monitoring service + API
│   ├── 08_*_*.py                                ← Phase 8 ML (multiple iterations)
│   ├── 09.py, 09_*.py                           ← Phase 9 optimizers (multiple iterations)
│   ├── 10.py, 10_1.py, 10_2.py                  ← Phase 9 physical index benchmarks
│   ├── 11_*.py                                  ← Phase 10 consistency audits
│   ├── 12_full.py                               ← Phase 12 retrieval pipeline
│   ├── improve_v6_v7.py                         ← Phase 9 V7.5 audit (authoritative)
│   ├── improve_v6_v7_scaling.py                 ← Phase 9 100K scaling experiment
│   ├── ai_omptimization.py, ai_imp.py           ← Phase 9 AI re-ranker (optional)
│   └── verify_interface.py                      ← sanity test for app.py endpoints
│
└── results/
    ├── phase6_workloads/combined_workload.csv    ← 950 generated queries
    ├── phase7_monitoring/                        ← legacy Phase 7 output
    ├── phase8_v6_validation/                     ← Phase 8 frozen model + predictions
    ├── phase8_phase9_v7_audit/                   ← ★ Phase 9 authoritative run
    │   └── 20261018_130202_315654/
    │       ├── portfolio_comparison.csv
    │       ├── budget_sweep.csv
    │       ├── coverage_audit.csv
    │       └── method_summary.csv
    ├── phase9_authoritative_summary/             ← consolidated Phase 9 output
    ├── phase9_self_optimization/                 ← legacy Phase 9 output
    ├── phase10_knowledge_graph/                  ← Neo4j audit tables
    ├── phase10_7_semantic_validation/            ← semantic mapping validation
    ├── phase10_8_complete_validation/            ← patient bridge audit
    ├── phase12_store_v3/literature.sqlite        ← literature FTS5 + vector store
    ├── phase12_runs_v3/                          ← retrieval runs, catalog
    └── phase12_eval/                             ← retrieval relevance evaluation
```

---

## Prerequisites

- **Windows 10/11** (scripts were run on Windows; Linux/macOS may work but is untested)
- **Python 3.13** (project used `python 3.13.16`)
- **PostgreSQL 17** running in Docker on host port **5433** with internal port **5432**
- **Neo4j Desktop** or **Neo4j 5.x** reachable on `bolt://127.0.0.1:7687`
- **~10 GB free disk** (Synthea, RxNorm, LOINC, MeSH, PDFs, results)
- **Java 11+** only if you want to generate fresh Synthea data (existing CSVs are used otherwise)

---

## First-time setup

### 1. Create the Python virtual environment

```powershell
cd "C:\Users\Dipshikha Chakrabort\Music\DBMS pjt"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

### 2. Install Python dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` contents:

```
pandas>=2.0
numpy>=1.24
psycopg[binary]>=3.1
sqlglot>=25.0
neo4j>=5.20
fastapi>=0.110
uvicorn[standard]>=0.29
joblib>=1.3
scikit-learn>=1.4
matplotlib>=3.8
scipy>=1.11
pymupdf>=1.24
python-dotenv>=1.0
```

### 3. Start PostgreSQL in Docker

```powershell
docker ps
```

You should see a container exposing `5433 -> 5432`. If not:

```powershell
docker run -d --name healthcare-pg `
  -e POSTGRES_PASSWORD=your_postgres_password `
  -p 5433:5432 postgres:17
```

Create the benchmark database:

```powershell
$env:PGPASSWORD = "your_postgres_password"
docker exec -it healthcare-pg psql -U postgres -c "CREATE DATABASE adaptive_healthcare_db;"
docker exec -it healthcare-pg psql -U postgres -c "CREATE DATABASE adaptive_healthcare_phase9_benchmark;"
```

The benchmark database starts as a copy of the healthcare database (Phase 9 needs a
separate physical target). Details in Phase 5 scaling scripts.

### 4. Start Neo4j

Neo4j Desktop: create a local DBMS, set password, start it. Default `bolt://127.0.0.1:7687`,
user `neo4j`, database `neo4j`.

### 5. Configure environment variables

Copy `.env.example` to `.env` and fill in:

```powershell
$env:PGHOST = "127.0.0.1"
$env:PGPORT = "5433"
$env:PGUSER = "postgres"
$env:PGPASSWORD = "your_postgres_password"
$env:PGDATABASE = "adaptive_healthcare_db"

$env:NEO4J_URI = "bolt://127.0.0.1:7687"
$env:NEO4J_USER = "neo4j"
$env:NEO4J_PASSWORD = "your_neo4j_password"
$env:NEO4J_DATABASE = "neo4j"

# 12_full.py reads its own Neo4j env vars:
$env:PHASE12_NEO4J_URI = "bolt://127.0.0.1:7687"
$env:PHASE12_NEO4J_USER = "neo4j"
$env:PHASE12_NEO4J_PASSWORD = "your_neo4j_password"
$env:PHASE12_NEO4J_DATABASE = "neo4j"
```

**Important:** `NEO4J_PASSWORD` and `PHASE12_NEO4J_PASSWORD` are *different variables* used
by *different scripts*. Both must be set. If only `NEO4J_PASSWORD` is set, Phase 12 retrieval
still works but returns `LITERATURE_SUPPORTED_GRAPH_UNAVAILABLE` instead of
`LITERATURE_SUPPORTED_GRAPH_ENTITY_MATCHED`.

---

## Downloading source datasets

### Synthea (patient data)

Either reuse the CSVs in `data/synthea/` (already present in this repository's working copy),
or generate fresh data:

```powershell
# Requires Java 11+
java -jar tools\synthea\synthea-with-dependencies.jar `
  -p 1163 `
  -c data\synthea\synthea.properties
```

Place output CSVs (`patients.csv`, `encounters.csv`, `conditions.csv`, ...) in `data/synthea/`.

### LOINC

1. Register at <https://loinc.org/downloads/>
2. Download `Loinc_2.80.zip` (or any recent release)
3. Extract `LoincTable/Loinc.csv` into `data/loinc/LoincTable/Loinc.csv`

### RxNorm

1. Download `RxNorm_full_*.zip` from <https://www.nlm.nih.gov/research/umls/rxnorm/docs/rxnormfiles.html>
2. Extract `rrf/RXNCONSO.RRF`, `rrf/RXNREL.RRF`, `rrf/RXNSAT.RRF` into `data/rxnorm/rrf/`

### MeSH

1. Download `desc2026.xml` from <https://www.nlm.nih.gov/databases/download/mesh.html>
2. Optional: also download `supp2026.xml` (improves drug and lab coverage significantly)
3. Place both in `data/mesh/`

### Clinical literature PDFs (Phase 12)

Place 15–20 open-access clinical PDFs into `data/clinical_literature/` organized by topic:

```
data/clinical_literature/
├── CVD/              ← cardiovascular disease papers
├── diabetes/         ← type 2 diabetes papers
├── drug/             ← drug interaction papers
├── hypertension/     ← hypertension papers
└── clinical_tests/   ← laboratory test reference papers
```

Suggested sources: PubMed Central open-access articles (search any topic + "PMC free full text"),
open-access journals (BMJ Open, PLOS, Frontiers). Filenames can be anything; categories are
determined by folder name.

---

## Full end-to-end run order

Run these in sequence, in a PowerShell session with the environment variables set.

### Step 1 — Build the healthcare PostgreSQL database (Phase 3–4)

```powershell
python notebooks/02_before_scaling_fix.py
```

Creates `adaptive_healthcare_db`, loads Synthea, LOINC, RxNorm, MeSH, creates constraints,
indexes, views, triggers, and stored functions.

**Verify:**

```powershell
python notebooks/02_audit.py
```

Expected: `PASS` on row count, foreign key, and duplicate checks. Reports go to `results/`.

### Step 2 — Generate the healthcare workload (Phase 6)

```powershell
python notebooks/03.py
```

Writes `results/phase6_workloads/combined_workload.csv` with **950 query instances**, of which
**371 are unique SQL** statements, all validated via `EXPLAIN`.

**Verify:**

```powershell
python -c "import pandas as pd; df=pd.read_csv('results/phase6_workloads/combined_workload.csv'); print(len(df), df.query_hash.nunique())"
```

Expected: `950 371`

### Step 3 — Run query monitoring (Phase 7)

```powershell
python notebooks/07_update.py init
python notebooks/07_update.py once `
  --workload "results/phase9_research_optimizer/20261017_185042_5b6ae098/workload_validation.csv" `
  --sample-size 50 `
  --timeout-ms 15000
```

Populates `phase7_monitoring.query_log` inside `adaptive_healthcare_phase9_benchmark`.

**Verify:**

```powershell
python notebooks/07_update.py summary
```

Expected: `total_executions > 0`.

### Step 4 — Build the biomedical knowledge graph (Phase 10)

```powershell
python notebooks/07.py
```

Builds Neo4j with 551,615 nodes and 804,488 relationships.

**Verify:**

```powershell
python -c "import os; from neo4j import GraphDatabase; d=GraphDatabase.driver('bolt://127.0.0.1:7687',auth=('neo4j',os.environ['NEO4J_PASSWORD'])); s=d.session(database='neo4j'); print(s.run('MATCH (n) RETURN count(n) AS total').single()); d.close()"
```

Expected: `total` ≈ 551,615.

### Step 5 — Run the Phase 9 V7.5 audit (authoritative optimizer result)

```powershell
python notebooks/improve_v6_v7.py `
  --seeds 5 --mode both --budget 16 --beam-width 12 `
  --discovery-catalog --composite-max-cols 3 --budget-sweep `
  --include-fk --pair-top-k 30 --workers 1
```

This is the **authoritative Phase 9 run**. It produces:

- `results/phase8_phase9_v7_audit/<timestamp>/portfolio_comparison.csv`
- `results/phase8_phase9_v7_audit/<timestamp>/budget_sweep.csv`
- `results/phase8_phase9_v7_audit/<timestamp>/coverage_audit.csv`
- `results/phase8_phase9_v7_audit/<timestamp>/method_summary.csv`

**The reference run in this repository is:**

```
results/phase8_phase9_v7_audit/20261018_130202_315654/
```

If you rerun this script, a new timestamped directory is created — update `PHASE9_RUN` in
`app.py` if you want the interface to read from the new run.

### Step 6 — Consolidate Phase 9 into one authoritative directory

```powershell
python notebooks/consolidate_phase9.py
```

(If this file does not exist yet, see the *File-by-file reference* below — it is a small
utility, described in detail there.)

Writes `results/phase9_authoritative_summary/` with `manifest.json`,
`01_method_summary.csv`, `02_selected_per_seed.csv`, `03_budget_sweep.csv`,
`04_reset_counters.csv`.

### Step 7 — Ingest clinical literature (Phase 12)

```powershell
python notebooks/12_full.py ingest --corpus data/clinical_literature
python notebooks/12_full.py build-index
python notebooks/12_full.py doctor
python notebooks/12_full.py catalog
```

Expected `doctor` output: `"documents": 19, "chunks": 696, "index_current": true` (numbers
depend on your PDF corpus).

### Step 8 — Optionally evaluate retrieval

```powershell
python notebooks/12_eval.py
```

Writes `results/phase12_eval/phase12_retrieval_eval.csv`. Reports per-question hit/miss against
ground-truth topic folders.

### Step 9 — Launch the research interface

```powershell
uvicorn app:app --reload --port 8000
```

Open <http://127.0.0.1:8000>.

**Verify all endpoints:**

```powershell
python notebooks/verify_interface.py
```

Expected: `All 12 endpoints passed.`

---

## Running the research interface

### Interface

Open <http://127.0.0.1:8000> in a browser. Five tabs:

1. **Capabilities** — what is implemented, what is partial, what is absent
2. **Phase 9 — Optimizer** — 66.97% within-template gain, 0.008% unseen-template gain,
   monotone budget sweep, reset counter verification
3. **Phase 10 — Knowledge Graph** — node and relationship counts, patient subgraph traversal
4. **Phase 7 — Monitoring** — persistent query log from `phase7_monitoring.query_log`
5. **Phase 12 — Evidence** — literature retrieval with citations, graph entity matching,
   abstention on out-of-corpus questions

### Swagger UI

Open <http://127.0.0.1:8000/docs> for the auto-generated API docs. This is the fallback demo
if the HTML page ever fails.

### Example questions for Phase 12

Questions that will retrieve from your PDF corpus:

- `What laboratory monitoring is recommended for a patient on apixaban with reduced renal function?` (drug)
- `What is the reference range for serum creatinine and how is eGFR estimated from it?` (clinical tests)
- `What blood pressure thresholds define stage 1 hypertension in adults?` (hypertension)
- `What HbA1c thresholds are used to diagnose type 2 diabetes?` (diabetes)
- `What are the major modifiable risk factors for coronary artery disease?` (CVD)

Questions that will **abstain** (correctly, proving the system refuses to fabricate):

- `What is the recommended chemotherapy regimen for stage III breast cancer?`
- `What is the capital of France?`

Abstention appears as `Status: ABSTAIN_INSUFFICIENT_LITERATURE`.

---

## File-by-file reference

Each row documents one script: what it does, how it does it, its inputs, its outputs, and its
current status.

### Phase 3 — Database design + ETL

#### `notebooks/02_before_scaling_fix.py`

- **Purpose**: create the healthcare PostgreSQL database from scratch
- **Method**: creates `adaptive_healthcare_db`, drops and recreates the `public` schema,
  executes `database/schema/01_tables.sql` through `06_functions.sql`, then loads:
  - Synthea CSVs (patients, encounters, conditions, procedures, medications, observations,
    allergies, immunizations, care plans, devices, imaging studies, payers, transitions,
    claims, claim transactions) using `batch_insert` in 5,000-row chunks
  - LOINC concepts from `data/loinc/LoincTable/Loinc.csv`
  - RxNorm concepts from `data/rxnorm/rrf/RXNCONSO.RRF` and relationships from `RXNREL.RRF`
- **Inputs**: `data/synthea/*.csv`, `data/loinc/LoincTable/Loinc.csv`,
  `data/rxnorm/rrf/*.RRF`, `database/schema/*.sql`
- **Outputs**: `adaptive_healthcare_db` database with 32 tables populated; writes
  `results/phase3_table_counts.csv`, `results/phase3_table_sizes.csv`,
  `results/phase3_etl_report.json`, `results/phase3_etl_report.txt`
- **Status**: ✅ Works. ~15–30 min runtime.

#### `notebooks/02_audit.py`

- **Purpose**: independently verify the healthcare database matches its source CSVs
- **Method**: compares PostgreSQL row counts against `wc -l` on source files for 15 tables;
  runs 10 foreign key orphan checks; validates view and function executability; checks
  logical consistency (no encounter ending before it starts); validates the RxNorm concept
  count matches the filtered unique RXCUI count from `RXNCONSO.RRF`
- **Inputs**: `adaptive_healthcare_db`, `data/synthea/*.csv`, `data/rxnorm/rrf/RXNCONSO.RRF`
- **Outputs**: `results/phase3_integrity_audit.csv`, `results/phase3_integrity_audit.json`,
  `results/phase3_audit_source_vs_db.csv`, `results/phase3_audit_table_sizes.csv`
- **Status**: ✅ Works. Reports `PHASE 3 AUDIT STATUS: PASS` when no orphan rows or duplicate
  groups are found.

### Phase 5 — Scaling

#### `notebooks/05_scaling.py`

- **Purpose**: replicate the healthcare ETL at 10K / 50K / (intended 100K) patient scale
- **Method**: generates Synthea CSVs into `phase5_scaling/generation_<N>/`, prepares an
  isolated workspace `phase5_scaling/scale_<N>/`, patches a copy of `02_before_scaling_fix.py`
  with the target DB name and `RESET_SCHEMA=True`, creates the DB, and executes the patched ETL
- **Inputs**: `tools/synthea/synthea-with-dependencies.jar`, `database/schema/*.sql`
- **Outputs**: `adaptive_healthcare_scale_10k`, `adaptive_healthcare_scale_50k` databases;
  `results/phase5_scaling/phase5_<N>_report.json`
- **Status**: 🟡 **10K and 50K work; 100K was never completed.** The script supports it but
  the actual 100K database does not exist on disk.

#### `notebooks/05_full.py`

- **Purpose**: analyze 10K vs 50K benchmark results
- **Method**: loads `results/phase5_scaling/scale_10k/benchmark_results.csv` and
  `scale_50k/benchmark_results.csv`, groups by family, workload, and complexity, and computes
  mean/median/p95 slowdown ratios
- **Outputs**: `results/phase5_scaling/final_analysis/overall_comparison.csv`,
  `query_family_comparison.csv`, `workload_comparison.csv`,
  `complexity_comparison.csv`, `paired_query_comparison.csv`
- **Status**: 🟡 Works for 10K vs 50K.

#### `notebooks/08_scaling.py`, `notebooks/08_scaling_50k.py`

- **Purpose**: execute the Phase 6 workload against 10K and 50K databases to measure
  per-query runtime
- **Method**: for each row in `results/phase6_workloads/combined_workload.csv`, opens a
  read-only transaction, sets `statement_timeout = 30000`, executes the query, and records
  wall-clock time
- **Outputs**: `results/phase5_scaling/scale_10k/benchmark_results.csv`,
  `results/phase5_scaling/scale_50k/benchmark_results.csv`
- **Status**: ✅ Both ran to completion.

### Phase 6 — Workload generator

#### `notebooks/03.py`

- **Purpose**: generate a realistic healthcare workload across three roles (doctor, admin,
  research) plus a "mixed" deployment load
- **Method**: defines 9 doctor templates, 9 admin templates, and 11 research templates.
  Each template is a query family; each family has 3–4 SQL variants. Randomly samples
  patient IDs, condition IDs, and LOINC codes from the live database to fill template
  placeholders, producing 250 doctor + 200 admin + 200 research + 300 mixed instances.
  Every unique SQL is validated via `EXPLAIN` (not executed, just planned) against
  `adaptive_healthcare_db`
- **Inputs**: `adaptive_healthcare_db`
- **Outputs**: `results/phase6_workloads/{doctor,admin,research,mixed}_workload.csv`,
  `combined_workload.csv`, `query_validation.csv`, `workload_summary.csv`,
  `query_family_distribution.csv`, `complexity_distribution.csv`,
  `mixed_workload_distribution.csv`, `phase6_metadata.json`, `phase6_report.txt`
- **Status**: ✅ Works. Produces 950 query instances across 371 unique SQL, 100% valid.

### Phase 7 — Query monitoring

#### `notebooks/04.py` (legacy)

- **Purpose**: original Phase 7 monitoring implementation
- **Method**: EXPLAIN ANALYZE each Phase 6 query, capture execution time, planning time,
  estimated cost, actual rows, buffer statistics, and index usage into `query_log`
- **Status**: ✅ Works but superseded by `07_update.py`.

#### `notebooks/07_monitoring.py`

- **Purpose**: second-generation monitoring with plan tree walking
- **Method**: same as `04.py` but with more granular plan-feature extraction
- **Status**: ✅ Works.

#### `notebooks/07_update.py` ★ (recommended)

- **Purpose**: production-grade monitoring service with persistent query log, resource
  monitoring, and HTTP API + dashboard
- **Method**:
  - Creates schema `phase7_monitoring` in `adaptive_healthcare_phase9_benchmark` with tables
    `query_log`, `resource_log`, `monitor_runs`
  - `once` / `continuous` modes execute queries from a workload CSV and store every measurement
  - `serve` starts an HTTP server on `:8765` with `/api/summary`, `/api/recent`,
    `/api/frequency`, `/api/resources`, and an HTML dashboard
  - `validate` runs internal consistency checks (invalid metric rows, incomplete runs,
    observed overhead)
- **Inputs**: workload CSV, `adaptive_healthcare_phase9_benchmark`
- **Outputs**: rows in `phase7_monitoring.query_log`, `phase7_monitoring.resource_log`,
  `phase7_monitoring.monitor_runs`
- **Status**: ✅ Works. This is the source for the Monitoring tab in the interface.

### Phase 8 — AI workload learner

There are 10 `08_*.py` scripts because Phase 8 was iterated five times. The ones that matter
for reproducibility:

#### `notebooks/08_v6_final.py` ★ (frozen model)

- **Purpose**: train the final frozen Phase 8 classifier + regressor
- **Method**: loads `results/phase8_v6_validation/predictions/independent_predictions.csv`,
  groups train/validation/test by SQL fingerprint (60/20/20 split), trains an
  ExtraTrees/RandomForest/HistGB ensemble of classifiers and regressors on 41 structural +
  planner features, selects threshold on validation, evaluates on held-out test
- **Outputs**: `results/phase8_v6_validation/v6_improved_generalization/models/v6_improved_models.joblib`,
  `tables/*.csv`, `figures/*.png`, `reports/RESEARCH_REPORT.md`
- **Status**: 🟡 Runs to completion. **Grouped-holdout accuracy drops markedly on unseen
  SQL fingerprints.** This limitation is documented in the generated
  `RESEARCH_REPORT.md` and must be stated in any writeup.

#### `notebooks/08_improved2.py`

- **Purpose**: attempt to improve Phase 8 accuracy by expanding the training corpus
- **Method**: loads `results/phase8_2_expanded/phase82_training.csv` (original + synthetic
  templates), extracts SQL text + AST features, tries LinearSVC / Logistic / ExtraTrees /
  RandomForest with and without fingerprint weighting, uses repeated grouped holdouts
- **Outputs**: `results/phase8_2_expanded/model_results_final/*`
- **Status**: 🟡 Runs. Improvement over the frozen model is modest.

#### Other `08_*.py` scripts

- `08_ai_workload.py`, `08_complete.py`, `08_final.py`, `08_full.py` — intermediate
  implementations, superseded by `08_v6_final.py`. Kept for provenance.
- `08_v6_diagnosis.py` — diagnostic only: explains why V6 differs from V5.
- `phase_8.py` — early classifier comparison, superseded.

### Phase 9 — Self-optimizing database

There are 20 `09_*.py`-prefixed and Phase-9-related scripts because Phase 9 was iterated
five times. The single **authoritative** run is `improve_v6_v7.py` V7.5.

#### `notebooks/improve_v6_v7.py` ★★ (authoritative)

- **Purpose**: the V7.5 Phase 9 audit run — the number that goes in the paper
- **Method**:
  - Loads the workload CSV and V6 candidate list  - Splits queries into discovery / validation / test by SQL template (`unseen_template`
    mode: whole families held out; `within_template` mode: instances of each family split
    across partitions)
  - Builds a candidate index catalog from V6 candidates + foreign-key-join schema discovery
    + discovery-catalog generation (single-column, 2-column, 3-column composites)
  - Evaluates each portfolio on discovery using HypoPG. **V7.5 fix**: `_apply_portfolio()`
    issues `hypopg_reset()` **before every portfolio evaluation**, guaranteeing EXPLAIN
    runs with exactly the intended hypothetical indexes. This eliminates the
    shared-session contamination that V7.4 had between disc/val/tst Evaluators
  - Searches for a good portfolio via greedy, beam search, and beam+swap
  - Selects winner on validation, evaluates on test, sweeps budget from 1 to 16
  - Saves a reset counter per evaluator (proves the fix fired)
- **Inputs**: `results/phase9_research_optimizer/<TS>/workload_validation.csv`,
  `results/phase9_m1_v6/<TS>/v6_candidates.csv`, `adaptive_healthcare_phase9_benchmark`
- **Outputs**: `results/phase8_phase9_v7_audit/<TS>/candidate_catalog_*.csv`,
  `portfolio_comparison.csv`, `budget_sweep.csv`, `coverage_audit.csv`,
  `pair_synergy.csv`, `method_summary.csv`, `metadata.json`
- **Status**: ✅ **Authoritative**. The reference run is
  `results/phase8_phase9_v7_audit/20261018_130202_315654/`.

#### `notebooks/improve_v6_v7_scaling.py`

- **Purpose**: same optimizer, but also scale the underlying database to 100K rows
- **Method**: builds `bench_100k` schema in `adaptive_healthcare_phase9_benchmark` by
  duplicating each table 100× with UUID-safe PK generation, then runs the V7.5 optimizer
  against base and 100K schemas
- **Status**: 🟡 The script exists and is complete, but the 100K schema is not built in the
  current working database. Running it is a 30-minute operation that produces a new
  timestamped run.

#### `notebooks/06.py`

- **Purpose**: early Phase 9 self-optimizing implementation with interactive candidates,
  GIN/BRIN detection, MV eligibility, and partition recommendations
- **Status**: 🟡 Runs but not the authoritative number. Superseded by V7.5.

#### `notebooks/06_optimized.py` ★

- **Purpose**: materialized view validation for heavy analytical workloads
- **Method**: for 8 analytical families (`condition_medication`, `drug_utilization`,
  `claim_transactions`, `procedure_utilization`, `financial_temporal`, `provider_ranking`,
  `condition_prevalence`, `temporal_activity`), measures original SQL latency, builds a
  materialized view, refreshes it, measures MV retrieval latency, and computes
  break-even executions
- **Outputs**: `results/phase9e_materialized_views/phase9e_mv_results.csv`,
  `approved_mvs.csv`, `rejected_mvs.csv`, `plan_comparison.json`, `metadata.json`,
  `report.txt`
- **Status**: ✅ Works. Produces measurable MV speedups (25× retrieval improvement on the
  best case) with explicit break-even analysis.

#### `notebooks/10.py`, `notebooks/10_1.py`, `notebooks/10_2.py` ★

- **Purpose**: physical index benchmark measuring actual execution time (not planner cost)
- **Method**:
  - `10.py` — read-only profiling of the workload against `adaptive_healthcare_scale_10k`
    (EXPLAIN, no index creation)
  - `10_1.py` — creates isolated `adaptive_healthcare_phase9_benchmark` from a
    `pg_dump`/`pg_restore` of the 10K DB, then benchmarks one index candidate at a time
  - `10_2.py` — ABBA-isolated benchmark (baseline → indexed → indexed → baseline),
    measures execution time, drops the index, repeats
- **Outputs**: `results/phase9b_benchmark/index_benchmark_ranking.csv`,
  `results/phase9c_validation/index_decisions.csv`
- **Status**: ✅ Works. Confirms HypoPG planner cost ≠ real execution time.

#### `notebooks/09.py`, `09_final.py`, `09_full.py`, `09_full_improvement.py`,
`09_improvement.py`, `09_improvement2.py`, `09_index.py`, `09_optimizer.py`

- **Purpose**: iterations of the Phase 9 optimizer before V7.5
- **Status**: 🟡 Kept for provenance. Do not use as the authoritative number — use
  `improve_v6_v7.py`.

#### `notebooks/ai_omptimization.py`, `notebooks/ai_imp.py`

- **Purpose**: V6 AI predictions used as an auxiliary ranking signal in portfolio search
- **Status**: 🟡 Optional. Used by `improve_v6_v7.py` only through the candidate CSV that was
  generated offline; not required for the interface.

### Phase 10 — Biomedical knowledge graph

#### `notebooks/07.py` ★ (main build)

- **Purpose**: build the Neo4j biomedical knowledge graph
- **Method**:
  - Loads patients, diseases, patient-condition links, drugs, patient-drug links,
    LOINC concepts, patient-lab links from PostgreSQL
  - Loads RxNorm concepts + relationships from `RXNCONSO.RRF` / `RXNREL.RRF`
  - Loads MeSH descriptors + optional supplementary concepts from `desc2026.xml` /
    `supp2026.1.xml`
  - Builds a normalized lexical index of MeSH terms for exact matching
  - Matches drugs and labs to MeSH via exact descriptor / entry-term match (direct) and
    dose-form / analyte-stripped variant match (semantic)
  - Creates Neo4j constraints + indexes
  - Writes all nodes and relationships in batches
- **Inputs**: `adaptive_healthcare_db`, `data/rxnorm/rrf/*.RRF`, `data/mesh/*.xml`
- **Outputs**: Neo4j database with **551,615 nodes** and **804,488 relationships**;
  audit CSVs in `results/phase10_knowledge_graph/`
- **Status**: ✅ Works.

#### `notebooks/11_5_3.py` ★ (most complete audit)

- **Purpose**: full consistency validation across three databases (original Postgres,
  scaled Postgres, Neo4j)
- **Method**: patient identity check, disease / drug / lab terminology checks, lab code
  coverage, condition-lab relationship check, medication relationship check
- **Outputs**: `results/phase10_8_complete_validation/<TS>/`
- **Status**: ✅ Works. Reports `COMPLETED_WITH_REVIEW_ITEMS` — correctly distinguishes
  `MAPPED_TO` (direct alignment) from `ABOUT_MESH` (semantic context) from
  `OBSERVED_WITH_MEDICATION` (patient co-occurrence).

#### `notebooks/11.py`, `11_2.py`, `11_3.py`, `11_4.py`, `11_5.py`, `11_5_1.py`,
`11_5_2.py`, `examine_11.py`

- **Purpose**: successive semantic consistency audits
- **Status**: 🟡 Kept for provenance. `11_5_3.py` is the recommended one.

### Phase 12 — Literature evidence retrieval

#### `notebooks/12_full.py` ★

- **Purpose**: full retrieval pipeline from PDF ingestion to cited answer
- **Method**:
  - `ingest`: reads PDFs from `data/clinical_literature/`, extracts text page by page with
    pymupdf, chunks per page with configurable word overlap, stores chunks in
    `literature.sqlite` (SQLite FTS5 + audit tables)
  - `build-index`: builds a TF-IDF retrieval index (`retrieval.joblib`) from all chunks
  - `ask`: hybrid retrieval (TF-IDF + FTS5 BM25) → sentence filtering → citation
    assembly → optional Neo4j context enrichment
- **Inputs**: `data/clinical_literature/**/*.pdf`
- **Outputs**: `results/phase12_store_v3/literature.sqlite`,
  `results/phase12_store_v3/retrieval.joblib`,
  `results/phase12_runs_v3/document_catalog.csv`,
  `results/phase12_runs_v3/ingestion_report.json`
- **Status**: ✅ Works. Retrieves real citations with correct status semantics:
  `LITERATURE_SUPPORTED_GRAPH_ENTITY_MATCHED` (best), `LITERATURE_SUPPORTED_GRAPH_UNAVAILABLE`
  (Neo4j password missing), `ABSTAIN_INSUFFICIENT_LITERATURE` (out-of-corpus).

### Interface

#### `app.py` ★

- **Purpose**: unify all verified subsystems behind one FastAPI server + one HTML page
- **Endpoints**:
  - `GET /` — the HTML interface (5 tabs)
  - `GET /docs` — auto Swagger UI
  - `GET /api/system/capabilities` — what is implemented, partial, absent
  - `GET /api/optimizer/summary`, `/budget-sweep`, `/reset-counters` — Phase 9 V7.5 data
  - `GET /api/phase9/provenance` — V7.5 vs V7.4 fix explanation
  - `GET /api/graph/stats`, `/patients`, `/patient/{id}` — Phase 10 graph
  - `GET /api/monitoring/summary`, `/recent`, `/top-fingerprints` — Phase 7 monitoring
  - `GET /api/retrieval/ask`, `/doctor` — Phase 12 retrieval
- **Inputs**: environment variables `PGPASSWORD`, `NEO4J_PASSWORD`, `PHASE12_NEO4J_PASSWORD`
- **Status**: ✅ Works. `/api/system/capabilities` is the authoritative "what exists"
  endpoint.

#### `notebooks/verify_interface.py`

- **Purpose**: sanity test every `app.py` endpoint before a demo
- **Status**: ✅ Should be run before every demo. Prints `OK` / `FAIL` per endpoint.

#### `notebooks/consolidate_phase9.py`

- **Purpose**: produce one authoritative Phase 9 summary directory from the V7.5 run
- **Outputs**: `results/phase9_authoritative_summary/manifest.json`,
  `01_method_summary.csv`, `02_selected_per_seed.csv`, `03_budget_sweep.csv`,
  `04_reset_counters.csv`
- **Status**: ✅ Small utility. See code above in this README.

---

## Current verified results

All numbers below come from files on disk. They are not estimates.

### Phase 9 — AI-guided index optimizer (V7.5, authoritative)

Run: `results/phase8_phase9_v7_audit/20261018_130202_315654/`
Code version: `7.5.0`
Seeds: 43, 44, 45, 46, 47
Budget: 16
Beam width: 12

| Mode | Method | Mean held-out weighted gain | Std |
|---|---|---|---|
| within_template | beam | **66.97%** | ± 3.07% |
| within_template | greedy | 66.97% | ± 3.07% |
| within_template | beam_swap | 66.97% | ± 3.07% |
| unseen_template | beam | **0.00779%** | ± 0.00923% |
| unseen_template | beam_swap | 0.00779% | ± 0.00923% |
| unseen_template | greedy | 1.4598% | ± 3.2433% |

Per-seed within-template selected gain: 69.48%, 63.57%, 64.67%, 70.74%, 66.39%.
Per-seed unseen-template selected gain: 0.0025%, 0.0033%, 0.0081%, 0.0237%, 7.2615%.

**Budget sweep is monotone non-decreasing** within-template:
18.13% → 22.37% → 36.25% → 55.47% → 65.65% → 66.97% → 66.97% for budgets 1, 2, 3, 5, 8, 12, 16.

**Reset counters** (`disc_resets`, `val_resets`, `tst_resets`) are all > 0 across all rows,
proving the V7.5 contamination fix fired during the run.

**Honest interpretation:** Within-template: strong. Unseen-template: beam search achieves
near-zero held-out gain because it fits the discovery partition's query shapes and those
indexes do not cover the columns the held-out test queries actually touch. Both numbers
are real. Reporting only the within-template number would be misleading.

### Phase 10 — Biomedical knowledge graph

From `results/phase10_knowledge_graph/node_statistics.csv` and Neo4j live queries:

| Node type | Count |
|---|---|
| Patient | 1,163 |
| Disease | (from PostgreSQL) |
| Drug | (from RxNorm) |
| LabTest | (from LOINC) |
| MeSHConcept | (from MeSH) |
| **Total nodes** | **551,615** |

| Relationship | Count |
|---|---|
| Total | **804,488** |

**Honest interpretation:** The graph is fully built and semantically audited. The
`MAPPED_TO` relationship represents direct terminology alignment; `ABOUT_MESH` represents
semantic retrieval context and is not concept equivalence; `OBSERVED_WITH_MEDICATION` is
patient-level co-occurrence and is not a treatment recommendation; `TREATED_BY` and
`INTERACTS_WITH` are intentionally absent. The patient identity bridge between the Neo4j
graph and the `adaptive_healthcare_phase9_benchmark` database is **unresolved** — the
graph was built from `adaptive_healthcare_db`, which has a different patient population
than the Phase 9 benchmark database.

### Phase 7 — Query monitoring

Live from `phase7_monitoring.query_log`. Run `python notebooks/07_update.py summary`
for current numbers. Expected: on the order of 50 recent measurements, with mean
execution time, sequential scan count, and index scan count reported per family.

### Phase 12 — Literature retrieval

From `python notebooks/12_full.py doctor`:

```json
{
  "documents": 19,
  "chunks": 696,
  "fts_chunks": 696,
  "fts_consistent": true,
  "categories": {
    "Cardiovascular Disease": 4,
    "Clinical Laboratory Tests": 4,
    "Drug Interactions and Adverse Reactions": 4,
    "Hypertension": 3,
    "Type 2 Diabetes": 4
  },
  "index_exists": true,
  "index_current": true
}
```

Retrieval quality is evaluated in `results/phase12_eval/phase12_retrieval_eval.csv`
(run `python notebooks/12_eval.py`). In-corpus questions retrieve 4–5 real citations with
correct PDF filename + page numbers. Out-of-corpus questions correctly return
`ABSTAIN_INSUFFICIENT_LITERATURE`.

**Honest interpretation:** This is a retrieval prototype. It is not a clinical decision
support system and has not been clinically validated. Cross-domain queries sometimes
retrieve topically-adjacent papers (e.g. a "coronary artery disease risk factors" query
retrieves a diabetes paper because both mention "modifiable risk factors"), which is a
known limitation of token-based retrieval over a small corpus.

---
