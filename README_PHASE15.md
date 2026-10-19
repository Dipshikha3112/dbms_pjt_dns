# CareScope Phase 15 — Windows installation and file mapping

## Folder placement

Copy the ZIP contents **into the existing project root**:

`C:\Users\Dipshikha Chakrabort\Music\DBMS pjt\`

```text
DBMS pjt/
├── phase14_full.py                    # Your EXISTING Phase 14 file; keep unchanged
├── notebooks/
│   ├── phase12_graph_rag_v7.py        # Existing or supplied copy; V8 code internally
│   └── phase13_hybrid_query_planner.py # Existing or supplied copy
├── results/
│   ├── phase12_store_v3/literature.sqlite  # EXISTING, don't move
│   ├── phase13/cost_profiles.json          # EXISTING; fallback included
│   └── phase9_benchmark_results.csv        # OPTIONAL if desired; fallback included
├── phase15/
│   ├── main.py                        # New API + app entry point
│   ├── phase14_legacy.py              # Fallback copy if phase14_full.py is missing
│   ├── phase12_graph_rag_v7.py        # Fallback source copy (optional)
│   ├── phase13_hybrid_query_planner.py# Fallback source copy
│   ├── phase13_api.py                 # Reference Phase 13 API (not separately launched)
│   ├── cost_profiles.json            # Fallback historical samples
│   ├── phase9_benchmark_results.csv   # Fallback historical optimizer report
│   ├── static/index.html
│   ├── static/styles.css
│   ├── static/app.js
│   └── tests/test_phase15.py
└── requirements_phase15.txt
```

## Commands (PowerShell in project root)

```powershell
cd 'C:\Users\Dipshikha Chakrabort\Music\DBMS pjt'
.\.venv\Scripts\Activate.ps1
pip install -r requirements_phase15.txt
$env:PHASE15_API_KEY = 'choose-a-long-random-local-secret'
$env:PHASE14_API_KEY = $env:PHASE15_API_KEY
# Use the same existing PostgreSQL/Neo4j environment variables as Phase 14.
# PGHOST, PGPORT, PGUSER, PGPASSWORD, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
python -m unittest discover -s phase15/tests -v
python phase15/main.py
```

Open **http://127.0.0.1:8015/portal** and enter the key in the sidebar.

If you prefer not to type a key at every launch, set it through a private local startup script **outside version control**. Never share passwords or .env files.

## What is connected

- Doctor: Patient search, 360 view, diagnoses, laboratory results, encounters, prescription items joined to prescriptions, recorded allergies, timeline, observational Neo4j relationships, Phase 12 SQLite literature excerpts.
- Medication review: research-only review of recorded prescriptions and graph observations, retrieved excerpts, and missing safety evidence. **It deliberately does not assert drug indications, generate prescriptions, or recommend a patient-specific treatment.**
- DBA: Phase 13 planner and execution (lazy-loaded), strategy latency from historical cost profiles, Phase 9 historical benchmark results including rejected index candidates, system health and process-local activity.

## Important limitations

- This is a research prototype, not a clinical decision support system.
- The provided RxNorm relationship types describe drug terminology, not verified drug–disease indications, contraindications or interactions.
- Phase 12 keyword excerpt retrieval is not clinical guideline validation. Evidence sources require expert review before use.
- No clinical safety clearance can be inferred from an empty allergy table.
- Existing Phase 14 APIs are preserved; configure `PHASE14_API_KEY` and `PHASE15_API_KEY` consistently.
- The copied Phase 13 module expects the real Phase 12 module at `notebooks/phase12_graph_rag_v7.py` and the literature SQLite in `results/phase12_store_v3/`. Existing notebooks should be preserved. Some vector/hybrid queries require additional Phase 12 packages/models.
- Phase 9 optimizer files are **historical evidence**; Phase 15 does not run index DDL or mutate the database.
- Running tests without live databases validates API routes and static UI only. For full acceptance, manually exercise PostgreSQL, Neo4j, Phase 12 and Phase 13 connections on the user's Windows system.
- Restrict deployment to localhost and de-identified/synthetic patient data. Use a least-privilege database account for real deployments.


## Phase 15.1: Human-readable UI + Ollama

This update adds guided navigation, readable column labels, filtering and paging, patient condition dropdowns, and a local Ollama literature explanation tool. No patient data is sent to Ollama. Source excerpts are displayed for manual verification. This is not prescribing support.

Start Ollama separately in another terminal (if it is not running): `ollama serve`. Install the chosen model: `ollama pull qwen2.5:7b`. Set `$env:OLLAMA_MODEL = "qwen2.5:7b"` in the CareScope PowerShell terminal before launching. Use the original `RUN_PHASE15.ps1` or `python phase15/main.py`. Open `http://127.0.0.1:8015/portal`, connect, and open **Research & AI**.

Endpoints: GET `/api/v15/ollama-status` and POST `/api/v15/ai-evidence` with JSON `{ "question": "...", "max_sources": 5 }`. Both require the Phase 15 API key. The AI endpoint uses retrieved SQLite literature excerpts; model output is unverified and must not be treated as a clinical guideline.
