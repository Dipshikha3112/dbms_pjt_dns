# Phase 14 — Doctor-facing Research Workspace

## Important
This is a **research-only** patient record and evidence review prototype. It does **not** diagnose disease, recommend treatment, prescribe drugs, identify clinical emergencies, or make claims that synthetic graph associations are clinically validated. It does not resolve patient identities across PostgreSQL and Neo4j. Do not expose it to the internet or use identifiable patient data without appropriate security review and institutional authorization.

## Install
1. Back up your original project-root `app.py`.
2. Copy this package's `app.py` to your project root, and copy `static/index.html` to `<project>/static/index.html`.
3. Keep existing `notebooks/phase13_hybrid_query_planner.py`, `notebooks/phase12_graph_rag_v7.py`, and `results/` in place.
4. Activate your venv and install `fastapi uvicorn psycopg[binary] neo4j pydantic` plus your existing Phase 12/13 dependencies.
5. Configure the database and API key in the current PowerShell session:

```powershell
$env:PHASE14_API_KEY = 'CHOOSE_A_LONG_RANDOM_LOCAL_SECRET'
$env:PGHOST = '127.0.0.1'
$env:PGPORT = '5432'
$env:PGUSER = 'postgres'
$env:PGPASSWORD = 'YOUR_EXISTING_PASSWORD'
$env:PHASE14_PATIENT_DB = 'adaptive_healthcare_db'
$env:PHASE14_PATIENT_SCHEMA = 'public'
$env:NEO4J_URI = 'bolt://127.0.0.1:7687'
$env:NEO4J_USER = 'neo4j'
$env:NEO4J_PASSWORD = 'YOUR_EXISTING_PASSWORD'
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Open http://127.0.0.1:8000/ and click **Set access key**. API docs: http://127.0.0.1:8000/docs. In Swagger use **Authorize** for X-API-Key.

### Endpoints
- `GET /api/v1/patients/search?q=&limit=20&offset=0`
- `GET /api/v1/patients/{patient_id}`
- `GET /api/v1/patients/{patient_id}/history`
- `GET /api/v1/graph/patients`
- `GET /api/v1/graph/patient/{patient_id}`
- `POST /api/v1/graph/explore`
- `POST /api/v1/planner/execute`
- `GET /api/v1/planner/costs`
- `POST /api/v1/rag/retrieve`
- `POST /api/v1/rag/answer` (retrieval only; generation not configured)
- `GET /api/v1/optimizer/summary`
- `GET /api/v1/optimizer/recommendations`
- `GET /api/v1/system/health` (public liveness/dependency summary)
- `GET /api/v1/system/capabilities`

### Integration caveats
- Patient search discovers a table named `patient` or `patients`, plus common ID/name columns. Patient history looks for conventional table names and direct `patient_id`-type keys. It **does not guess joins**. If your schema uses different names, update `patient_config()` / `get_history()` to match actual schema.
- PostgreSQL role should have `SELECT` only, with row-level security if required. The API key is a single local demonstration secret, **not a production authentication/authorization system**. Do not treat it as sufficient for real clinical records.
- The API deliberately returns a reduced patient profile but history records may contain additional columns; ensure only synthetic/de-identified data are used.
- The optimizer endpoint reports measured database performance suggestions, not clinical recommendations.
- Phase 12 retrieval and Phase 13 routing may require additional local dependencies, environment settings and exact schemas; endpoint registration does not prove those dependencies work.
- The existing original app.py has additional Phase 7 and Phase 9 endpoints; this package preserves core Phase 9 audit through `/api/v1/optimizer/summary` but **does not preserve every legacy endpoint**. Keep a backup if you need them.

### Smoke tests
```powershell
python -m py_compile app.py
python -m pytest tests/test_phase14_smoke.py -q
```

All workflows must be exercised with your live local database before marking Phase 14 accepted.
