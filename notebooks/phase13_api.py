"""Optional API: uvicorn phase13_api:app --host 127.0.0.1 --port 8013"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from phase13_hybrid_query_planner import execute, CostRegistry, plan

app = FastAPI(title="Phase 13 Hybrid Query Planner", version="1.0.0")

class Request(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    limit: int = Field(default=8, ge=1, le=100)
    plan_only: bool = False

@app.post("/query")
def query(req: Request):
    try:
        registry = CostRegistry()
        if req.plan_only:
            return plan(req.question, registry)
        return execute(req.question, registry, req.limit)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{type(exc).__name__}: {exc}")

@app.get("/costs")
def costs():
    return CostRegistry().summary()

@app.get("/")
def home():
    return {"project":"Phase 13 Hybrid Query Planner", "docs":"/docs", "status":"running"}

@app.get("/health")
def health():
    return {"status":"running", "database_connections":"not checked"}
