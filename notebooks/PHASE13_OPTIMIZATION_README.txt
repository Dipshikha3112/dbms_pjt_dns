PHASE 13 OPTIMIZATION - INSTALL AND TEST

1. Stop Uvicorn (Ctrl+C). Back up existing notebooks/phase13_hybrid_query_planner.py.
2. Extract the four Python files from this ZIP into your project notebooks folder.
3. Keep notebooks/phase12_graph_rag_v7.py unchanged.
4. Run from project root:
   python .\notebooks\phase13_hybrid_query_planner.py --question "Which kidney lab tests are associated with hypertension?"
   python .\notebooks\phase13_graph_comparison.py --repeats 3
   python .\notebooks\phase13_unseen_validation.py
5. Start API from notebooks:
   uvicorn phase13_api:app --host 127.0.0.1 --port 8013 --reload
6. Open http://127.0.0.1:8013/docs

IMPORTANT: The module import is cached; the Phase 12 embedding model is NOT
necessarily cached. Repeated model-loading warnings mean Phase 12's model
creation must be patched separately after inspecting its actual source.
Generic graph questions may remain alphabetically ordered because all
lexical relevance scores tie. No clinical ranking validity is claimed.
