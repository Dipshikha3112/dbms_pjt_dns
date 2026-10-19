"""Compare graph retrieval baseline and query-aware reranking. No clinical ground truth."""
import argparse, csv, json, statistics, time
from pathlib import Path
from phase13_hybrid_query_planner import graph_execute, OUT, percentile

QUESTIONS = [
    "Which lab tests are associated with hypertension?",
    "Which kidney lab tests are associated with hypertension?",
    "Which cholesterol lab tests are associated with hypertension?",
    "Which blood lab tests are associated with diabetes?",
    "Which medications are associated with hypertension?",
]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repeats",type=int,default=3)
    ap.add_argument("--limit",type=int,default=8)
    args=ap.parse_args()
    if not 1<=args.repeats<=20 or not 1<=args.limit<=100:
        ap.error("repeats must be 1..20; limit must be 1..100")
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    for question in QUESTIONS:
        for mode in ("baseline","optimized"):
            for run in range(args.repeats):
                t=time.perf_counter()
                try:
                    result=graph_execute(question,args.limit,ranking=mode)
                    status="success"; error=""
                except Exception as exc:
                    result={"rows":[]};status="error";error=str(exc)
                rows.append({"question":question,"mode":mode,"repeat":run+1,
                             "status":status,"latency_ms":round((time.perf_counter()-t)*1000,3),
                             "candidate_count":result.get("candidate_count",0),
                             "top_codes":json.dumps([x.get("concept_code") for x in result["rows"]]),
                             "error":error})
    with (OUT/"graph_before_after.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    summary={}
    for mode in ("baseline","optimized"):
        values=[r["latency_ms"] for r in rows if r["mode"]==mode and r["status"]=="success"]
        summary[mode]={"successful_runs":len(values),"median_ms":statistics.median(values) if values else None,
                       "p95_ms":percentile(values,95)}
    summary["limitation"]="Timing and ranking comparison only; relevance improvement requires independent manual labels."
    (OUT/"graph_before_after_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":main()
