"""Run from notebooks: python phase13_unseen_validation.py"""
import csv, json
from pathlib import Path
from phase13_hybrid_query_planner import classify, capabilities, CostRegistry, plan, OUT

# Independent challenge prompts, not used in the nine-question development benchmark.
CASES = [
('Give the total number of patients.', {'sql'}),
('Count patient entries.', {'sql'}),
('What is the number of diagnoses?', {'sql'}),
('How many lab results exist?', {'sql'}),
('Count prescriptions in the database.', {'sql'}),
('How many documents are indexed?', {'sql'}),
('Give the total number of literature chunks.', {'sql'}),
('What is the average number of lab results?', {'sql'}),
('Show the number of patient records.', {'sql'}),
('Calculate the sum of prescription records.', {'sql'}),
('Show graph neighbors of hypertension.', {'graph'}),
('What is connected to diabetes in the knowledge graph?', {'graph'}),
('Which lab tests are linked to hypertension?', {'graph'}),
('Identify medication associations for hypertension.', {'graph'}),
('Find the graph path from diabetes to lab tests.', {'graph'}),
('List diseases related to stroke in the graph.', {'graph'}),
('What MeSH concepts are mapped to hypertension?', {'graph'}),
('Find graph relationships for obesity.', {'graph'}),
('Which biomarkers are associated with diabetes?', {'graph'}),
('Explore neighboring concepts of heart failure.', {'graph'}),
('Locate papers on endothelial injury.', {'vector'}),
('Summarize literature about blood pressure variability.', {'vector'}),
('Find studies of atherosclerosis.', {'vector'}),
('Explain mechanisms of vascular inflammation.', {'vector'}),
('Find similar articles on lipid metabolism.', {'vector'}),
('Discuss research about cardiovascular risk.', {'vector'}),
('What publications discuss renal dysfunction?', {'vector'}),
('Summarize evidence on stroke prevention.', {'vector'}),
('Why does arterial stiffness occur?', {'vector'}),
('Find documents about hypertension therapy.', {'vector'}),
('Count patients and find research about hypertension.', {'sql','vector'}),
('How many diagnoses exist and what do studies say about diabetes?', {'sql','vector'}),
('Count prescriptions and summarize literature about statins.', {'sql','vector'}),
('Give the number of patients and explain cardiovascular mechanisms.', {'sql','vector'}),
('Count lab results and find articles on kidney disease.', {'sql','vector'}),
('Which lab tests are linked to hypertension and what does literature report?', {'graph','vector'}),
('Find medication associations for diabetes and supporting research.', {'graph','vector'}),
('What graph relationships exist for obesity and what do studies say?', {'graph','vector'}),
('Explore heart failure graph neighbors and summarize related papers.', {'graph','vector'}),
('Which biomarkers are associated with hypertension and what evidence exists?', {'graph','vector'}),
('Count patients and find hypertension graph relationships.', {'sql','graph'}),
('How many diagnoses exist and which lab tests are associated with diabetes?', {'sql','graph'}),
('Count prescriptions and show graph medication associations for hypertension.', {'sql','graph'}),
('Give the total number of patients and explore hypertension graph neighbors.', {'sql','graph'}),
('How many lab results exist and which biomarkers are associated with hypertension?', {'sql','graph'}),
('Count patients, explore hypertension graph links, and find research evidence.', {'sql','graph','vector'}),
('How many diagnoses exist, which lab tests are linked to diabetes, and what studies discuss it?', {'sql','graph','vector'}),
('Count prescriptions, show medication associations for hypertension, and summarize literature.', {'sql','graph','vector'}),
('Give the total number of patients, explore heart failure graph neighbors, and find related papers.', {'sql','graph','vector'}),
('Count lab results, identify hypertension biomarkers, and find research articles.', {'sql','graph','vector'}),
]

def main():
    registry=CostRegistry()
    rows=[]
    for q, gold in CASES:
        pred=plan(q,registry)
        found=set(pred['required_capabilities'])
        selected=pred['selected_strategy']
        rows.append({'question':q,'expected_capabilities':'+'.join(sorted(gold)),
            'predicted_capabilities':'+'.join(sorted(found)),
            'selected_strategy':selected,'exact_capability_match':gold==found,
            'capability_recall':round(len(gold & found)/len(gold),3),
            'capability_precision':round(len(gold & found)/max(1,len(found)),3),
            'strategy_covers_gold':gold.issubset(capabilities(selected)),
            'justification':pred['justification']})
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'unseen_routing_results.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={'total':len(rows),'exact_capability_accuracy':sum(r['exact_capability_match'] for r in rows)/len(rows),
             'mean_capability_recall':sum(r['capability_recall'] for r in rows)/len(rows),
             'mean_capability_precision':sum(r['capability_precision'] for r in rows)/len(rows),
             'strategy_coverage_rate':sum(r['strategy_covers_gold'] for r in rows)/len(rows),
             'mismatches':[r for r in rows if not r['exact_capability_match']],
             'caveat':'Independent of original nine examples, but still author-generated; have a reviewer verify gold capabilities. Routing only, not execution.'}
    (OUT/'unseen_routing_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
