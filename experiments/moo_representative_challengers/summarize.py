"""Separate challenger evidence registry; never writes canonical results.csv."""
import json
from .common import HERE, write_json

REFERENCES={
    'epo':{'stage1_ndcg10':.0584,'stage2_ndcg10':.0588},
    'gradhv':{'stage1_ndcg10':.0486,'stage2_ndcg10':.0488,'selection_caveat':'Historical GradHV used best validation NDCG across solutions; MosT uses frozen scalarized rank_heavy score.'},
    'phn_adapter':{'stage1_ndcg10':.0423}}


def summarize():
    runs=[]
    for path in sorted((HERE/'runs').glob('*.json')):
        r=json.loads(path.read_text())
        if r['test_evaluation_count']!=0: raise RuntimeError('TEST-contaminated artifact')
        if r['status']=='completed' and not r['gates']['passed']: raise RuntimeError('Completed run failed gate')
        best=(r.get('validation') or {}).get('ranking_operating_point') or {}
        runs.append({k:r.get(k) for k in ['run_id','method','implementation_name','representative_fidelity',
            'exact_method_reproduction','status','stage','git_commit','source_digest','test_evaluation_count']} |
            {'metrics':best.get('metrics'),'best_epoch':r['training'].get('best_epoch'),
             'runtime_seconds':r.get('runtime_seconds'),'source_json':str(path.relative_to(HERE.parents[1]))})
    summary={'test_evaluation_count':0,'canonical':False,'references':REFERENCES,'runs':runs,
             'interpretation':'Compare screening with Stage 1; Stage 2 tuning is supplementary context. No representative changes before human review.'}
    write_json(HERE/'summary.json',summary)
    return summary


if __name__=='__main__': print(json.dumps(summarize(),ensure_ascii=False,indent=2))
