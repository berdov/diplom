"""Replay the full stored MosT validation trajectory with historical GradHV selection."""
import json
from pathlib import Path
from experiments.target_combination_analysis.common import sha, write
ROOT=Path(__file__).resolve().parents[2]

def replay():
    path=ROOT/'experiments/moo_representative_challengers/runs/most_convergence_001.json'
    r=json.loads(path.read_text());cfg=r['frozen_config']['run']
    assert r['status']=='completed' and r['gates']['passed'] and r['test_evaluation_count']==0
    rows=[];best=None;stale=0;stop=None;best_point=None
    epochs=[e for e in r['training']['epochs'] if 'validation' in e]
    assert [e['epoch'] for e in epochs]==list(range(5,r['training']['stop_epoch']+1,5))
    for e in epochs:
        points=e['validation']['points']
        assert len(points)==3 and [p['solution_index'] for p in points]==[0,1,2]
        p=max(points,key=lambda p:p['metrics']['NDCG@10']) # historical stable max tie rule
        score=p['metrics']['NDCG@10'];improved=best is None or score>best+cfg['convergence_min_delta']
        if improved:best=score;best_epoch=e['epoch'];best_point=p;stale=0
        else:stale+=1
        rows.append({'epoch':e['epoch'],'solution_index':p['solution_index'],'NDCG@10':score,'old_selected_solution':e['validation']['ranking_operating_point']['solution_index'],'improved':improved,'checks_without_improvement':stale})
        if e['epoch']>=cfg['convergence_min_epochs'] and stale>=cfg['convergence_patience']:
            stop=e['epoch'];break
    assert stop==r['training']['stop_epoch']==25,'Rerun required: trajectory insufficient or stopping differs'
    assert best_epoch==r['training']['best_epoch']==10
    assert best_point==r['validation']['ranking_operating_point'],'Rerun/checkpoint recovery required'
    return {'status':'passed','fairness_operating_point_selection_closed':True,'scope':'operating-point / checkpoint / early-stopping selection only; not all method differences','method':'most','derivation':'posthoc replay of every stored validation checkpoint; no new evaluation or training','source_run':str(path.relative_to(ROOT)),'source_sha256':sha(path),'source_code_sha':r['git_commit'],'new_selection':'max VALID NDCG@10 among three solutions; stable first tie, same as historical GradHV','selection_is_validation_oracle':True,'validation_trajectory':rows,'best_epoch':best_epoch,'stop_epoch':stop,'ranking_operating_point':best_point,'best_checkpoint':r['best_checkpoint'],'test_evaluation_count':0,'rerun_required':False,'original_result_modified':False}

if __name__=='__main__':write(ROOT/'reports/evidence/most_gradhv_selection_replay.json',replay())
