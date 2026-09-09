"""Partial-safe, single-seed factorial summary; never writes canonical results."""
import argparse
import csv
import itertools
import math
import statistics as st
from .common import HERE,ROOT,TARGETS,METRICS,config,combinations,read,write,sha,git,weights

def describe(values):
    return {'count':len(values),'mean':st.mean(values) if values else None,'median':st.median(values) if values else None,'min':min(values) if values else None,'max':max(values) if values else None,'number_positive':sum(x>0 for x in values),'number_negative':sum(x<0 for x in values)}

def factorial(values):
    marginal=[];pairs=[]
    for t in TARGETS:
        deltas=[]
        for n in range(4):
            for s in itertools.combinations([x for x in TARGETS if x!=t],n):
                a=frozenset(s);b=a|{t}
                if a in values and b in values: deltas.append({'background':list(s),'delta':values[b]-values[a]})
        marginal.append({'target':t,'expected_comparisons':8,**describe([r['delta'] for r in deltas]),'matched_deltas':deltas})
    for a,b in itertools.combinations(TARGETS,2):
        contrasts=[];remaining=[t for t in TARGETS if t not in [a,b]]
        for n in range(3):
            for s in itertools.combinations(remaining,n):
                base=frozenset(s);keys=[base,base|{a},base|{b},base|{a,b}]
                if all(k in values for k in keys): contrasts.append({'background':list(s),'delta':values[keys[3]]-values[keys[1]]-values[keys[2]]+values[keys[0]]})
        pairs.append({'pair':f'{a} + {b}','expected_comparisons':4,**describe([r['delta'] for r in contrasts]),'matched_deltas':contrasts})
    return marginal,pairs

def pearson(pairs):
    if len(pairs)<2:return None
    x,y=zip(*pairs);mx,my=st.mean(x),st.mean(y)
    den=math.sqrt(sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))
    return sum((a-mx)*(b-my) for a,b in pairs)/den if den else None

def build(cells,runs,commit,cfg,hashes):
    canonical=[];values={};gradients=[];regression=[]
    for cell in cells:
        row=dict(cell,status='missing',rank=None,**{m:None for m in METRICS},best_epoch=None,runtime_sec=None,delta_NDCG10_vs_primary=None,relative_delta_pct=None)
        run=runs.get(cell['run_id'])
        if run:
            row['status']=run.get('status','invalid')
            valid=(run.get('status')=='completed' and run.get('scientific_run') is True and run.get('git_commit')==commit and run.get('fingerprint')==cfg['fingerprint'] and run.get('evaluation_split')=='validation' and run.get('test_usage')=='forbidden' and run.get('test_evaluation_count')==0 and run.get('active_targets')==cell['active_targets'] and run.get('array_index')==cell['array_index'] and run.get('combination_id')==cell['combination_id'] and run.get('source_digest')==hashes['source_digest'] and run.get('config_sha256')==hashes['config'] and run.get('combinations_sha256')==hashes['combinations'] and run.get('loss_weight_mode')=='uniform_normalized_aux' and run.get('auxiliary_loss_weights')==weights(cell['active_targets']) and run.get('lambda_aux')==cfg['optimization']['lambda_aux'] and run.get('seed')==cfg['training']['seed'] and run.get('gates',{}).get('passed') is True)
            metrics=run.get('ranking_metrics',{})
            valid=valid and all(isinstance(metrics.get(m),(int,float)) and math.isfinite(metrics[m]) and 0<=metrics[m]<=1 for m in METRICS)
            valid=valid and all(metrics.get(f'HR@{k}')==metrics.get(f'Recall@{k}') for k in [5,10,20,50])
            if valid:
                row.update(**{m:metrics[m] for m in METRICS},best_epoch=run['best_epoch'],runtime_sec=run['runtime_sec'])
                values[frozenset(cell['active_targets'])]=metrics['NDCG@10']
                for target,g in run.get('gradient_diagnostics',{}).get('summary',{}).get('per_auxiliary_vs_primary',{}).items():
                    gradients.append({'combination':cell['display_name'],'target':target,'mean_cosine':g['cosine_to_primary'].get('mean'),'median_cosine':g['cosine_to_primary'].get('median'),'norm_mean':g['aux_norm'].get('mean'),'norm_ratio_mean':g['norm_ratio_to_primary'].get('mean'),'conflict_fraction':g.get('conflict_fraction_with_primary'),'batches':g['diagnostic_batches']})
                for pair,g in run.get('gradient_diagnostics',{}).get('summary',{}).get('auxiliary_pairwise',{}).items():
                    gradients.append({'combination':cell['display_name'],'target':pair,'mean_cosine':g['cosine'].get('mean'),'median_cosine':g['cosine'].get('median'),'norm_mean':None,'norm_ratio_mean':None,'conflict_fraction':g.get('conflict_fraction'),'batches':g['diagnostic_batches']})
            elif row['status']=='completed':row['status']='invalid_provenance_or_metrics'
        canonical.append(row)
    primary=values.get(frozenset())
    ranked=sorted([r for r in canonical if r['NDCG@10'] is not None],key=lambda r:(-r['NDCG@10'],r['array_index']))
    for i,row in enumerate(ranked,1):
        row['rank']=i
        if primary is not None:
            delta=row['NDCG@10']-primary;row['delta_NDCG10_vs_primary']=delta;row['relative_delta_pct']=100*delta/primary if primary else None
    best_count=[{'n_aux':n,'best':next((r for r in ranked if r['n_aux_targets']==n),None),'mean_ndcg10':st.mean([r['NDCG@10'] for r in ranked if r['n_aux_targets']==n]) if any(r['n_aux_targets']==n for r in ranked) else None} for n in range(5)]
    marginal,pairs=factorial(values)
    relation=[]
    for r in ranked:
        gs=[g['conflict_fraction'] for g in gradients if g['combination']==r['display_name'] and g['target'] in TARGETS and g['conflict_fraction'] is not None]
        if gs and r['delta_NDCG10_vs_primary'] is not None:relation.append((st.mean(gs),r['delta_NDCG10_vs_primary']))
    gap=ranked[0]['NDCG@10']-ranked[1]['NDCG@10'] if len(ranked)>1 else None
    return {'git_commit':commit,'fingerprint':cfg['fingerprint'],'seed':cfg['training']['seed'],'loss_formula':cfg['loss_formula'],'lambda_aux':cfg['optimization']['lambda_aux'],'evaluation_split':'validation','test_usage':'forbidden','test_evaluation_count':0,'completeness':f'{len(ranked)}/16','complete':len(ranked)==16,'canonical_order':canonical,'ranking':ranked,'best_by_aux_count':best_count,'main_effects':marginal,'pair_interactions':pairs,'gradient_summary':gradients,'top_gap':gap,'near_tie':gap is not None and gap<=cfg['near_tie_ndcg10'],'exploratory_conflict_delta_pearson':pearson(relation),'exploratory_cell_count':len(relation),'interpretation':'Descriptive fixed-hyperparameter single-seed screening. No p-values, significance or causal gradient claims. Partial effects use only available matched backgrounds.'}

def table(rows,keys):
    def fmt(v):
        if v is None:return '—'
        if isinstance(v,float):return f'{v:.6g}'
        if isinstance(v,list):return ', '.join(v) or 'none'
        return str(v).replace('|',' / ')
    return '\n'.join(['| '+' | '.join(keys)+' |','| '+' | '.join(['---']*len(keys))+' |']+['| '+' | '.join(fmt(r.get(k)) for k in keys)+' |' for r in rows])

def markdown(s):
    rows=s['ranking']+[r for r in s['canonical_order'] if r['rank'] is None]
    count=[{'n_aux':r['n_aux'],'best':r['best']['display_name'] if r['best'] else None,'NDCG@10':r['best']['NDCG@10'] if r['best'] else None,'mean_ndcg10':r['mean_ndcg10']} for r in s['best_by_aux_count']]
    parts=['# Анализ всех комбинаций auxiliary targets',f"KuaiRand-Pure, Protocol B, validation-only. Completeness **{s['completeness']}**.",f"Code SHA `{s['git_commit']}`; fingerprint `{s['fingerprint']}`; seed {s['seed']}.",f"Loss: {s['loss_formula']}; lambda={s['lambda_aux']}. TEST=0. No per-combination tuning.",'## Все комбинации',table(rows,['rank','display_name','status','n_aux_targets','active_targets','best_epoch',*METRICS,'delta_NDCG10_vs_primary','relative_delta_pct','runtime_sec']),'## Лучшие по числу auxiliary',table(count,['n_aux','best','NDCG@10','mean_ndcg10']),'Средние по размеру subset описательны: больше auxiliary не гарантирует улучшения.','## Matched marginal effects',table(s['main_effects'],['target','count','expected_comparisons','mean','median','min','max','number_positive','number_negative']),'## Pairwise interactions',table(s['pair_interactions'],['pair','count','expected_comparisons','mean','median','min','max']),'## Gradient diagnostics',table(s['gradient_summary'],['combination','target','mean_cosine','median_cosine','norm_mean','norm_ratio_mean','conflict_fraction','batches']),'## Ограничения',s['interpretation'],f"Exploratory conflict/delta Pearson: {s['exploratory_conflict_delta_pearson']} ({s['exploratory_cell_count']} cells). Корреляция не доказывает причинность."]
    if s['ranking']:parts.append(f"Максимальный наблюдаемый validation NDCG@10: {s['ranking'][0]['display_name']}. Это не доказательство универсально лучшей комбинации.")
    if s['near_tie']:parts.append('Top-1 и top-2 численно близки (gap ≤ 0.0001); однозначный научный победитель не установлен.')
    parts.extend(['## Historical regression anchors',table(s.get('historical_regression',[]),['combination','historical_ndcg10','new_ndcg10','delta','requires_investigation']),'При существенном расхождении с primary/single Stage3 anchors интерпретация откладывается до выяснения причины. Старый all-four имеет другой weighting и не является regression-equivalent.'])
    return '\n\n'.join(parts)+'\n'

def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    from .common import provenance,source_digest
    commit=provenance();cfg=config();cells=combinations();runs={}
    for c in cells:
        path=HERE/'runs'/f"{c['run_id']}.json"
        if path.exists():
            try:runs[c['run_id']]=read(path)
            except (ValueError,OSError) as e:runs[c['run_id']]={'status':'invalid_json','error':str(e)}
    summary=build(cells,runs,commit,cfg,{'source_digest':source_digest(),'config':sha(HERE/'config.yaml'),'combinations':sha(HERE/'combinations.yaml')})
    anchors=['stage3_primary_only_001','stage3_aux_click_001','stage3_aux_long_view_001','stage3_aux_like_001','stage3_aux_profile_enter_001'];reg=[]
    for c,anchor in zip(cells[:5],anchors):
        old=read(ROOT/'experiments/stage3_auxiliary_analysis/runs'/f'{anchor}.json')['best_validation_metrics']['NDCG@10']
        new=summary['canonical_order'][c['array_index']]['NDCG@10'];delta=None if new is None else new-old
        reg.append({'combination':c['display_name'],'historical_ndcg10':old,'new_ndcg10':new,'delta':delta,'requires_investigation':delta is not None and abs(delta)>cfg['historical_discrepancy_ndcg10']})
    summary['historical_regression']=reg;summary['interpretation_blocked_by_regression']=any(r['requires_investigation'] for r in reg)
    # Missing JSONs may include jobs killed before the Python exception handler.
    import subprocess
    submission=HERE/'submissions/pipeline.json'
    if submission.exists():
        job=read(submission).get('array_job_id')
        if job:
            result=subprocess.run(['sacct','-j',str(job),'-X','--format=JobID,State,ExitCode,Elapsed','-P'],capture_output=True,text=True)
            summary['scheduler_accounting']={'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
    write(HERE/'summary.json',summary)
    keys=['rank','array_index','combination_id','display_name','status','n_aux_targets','active_targets','best_epoch',*METRICS,'delta_NDCG10_vs_primary','relative_delta_pct','runtime_sec','run_id']
    for name,rows in [('target_combinations.csv',summary['ranking']+[r for r in summary['canonical_order'] if r['rank'] is None]),('target_combinations_canonical.csv',summary['canonical_order'])]:
        with (HERE/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({**r,'active_targets':';'.join(r['active_targets'])} for r in rows)
    (ROOT/'reports/TARGET_COMBINATION_ANALYSIS.md').write_text(markdown(summary))
    print(summary['completeness'])

if __name__=='__main__':main()
