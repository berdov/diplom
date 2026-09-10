"""Aggregate paired seeds without tuning or model/data evaluation."""
import json
from pathlib import Path
from statistics import mean,stdev
from experiments.stage_confirmation.run import HERE,ROOT,SEEDS,CELLS,case
from experiments.target_combination_analysis.common import combinations,METRICS,write

def aggregate():
    groups=[];missing=[]
    for pos,idx in enumerate(CELLS):
        cell=combinations()[idx]; records=[]
        for seed in SEEDS:
            path=(ROOT/'reports/evidence/target_combinations/runs'/f"{cell['run_id']}.json") if seed==2026 else HERE/'runs'/f"target_confirm_{cell['combination_id']}_seed{seed}_001.json"
            if not path.exists():missing.append(str(path.relative_to(ROOT)));continue
            r=json.loads(path.read_text())
            if r['status']!='completed' or not r['gates']['passed']:
                missing.append(str(path.relative_to(ROOT)));continue
            assert r['seed']==seed and r['active_targets']==cell['active_targets'] and r['test_evaluation_count']==0
            assert set(r['ranking_metrics'])==set(METRICS)
            records.append({'seed':seed,'metrics':r['ranking_metrics'],'best_epoch':r['best_epoch'],'source':str(path.relative_to(ROOT)),'code_sha':r['git_commit']})
        groups.append({'combination':cell['display_name'],'n_aux':cell['n_aux_targets'],'records':records,'metrics':{m:{'mean':mean(r['metrics'][m] for r in records),'std_sample':stdev(r['metrics'][m] for r in records) if len(records)>1 else None} for m in METRICS} if records else {}})
    complete=not missing;pairwise=[]
    for a,b in [(1,0),(2,0),(2,1)]:
        left={r['seed']:r for r in groups[a]['records']};right={r['seed']:r for r in groups[b]['records']}
        ds=[{'seed':s,'delta':left[s]['metrics']['NDCG@10']-right[s]['metrics']['NDCG@10']} for s in SEEDS if s in left and s in right]
        pairwise.append({'comparison':groups[a]['combination']+' minus '+groups[b]['combination'],'deltas':ds,'mean':mean(d['delta'] for d in ds) if ds else None,'std_sample':stdev(d['delta'] for d in ds) if len(ds)>1 else None})
    return {'complete':complete,'completeness':f"{sum(len(g['records']) for g in groups)}/9",'seeds':SEEDS,'std_definition':'sample standard deviation, ddof=1','test_evaluation_count':0,'groups':groups,'paired_ndcg10':pairwise,'missing_or_failed':missing}

def render(s):
    lines=['# Закрытие старого экспериментального этапа','',
    'Canonical docs merged fast-forward в main: `93053d86eb3b7516ecf748fe060baa179d4e678b`.','',
    '## MOO representatives','',
    'Fairness operating-point selection MosT vs GradHV закрыта полным post-hoc replay: на epochs 5/10/15/20/25 max VALID NDCG@10 выбирает то же solution_index=1; best epoch=10, stop epoch=25, NDCG@10=0.0522. Новый rerun не нужен. [Проверка](evidence/most_gradhv_selection_replay.json). Это устраняет конкретное различие selection, не все различия методов и бюджетов.','',
    'Рабочий representative set можно зафиксировать: **STCH, FAMO, PCGrad, EPO, MosT-style, PHN-HVI-adapter, COSMOS-style, PaLoRA**. Это выбор для следующего этапа по имеющимся VALID экспериментам, без заявления о статистически доказанном превосходстве семейств.','',
    'Historical EPO+MoE M0/M2/M4/M8: **technical failure / no scientific result**, jobs 4300861–4300864 FAILED (1:0). Не являются отрицательными научными результатами; не перезапускались.','',
    '## MTL multi-seed confirmation','',
    f"Completeness **{s['completeness']}**. Seeds 2026/2027/2028 общие для трёх вариантов; 2026 переиспользован из screening. Fixed protocol, без tuning и TEST. std — выборочное стандартное отклонение (ddof=1). До 3/3 seeds средние описывают только доступную часть.",'',
    '| Combination | Seeds | NDCG@10 mean ± std |','| --- | ---: | ---: |']
    for g in s['groups']:
        v=g['metrics'].get('NDCG@10'); std='—' if not v or v['std_sample'] is None else f"{v['std_sample']:.6f}"
        lines.append(f"| {g['combination']} | {len(g['records'])}/3 | {v['mean']:.6f} ± {std} |" if v else f"| {g['combination']} | 0/3 | — |")
    lines+=['','## Все canonical metrics','', '| Combination | Metric | Mean | Sample std |','| --- | --- | ---: | ---: |']
    for g in s['groups']:
        for m,v in g['metrics'].items():
            std='—' if v['std_sample'] is None else f"{v['std_sample']:.6f}"
            lines.append(f"| {g['combination']} | {m} | {v['mean']:.6f} | {std} |")
    lines+=['','## Paired NDCG@10 differences','', '| Comparison | Seed deltas | Mean |','| --- | --- | ---: |']
    for p in s['paired_ndcg10']:
        if p['mean'] is not None:lines.append(f"| {p['comparison']} | "+', '.join(f"{d['seed']}: {d['delta']:+.4f}" for d in p['deltas'])+f" | {p['mean']:+.6f} |")
    lines+=['','## Можно ли заморозить MTL target set?','']
    if not s['complete']:lines+=['Пока нет: confirmation не завершён. Нельзя выдавать screening winner за multi-seed подтверждение.']
    else:
        pair,triple,contrast=s['paired_ndcg10']
        stable=[i for i,p in [(1,pair),(2,triple)] if all(d['delta']>0 for d in p['deltas'])]
        if stable:
            best=max(stable,key=lambda i:s['groups'][i]['metrics']['NDCG@10']['mean'])
            if len(stable)==2 and abs(contrast['mean'])<=0.0001:best=1
            lines += [f"Можно зафиксировать **{s['groups'][best]['combination']}** как рабочий target set: преимущество над primary-only сохраняется на всех трёх seeds. При разнице средних pair/triple ≤0.0001 предпочтена меньшая пара (порог near-tie из screening). Это описательное подтверждение, не тест статистической значимости."]
        else:lines+=['Устойчивое преимущество auxiliary subset над primary-only на всех трёх seeds не подтверждено. Замораживать screening winner как подтверждённое улучшение нельзя; primary-only остаётся контрольным вариантом. Дополнительные запуски автоматически не назначаются.']
    lines+=['','[Машиночитаемая сводка](../experiments/stage_confirmation/summary.json). Исходные screening/challenger результаты не изменялись.','']
    return '\n'.join(lines)

if __name__=='__main__':
    s=aggregate();write(HERE/'summary.json',s)
    # Write only inside experiment checkout; collect into report after completion.
    (HERE/'STAGE_CLOSURE.md').write_text(render(s))
    print(s['completeness'])
