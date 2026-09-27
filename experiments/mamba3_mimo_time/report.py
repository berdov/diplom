"""Saved-record summary only; no model imports, evaluation, or checkpoint load."""
import json
import math
from . import config as c
from .records import read, create, sha


def summarize(records):
    rows=[]
    for mode in c.MODES:
        r=records.get(mode)
        row=dict(mode=mode,parameters=c.COUNTS[mode],status='NOT_RUN',metrics=None,first27=None,
                 best_epoch=None,actual_epochs=0,train_seconds=None,valid_seconds=None,peak_memory_bytes=None)
        if r:
            if r.get('TEST')!='NOT_RUN' or r.get('test_evaluation_count')!=0 or r.get('mode')!=mode:
                raise ValueError('Result scope/TEST mismatch')
            row.update(status=r['status'],actual_epochs=r.get('actual_epochs',0),error=r.get('error'))
            if r['status']=='PASS':
                h=r['history']; scores=[v['valid_ndcg10'] for v in h]
                if ([v['epoch'] for v in h]!=list(range(r['actual_epochs'])) or not scores
                        or not all(math.isfinite(v) for v in scores)):
                    raise ValueError('Incomplete/nonfinite successful history')
                best=max(i for i,v in enumerate(scores) if v==max(scores))
                if (r['best_epoch']!=best or h[best]['valid_metrics']!=r['best_valid_metrics']
                        or h[best]['diagnostics']!=r['best_diagnostics'] or r['best_valid_score']!=scores[best]):
                    raise ValueError('Selected epoch/metrics/diagnostics drift')
                row.update(metrics=r['best_valid_metrics'],parameters=r['parameter_count'],best_epoch=best,
                           first27=max(scores[:27]),first27_complete=len(h)>=27,train_seconds=r['train_seconds'],
                           valid_seconds=r['valid_seconds'],peak_memory_bytes=r['peak_gpu_allocated_bytes'],
                           peak_reserved_bytes=r['peak_gpu_reserved_bytes'])
        rows.append(row)
    by={r['mode']:r for r in rows}
    comparisons=[]
    for left,right in [('dual','base'),('triple','dual'),('triple','base')]:
        a,b=by[left],by[right]
        available=a['status']==b['status']=='PASS'
        av=a['metrics']['ndcg@10'] if available else None
        bv=b['metrics']['ndcg@10'] if available else None
        comparisons.append(dict(comparison=left+' - '+right,absolute=av-bv if available else None,
                                relative_percent=100*(av/bv-1) if available and bv else None))
    return dict(status='PASS' if all(r['status']=='PASS' for r in rows) else 'INCOMPLETE',rows=rows,
                comparisons=comparisons,seed=2026,split='VALID',TEST='NOT_RUN',test_evaluation_count=0,
                scientific_fits_started=sum(bool(r.get('scientific_fit_started')) for r in records.values()),
                scientific_fits_completed=sum(r['status']=='PASS' for r in rows))


def markdown(summary):
    lines=['# MIMO base / dual / triple: exploratory pilot','',
           'KuaiRand: хронологический leave-one-out, полный каталог. Только VALID, TEST не использовался.',
           'Один seed2026; статистическая значимость/SOTA не заявляются. Частичная серия не выбирает победителя.', '',
           '| mode | parameters | VALID NDCG@10 | HR@10 | best 0–26 | best epoch (0-based) | actual epochs | TRAIN / VALID seconds | peak allocated / reserved GiB | status |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    f=lambda v: 'нет данных' if v is None else f'{v:.4f}'
    for r in summary['rows']:
        metrics=r['metrics'] or {}
        window=f(r['first27'])+(' (неполное окно)' if r.get('first27_complete') is False else '')
        memory='нет данных' if r['peak_memory_bytes'] is None else f"{r['peak_memory_bytes']/2**30:.3f} / {r['peak_reserved_bytes']/2**30:.3f}"
        lines.append(f"| {r['mode']} | {r['parameters']} | {f(metrics.get('ndcg@10'))} | {f(metrics.get('hit@10'))} | {window} | {r['best_epoch']} | {r['actual_epochs']} | {f(r['train_seconds'])} / {f(r['valid_seconds'])} | {memory} | {r['status']} |")
    lines+=['','## Парные разницы VALID NDCG@10','']
    for r in summary['comparisons']:
        lines.append(f"- {r['comparison']}: absolute={f(r['absolute'])}; relative={f(r['relative_percent'])}%.")
    lines+=['','## Все cutoff','', '| mode | metric | @5 | @10 | @20 | @50 |','|---|---|---:|---:|---:|---:|']
    for r in summary['rows']:
        for label,key in [('HR','hit'),('Recall','recall'),('NDCG','ndcg')]:
            lines.append('| '+r['mode']+' | '+label+' | '+' | '.join(f((r['metrics'] or {}).get(f'{key}@{k}')) for k in (5,10,20,50))+' |')
    lines+=['','Первые 27 эпох являются срезом той же history, не независимым подтверждением.',
            'Исторический SISO pilot2026: dual=.0615 (610572 параметра), triple=.0623 (610638); другая архитектура. Не подменяет MIMO base и не объединяется с этим seed в парное доказательство.',
            'VALID не сравнивается напрямую с опубликованным TEST TiM4Rec. Внешнее сравнение требует отдельного TEST-плана.',
            'Admission разрешает ограниченный пилот по mimo_numeric_acceptance_v1, не отменяет legacy exact-zero/near-zero failures.', '']
    return '\n'.join(lines)


def write(base):
    records={}
    for mode in c.MODES:
        p=c.paths(mode)
        if p['result'].exists():
            r=read(p['result'])
            if any(r.get(k)!=v for k,v in base.items()):
                raise ValueError('Report result ownership')
            if r['status']=='PASS':
                meta=read(p['metadata'])
                if meta['metrics']!=r['best_valid_metrics'] or meta['epoch']!=r['best_epoch'] or sha(p['checkpoint'])!=r['checkpoint_sha256']:
                    raise ValueError('Report checkpoint evidence')
            records[mode]=r
    summary=dict(base,**summarize(records))
    create(c.SUMMARY,summary)
    c.SUMMARY.with_suffix('.md').parent.mkdir(parents=True,exist_ok=True)
    with c.SUMMARY.with_suffix('.md').open('x') as out:
        out.write(markdown(summary))
    return summary
