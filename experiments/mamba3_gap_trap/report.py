"""Partial-safe numeric summary; no model, dataset or checkpoint loading."""
import math
from . import config as c
from experiments.mamba3_mimo_time.records import read,create,now,sha,finite_tree


def validate_record(r,variant):
    if r.get('status')!='PASS':return
    if not finite_tree(r):raise ValueError('Nonfinite scientific record')
    p=c.paths(variant);h=r.get('history',[])
    if (r.get('run_id')!=p['run_id'] or r.get('gap_trap_mode')!=variant or r.get('mode')!='dual' or r.get('seed')!=2026
        or r.get('parameter_count')!=c.COUNTS[variant] or r.get('TEST')!='NOT_RUN' or r.get('test_evaluation_count')!=0
        or r.get('scientific_fit_started') is not True or not h
        or [x['epoch'] for x in h]!=list(range(r.get('actual_epochs',-1)))):
        raise ValueError('Invalid successful run identity/history')
    scores=[x['valid_ndcg10'] for x in h]
    if not all(math.isfinite(v) for v in scores):raise ValueError('Nonfinite history')
    metric_keys={f'{kind}@{k}' for kind in ('hit','ndcg','recall') for k in (5,10,20,50)}
    for row in h:
        if set(row['valid_metrics'])!=metric_keys or row['valid_metrics']['ndcg@10']!=row['valid_ndcg10']:
            raise ValueError('Metric keys/selection mismatch')
    best=max(i for i,v in enumerate(scores) if v==max(scores))
    if (r.get('best_epoch')!=best or r.get('best_valid_score')!=scores[best]
        or r.get('best_valid_metrics')!=h[best]['valid_metrics'] or r.get('best_diagnostics')!=h[best]['diagnostics']):
        raise ValueError('Best/last-tie mismatch')
    best_so_far=-math.inf;stale=0;stop=None
    for i,score in enumerate(scores):
        if score>=best_so_far:best_so_far=score;stale=0
        else:stale+=1
        if stale>10 and stop is None:stop=i
    if (len(h)<300 and stop!=len(h)-1) or (stop is not None and stop!=len(h)-1) or len(h)>300:
        raise ValueError('Not a complete unchanged early-stopped fit')
    if r.get('first27_complete')!=(len(h)>=27) or r.get('first27_best_ndcg10')!=(max(scores[:27]) if len(h)>=27 else None):
        raise ValueError('First27 mismatch')



def validate_checkpoint(r,variant):
    p=c.paths(variant);meta=read(p['metadata'])
    if sha(p['checkpoint'])!=r['checkpoint_sha256'] or meta['checkpoint_sha256']!=r['checkpoint_sha256']:
        raise ValueError('Checkpoint SHA mismatch')
    for k in ('run_id','mode','gap_trap_mode','seed','execution_commit','config_sha256','source_hash','core_hash'):
        if meta[k]!=r[k]:raise ValueError('Checkpoint metadata identity: '+k)
    if meta['epoch']!=r['best_epoch'] or meta['metrics']!=r['best_valid_metrics']:
        raise ValueError('Checkpoint metadata best mismatch')


def summarize(records):
    rows=[]
    for variant in c.MODES:
        r=records.get(variant,{})
        error=r.get('validation_error')
        try:validate_record(r,variant)
        except (ValueError,KeyError,TypeError) as exc:
            error=str(exc);r=dict(r,status='FAIL',best_valid_metrics={},history=[])
        h=r.get('history',[])
        complete=len(h)>=27 and [x['epoch'] for x in h[:27]]==list(range(27))
        metric=r.get('best_valid_metrics',{})
        rows.append(dict(variant=variant,parameters=c.COUNTS[variant],status=r.get('status','NOT_RUN'),validation_error=error,
                         ndcg10=metric.get('ndcg@10'),hr10=metric.get('hit@10'),best_epoch=r.get('best_epoch'),
                         actual_epochs=r.get('actual_epochs',0),first27_complete=complete,
                         first27_ndcg10=max(x['valid_ndcg10'] for x in h[:27]) if complete else None,
                         train_seconds=r.get('train_seconds'),valid_seconds=r.get('valid_seconds'),
                         peak_allocated_bytes=r.get('peak_gpu_allocated_bytes'),peak_reserved_bytes=r.get('peak_gpu_reserved_bytes'),
                         scientific_fit_started=r.get('scientific_fit_started',False),gap_trap_diagnostics=r.get('best_diagnostics',{}).get('gap_trap')))
    by={r['variant']:r for r in rows}; contrasts=[]
    for left,right in c.plan()['contrasts']:
        a,b=by[left],by[right]
        valid=a['status']==b['status']=='PASS' and a['ndcg10'] is not None and b['ndcg10'] is not None
        delta=a['ndcg10']-b['ndcg10'] if valid else None
        contrasts.append(dict(comparison=left+' - '+right,status='COMPLETE' if valid else 'NOT_AVAILABLE',
                              delta=delta,relative_percent=100*delta/b['ndcg10'] if valid and b['ndcg10']!=0 else None))
    return dict(status='PASS' if all(r['status']=='PASS' for r in rows) else 'INCOMPLETE',rows=rows,contrasts=contrasts,
                scientific_fits_started=sum(r['scientific_fit_started'] for r in rows),
                scientific_fits_completed=sum(r['status']=='PASS' for r in rows),scientific_fits_expected=2,
                seed=2026,selection_split='VALID',TEST='NOT_RUN',test_evaluation_count=0,
                interpretation='Exploratory single seed; no significance/backbone selection; historical dual replay is not a new independent seed')


def markdown(r):
    lines=['# Gap Trap: pilot seed2026','',
           'KuaiRand: хронологический leave-one-out, полный каталог. Только VALID; TEST не запускался.','',
           '| Variant | Parameters | VALID NDCG@10 | HR@10 | Best epoch | Epochs | First27 NDCG / complete | TRAIN / VALID s | Peak allocated / reserved bytes | Status |',
           '|---|---:|---:|---:|---:|---:|---|---:|---:|---|']
    def value(v):return '—' if v is None else str(v)
    for x in r['rows']:
        vals=[x['variant'],x['parameters'],x['ndcg10'],x['hr10'],x['best_epoch'],x['actual_epochs'],f"{value(x['first27_ndcg10'])} / {x['first27_complete']}",
              f"{value(x['train_seconds'])} / {value(x['valid_seconds'])}",f"{value(x['peak_allocated_bytes'])} / {value(x['peak_reserved_bytes'])}",x['status']]
        lines.append('| '+' | '.join(value(v) for v in vals)+' |')
    lines+=['','| Contrast | Δ VALID NDCG@10 | Relative % | Status |','|---|---:|---:|---|']
    for x in r['contrasts']:lines.append(f"| {x['comparison']} | {value(x['delta'])} | {value(x['relative_percent'])} | {x['status']} |")
    lines+=['','Один seed не устанавливает устойчивость. First27 только при полном окне0–26; early stopping до27 не отменяет успешный fit. Время включает JIT/cache эффекты. α — один общий коэффициент поправки Trap; результат не доказывает устойчивый эффект.', '']
    return '\n'.join(lines)


def write(base,reason=None):
    records={}
    for variant in c.MODES:
        p=c.paths(variant)
        if not p['result'].exists():
            create(p['result'],dict(**base,gap_trap_mode=variant,mode='dual',seed=2026,run_id=p['run_id'],
                                    status='NOT_RUN',scientific_fit_started=False,history=[],actual_epochs=0,reason=reason,finished_at=now()))
        r=read(p['result'])
        if any(r.get(k)!=v for k,v in base.items()) or r['gap_trap_mode']!=variant:
            raise ValueError('Foreign result in summary')
        if r['status']=='PASS':
            try:
                validate_record(r,variant);validate_checkpoint(r,variant)
            except (ValueError,KeyError,TypeError,OSError) as exc:
                # Keep the original run evidence, but exclude unverified metrics.
                r=dict(r,status='FAIL',best_valid_metrics={},history=[],validation_error=str(exc))
        records[variant]=r
    summary=dict(base)
    summary.update(summarize(records))
    summary['blocking_reason']=reason
    create(c.SUMMARY,summary)
    # Numeric JSON is durable even if Markdown rendering fails.
    with c.SUMMARY.with_suffix('.md').open('x') as stream:stream.write(markdown(summary))
    return summary
