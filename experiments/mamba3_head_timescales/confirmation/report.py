"""Partial-safe saved-JSON aggregation, independent pair subsets, no model work."""
import math
import statistics
from . import config as c
from experiments.mamba3_mimo_time.records import read, create, sha, now, finite_tree


def stats(values):
    return dict(n_available=len(values),mean=statistics.mean(values) if values else None,
                sample_std=statistics.stdev(values) if len(values)>1 else None,ddof=1)


def validate_record(r,task):
    if r.get('status')!='PASS':return
    if not finite_tree(r):raise ValueError('Nonfinite scientific record')
    expected=dict(run_id=task['run_id'],time_scale_mode=task['variant'],seed=task['seed'],mode='dual',
                  architecture='MIMO',backend='upstream',rank=4,chunk=8,parameter_count=c.COUNTS[task['variant']],
                  TEST='NOT_RUN',test_evaluation_count=0,scientific_fit_started=True)
    if any(r.get(k)!=v for k,v in expected.items()):raise ValueError('Scientific identity/counts/TEST mismatch')
    h=r.get('history',[])
    if not h or len(h)!=r.get('actual_epochs') or [x['epoch'] for x in h]!=list(range(len(h))):
        raise ValueError('Missing/discontinuous history')
    scores=[x['valid_ndcg10'] for x in h]
    if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in scores):raise ValueError('Nonfinite history')
    metric_keys={f'{kind}@{k}' for kind in ('hit','ndcg','recall') for k in (5,10,20,50)}
    for row in h:
        if set(row['valid_metrics'])!=metric_keys:raise ValueError('Missing/unexpected metric keys')
        if not all(math.isfinite(v) for v in row['valid_metrics'].values()):raise ValueError('Nonfinite metrics')
        if row['valid_metrics']['ndcg@10']!=row['valid_ndcg10']:raise ValueError('Selection/metric mismatch')
    best=max(i for i,v in enumerate(scores) if v==max(scores))
    if (r.get('best_epoch')!=best or r.get('best_valid_score')!=scores[best]
        or r.get('best_valid_metrics')!=h[best]['valid_metrics'] or r.get('best_diagnostics')!=h[best]['diagnostics']):
        raise ValueError('Best metrics/diagnostics/last tie mismatch')
    best_so_far=-math.inf;stale=0;stop=None
    for i,score in enumerate(scores):
        if score>=best_so_far:best_so_far=score;stale=0
        else:stale+=1
        if stale>10 and stop is None:stop=i
    if (len(h)<300 and stop!=len(h)-1) or (stop is not None and stop!=len(h)-1) or len(h)>300:
        raise ValueError('Not a complete unchanged early-stopped fit')
    if r.get('first27_complete')!=(len(h)>=27):raise ValueError('first27 completeness')
    wanted=max(scores[:27]) if len(h)>=27 else None
    if r.get('first27_best_ndcg10')!=wanted:raise ValueError('first27 score')
    pilot=read(c.pilot_path(task['variant']))
    for config in ('config','effective_config'):
        allowed={'seed','checkpoint_dir'}
        a,b=r[config],pilot[config]
        if any(a.get(k)!=b.get(k) for k in set(a)|set(b) if k not in allowed) or a['seed']!=task['seed']:
            raise ValueError('Frozen '+config+' drift')
    for key in ('protocol','manifest_sha256','train_time_stats_sha256','verified_history_stats','precision','optimizer_settings'):
        # Exact JSON representation distinguishes numeric types in canonical Adam.
        import json
        if json.dumps(r[key],sort_keys=True)!=json.dumps(pilot[key],sort_keys=True):raise ValueError('Frozen data/settings: '+key)
    required=('initial_backbone_sha256','initial_common_calibrator_hashes','initial_alpha','rng_components','first_train_batch_sha256')
    if any(k not in r for k in required):raise ValueError('Missing paired initialization/batch evidence')
    variant=task['variant']
    shape=1 if variant=='shared_tau' else 2
    alpha={} if variant=='fixed' else {m:dict(shape=[shape],values=[0.0]*shape) for m in ('decay','scan')}
    if r['initial_alpha']!=alpha or set(r['initial_common_calibrator_hashes'])!={'decay','scan'}:
        raise ValueError('Initial alpha/common mapping')
    if set(r['rng_components'])!={'python','numpy','cpu','cuda','aggregate','loader_generator'}:
        raise ValueError('Missing RNG components')
    if not r['first_train_batch_sha256']:raise ValueError('Missing consumed batch')


def validate_checkpoint(r,entry,weights=True):
    p=c.paths(entry['variant'],entry['seed'],entry['attempt'])
    meta=read(p['metadata'])
    if meta['checkpoint_sha256']!=r['checkpoint_sha256'] or (weights and sha(p['checkpoint'])!=r['checkpoint_sha256']):
        raise ValueError('Checkpoint SHA mismatch')
    for key in ('run_id','mode','time_scale_mode','seed','execution_commit','config_sha256','source_hash','core_hash'):
        if meta[key]!=r[key]:raise ValueError('Checkpoint metadata identity: '+key)
    if (meta['epoch']!=r['best_epoch'] or meta['metrics']!=r['best_valid_metrics']
        or meta['head_timescales']!=r['best_diagnostics']['head_timescales']):raise ValueError('Checkpoint selected epoch metadata')


def source_record(entry,base=None):
    path=c.ROOT/entry['result']
    r=read(path)
    if any(r.get(k)!=entry[v] for k,v in [('run_id','run_id'),('time_scale_mode','variant'),('seed','seed')]):
        raise ValueError('Wrong source-index result identity')
    if r.get('execution_attempt')!=entry['attempt']:raise ValueError('Wrong source allocation')
    saved=entry.get('preserved')
    if saved:
        if entry['attempt']!='001' or (base is not None and base['execution_attempt']!='002'):
            raise ValueError('Preserved sources only from first allocation into second')
        if sha(path)!=saved['result_sha256'] or r.get('status')!='PASS':raise ValueError('Preserved source changed')
        for key in ('job_id','execution_commit','source_hash','reservation_sha256','checkpoint_sha256'):
            if r.get(key)!=saved[key]:raise ValueError('Preserved lineage: '+key)
        if sha(c.paths(entry['variant'],entry['seed'],entry['attempt'])['metadata'])!=saved['metadata_sha256']:
            raise ValueError('Preserved metadata changed')
        parent=c.allocation('001');reservation=read(parent['reservation']);submission=read(parent['submission'])
        if r['job_id']!=submission.get('job_id') or r['reservation_sha256']!=sha(parent['reservation']):
            raise ValueError('Preserved source differs from real parent allocation')
        for key in ('execution_commit','source_hash','source_manifest_sha256','source_index_sha256','plan_sha256'):
            if r.get(key)!=reservation.get(key):raise ValueError('Preserved reservation binding: '+key)
    elif base is not None and any(r.get(k)!=v for k,v in base.items()):
        raise ValueError('Foreign result allocation/ownership')
    return r


def previous_records(variant,seed,attempt,base):
    wanted=c.tasks().index(dict(variant=variant,seed=seed,run_id=c.paths(variant,seed)['run_id']))
    previous=[]
    for entry in c.index(attempt)['entries'][:wanted]:
        r=source_record(entry,base)
        validate_record(r,entry)
        if r['status']!='PASS':raise ValueError('Earlier planned fit is not complete')
        if entry['seed']==seed:previous.append(r)
    return previous


def rows_from(records,seeds):
    rows=[]
    for seed in seeds:
        for variant in c.MODES:
            r=records.get((seed,variant),{})
            valid=r.get('status')=='PASS' and not r.get('validation_error')
            h=r.get('history',[]) if valid else []
            if not isinstance(h,list):h=[]
            complete=len(h)>=27 and [x['epoch'] for x in h[:27]]==list(range(27))
            metrics=r.get('best_valid_metrics',{}) if valid else {}
            rows.append(dict(seed=seed,variant=variant,status=r.get('status','NOT_RUN'),
                run_id=r.get('run_id'),job_id=r.get('job_id'),execution_attempt=r.get('execution_attempt'),
                source_hash=r.get('source_hash'),execution_commit=r.get('execution_commit'),
                parameters=r.get('parameter_count',c.COUNTS[variant]),ndcg10=metrics.get('ndcg@10'),hr10=metrics.get('hit@10'),
                best_epoch=r.get('best_epoch') if valid else None,actual_epochs=r.get('actual_epochs',0),
                first27_complete=complete,first27_ndcg10=max(x['valid_ndcg10'] for x in h[:27]) if valid and complete else None,
                best_available_window_ndcg10=max(x['valid_ndcg10'] for x in h[:27]) if valid and h else None,
                scientific_fit_started=r.get('scientific_fit_started',False),validation_error=r.get('validation_error'),
                train_seconds=r.get('train_seconds'),valid_seconds=r.get('valid_seconds'),
                peak_allocated_bytes=r.get('peak_gpu_allocated_bytes'),peak_reserved_bytes=r.get('peak_gpu_reserved_bytes'),
                best_diagnostics=r.get('best_diagnostics') if valid else None))
    return rows


def cohort(rows,seeds,field):
    by={(x['seed'],x['variant']):x for x in rows}
    def usable(seed,variant):
        x=by[(seed,variant)]
        return x['status']=='PASS' and x[field] is not None and not x['validation_error']
    models={}
    for variant in c.MODES:
        available=[s for s in seeds if usable(s,variant)]
        models[variant]=dict(stats([by[s,variant][field] for s in available]),seeds=available,n_expected=len(seeds))
    triples=[s for s in seeds if all(usable(s,v) for v in c.MODES)]
    common={v:dict(stats([by[s,v][field] for s in triples]),seeds=triples,n_expected=len(seeds)) for v in c.MODES}
    contrasts={}
    for left,right in c.plan()['contrasts']:
        available=[s for s in seeds if usable(s,left) and usable(s,right)]
        pairs=[dict(seed=s,left=by[s,left][field],right=by[s,right][field],delta=by[s,left][field]-by[s,right][field]) for s in available]
        diffs=[p['delta'] for p in pairs]
        a,b=stats([p['left'] for p in pairs]),stats([p['right'] for p in pairs])
        contrasts[left+'-'+right]=dict(stats(diffs),pairs=pairs,seeds=available,missing_seeds=[s for s in seeds if s not in available],
            n_expected=len(seeds),positive=sum(v>0 for v in diffs),zero=sum(v==0 for v in diffs),negative=sum(v<0 for v in diffs),
            left=a,right=b,relative_percent=100*(a['mean']/b['mean']-1) if pairs and b['mean'] else None)
    return dict(n_expected=len(seeds),complete_triples=triples,models_available=models,
                models_same_complete_triples=common,contrasts=contrasts,
                note='Compare means only on the same seeds; first27 pairs do not require a complete third variant')


def summarize(records,pilots=None):
    all_records=dict(records)
    if pilots is not None:all_records.update({(2026,v):pilots[v] for v in c.MODES})
    for (seed,variant),r in list(all_records.items()):
        try:
            name=c.paths(variant,seed)['run_id'] if seed!=2026 else f'mamba3_headtime_{variant}_seed2026_001'
            validate_record(r,dict(seed=seed,variant=variant,run_id=name))
        except (ValueError,KeyError,TypeError,AttributeError,OverflowError) as exc:
            all_records[seed,variant]=dict(r,status='FAIL',validation_error=repr(exc))
    for seed in (2026,*c.SEEDS):
        available=[(v,all_records[seed,v]) for v in c.MODES if (seed,v) in all_records and all_records[seed,v].get('status')=='PASS']
        if len(available)>1:
            reference=available[0][1]
            keys=('initial_backbone_sha256','initial_common_calibrator_hashes','rng_components','first_train_batch_sha256')
            if any(any(r[k]!=reference[k] for k in keys) for _,r in available[1:]):
                for variant,r in available:
                    all_records[seed,variant]=dict(r,status='FAIL',validation_error='Paired initial/RNG/batch mismatch')
    rows=rows_from(all_records,[2026,*c.SEEDS])
    new=[r for r in rows if r['seed'] in c.SEEDS]
    cohorts={}
    for name,seeds in [('new4',list(c.SEEDS)),('all5',[2026,*c.SEEDS])]:
        for horizon,field in [('full','ndcg10'),('first27','first27_ndcg10')]:
            cohorts[name+'_'+horizon]=cohort(rows,seeds,field)
    return dict(status='PASS' if all(r['status']=='PASS' and not r['validation_error'] for r in new) else 'INCOMPLETE',
        rows=rows,cohorts=cohorts,scientific_fits_started=sum(r['scientific_fit_started'] is True for r in new),
        scientific_fits_start_unknown=sum(r['scientific_fit_started'] is None for r in new),
        scientific_fits_completed=sum(r['status']=='PASS' and not r['validation_error'] for r in new),scientific_fits_expected=12,
        complete_new_triples=len(cohorts['new4_full']['complete_triples']),selection_split='VALID',TEST='NOT_RUN',test_evaluation_count=0,
        pilot_role='Exploratory, not a new independent historical MIMO seed',
        first27_caveat='Same histories; neither independent replication nor strictly equal GPU budget')


def markdown(summary):
    def val(x):return '—' if x is None else str(x)
    def pm(x):return '—' if x['mean'] is None else f"{x['mean']:.6f} ± {x['sample_std']:.6f}" if x['sample_std'] is not None else f"{x['mean']:.6f} (n=1)"
    lines=['# Head-timescales confirmation','',f"Status: {summary['status']}; new fits {summary['scientific_fits_completed']}/12; TEST=0.",'',
           '| Seed | Variant | Status | NDCG@10 | HR@10 | Best epoch | Epochs | First27 complete | First27 |',
           '|---|---|---|---:|---:|---:|---:|---|---:|']
    for r in summary['rows']:
        lines.append('| '+' | '.join(val(r[k]) for k in ('seed','variant','status','ndcg10','hr10','best_epoch','actual_epochs','first27_complete','first27_ndcg10'))+' |')
    for name,group in summary['cohorts'].items():
        lines+=['',f'## {name}','','| Variant | Available / expected | Seeds | Mean ± sample std |','|---|---:|---|---:|']
        for v,s in group['models_available'].items():lines.append(f"| {v} | {s['n_available']}/{s['n_expected']} | {s['seeds']} | {pm(s)} |")
        lines+=['','| Paired contrast | n / expected | Seeds | Δ mean ± std | + / 0 / − | Relative % |','|---|---:|---|---:|---:|---:|']
        for contrast,s in group['contrasts'].items():
            lines.append(f"| {contrast} | {s['n_available']}/{s['n_expected']} | {s['seeds']} | {pm(s)} | {s['positive']}/{s['zero']}/{s['negative']} | {val(s['relative_percent'])} |")
    lines+=['','First27 uses only complete observed windows0–26, with independent pair subsets. '
            'Pilot2026 is exploratory. No significance, personalization or head-specialization claim. '
            'R denotes global normalization scales. VALID is not compared to published TEST.','']
    if summary.get('blocking_reason'):lines+=['Blocking reason: '+str(summary['blocking_reason']),'']
    return '\n'.join(lines)


def write(base,attempt,reason=None):
    allocation=c.allocation(attempt);records={}
    for entry in c.index(attempt)['entries']:
        p=c.ROOT/entry['result']
        if not p.exists():
            if entry['attempt']!=attempt:raise ValueError('Missing preserved source')
            create(p,dict(**base,run_id=entry['run_id'],mode='dual',time_scale_mode=entry['variant'],seed=entry['seed'],
                          status='NOT_RUN',scientific_fit_started=False,history=[],actual_epochs=0,blocking_reason=reason,finished_at=now()))
        raw=dict(run_id=entry['run_id'],time_scale_mode=entry['variant'],seed=entry['seed'],
                 status='FAIL',scientific_fit_started=None,history=[],actual_epochs=0)
        try:
            decoded=read(p)
            if not isinstance(decoded,dict):raise ValueError('Result must be a JSON object')
            raw=decoded
            r=source_record(entry,base)
            validate_record(r,entry)
            if r['status']=='PASS':validate_checkpoint(r,entry)
        except (ValueError,KeyError,TypeError,AttributeError,OverflowError,OSError) as exc:
            r=dict(raw,status='FAIL',validation_error=repr(exc))
        records[entry['seed'],entry['variant']]=r
    result=dict(base)
    result.update(summarize(records,{v:read(c.pilot_path(v)) for v in c.MODES}))
    result['blocking_reason']=reason
    create(allocation['summary'],result)
    with allocation['summary'].with_suffix('.md').open('x') as stream:stream.write(markdown(result))
    return result
