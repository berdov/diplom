"""Exact replay, prefix admission, paired summaries; no model forward."""
import copy
import math
import statistics
from types import SimpleNamespace
from . import config as c
from ... import report as pilot_report
from ..reuse import bind
from experiments.mamba3_mimo_time.records import read, create, sha


def replay_check(record):
    ref=read(c.historical(record['seed']))
    keys=('seed','checkpoint_sha256','best_epoch','best_valid_metrics','actual_epochs','first27_best_ndcg10',
          'first_train_batch_sha256','initial_backbone_sha256','rng_components','optimizer_settings','precision',
          'protocol','manifest_sha256','train_time_stats_sha256','verified_history_stats')
    differences=[k for k in keys if record.get(k)!=ref.get(k)]
    if record.get('initial_common_calibrator_hashes')!=ref['initial_calibrator_hashes']:differences.append('initial_calibrators')
    for key in ('config','effective_config'):
        a,b=record[key],ref[key]
        if any(a.get(k)!=b.get(k) for k in set(a)|set(b) if k not in ('checkpoint_dir','gap_trap_mode')):differences.append(key)
    if len(record['history'])!=len(ref['history']):differences.append('history_length')
    for i,(a,b) in enumerate(zip(record['history'],ref['history'])):
        for key in ('epoch','valid_ndcg10','valid_metrics','train_loss'):
            if a[key]!=b[key]:differences.append(f'history[{i}].{key}')
        if {k:v for k,v in a['diagnostics'].items() if k!='gap_trap'}!=b['diagnostics']:differences.append(f'history[{i}].diagnostics')
    if differences:raise ValueError('Historical fixed replay mismatch: '+repr(differences))
    return dict(status='PASS',historical_run_id=ref['run_id'],historical_job_id=ref['job_id'],
                checkpoint_sha256=ref['checkpoint_sha256'],scientific_history_exact=True,timing_memory_excluded=True)


def validate_record(record,variant,seed):
    # Reuse unchanged selection/history validator; normalize only its literal
    # seed2026 identity after validating the real planned seed independently.
    if record.get('seed')!=seed or seed not in c.SEEDS:raise ValueError('Unplanned run seed')
    adapter=SimpleNamespace(paths=lambda v:c.paths(v,seed),COUNTS=c.COUNTS)
    normalized=dict(record,seed=2026)
    bind(pilot_report,{'c':adapter})['validate_record'](normalized,variant)
    if record.get('status')=='PASS':
        from .state import require_initial
        require_initial(record)
        reference=read(c.pilot_path(variant))
        for key in ('protocol','manifest_sha256','train_time_stats_sha256','verified_history_stats','precision','optimizer_settings'):
            if record[key]!=reference[key]:raise ValueError('Saved data/protocol/optimizer drift: '+key)
        for row in record['history']:
            alpha=row['diagnostics']['gap_trap']['alpha']
            if not isinstance(alpha,(int,float)) or not 0<=alpha<=1 or (variant=='fixed_replay' and alpha!=0):raise ValueError('Saved alpha diagnostics')
        if record['first_train_batch_sha256']!=read(c.historical(seed))['first_train_batch_sha256']:raise ValueError('Saved consumed batch drift')
        for setting in ('config','effective_config'):
            a,b=record[setting],reference[setting]
            if a.get('seed')!=seed:raise ValueError('Saved config seed drift')
            if any(a.get(k)!=b.get(k) for k in set(a)|set(b) if k not in ('seed','checkpoint_dir')):raise ValueError('Pilot config drift')
        if variant=='fixed_replay':replay_check(record)


def validate_checkpoint(record,variant,seed):
    adapter=SimpleNamespace(paths=lambda v:c.paths(v,seed))
    bind(pilot_report,{'c':adapter})['validate_checkpoint'](record,variant)
    meta=read(c.paths(variant,seed)['metadata'])
    if meta['gap_trap']!=record['best_diagnostics']['gap_trap']:raise ValueError('Checkpoint alpha diagnostics')


def successful(task,base=None):
    from .provenance import validate_record_owner
    record=read(c.paths(task['variant'],task['seed'])['result'])
    if record.get('status')!='PASS':raise ValueError('Earlier fit incomplete/failed; no progression')
    validate_record_owner(record);validate_record(record,task['variant'],task['seed'])
    validate_checkpoint(record,task['variant'],task['seed'])
    if base and any(record[k]!=base[k] for k in ('study_id','execution_commit','source_hash')):raise ValueError('Source changed between fits')
    return record


def completed_prefix(base=None):
    records=[];missing=False
    for task in c.tasks():
        p=c.paths(task['variant'],task['seed'])
        # The parent creates process stdout/stderr before the child can claim
        # a run. A process directory alone is not a scientific start.
        occupied=any(p[k].exists() or p[k].is_symlink() for k in ('result','lock','checkpoint','metadata'))
        if not occupied:missing=True;continue
        if missing:raise ValueError('Scientific task order or incomplete output namespace')
        records.append(successful(task,base))
    return records


def assert_next_task(variant,seed,base):
    records=completed_prefix(base);tasks=c.tasks()
    if len(records)==len(tasks) or tasks[len(records)]!=dict(variant=variant,seed=seed,run_id=c.paths(variant,seed)['run_id']):
        raise ValueError('Only next never-started planned fit allowed')
    reservation=read(c.allocation(base['execution_attempt'])['reservation'])
    if tasks[len(records)] not in reservation['tasks']:raise ValueError('Fit not reserved for this allocation')


def previous_records(variant,seed,base):
    if variant=='fixed_replay':return []
    return [successful(dict(variant='fixed_replay',seed=seed),base)]


def stats(values):
    return dict(n=len(values),mean=statistics.mean(values) if values else None,
                sample_std=statistics.stdev(values) if len(values)>1 else None)


def cohort(records,seeds,key):
    pairs=[]
    for seed in seeds:
        a,b=(records.get((seed,v)) for v in c.MODES)
        if not a or not b or a.get('status')!='PASS' or b.get('status')!='PASS':continue
        x,y=a.get(key),b.get(key)
        if x is None or y is None:continue
        pairs.append(dict(seed=seed,fixed=x,centered=y,delta=y-x))
    x=[r['fixed'] for r in pairs];y=[r['centered'] for r in pairs];d=[r['delta'] for r in pairs]
    return dict(pairs=pairs,fixed=stats(x),centered=stats(y),paired_delta=stats(d),
                signs=dict(positive=sum(v>0 for v in d),negative=sum(v<0 for v in d),zero=sum(v==0 for v in d)),
                relative_means_percent=100*(statistics.mean(y)/statistics.mean(x)-1) if x and statistics.mean(x) else None)


def summarize(records):
    rows=[];lookup={};errors=[]
    for task in c.tasks():
        r=records.get(task['run_id'],{});variant,seed=task['variant'],task['seed']
        error=r.get('validation_error')
        if r.get('status')=='PASS':
            try:validate_record(r,variant,seed)
            except Exception as exc:error=repr(exc);r=dict(r,status='INVALID')
        if error:errors.append(dict(run_id=task['run_id'],error=error))
        if r.get('status')!='PASS':
            # A process can stop between the parent validation save and the
            # centered diagnostics save. Preserve raw files, not invalid metrics.
            r={k:r[k] for k in ('status','scientific_fit_started','actual_epochs') if k in r}
            if not isinstance(r.get('actual_epochs',0),int):r['actual_epochs']=0
            if r.get('scientific_fit_started',False) not in (True,False):r['scientific_fit_started']=None
        h=r.get('history',[]);d=r.get('best_diagnostics',{}).get('gap_trap',{})
        alpha=[x['diagnostics']['gap_trap']['alpha'] for x in h]
        row=dict(**task,status=r.get('status','NOT_RUN'),validation_error=error,scientific_fit_started=r.get('scientific_fit_started',False),
            best_valid_metrics=r.get('best_valid_metrics'),ndcg10=r.get('best_valid_metrics',{}).get('ndcg@10'),
            hr10=r.get('best_valid_metrics',{}).get('hit@10'),first27=r.get('first27_best_ndcg10'),
            first27_complete=r.get('first27_complete',False),best_epoch=r.get('best_epoch'),actual_epochs=r.get('actual_epochs',0),
            train_seconds=r.get('train_seconds'),valid_seconds=r.get('valid_seconds'),
            peak_gpu_allocated_bytes=r.get('peak_gpu_allocated_bytes'),peak_gpu_reserved_bytes=r.get('peak_gpu_reserved_bytes'),
            checkpoint_sha256=r.get('checkpoint_sha256'),alpha_best=d.get('alpha'),alpha_max=max(alpha) if alpha else None,
            alpha_final=alpha[-1] if alpha else None,zero_alpha_epochs=alpha.count(0.),alpha_trajectory=alpha,best_gap_trap_diagnostics=d)
        rows.append(row);lookup[seed,variant]=row
    primary=cohort(lookup,c.SEEDS,'ndcg10');first27=cohort(lookup,c.SEEDS,'first27')
    complete=all(r['status']=='PASS' for r in rows) and len(primary['pairs'])==4 and not errors
    all5=None
    if complete:
        extended=dict(lookup)
        for variant in c.MODES:
            p=read(c.pilot_path(variant));extended[2026,variant]=dict(status='PASS',ndcg10=p['best_valid_metrics']['ndcg@10'],first27=p['first27_best_ndcg10'])
        all5=dict(label='including exploratory pilot',full_run=cohort(extended,(2026,*c.SEEDS),'ndcg10'),first27=cohort(extended,(2026,*c.SEEDS),'first27'))
    return dict(status='PASS' if complete else 'INCOMPLETE',rows=rows,validation_errors=errors,
        scientific_fits_started=sum(r['scientific_fit_started'] is True for r in rows),unknown_scientific_starts=sum(r['scientific_fit_started'] is None for r in rows),
        scientific_fits_completed=sum(r['status']=='PASS' for r in rows),
        complete_pairs=len(primary['pairs']),scientific_fits_expected=8,pairs_expected=4,
        primary_new_seeds=primary if complete else None,first27_new_seeds=first27 if complete else None,
        all5_including_exploratory_pilot=all5,TEST='NOT_RUN',test_evaluation_count=0,
        interpretation='Complete new4 primary; pilot separate; first27 descriptive; sample std ddof1; no p-values, TEST or causal claims')


def write(base,reason=None):
    from .provenance import validate_record_owner
    records={}
    for task in c.tasks():
        path=c.paths(task['variant'],task['seed'])['result']
        if path.exists():
            r={}
            try:
                r=read(path);validate_record_owner(r)
                if r['status']=='PASS':validate_checkpoint(r,task['variant'],task['seed'])
            except Exception as exc:r=dict(r,status='INVALID',scientific_fit_started=r.get('scientific_fit_started'),validation_error=repr(exc))
            records[task['run_id']]=r
        elif c.paths(task['variant'],task['seed'])['lock'].exists():
            records[task['run_id']]=dict(status='INVALID',scientific_fit_started=None,validation_error='Run lock exists without readable result')
    value=dict(base);value.update(summarize(records));value['blocking_reason']=reason
    create(c.allocation(base['execution_attempt'])['summary'],value)
    return value
