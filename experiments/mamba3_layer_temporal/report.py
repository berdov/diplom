"""No forward or weight loading; exact control replay and partial-safe summaries."""
from . import config as c
from experiments.mamba3_gap_trap import report as history_report
from experiments.mamba3_gap_trap.centered.reuse import bind
from experiments.mamba3_mimo_time.records import read,sha,create,now

_validate=bind(history_report,{'c':c})['validate_record']


def replay_check(r):
    old=read(c.PILOT)
    keys=('seed','checkpoint_sha256','best_epoch','best_valid_metrics','actual_epochs','first27_best_ndcg10',
          'first_train_batch_sha256','initial_backbone_sha256','rng_components','optimizer_settings','precision',
          'protocol','manifest_sha256','train_time_stats_sha256','verified_history_stats')
    differences=[k for k in keys if r.get(k)!=old.get(k)]
    if r.get('initial_common_calibrator_hashes')!=old['initial_calibrator_hashes']:differences.append('initial_calibrators')
    for key in ('config','effective_config'):
        a,b=r[key],old[key]
        if any(a.get(k)!=b.get(k) for k in set(a)|set(b) if k not in ('checkpoint_dir','temporal_sharing')):differences.append(key)
    if len(r['history'])!=len(old['history']):differences.append('history_length')
    for i,(a,b) in enumerate(zip(r['history'],old['history'])):
        for key in ('epoch','valid_ndcg10','valid_metrics','train_loss'):
            if a[key]!=b[key]:differences.append(f'history[{i}].{key}')
        if {k:v for k,v in a['diagnostics'].items() if k!='layer_temporal'}!=b['diagnostics']:differences.append(f'history[{i}].diagnostics')
    if differences:raise ValueError('Historical control replay mismatch: '+repr(differences))
    return dict(status='PASS',historical_run_id=old['run_id'],historical_job_id=old['job_id'],checkpoint_sha256=old['checkpoint_sha256'],
                scientific_history_exact=True,timing_memory_excluded=True)


def validate_record(r,variant):
    if r.get('status')!='PASS':return
    if r.get('temporal_sharing')!=variant:raise ValueError('Variant identity')
    _validate(dict(r,gap_trap_mode=variant),variant)
    old=read(c.PILOT)
    for key in ('protocol','manifest_sha256','train_time_stats_sha256','verified_history_stats','precision','optimizer_settings','first_train_batch_sha256','initial_backbone_sha256','rng_components'):
        if r[key]!=old[key]:raise ValueError('Frozen state/data drift '+key)
    if r['initial_common_calibrator_hashes']!=old['initial_calibrator_hashes'] or r['initial_layer_calibrator_hashes']!=[old['initial_calibrator_hashes']]*2:raise ValueError('Initial temporal maps')
    for key in ('config','effective_config'):
        a,b=r[key],old[key]
        if a.get('temporal_sharing')!=variant:raise ValueError('Config sharing identity')
        if any(a.get(k)!=b.get(k) for k in set(a)|set(b) if k not in ('checkpoint_dir','temporal_sharing')):raise ValueError('Frozen config drift')
    if variant=='shared_layers':replay_check(r)


def validate_checkpoint(r,variant):
    p=c.paths(variant);meta=read(p['metadata'])
    if sha(p['checkpoint'])!=r['checkpoint_sha256'] or meta['checkpoint_sha256']!=r['checkpoint_sha256']:raise ValueError('Checkpoint SHA')
    for k in ('run_id','mode','temporal_sharing','seed','execution_commit','config_sha256','source_hash','core_hash'):
        if meta[k]!=r[k]:raise ValueError('Checkpoint owner '+k)
    if meta['epoch']!=r['best_epoch'] or meta['metrics']!=r['best_valid_metrics'] or meta['layer_temporal']!=r['best_diagnostics']['layer_temporal']:raise ValueError('Checkpoint selection')


def summarize(records):
    rows=[]
    for variant in c.MODES:
        r=records.get(variant,{});error=r.get('validation_error')
        try:validate_record(r,variant)
        except (ValueError,KeyError,TypeError) as exc:error=str(exc);r=dict(r,status='INVALID')
        good=r.get('status')=='PASS'
        rows.append(dict(variant=variant,status=r.get('status','NOT_RUN'),validation_error=error,parameters=c.COUNTS[variant],
            scientific_fit_started=r.get('scientific_fit_started',False),actual_epochs=r.get('actual_epochs',0),
            **{k:r.get(k) if good else None for k in ('best_valid_metrics','best_epoch','first27_best_ndcg10','train_seconds','valid_seconds',
            'peak_gpu_allocated_bytes','peak_gpu_reserved_bytes','checkpoint_sha256','best_diagnostics')}))
    complete=all(r['status']=='PASS' for r in rows);delta=relative=first=None
    if complete:
        f,z=(r['best_valid_metrics']['ndcg@10'] for r in rows);delta=z-f;relative=100*delta/f if f else None
        if all(r['first27_best_ndcg10'] is not None for r in rows):first=rows[1]['first27_best_ndcg10']-rows[0]['first27_best_ndcg10']
    return dict(status='PASS' if complete else 'INCOMPLETE',rows=rows,primary_contrast='layer_specific - shared_layers',
        delta=delta,relative_percent=relative,first27_delta=first,scientific_fits_started=sum(r['scientific_fit_started'] is True for r in rows),
        scientific_fits_completed=sum(r['status']=='PASS' for r in rows),scientific_fits_expected=2,seed=2026,TEST='NOT_RUN',test_evaluation_count=0,
        interpretation='Single paired seed; VALID only; no significance or automatic confirmation')


def write(base,reason=None):
    records={}
    for v in c.MODES:
        p=c.paths(v)
        if not p['result'].exists():
            create(p['result'],dict(**base,temporal_sharing=v,mode='dual',seed=2026,run_id=p['run_id'],status='NOT_RUN',
                scientific_fit_started=False,actual_epochs=0,history=[],reason=reason,finished_at=now()))
        r=read(p['result'])
        if any(r.get(k)!=value for k,value in base.items()) or r.get('temporal_sharing')!=v:raise ValueError('Foreign run record')
        if r['status']=='PASS':
            try:validate_record(r,v);validate_checkpoint(r,v)
            except (ValueError,KeyError,TypeError,OSError) as exc:r=dict(r,status='INVALID',validation_error=str(exc))
        records[v]=r
    result=dict(base);result.update(summarize(records));result['blocking_reason']=reason
    if result['status']=='PASS':
        from .state import paired
        paired(records[c.MODES[0]],records[c.MODES[1]],first_batch=True)
        result['fresh_control_replay']=replay_check(records[c.MODES[0]])
    create(c.SUMMARY,result)
    return result
