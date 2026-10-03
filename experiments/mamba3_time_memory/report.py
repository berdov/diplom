"""Saved-record validation and partial-safe summaries; no model or weights."""
from . import config as c
from experiments.mamba3_gap_trap import report as history_report
from experiments.mamba3_gap_trap.centered.reuse import bind
from experiments.mamba3_mimo_time.records import read,sha,create

_validate=bind(history_report,{'c':c})['validate_record']
PAIRING=('initial_backbone_sha256','initial_common_calibrator_hashes','rng_components','protocol','manifest_sha256','train_time_stats_sha256','verified_history_stats','precision','optimizer_settings','first_train_batch_sha256')


def replay_check(r):
    old=read(c.PILOT)
    keys=('seed','checkpoint_sha256','best_epoch','best_valid_metrics','actual_epochs','first27_best_ndcg10','first_train_batch_sha256','initial_backbone_sha256','rng_components','optimizer_settings','precision','protocol','manifest_sha256','train_time_stats_sha256','verified_history_stats')
    differences=[k for k in keys if r.get(k)!=old.get(k)]
    if r.get('initial_common_calibrator_hashes')!=old['initial_calibrator_hashes']:differences.append('initial_calibrators')
    for key in ('config','effective_config'):
        a,b=r[key],old[key]
        if any(a.get(k)!=b.get(k) for k in set(a)|set(b) if k not in ('checkpoint_dir','memory_mode')):differences.append(key)
    if len(r['history'])!=len(old['history']):differences.append('history_length')
    for i,(a,b) in enumerate(zip(r['history'],old['history'])):
        for key in ('epoch','valid_ndcg10','valid_metrics','train_loss','diagnostics'):
            if a[key]!=b[key]:differences.append(f'history[{i}].{key}')
    if differences:raise ValueError('Historical control replay mismatch; investigate without refit: '+repr(differences))
    return dict(status='PASS',historical_run_id=old['run_id'],historical_job_id=old['job_id'],checkpoint_sha256=old['checkpoint_sha256'],scientific_history_exact=True,timing_memory_excluded=True)


def validate_record(r,variant):
    if r.get('status')!='PASS':return
    if r.get('memory_mode')!=variant:raise ValueError('Variant identity')
    _validate(dict(r,gap_trap_mode=variant),variant)
    old=read(c.PILOT)
    for key in ('protocol','manifest_sha256','train_time_stats_sha256','verified_history_stats','precision','optimizer_settings','first_train_batch_sha256','initial_backbone_sha256','rng_components'):
        if r[key]!=old[key]:raise ValueError('Frozen state/data drift '+key)
    if r['initial_common_calibrator_hashes']!=old['initial_calibrator_hashes'] or r['initial_beta']!=(None if variant=='no_memory' else 0.):raise ValueError('Initial temporal maps/beta')
    for key in ('config','effective_config'):
        a,b=r[key],old[key]
        if a.get('memory_mode')!=variant:raise ValueError('Config variant identity')
        if any(a.get(k)!=b.get(k) for k in set(a)|set(b) if k not in ('checkpoint_dir','memory_mode')):raise ValueError('Frozen config drift')
    if variant=='no_memory':replay_check(r)


def validate_checkpoint(r,variant):
    p=c.paths(variant);meta=read(p['metadata'])
    if sha(p['checkpoint'])!=r['checkpoint_sha256'] or meta['checkpoint_sha256']!=r['checkpoint_sha256']:raise ValueError('Checkpoint SHA')
    for k in ('run_id','mode','memory_mode','seed','execution_commit','config_sha256','source_hash','core_hash'):
        if meta[k]!=r[k]:raise ValueError('Checkpoint owner '+k)
    if meta['epoch']!=r['best_epoch'] or meta['metrics']!=r['best_valid_metrics'] or meta['memory']!=r['best_diagnostics'].get('memory'):raise ValueError('Checkpoint selection')


def summarize(records):
    rows=[]
    for variant in c.MODES:
        r=records.get(variant,{});error=r.get('validation_error')
        try:validate_record(r,variant)
        except (ValueError,KeyError,TypeError) as exc:error=str(exc);r=dict(r,status='INVALID')
        good=r.get('status')=='PASS'
        row=dict(variant=variant,status=r.get('status','NOT_RUN'),validation_error=error,parameters=c.COUNTS[variant],scientific_fit_started=r.get('scientific_fit_started',False),actual_epochs=r.get('actual_epochs',0),**{k:r.get(k) if good else None for k in ('best_valid_metrics','best_epoch','first27_complete','first27_best_ndcg10','train_seconds','valid_seconds','peak_gpu_allocated_bytes','peak_gpu_reserved_bytes','checkpoint_sha256','best_diagnostics')})
        if good and variant!='no_memory':
            values=[x['diagnostics']['memory'] for x in r['history']]
            row['gate_boundaries']={k:dict(best=r['best_diagnostics']['memory'][k],final=values[-1][k],max_abs=max(abs(x[k]) for x in values),negative_count=sum(x[k]<0 for x in values),observed_epochs=len(values),negative_fraction=sum(x[k]<0 for x in values)/len(values)) for k in ('beta','lambda')}
        rows.append(row)
    by={x['variant']:x for x in rows};contrasts=[]
    for left,right in c.plan()['contrasts']:
        a,b=by[left],by[right];complete=a['status']==b['status']=='PASS';delta=relative=first=None
        if complete:
            x,y=a['best_valid_metrics']['ndcg@10'],b['best_valid_metrics']['ndcg@10'];delta=x-y;relative=100*delta/y if y else None
            if a['first27_complete'] and b['first27_complete']:first=a['first27_best_ndcg10']-b['first27_best_ndcg10']
        contrasts.append(dict(comparison=left+' - '+right,status='COMPLETE' if complete else 'NOT_AVAILABLE',delta=delta,relative_percent=relative,first27_delta=first))
    return dict(status='PASS' if all(x['status']=='PASS' for x in rows) else 'INCOMPLETE',rows=rows,contrasts=contrasts,primary_contrast='time_memory - index_memory',scientific_fits_started=sum(x['scientific_fit_started'] is True for x in rows),scientific_fits_completed=sum(x['status']=='PASS' for x in rows),scientific_fits_expected=3,seed=2026,TEST='NOT_RUN',test_evaluation_count=0,interpretation='Single paired seed; supplied 50-event window representations, not native SSM states or persistent memory; VALID only')


def write(base,reason=None):
    records={}
    for v in c.MODES:
        p=c.paths(v)
        if not p['result'].exists():
            create(p['result'],dict(**base,memory_mode=v,mode='dual',seed=2026,run_id=p['run_id'],status='NOT_RUN',scientific_fit_started=False,actual_epochs=0,history=[],reason=reason))
        r=read(p['result'])
        if any(r.get(k)!=value for k,value in base.items()) or r.get('memory_mode')!=v:raise ValueError('Foreign run record')
        if r['status']=='PASS':
            try:validate_record(r,v);validate_checkpoint(r,v)
            except (ValueError,KeyError,TypeError,OSError) as exc:r=dict(r,status='INVALID',validation_error=str(exc))
        records[v]=r
    result=dict(base);result.update(summarize(records));result['blocking_reason']=reason
    if result['status']=='PASS':
        for v in c.MODES[1:]:
            if any(records[v][k]!=records['no_memory'][k] for k in PAIRING):raise ValueError('Within-triple pairing mismatch')
        result['fresh_control_replay']=replay_check(records['no_memory'])
    create(c.SUMMARY,result);return result
