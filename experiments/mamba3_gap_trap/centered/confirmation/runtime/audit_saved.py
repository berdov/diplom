"""Independent saved-artifact audit; standard library, no weights or data forward."""
import argparse
import hashlib
import json
import math
import re
import struct
import subprocess
from pathlib import Path
from types import SimpleNamespace
from experiments.mamba3_gap_trap.centered.confirmation import config as c, report, provenance as p
from experiments.mamba3_gap_trap import report as history_report
from experiments.mamba3_gap_trap.centered.reuse import bind
from experiments.mamba3_mimo_time.records import read, sha, create, now

PAIRING=('initial_backbone_sha256','initial_common_calibrator_hashes','rng_components','protocol','manifest_sha256',
         'train_time_stats_sha256','verified_history_stats','precision','optimizer_settings','first_train_batch_sha256')


def analytic_equal(saved,expected):
    return math.isfinite(saved) and math.isfinite(expected) and abs(saved-expected)<=math.ulp(expected)


def validate_saved_record(r,mode,seed):
    if r.get('status')!='PASS' or r.get('seed')!=seed or seed not in c.SEEDS:raise ValueError('Incomplete or foreign record')
    adapter=SimpleNamespace(paths=lambda v:c.paths(v,seed),COUNTS=c.COUNTS)
    bind(history_report,{'c':adapter})['validate_record'](dict(r,seed=2026),mode)
    pilot,reference=read(c.pilot_path(mode)),read(c.historical(seed))
    for setting in ('config','effective_config'):
        a,b=r[setting],pilot[setting]
        if a['seed']!=seed or any(a.get(k)!=b.get(k) for k in set(a)|set(b) if k not in ('seed','checkpoint_dir')):raise ValueError('Frozen '+setting)
    if hashlib.sha256(json.dumps(r['config'],sort_keys=True).encode()).hexdigest()!=r['config_sha256']:raise ValueError('Config SHA')
    for key in ('protocol','manifest_sha256','train_time_stats_sha256','verified_history_stats','precision','optimizer_settings'):
        if r[key]!=pilot[key]:raise ValueError('Frozen pilot '+key)
    for actual,old in [('initial_backbone_sha256','initial_backbone_sha256'),('initial_common_calibrator_hashes','initial_calibrator_hashes'),
                       ('rng_components','rng_components'),('first_train_batch_sha256','first_train_batch_sha256')]:
        if r[actual]!=reference[old]:raise ValueError('Historical initialization/RNG/batch '+actual)
    if r['initial_alpha']!=({} if mode=='fixed_replay' else {'gap_trap.alpha':0.}):raise ValueError('Initial alpha')
    if set(r['rng_components'])!={'python','numpy','cpu','cuda','aggregate','loader_generator'}:raise ValueError('RNG fields')
    if set(r['initial_common_calibrator_hashes'])!={'decay','scan'}:raise ValueError('Calibrator fields')
    if mode=='fixed_replay':report.replay_check(r)


def audit_run(r,runtime,checkpoints,analytic_rounding):
    mode,seed,run_id=r['gap_trap_mode'],r['seed'],r['run_id']
    validate_saved_record(r,mode,seed)
    meta=read(runtime/'checkpoints/best_metadata.json')
    rel=str(c.paths(mode,seed)['checkpoint'].relative_to(c.HERE))
    if checkpoints[rel]['sha256']!=r['checkpoint_sha256'] or meta['checkpoint_sha256']!=r['checkpoint_sha256']:raise ValueError('Checkpoint SHA')
    expected_path='/home/daryumin/iberdov/diplom/'+str(c.paths(mode,seed)['checkpoint'].relative_to(c.ROOT))
    if r['checkpoint_path']!=expected_path:raise ValueError('Checkpoint path binding')
    for key in ('run_id','mode','gap_trap_mode','seed','execution_commit','config_sha256','source_hash','core_hash'):
        if meta[key]!=r[key]:raise ValueError('Checkpoint metadata '+key)
    if meta['epoch']!=r['best_epoch'] or meta['metrics']!=r['best_valid_metrics'] or meta['gap_trap']!=r['best_diagnostics']['gap_trap']:raise ValueError('Checkpoint selection')
    paths=list((runtime/'log').rglob('*.log'))
    if len(paths)!=1:raise ValueError('Ambiguous metric log')
    content=paths[0].read_text()
    process=re.sub(r'\[[0-?]*[ -/]*[@-~]','',(runtime/'process/stderr.log').read_text())
    evaluations=re.findall(r'epoch (\d+) evaluating \[time: [^,]+, valid_score: ([0-9.]+)\]',content)
    metrics=[dict((k,float(v)) for k,v in re.findall(r'((?:hit|ndcg|recall)@\d+) : ([0-9.]+)',line)) for line in content.splitlines() if line.startswith('hit@5 :')]
    training=re.findall(r'epoch (\d+) training \[time: [^,]+, train loss: ([0-9.]+)\]',content)
    if len(evaluations)!=len(metrics) or len(evaluations)!=len(training) or len(evaluations)!=r['actual_epochs']:raise ValueError('Epoch log count')
    if re.findall(r'epoch (\d+) evaluating \[time: [^,]+, valid_score: ([0-9.]+)\]',process)!=evaluations:raise ValueError('Process evaluating log')
    process_metrics=[dict((k,float(v)) for k,v in re.findall(r'((?:hit|ndcg|recall)@\d+) : ([0-9.]+)',line)) for line in process.splitlines() if line.startswith('hit@5 :')]
    if process_metrics!=metrics or re.findall(r'epoch (\d+) training \[time: [^,]+, train loss: ([0-9.]+)\]',process)!=training:raise ValueError('Process metrics/loss log')
    for index,((epoch,score),metric,(te,loss),row) in enumerate(zip(evaluations,metrics,training,r['history'])):
        if int(epoch)!=index or int(te)!=index or float(score)!=row['valid_ndcg10'] or metric!=row['valid_metrics']:raise ValueError('Epoch metrics/log')
        if float(loss)!=round(row['train_loss'],4):raise ValueError('Train loss/log')
        diag=row['diagnostics']['gap_trap'];alpha=diag['alpha']
        q=[(g-1)/(g+1) for g in c.plan()['diagnostic_grid']]
        shifts=[alpha*x for x in q]
        if not 0<=alpha<=1 or diag['q_centered']!=q or diag['logit_shift']!=shifts:raise ValueError('Centered diagnostics')
        if diag['alpha_bounds']!=[0.,1.] or diag['at_lower_bound']!=(alpha==0.) or diag['at_upper_bound']!=(alpha==1.) or diag['shared_across_heads_and_layers'] is not True:raise ValueError('Alpha flags')
        if diag['gaps_over_R0']!=c.plan()['diagnostic_grid'] or diag['content_logits']!=c.plan()['diagnostic_content_logits']:raise ValueError('Diagnostic grid')
        analytic=[('odds_multiplier',diag['odds_multiplier'],[math.exp(x) for x in shifts])]
        fractions=diag['current_fraction'];logits=c.plan()['diagnostic_content_logits']
        if len(fractions)!=len(logits):raise ValueError('Sigmoid diagnostic shape')
        analytic.extend(('current_fraction_'+str(t),actual,[1/(1+math.exp(-t-v)) for v in shifts]) for t,actual in zip(logits,fractions))
        for name,actual,expected in analytic:
            if len(actual)!=len(expected):raise ValueError('Analytic diagnostic shape')
            for grid_index,(a,b) in enumerate(zip(actual,expected)):
                if not analytic_equal(a,b):raise ValueError('Odds/sigmoid diagnostics')
                if a!=b:analytic_rounding.append(dict(run_id=r['run_id'],variant=mode,epoch=index,field=name,grid_index=grid_index,saved=a,recomputed=b,ulps=abs(a-b)/math.ulp(b)))
        def fp32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
        def bf16(x):
            bits=struct.unpack('<I',struct.pack('<f',x))[0]
            bits=(bits+0x7fff+((bits>>16)&1))&0xffff0000
            return struct.unpack('<f',struct.pack('<I',bits))[0]
        actual=[fp32(fp32(alpha)*fp32(x)) for x in q]
        effect=[[fp32(bf16(fp32(fp32(t)+v))-fp32(t)) for v in actual] for t in c.plan()['diagnostic_content_logits']]
        count=3*sum(x!=0 for x in actual);zeros=sum(e==0 and v!=0 for row_effect in effect for e,v in zip(row_effect,actual))
        expected_bf=dict(shift_before_cast=actual,shift_cast_alone=[bf16(x) for x in actual],effective_shift_after_add_and_cast=effect,nonzero_shift_grid_cases=count,rounded_to_zero_cases=zeros,rounded_to_zero_fraction=zeros/count if count else None)
        if diag['bf16']!=expected_bf:raise ValueError('BF16 analytic evidence')
    if len(r['history'])<r['config']['epochs'] and re.findall(r'Finished training, best eval result in epoch (\d+)\b',content)!=[str(r['best_epoch'])]:raise ValueError('Terminal selection log')
    for key in ('train_seconds','valid_seconds'):
        if r[key]!=sum(x[key] for x in r['history']):raise ValueError('Time sum')
    for kind in ('allocated','reserved'):
        if r['peak_gpu_'+kind+'_bytes']!=max(max(x['train_peak_'+kind+'_bytes'],x['valid_peak_'+kind+'_bytes']) for x in r['history']):raise ValueError('Memory peak')
    alpha=[x['diagnostics']['gap_trap']['alpha'] for x in r['history']]
    if mode=='fixed_replay' and any(alpha):raise ValueError('Fixed alpha nonzero')
    return dict(run_id=run_id,variant=mode,seed=seed,epochs=len(alpha),metric_cells=12*len(alpha),
        checkpoint_sha256=r['checkpoint_sha256'],best_alpha=r['best_diagnostics']['gap_trap']['alpha'],
        final_alpha=alpha[-1],max_epoch_alpha=max(alpha),zero_alpha_epochs=alpha.count(0.))


def compare_summary(saved,expected,path='',rounding=None):
    if isinstance(expected,dict):
        for k,v in expected.items():compare_summary(saved[k],v,path+'.'+k,rounding)
    elif isinstance(expected,list):
        if len(saved)!=len(expected):raise ValueError('Summary list shape '+path)
        for i,v in enumerate(expected):compare_summary(saved[i],v,path+f'[{i}]',rounding)
    elif saved!=expected:
        # Python versions use different final square-root paths for sample std.
        if path.endswith('.sample_std') and isinstance(expected,float) and math.isfinite(saved) and abs(saved-expected)<=2*math.ulp(expected):
            rounding.append(dict(path=path,saved=saved,recomputed=expected,ulps=abs(saved-expected)/math.ulp(expected)))
        else:raise ValueError('Summary recomputation '+path+': '+repr((saved,expected)))


def audit(folder,terminal=True):
    saved=read(folder/'preservation_manifest.json');files=folder/'files';manifest=p.verify()
    execution=saved.get('checkout_commit',saved.get('execution_commit'))
    if execution!='23468e74389aed841aab0ac16b370c5ce376e328':raise ValueError('Unplanned execution')
    for row in saved['files']:
        path=files/row['path']
        if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:raise ValueError('Preservation SHA: '+row['path'])
    for name,expected in manifest['files'].items():
        if hashlib.sha256(subprocess.check_output(['git','show',execution+':'+name],cwd=c.ROOT)).hexdigest()!=expected:raise ValueError('Execution source '+name)
    bases={};pipelines={};all_stages=[]
    if terminal:
        schedulers=read(folder/'scheduler_terminal.json')
        for attempt in ('001','002'):
            logs=files/'slurm_logs'/('attempt_'+attempt)
            if not (logs/'submission.json').exists():continue
            login,reservation,submission=(read(logs/name) for name in ('login_verification.json','reservation.json','submission.json'))
            job=submission['job_id'];scheduler=schedulers['jobs'][job]
            if scheduler['state']!='COMPLETED' or scheduler['exit_code']!='0:0':raise ValueError('Terminal scheduler')
            raw=schedulers['sacct_raw'].splitlines()
            scheduler_rows=[dict(zip(raw[0].split('|'),line.split('|'))) for line in raw[1:] if line]
            if scheduler_rows!=schedulers['steps']:raise ValueError('Scheduler raw/parsed mismatch')
            actual=[r for r in scheduler_rows if r['JobIDRaw']==job]
            if len(actual)!=1 or actual[0]['State']!=scheduler['state'] or actual[0]['ExitCode']!=scheduler['exit_code']:raise ValueError('Scheduler job binding')
            base=p.bindings(execution,manifest,attempt)
            adapter=SimpleNamespace(**vars(c));adapter.HERE=files
            bind(p,{'c':adapter})['validate_ownership'](login,reservation,sha(logs/'login_verification.json'),base,job,submission)
            base.update(job_id=job,reservation_token=reservation['token'],reservation_sha256=sha(logs/'reservation.json'),login_verification_sha256=sha(logs/'login_verification.json'))
            bases[attempt]=base
            for prefix in ('cpu_preflight_','no_git_preflight_'):
                cpu=read(logs/(prefix+execution+'.json'));tests=cpu['cpu_tests']
                if cpu['status']!='PASS' or cpu['execution_commit']!=execution or cpu['source_hash']!=manifest['source_hash'] or cpu['cuda_initialized'] is not False:raise ValueError('CPU provenance')
                if tests['run']!=38 or any(tests[k] for k in ('failures','errors','skipped')):raise ValueError('CPU tests')
            inherited=read(logs/'inherited_admission.json')
            if any(inherited.get(k)!=v for k,v in base.items()) or inherited['status']!='PASS' or inherited['inherited']!=p.inherited():raise ValueError('Inherited admission')
            pipeline=read(logs/'pipeline_status.json');pipelines[attempt]=pipeline
            for artifact in (pipeline,read(logs/'terminal_metadata.json')):
                if any(artifact.get(k)!=v for k,v in base.items()) or artifact['status'] not in ('PASS','PAUSED_DEADLINE'):raise ValueError('Pipeline/terminal binding')
            if pipeline.get('unknown_scientific_starts',0) or any(k in pipeline for k in ('error','traceback','report_traceback')):raise ValueError('Unknown start or pipeline error')
            terminal_record=read(logs/'terminal_metadata.json')
            if terminal_record['pipeline_sha256']!=sha(logs/'pipeline_status.json'):raise ValueError('Terminal pipeline SHA')
            for key in ('scientific_fits_started','scientific_fits_completed'):
                if terminal_record[key]!=pipeline[key]:raise ValueError('Terminal counters')
            for key,value in terminal_record['checkpoints'].items():
                record_path='slurm_logs/fits/'+key+'/checkpoints/best_state_dict.pth'
                if value['sha256']!=saved['checkpoints'][record_path]['sha256'] or value['bytes']!=saved['checkpoints'][record_path]['bytes']:raise ValueError('Terminal checkpoint SHA/bytes')
            all_stages.extend(pipeline['stages'])
        if not pipelines:raise ValueError('No completed allocation')
        last_attempt=max(pipelines);last=pipelines[last_attempt]
        if last['status']!='PASS' or (last['scientific_fits_started'],last['scientific_fits_completed'],last['complete_pairs'])!=(8,8,4):raise ValueError('Incomplete terminal series')
        if [{k:r[k] for k in ('variant','seed','run_id')} for r in all_stages]!=c.tasks() or any(r['status']!='PASS' for r in all_stages):raise ValueError('Eight ordered stages')
    records={};rows=[];analytic_rounding=[];replays=[]
    for task in c.tasks():
        path=files/'runs'/(task['run_id']+'.json')
        if not path.exists():
            if terminal:raise ValueError('Missing planned result')
            continue
        r=read(path)
        if terminal:
            base=bases[r['execution_attempt']]
            if any(r.get(k)!=v for k,v in base.items()):raise ValueError('Foreign run identity')
            lock=read(files/'slurm_logs/fits'/task['run_id']/'run.lock')
            if any(lock.get(k)!=v for k,v in base.items()) or any(lock.get(k)!=r[k] for k in ('run_id','seed','gap_trap_mode','inherited_record_sha256')):raise ValueError('Run lock identity')
            admission=files/'slurm_logs'/('attempt_'+r['execution_attempt'])/'inherited_admission.json'
            if r['inherited_record_sha256']!=sha(admission):raise ValueError('Run admission binding')
        rows.append(audit_run(r,files/'slurm_logs/fits'/task['run_id'],saved['checkpoints'],analytic_rounding));records[task['run_id']]=r
        if task['variant']=='fixed_replay':replays.append(report.replay_check(r))
    pairs=[]
    for seed in c.SEEDS:
        names=[c.paths(v,seed)['run_id'] for v in c.MODES]
        if not all(n in records for n in names):continue
        fixed,centered=(records[n] for n in names)
        if any(fixed[k]!=centered[k] for k in PAIRING):raise ValueError('Paired initialization/data/precision')
        pair=read(files/'runs'/f'pair_seed{seed}.json')
        if pair['status']!='PASS' or pair['seed']!=seed or pair['TEST']!='NOT_RUN' or pair['test_evaluation_count']!=0:raise ValueError('Pair audit identity')
        for label,name in zip(('fixed','centered'),names):
            if pair[label+'_run_id']!=name or pair[label+'_result_sha256']!=sha(files/'runs'/(name+'.json')):raise ValueError('Pair result SHA')
        if pair['shared']!={k:centered[k] for k in PAIRING} or pair['historical_fixed_replay']!=report.replay_check(fixed):raise ValueError('Pair state/replay evidence')
        pairs.append(seed)
    summary_rounding=[];summary=None
    if terminal:
        summary=read(files/'slurm_logs'/('attempt_'+last_attempt)/'confirmation_summary.json')
        if any(summary.get(k)!=v for k,v in bases[last_attempt].items()) or summary.get('blocking_reason') is not None:raise ValueError('Summary ownership')
        recomputed=bind(report,{'validate_record':validate_saved_record})['summarize'](records)
        compare_summary(summary,recomputed,rounding=summary_rounding)
        if len(records)!=8 or len(pairs)!=4:raise ValueError('Eight fits/four pairs required')
    errors=[];warnings=[]
    for path in (files/'slurm_logs').rglob('*'):
        if path.suffix not in ('.log','.out','.err'):continue
        for line in path.read_text().splitlines():
            if re.search(r'Traceback \(most recent call last\)|CUDA out of memory|RuntimeError:|\b(?:nan|inf)\b',line,re.I):errors.append(dict(path=str(path.relative_to(files)),line=line))
            if 'Warning:' in line:warnings.append(dict(path=str(path.relative_to(files)),line=line))
    if errors:raise ValueError('Runtime errors: '+repr(errors[:4]))
    value=dict(status='PASS' if terminal else 'INTERIM_VERIFIED',job_id=saved['job_id'],execution_commit=execution,
        source_hash=manifest['source_hash'],source_blobs=len(manifest['files']),audited_at=now(),terminal=terminal,
        preserved_files=len(saved['files']),preserved_bytes=sum(x['bytes'] for x in saved['files']),
        scientific_fits_verified=len(records),complete_pairs_verified=len(pairs),TEST='NOT_RUN',test_evaluation_count=0,
        rows=rows,metric_cells=sum(x['metric_cells'] for x in rows),fixed_replays=replays,paired_fields=list(PAIRING),
        runtime_errors=errors,warnings=warnings,analytic_libm_tolerance_ulps=1,analytic_rounding=analytic_rounding,
        summary_std_tolerance_ulps=2,summary_rounding=summary_rounding,checkpoint_loading=False,
        result_sha256={name:sha(files/'runs'/(name+'.json')) for name in records})
    if summary is not None:value['summary_sha256']=sha(files/'slurm_logs'/('attempt_'+last_attempt)/'confirmation_summary.json')
    output=folder/'independent_audit.json'
    if output.exists():
        prior=read(output)
        if {k:v for k,v in prior.items() if k!='audited_at'}!={k:v for k,v in value.items() if k!='audited_at'}:raise ValueError('Saved audit changed')
        return prior
    create(output,value);return value


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);parser.add_argument('--interim',action='store_true');args=parser.parse_args()
    result=audit(args.folder,terminal=not args.interim)
    print(json.dumps({k:v for k,v in result.items() if k not in ('warnings','result_sha256','paired_fields')},indent=2))
