"""Audit saved pilot evidence without importing Torch or loading checkpoints."""
import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from experiments.mamba3_layer_temporal import config as c, provenance as p, report
from experiments.mamba3_mimo_time.records import read, sha, create, now
from experiments.mamba3_gap_trap.provenance import accepted_cases

EXECUTION='30640e2be36b45c6f89e31f91aae83f76f2d2254'
PAIRING=('initial_backbone_sha256','initial_common_calibrator_hashes','initial_layer_calibrator_hashes',
         'rng_components','protocol','manifest_sha256','train_time_stats_sha256','verified_history_stats',
         'precision','optimizer_settings','first_train_batch_sha256')


def require(condition, label):
    if not condition:raise ValueError(label)


def diagnostics(d, variant, rounding):
    require(d['temporal_sharing']==variant and len(d['layers'])==2, 'Diagnostic identity')
    require(d['relative_l2_denominator']=='layer0 norm; null if zero' and d['near_bounds']=='scale < 0.51 / scale > 1.99','Diagnostic definitions')
    def close(a,b):
        # Only post-fit analytic reductions: Torch float64 vs Python/libm.
        # This tolerance does not affect metrics or GPU acceptance.
        require(math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-15),'Diagnostic reduction')
        if a!=b:
            rounding['different_reductions']+=1
            rounding['max_absolute_difference']=max(rounding['max_absolute_difference'],abs(a-b))
    for name in ('decay','scan'):
        for layer in d['layers']:
            row=layer[name];curve=row['scale_by_gap_and_head']
            require(row['gaps_over_R0']==c.plan()['diagnostic_grid'] and row['reference_ms']==838393.,'Grid/reference')
            require(len(curve)==10 and all(len(x)==2 and all(math.isfinite(v) and .5<=v<=2 for v in x) for x in curve),'Curve shape/bounds')
            require(row['at_reference']==curve[5],'Reference scales')
            for field,condition in [('near_lower',lambda x:x<.51),('near_upper',lambda x:x>1.99)]:
                for h in range(2):close(row[field][h],sum(condition(x[h]) for x in curve)/10)
        a,b=[x[name]['scale_by_gap_and_head'] for x in d['layers']];div=d['divergence'][name]
        for h in range(2):
            ratios=[abs(math.log(x[h]/y[h])) for x,y in zip(a,b)]
            close(div['mean_abs_log_ratio_per_head'][h],sum(ratios)/10)
            close(div['max_abs_log_ratio_per_head'][h],max(ratios))
        values=div['parameter_distances']
        require(set(values)=={'first.weight','first.bias','last.weight','last.bias'},'Parameter distance fields')
        for row in values.values():
            require(row['l2']>=0 and row['reference_l2']>=0,'Parameter norm sign')
            if row['reference_l2']:close(row['relative_l2'],row['l2']/row['reference_l2'])
            else:require(row['relative_l2'] is None,'Zero norm convention')
        close(div['all_parameters_l2'],math.sqrt(sum(x['l2']**2 for x in values.values())))
        close(div['reference_parameters_l2'],math.sqrt(sum(x['reference_l2']**2 for x in values.values())))
        if div['reference_parameters_l2']:close(div['relative_parameters_l2'],div['all_parameters_l2']/div['reference_parameters_l2'])
        else:require(div['relative_parameters_l2'] is None,'Combined zero norm')
        if variant=='shared_layers':
            require(a==b and div['all_parameters_l2']==0 and all(x['l2']==0 for x in values.values()),'Shared divergence')


def audit_run(r, variant, files, saved, terminal, rounding):
    require(r['status']=='PASS','Missing successful fit');report.validate_record(r,variant)
    require(hashlib.sha256(json.dumps(r['config'],sort_keys=True).encode()).hexdigest()==r['config_sha256'],'Config SHA')
    paths=c.paths(variant);runtime=files/paths['runtime'].relative_to(c.HERE)
    meta=read(runtime/'checkpoints/best_metadata.json');checkpoint=saved['checkpoints'][str(paths['checkpoint'].relative_to(c.HERE))]
    require(checkpoint==terminal['checkpoints'][variant],'Preserved checkpoint vs terminal')
    require(checkpoint['sha256']==r['checkpoint_sha256']==meta['checkpoint_sha256'],'Checkpoint SHA')
    expected_path='/home/daryumin/iberdov/diplom/'+str(paths['checkpoint'].relative_to(c.ROOT))
    require(r['checkpoint_path']==checkpoint['path']==expected_path and checkpoint['bytes']>0,'Checkpoint path/bytes')
    for k in ('run_id','mode','temporal_sharing','seed','execution_commit','config_sha256','source_hash','core_hash'):
        require(meta[k]==r[k],'Checkpoint owner '+k)
    require(meta['epoch']==r['best_epoch'] and meta['metrics']==r['best_valid_metrics'] and meta['layer_temporal']==r['best_diagnostics']['layer_temporal'],'Checkpoint selection')
    logs=list((runtime/'log').rglob('*.log'));require(len(logs)==1,'Metric log count')
    content=logs[0].read_text();process=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',(runtime/'process/stderr.log').read_text())
    def parse(text):
        evaluations=re.findall(r'epoch (\d+) evaluating \[time: [^,]+, valid_score: ([0-9.]+)\]',text)
        metrics=[dict((k,float(v)) for k,v in re.findall(r'((?:hit|ndcg|recall)@\d+) : ([0-9.]+)',line)) for line in text.splitlines() if line.startswith('hit@5 :')]
        training=re.findall(r'epoch (\d+) training \[time: [^,]+, train loss: ([0-9.]+)\]',text)
        return evaluations,metrics,training
    evaluations,metrics,training=parse(content)
    require(parse(process)==(evaluations,metrics,training),'Process/RecBole metrics and loss')
    require(len(evaluations)==len(metrics)==len(training)==r['actual_epochs'],'Epoch log count')
    for index,((epoch,score),metric,(te,loss),row) in enumerate(zip(evaluations,metrics,training,r['history'])):
        require(int(epoch)==int(te)==index and float(score)==row['valid_ndcg10'] and metric==row['valid_metrics'],'Epoch metrics/log')
        require(float(loss)==round(row['train_loss'],4),'Loss/log rounding')
        diagnostics(row['diagnostics']['layer_temporal'],variant,rounding)
    require(re.findall(r'Finished training, best eval result in epoch (\d+)\b',content)==[str(r['best_epoch'])],'Terminal selection log')
    for k in ('train_seconds','valid_seconds'):require(r[k]==sum(x[k] for x in r['history']),'Time sum')
    for kind in ('allocated','reserved'):
        require(r['peak_gpu_'+kind+'_bytes']==max(max(x['train_peak_'+kind+'_bytes'],x['valid_peak_'+kind+'_bytes']) for x in r['history']),'Memory peak')
    return dict(variant=variant,run_id=r['run_id'],epochs=r['actual_epochs'],metric_cells=12*r['actual_epochs'],checkpoint_sha256=r['checkpoint_sha256'],best_epoch=r['best_epoch'])


def audit(folder):
    saved=read(folder/'preservation_manifest.json');files=folder/'files';manifest=p.verify();job='4372822'
    require(saved['execution_commit']==EXECUTION and saved['job_id']==job and saved['source_hash']==manifest['source_hash'],'Preservation identity')
    for row in saved['files']:
        path=files/row['path'];require(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Preserved file '+row['path'])
    require(len({x['path'] for x in saved['files']})==len(saved['files']),'Duplicate preserved paths')
    for name,expected in manifest['files'].items():
        actual=hashlib.sha256(subprocess.check_output(['git','show',EXECUTION+':'+name],cwd=c.ROOT)).hexdigest()
        require(actual==expected,'Execution source '+name)
    require(sha(files/'source_manifest.json')==sha(c.MANIFEST) and sha(files/'study_plan.json')==sha(c.HERE/'study_plan.json'),'Frozen manifest/plan')
    scheduler=read(folder/'scheduler_terminal.json');raw=scheduler['sacct_raw'].splitlines()
    parsed=[dict(zip(raw[0].split('|'),line.split('|'))) for line in raw[1:] if line]
    require(parsed==scheduler['steps'] and [x for x in parsed if x['JobIDRaw']==job]==[scheduler['job']],'Scheduler raw binding')
    require(all(x['State']=='COMPLETED' and x['ExitCode']=='0:0' for x in parsed),'Terminal scheduler')
    logs=files/'slurm_logs/attempt_001';runs=files/'runs/attempt_001'
    login,reservation,submission=[read(logs/name) for name in ('login_verification.json','reservation.json','submission.json')]
    base=p.bindings(EXECUTION,manifest)
    p.validate_ownership(login,reservation,sha(logs/'login_verification.json'),base,job,submission)
    base.update(job_id=job,reservation_token=reservation['token'],reservation_sha256=sha(logs/'reservation.json'),login_verification_sha256=sha(logs/'login_verification.json'))
    def owner(record,label,status=True):
        require(all(record.get(k)==v for k,v in base.items()),label+' owner')
        if status:require(record.get('status')=='PASS',label+' status')
    cpu_rows=[]
    for prefix in ('cpu_preflight_','no_git_preflight_'):
        cpu=read(logs/(prefix+EXECUTION+'.json'));tests=cpu['cpu_tests']
        require(cpu['status']=='PASS' and cpu['execution_commit']==EXECUTION and cpu['source_hash']==manifest['source_hash'] and cpu['cuda_initialized'] is False,'CPU provenance')
        require(tests['run']==28 and not any(tests[k] for k in ('failures','errors','skipped')),'CPU tests')
        require(cpu['parameter_counts']==c.COUNTS and cpu['scientific_fits']==0 and cpu['test_evaluation_count']==0,'CPU scope/counts')
        cpu_rows.append(dict(file=prefix+EXECUTION+'.json',tests=28,status='PASS'))
    inherited=read(runs/'inherited_kernel.json');owner(inherited,'Inherited')
    require(inherited['inherited']==p.inherited(),'Inherited exact admission')
    gate=read(runs/'targeted_gate.json');owner(gate,'Gate')
    require(gate['required_cases']==c.plan()['required_cases'] and accepted_cases(gate['cases'],c.plan()['required_cases']),'Targeted gate')
    checks=sum(len(x['checks']) for x in gate['cases'])
    require(len(gate['cases'])==6 and checks==214 and gate['scientific_fits']==0,'Gate scope')
    smoke=read(runs/'smoke.json');owner(smoke,'Smoke');p.validate_smoke(smoke)
    require(smoke['targeted_gate_sha256']==sha(runs/'targeted_gate.json') and smoke['scientific_fits']==0,'Smoke gate/scope')
    pipeline=read(logs/'pipeline_status.json');terminal=read(runs/'terminal_metadata.json')
    for label,r in [('Pipeline',pipeline),('Terminal',terminal)]:
        owner(r,label);require((r['scientific_fits_started'],r['scientific_fits_completed'])==(2,2),label+' counters')
        require(not any(k in r for k in ('error','traceback','report_traceback')),label+' error')
    require([(x['stage'],x['status']) for x in pipeline['stages']]==[(x,'PASS') for x in ('gate','smoke',*c.MODES)],'Pipeline ordered stages')
    require(terminal['pipeline_sha256']==sha(logs/'pipeline_status.json'),'Terminal pipeline SHA')
    lock=read(logs/'pipeline.lock');owner(lock,'Pipeline lock',False)
    records={};rows=[];rounding=dict(different_reductions=0,max_absolute_difference=0.,relative_tolerance=1e-12,absolute_tolerance=1e-15)
    for variant in c.MODES:
        paths=c.paths(variant);r=read(files/paths['result'].relative_to(c.HERE));owner(r,variant)
        lock=read(files/paths['lock'].relative_to(c.HERE));owner(lock,variant+' lock',False)
        for k in ('run_id','temporal_sharing','seed','targeted_gate_sha256','smoke_sha256'):require(lock[k]==r[k],'Run lock '+k)
        require(r['targeted_gate_sha256']==sha(runs/'targeted_gate.json') and r['smoke_sha256']==sha(runs/'smoke.json'),'Fit admission')
        rows.append(audit_run(r,variant,files,saved,terminal,rounding));records[variant]=r
    a,b=[records[v] for v in c.MODES]
    require(all(a[k]==b[k] for k in PAIRING),'Paired fields')
    require(set(a['rng_components'])=={'python','numpy','cpu','cuda','aggregate','loader_generator'},'RNG components')
    replay=report.replay_check(a);summary=read(runs/'pilot_summary.json');owner(summary,'Summary')
    expected=report.summarize(records)
    require(all(summary[k]==v for k,v in expected.items()),'Recomputed summary')
    require(summary['fresh_control_replay']==replay and summary['blocking_reason'] is None,'Summary replay')
    errors=[];warnings=[]
    for path in logs.rglob('*'):
        if path.suffix not in ('.log','.out','.err'):continue
        for line in path.read_text().splitlines():
            if re.search(r'Traceback \(most recent call last\)|CUDA out of memory|RuntimeError:|\b(?:nan|inf)\b',line,re.I):errors.append(dict(path=str(path.relative_to(files)),line=line))
            if 'Warning:' in line:warnings.append(dict(path=str(path.relative_to(files)),line=line))
    require(not errors,'Runtime errors '+repr(errors[:4]))
    value=dict(status='PASS',job_id=job,execution_commit=EXECUTION,source_hash=manifest['source_hash'],source_blobs=len(manifest['files']),audited_at=now(),
        preserved_files=len(saved['files']),preserved_bytes=sum(x['bytes'] for x in saved['files']),scheduler=scheduler['job'],
        cpu=cpu_rows,gpu_cases=6,gpu_checks=checks,smoke_steps_per_variant=3,scientific_fits_verified=2,complete_pairs_verified=1,
        TEST='NOT_RUN',test_evaluation_count=0,rows=rows,epochs=sum(x['epochs'] for x in rows),metric_cells=sum(x['metric_cells'] for x in rows),
        fresh_control_replay=replay,paired_fields=list(PAIRING),runtime_errors=errors,warnings=warnings,diagnostic_reductions=rounding,
        parameter_distances_audit='Saved per-fit values: algebraic consistency only; checkpoint SHA and selected-epoch metadata bound. No weight loading.',
        checkpoint_loading=False,result_sha256={v:sha(files/c.paths(v)['result'].relative_to(c.HERE)) for v in c.MODES},summary_sha256=sha(runs/'pilot_summary.json'),
        preservation_manifest_sha256=sha(folder/'preservation_manifest.json'),scheduler_sha256=sha(folder/'scheduler_terminal.json'))
    output=folder/'independent_audit.json'
    if output.exists():
        prior=read(output);require({k:v for k,v in prior.items() if k!='audited_at'}=={k:v for k,v in value.items() if k!='audited_at'},'Saved audit changed');return prior
    create(output,value);return value


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);args=parser.parse_args()
    result=audit(args.folder)
    print(json.dumps({k:v for k,v in result.items() if k not in ('warnings','paired_fields')},indent=2))
