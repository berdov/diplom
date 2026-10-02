"""Read preserved terminal bytes; no model, checkpoint load, or dataset access."""
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from experiments.mamba3_gap_trap import config as c
from experiments.mamba3_gap_trap import provenance as p
from experiments.mamba3_gap_trap.report import validate_record, summarize
from experiments.mamba3_mimo_time.records import read, sha, create, accepted_cases, now


def main():
    folder=c.HERE/'evidence/job4370162';files=folder/'files'
    saved=read(folder/'preservation_manifest.json')
    execution='8ee54cf1faee22bb3abad3f31aa9268d77a125d1'
    if saved['checkout_commit']!=execution:raise ValueError('Execution changed')
    for row in saved['files']:
        path=files/row['path']
        if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:raise ValueError('Preserved bytes changed')
    manifest=p.verify()
    for name,wanted in manifest['files'].items():
        blob=subprocess.check_output(['git','show',execution+':'+name],cwd=c.ROOT)
        if hashlib.sha256(blob).hexdigest()!=wanted:raise ValueError('Execution blob drift: '+name)
    base=p.bindings(execution,manifest)
    logs=files/'slurm_logs/attempt_001';runs=files/'runs/attempt_001'
    login,reservation,submission=(read(logs/n) for n in ('login_verification_001.json','reservation_001.json','submission_001.json'))
    p.validate_ownership(login,reservation,sha(logs/'login_verification_001.json'),base,'4370162',submission)
    base.update(job_id='4370162',reservation_token=reservation['token'],reservation_sha256=sha(logs/'reservation_001.json'),login_verification_sha256=sha(logs/'login_verification_001.json'))
    def checked(path):
        r=read(path)
        if r.get('status')!='PASS' or any(r.get(k)!=v for k,v in base.items()):raise ValueError('Foreign/failed stage '+str(path))
        return r
    inherited=checked(runs/'inherited_kernel_001.json')
    if inherited['inherited']!=p.inherited():raise ValueError('Inherited admission changed')
    gate=checked(runs/'targeted_gate_001.json');smoke=checked(runs/'smoke_001.json')
    if not accepted_cases(gate['cases'],c.plan()['required_cases']):raise ValueError('Targeted checks failed')
    p.validate_smoke(smoke)
    if smoke['targeted_gate_sha256']!=sha(runs/'targeted_gate_001.json'):raise ValueError('Smoke gate binding')
    pipeline=checked(logs/'pipeline_status.json')
    if pipeline['scientific_fits_started']!=2 or pipeline['scientific_fits_completed']!=2:raise ValueError('Fit budget/count')
    records={};rows=[];old=read(c.PILOT)
    for variant in c.MODES:
        run_id=c.paths(variant)['run_id'];r=checked(runs/(run_id+'.json'));validate_record(r,variant)
        if r['targeted_gate_sha256']!=sha(runs/'targeted_gate_001.json') or r['smoke_sha256']!=sha(runs/'smoke_001.json'):raise ValueError('Fit admission binding')
        for setting in ('config','effective_config'):
            allowed={'checkpoint_dir','gap_trap_mode'};a,b=r[setting],old[setting]
            if any(a.get(k)!=b.get(k) for k in set(a)|set(b) if k not in allowed):raise ValueError('Frozen '+setting)
        for k in ('protocol','manifest_sha256','train_time_stats_sha256','verified_history_stats','precision','optimizer_settings'):
            if json.dumps(r[k],sort_keys=True)!=json.dumps(old[k],sort_keys=True):raise ValueError('Frozen '+k)
        meta=read(logs/run_id/'checkpoints/best_metadata.json')
        path='slurm_logs/attempt_001/'+run_id+'/checkpoints/best_state_dict.pth'
        if saved['checkpoints'][path]['sha256']!=r['checkpoint_sha256'] or meta['checkpoint_sha256']!=r['checkpoint_sha256']:raise ValueError('Checkpoint SHA')
        for key in ('run_id','mode','gap_trap_mode','seed','execution_commit','config_sha256','source_hash','core_hash'):
            if meta[key]!=r[key]:raise ValueError('Checkpoint identity')
        if meta['epoch']!=r['best_epoch'] or meta['metrics']!=r['best_valid_metrics'] or meta['gap_trap']!=r['best_diagnostics']['gap_trap']:raise ValueError('Checkpoint selection')
        recbole=list((logs/run_id/'log').rglob('*.log'))
        if len(recbole)!=1:raise ValueError('Ambiguous training log')
        content=recbole[0].read_text();lines=content.splitlines()
        evaluations=re.findall(r'epoch (\d+) evaluating \[time: [^,]+, valid_score: ([0-9.]+)\]',content)
        metrics=[dict((k,float(v)) for k,v in re.findall(r'((?:hit|ndcg|recall)@\d+) : ([0-9.]+)',line)) for line in lines if line.startswith('hit@5 :')]
        training=re.findall(r'epoch (\d+) training \[time: [^,]+, train loss: ([0-9.]+)\]',content)
        if len(evaluations)!=len(metrics) or len(evaluations)!=r['actual_epochs'] or len(training)!=r['actual_epochs']:raise ValueError('Log epoch count')
        for index,((epoch,score),metric,(te,loss),row) in enumerate(zip(evaluations,metrics,training,r['history'])):
            if int(epoch)!=index or int(te)!=index or float(score)!=row['valid_ndcg10'] or metric!=row['valid_metrics']:raise ValueError('Log metric mismatch')
            if float(loss)!=round(row['train_loss'],4):raise ValueError('Train loss/log mismatch')
            diag=row['diagnostics']['gap_trap'];alpha=diag['alpha']
            if not 0<=alpha<=1:raise ValueError('Alpha bounds')
            expected=[alpha*g/(1+g) for g in c.plan()['diagnostic_grid']]
            if diag['gaps_over_R0']!=c.plan()['diagnostic_grid'] or diag['logit_shift']!=expected:raise ValueError('Diagnostic grid/formula')
        if 'Finished training, best eval result in epoch '+str(r['best_epoch']) not in content:raise ValueError('Terminal log selection')
        for field,key in [('train_seconds','train_seconds'),('valid_seconds','valid_seconds')]:
            if r[field]!=sum(x[key] for x in r['history']):raise ValueError('Timing sum')
        records[variant]=r
        alpha=[x['diagnostics']['gap_trap']['alpha'] for x in r['history']]
        rows.append(dict(variant=variant,status='PASS',epochs=len(alpha),metric_cells=len(alpha)*12,checkpoint_sha256=r['checkpoint_sha256'],best_alpha=r['best_diagnostics']['gap_trap']['alpha'],final_alpha=alpha[-1],max_epoch_alpha=max(alpha),zero_alpha_epochs=sum(x==0 for x in alpha)))
    fixed,gap=(records[v] for v in c.MODES)
    pairing=('initial_backbone_sha256','initial_common_calibrator_hashes','rng_components','protocol','manifest_sha256','train_time_stats_sha256','verified_history_stats','precision','optimizer_settings','first_train_batch_sha256')
    for key in pairing:
        if json.dumps(fixed[key],sort_keys=True)!=json.dumps(gap[key],sort_keys=True):raise ValueError('Pair mismatch: '+key)
    if fixed['initial_alpha']!={} or gap['initial_alpha']!={'gap_trap.alpha':0.}:raise ValueError('Initial alpha')
    summary=checked(runs/'pilot_summary.json')
    recomputed=summarize(records)
    if any(summary[k]!=v for k,v in recomputed.items()):raise ValueError('Summary not reproducible')
    bad=[];warnings=[]
    for log in logs.rglob('*'):
        if log.suffix not in ('.log','.out','.err'):continue
        for line in log.read_text().splitlines():
            if any(word in line for word in ('Traceback (most recent call last)','CUDA out of memory','RuntimeError:')):bad.append(dict(path=str(log.relative_to(files)),line=line))
            if 'Warning:' in line:warnings.append(dict(path=str(log.relative_to(files)),line=line))
    if bad:raise ValueError('Runtime errors in logs')
    audit=dict(status='PASS',job_id='4370162',audited_at=now(),execution_commit=execution,source_hash=manifest['source_hash'],preserved_files=len(saved['files']),preserved_bytes=sum(x['bytes'] for x in saved['files']),source_blobs=len(manifest['files']),targeted_cases=4,targeted_required_checks=133,smoke_steps=6,scientific_fits_started=2,scientific_fits_completed=2,TEST='NOT_RUN',test_evaluation_count=0,paired_fields=list(pairing),rows=rows,metric_cells=sum(x['metric_cells'] for x in rows),runtime_errors=bad,warnings=warnings,contrasts=summary['contrasts'],checkpoint_loading=False)
    output=folder/'independent_audit.json'
    if output.exists():
        previous=read(output)
        if {k:v for k,v in previous.items() if k!='audited_at'}!={k:v for k,v in audit.items() if k!='audited_at'}:
            raise ValueError('Saved audit no longer reproducible')
        audit=previous
    else:create(output,audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in ('warnings','paired_fields')},indent=2))


if __name__=='__main__':main()
