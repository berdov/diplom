"""Terminal preserved-evidence audit; standard library, no data/model forward."""
import argparse
import hashlib
import json
import math
import struct
import re
import subprocess
from pathlib import Path
from experiments.mamba3_gap_trap.centered import config as c, provenance as p, report
from experiments.mamba3_mimo_time.records import read,sha,create,accepted_cases,now


def audit(job):
    folder=c.HERE/('evidence/job'+job);files=folder/'files'
    saved=read(folder/'preservation_manifest.json');execution=saved['checkout_commit']
    scheduler=read(folder/'scheduler_terminal.json')
    if scheduler['job_id']!=job or scheduler['state']!='COMPLETED' or scheduler['exit_code']!='0:0':raise ValueError('Slurm terminal status')
    for row in saved['files']:
        path=files/row['path']
        if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:raise ValueError('Preservation SHA: '+row['path'])
    manifest=p.verify()
    for name,expected in manifest['files'].items():
        blob=subprocess.check_output(['git','show',execution+':'+name],cwd=c.ROOT)
        if hashlib.sha256(blob).hexdigest()!=expected:raise ValueError('Execution source: '+name)
    logs,runs=files/c.LOGS.relative_to(c.HERE),files/c.RUNS.relative_to(c.HERE)
    base=p.bindings(execution,manifest)
    login,reservation,submission=(read(logs/n) for n in (c.LOGIN.name,c.RESERVATION.name,c.SUBMISSION.name))
    p.validate_ownership(login,reservation,sha(logs/c.LOGIN.name),base,job,submission)
    base.update(job_id=job,reservation_token=reservation['token'],reservation_sha256=sha(logs/c.RESERVATION.name),login_verification_sha256=sha(logs/c.LOGIN.name))
    def checked(path):
        r=read(path)
        if r.get('status')!='PASS' or any(r.get(k)!=v for k,v in base.items()):raise ValueError('Failed/foreign artifact: '+str(path))
        return r
    inherited=checked(runs/c.INHERITED.name)
    if inherited['inherited']!=p.inherited():raise ValueError('Inherited evidence mismatch')
    gate,smoke=checked(runs/c.GATE.name),checked(runs/c.SMOKE.name)
    if not accepted_cases(gate['cases'],c.plan()['required_cases']):raise ValueError('Required GPU leaves')
    p.validate_smoke(smoke)
    if smoke['targeted_gate_sha256']!=sha(runs/c.GATE.name):raise ValueError('Smoke binding')
    pipeline=checked(logs/'pipeline_status.json')
    if (pipeline['scientific_fits_started'],pipeline['scientific_fits_completed'])!=(2,2):raise ValueError('Fit count')
    if [row['stage'] for row in pipeline['stages']]!=['gate','smoke',*c.MODES] or any(row['status']!='PASS' for row in pipeline['stages']):raise ValueError('Pipeline stages')
    records={};rows=[];pilot=read(c.PILOT)
    for mode in c.MODES:
        run_id=c.paths(mode)['run_id'];r=checked(runs/(run_id+'.json'));report.validate_record(r,mode)
        if r['targeted_gate_sha256']!=sha(runs/c.GATE.name) or r['smoke_sha256']!=sha(runs/c.SMOKE.name):raise ValueError('Fit admission')
        for setting in ('config','effective_config'):
            a,b=r[setting],pilot[setting]
            if any(a.get(k)!=b.get(k) for k in set(a)|set(b) if k not in ('checkpoint_dir','gap_trap_mode')):raise ValueError('Frozen '+setting)
        if hashlib.sha256(json.dumps(r['config'],sort_keys=True).encode()).hexdigest()!=r['config_sha256']:raise ValueError('Config hash')
        for key in ('protocol','manifest_sha256','train_time_stats_sha256','verified_history_stats','precision','optimizer_settings'):
            if r[key]!=pilot[key]:raise ValueError('Frozen pilot '+key)
        meta=read(logs/run_id/'checkpoints/best_metadata.json')
        path=str(c.paths(mode)['checkpoint'].relative_to(c.HERE))
        if saved['checkpoints'][path]['sha256']!=r['checkpoint_sha256'] or meta['checkpoint_sha256']!=r['checkpoint_sha256']:raise ValueError('Checkpoint SHA')
        for key in ('run_id','mode','gap_trap_mode','seed','execution_commit','config_sha256','source_hash','core_hash'):
            if meta[key]!=r[key]:raise ValueError('Checkpoint metadata '+key)
        if meta['epoch']!=r['best_epoch'] or meta['metrics']!=r['best_valid_metrics'] or meta['gap_trap']!=r['best_diagnostics']['gap_trap']:raise ValueError('Checkpoint selection')
        paths=list((logs/run_id/'log').rglob('*.log'))
        if len(paths)!=1:raise ValueError('Ambiguous metric log')
        content=paths[0].read_text()
        process=re.sub(r'\[[0-?]*[ -/]*[@-~]','',(logs/run_id/'process/stderr.log').read_text())
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
            if diag['odds_multiplier']!=[math.exp(x) for x in shifts] or diag['current_fraction']!=[[1/(1+math.exp(-t-v)) for v in shifts] for t in c.plan()['diagnostic_content_logits']]:raise ValueError('Odds/sigmoid diagnostics')
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
        records[mode]=r;alpha=[x['diagnostics']['gap_trap']['alpha'] for x in r['history']]
        rows.append(dict(variant=mode,epochs=len(alpha),metric_cells=12*len(alpha),checkpoint_sha256=r['checkpoint_sha256'],best_alpha=r['best_diagnostics']['gap_trap']['alpha'],final_alpha=alpha[-1],max_epoch_alpha=max(alpha),zero_alpha_epochs=sum(a==0 for a in alpha)))
    fixed,centered=(records[m] for m in c.MODES)
    pairing=('initial_backbone_sha256','initial_common_calibrator_hashes','rng_components','protocol','manifest_sha256','train_time_stats_sha256','verified_history_stats','precision','optimizer_settings','first_train_batch_sha256')
    for key in pairing:
        if fixed[key]!=centered[key]:raise ValueError('Pairing '+key)
    if fixed['initial_alpha']!={} or centered['initial_alpha']!={'gap_trap.alpha':0.}:raise ValueError('Initial alpha')
    baseline=report.replay_check(fixed)
    summary=checked(runs/'pilot_summary.json');recomputed=report.summarize(records)
    if any(summary[k]!=v for k,v in recomputed.items()):raise ValueError('Summary recomputation')
    terminal=read(runs/'terminal_metadata.json')
    if terminal['status']!='PASS' or terminal['scientific_fits_completed']!=2 or terminal['scientific_fits_started']!=2 or terminal['test_evaluation_count']!=0 or terminal['TEST']!='NOT_RUN':raise ValueError('Terminal metadata')
    if terminal['pipeline_sha256']!=sha(logs/'pipeline_status.json') or terminal['checkpoints']!={mode:records[mode]['checkpoint_sha256'] for mode in c.MODES}:raise ValueError('Terminal hash bindings')
    errors=[];warnings=[]
    for path in logs.rglob('*'):
        if path.suffix not in ('.log','.out','.err'):continue
        for line in path.read_text().splitlines():
            if re.search(r'Traceback \(most recent call last\)|CUDA out of memory|RuntimeError:|\b(?:nan|inf)\b',line,re.I):errors.append(dict(path=str(path.relative_to(files)),line=line))
            if 'Warning:' in line:warnings.append(dict(path=str(path.relative_to(files)),line=line))
    if errors:raise ValueError('Runtime log errors: '+repr(errors[:4]))
    value=dict(status='PASS',job_id=job,execution_commit=execution,source_hash=manifest['source_hash'],audited_at=now(),
        source_blobs=len(manifest['files']),preserved_files=len(saved['files']),preserved_bytes=sum(x['bytes'] for x in saved['files']),
        targeted_cases=len(c.plan()['required_cases']),targeted_required_checks=sum(len(x['required_keys']) for x in c.plan()['required_cases']),
        smoke_steps=6,scientific_fits_started=2,scientific_fits_completed=2,TEST='NOT_RUN',test_evaluation_count=0,
        rows=rows,metric_cells=sum(x['metric_cells'] for x in rows),paired_fields=list(pairing),fixed_replay=baseline,
        contrasts=summary['contrasts'],runtime_errors=errors,warnings=warnings,checkpoint_loading=False)
    output=folder/'independent_audit.json'
    if output.exists():
        previous=read(output)
        if {k:v for k,v in previous.items() if k!='audited_at'}!={k:v for k,v in value.items() if k!='audited_at'}:raise ValueError('Audit changed')
        value=previous
    else:create(output,value)
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True);args=parser.parse_args()
    print(json.dumps(audit(args.job),indent=2))
