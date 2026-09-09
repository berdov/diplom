"""Atomic submission ledger; retain partial submissions on any failure."""
import os
from pathlib import Path
import subprocess
from .common import HERE,ROOT,config,read,write,provenance,source_digest,combinations

def main():
    sha=provenance();cfg=config()
    assert str(ROOT)=='/home/daryumin/iberdov/diplom_exp_target_combinations_002'
    audit=read(HERE/'artifacts/data_audit.json');assert audit['status']=='passed' and audit['source_digest']==source_digest()
    env_check=read(HERE/'artifacts/environment.json');assert env_check['status']=='passed'
    assert os.environ['TC_REPO']==str(ROOT) and Path(os.environ['TC_PYTHON']).is_file()
    assert env_check['python_executable']==os.environ['TC_PYTHON']
    for run_id in [cfg['smoke_run_id']]+[c['run_id'] for c in combinations()]:
        assert not (HERE/'runs'/f'{run_id}.json').exists() and not (HERE/'artifacts'/run_id).exists()
    folder=HERE/'submissions';folder.mkdir(exist_ok=True);(folder/'pipeline.lock').mkdir()
    log=ROOT/'logs/slurm';log.mkdir(parents=True,exist_ok=True)
    payload={'status':'submitting','git_commit':sha,'branch':cfg['branch'],'test_evaluation_count':0,'array':'0-15%4','python':os.environ['TC_PYTHON'],'data_audit_sha':audit['source_digest'],'submissions':[]}
    path=folder/'pipeline.json';write(path,payload)
    script=str(ROOT/'slurm/target_combinations.sh')
    def submit(mode,options):
        result=subprocess.run(['sbatch','--parsable','--no-requeue','--export=ALL',*options,script],cwd=ROOT,env=dict(os.environ,TC_MODE=mode),text=True,capture_output=True)
        item={'mode':mode,'options':options,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr};payload['submissions'].append(item);write(path,payload)
        job=result.stdout.strip().split(';')[0]
        if result.returncode or not job.isdigit():raise RuntimeError('Ambiguous/failed submit: inspect ledger, do not repeat pipeline')
        payload[{'smoke':'smoke_job_id','full':'array_job_id','summary':'summary_job_id'}[mode]]=job;write(path,payload)
        return job
    gpu=['--partition=rocky','--constraint=type_e','--gres=gpu:a100:1','--cpus-per-task=4','--mem=0','--time=03:00:00']
    smoke=submit('smoke',gpu+['--job-name=target-combo-smoke',f'--output={log}/target-combo-smoke-%j.out',f'--error={log}/target-combo-smoke-%j.err'])
    array=submit('full',gpu+['--array=0-15%4','--job-name=target-combo',f'--dependency=afterok:{smoke}','--kill-on-invalid-dep=yes',f'--output={log}/target-combo-%A_%a.out',f'--error={log}/target-combo-%A_%a.err'])
    submit('summary',['--partition=rocky','--cpus-per-task=1','--mem=0','--time=00:15:00','--job-name=target-combo-summary',f'--dependency=afterany:{array}',f'--output={log}/target-combo-summary-%j.out',f'--error={log}/target-combo-summary-%j.err'])
    payload['status']='submitted';write(path,payload);print(path.read_text())

if __name__=='__main__':main()
