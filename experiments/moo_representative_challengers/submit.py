"""Submit one gate-checked canonical stage; an ambiguous submission retains its lock."""
import argparse
import os
from pathlib import Path
import subprocess
from .common import HERE, ROOT, canonical_run_id, git, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=['ferero','most','phn_hvi'],required=True)
    p.add_argument('--stage',choices=['smoke','sanity','convergence_screening'],required=True)
    p.add_argument('--python',required=True)
    args=p.parse_args()
    if ROOT==Path('/home/daryumin/iberdov/diplom'):
        raise RuntimeError('Use separate challenger checkout')
    sha=git('rev-parse','HEAD')
    env=dict(os.environ,REPO_DIR=str(ROOT),MOO_GIT_COMMIT=sha,MOO_METHOD=args.method,
             MOO_STAGE=args.stage,MOO_PYTHON=str(Path(args.python).resolve()),PYTHONDONTWRITEBYTECODE='1')
    # Login preflight cannot load data or initialize CUDA. Runtime must import normally.
    subprocess.run([args.python,'-m','experiments.moo_representative_challengers.run',
                    '--method',args.method,'--stage',args.stage,'--preflight-only'],cwd=ROOT,env=env,check=True)
    stage='convergence' if args.stage=='convergence_screening' else args.stage
    run_id=canonical_run_id(args.method, args.stage)
    folder=HERE/'submissions'; folder.mkdir(exist_ok=True)
    lock=folder/(run_id+'.lock'); lock.mkdir()  # atomic duplicate prevention
    output=folder/(run_id+'.json')
    payload={'run_id':run_id,'method':args.method,'stage':args.stage,'git_commit':sha,
             'git_branch':git('branch','--show-current'),'test_evaluation_count':0,
             'status':'submission_in_progress','python':env['MOO_PYTHON']}
    write_json(output,payload)
    jobname=f'mooc-{args.method}-{stage}'
    (HERE/'slurm_logs').mkdir(exist_ok=True)
    result=subprocess.run(['sbatch','--parsable','--job-name='+jobname,'--export=ALL',
        str(ROOT/'slurm/moo_representative_challengers.sh')],cwd=ROOT,env=env,text=True,capture_output=True)
    payload.update(stdout=result.stdout,stderr=result.stderr,returncode=result.returncode)
    if result.returncode==0 and result.stdout.strip().split(';')[0].isdigit():
        payload.update(status='submitted',job_id=result.stdout.strip().split(';')[0])
    else:
        payload['status']='submission_failed_or_ambiguous'
    write_json(output,payload)
    print(output.read_text())
    if payload['status']!='submitted': raise RuntimeError('Inspect scheduler before any retry; lock retained')


if __name__=='__main__': main()
