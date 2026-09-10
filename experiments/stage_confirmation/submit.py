"""Submit exactly six fixed confirmations, once, then a dependent CPU summary."""
import os
import subprocess
from .run import HERE,ROOT,provenance
from experiments.target_combination_analysis.common import write

def main():
    commit=provenance();python=os.environ['SC_PYTHON']
    assert os.environ['SC_REPO']==str(ROOT)
    for i in range(6):
        subprocess.run([python,'-m','experiments.stage_confirmation.run','--index',str(i),'--preflight-only'],check=True,cwd=ROOT)
    folder=HERE/'submissions';folder.mkdir(exist_ok=True);(folder/'pipeline.lock').mkdir()
    log=HERE/'slurm_logs';log.mkdir(exist_ok=True)
    path=folder/'pipeline.json';p={'status':'submitting','git_commit':commit,'seeds':[2026,2027,2028],'reused_seed':2026,'new_scientific_runs':6,'test_evaluation_count':0,'submissions':[]};write(path,p)
    def sbatch(mode,opts):
        r=subprocess.run(['sbatch','--parsable','--no-requeue','--export=ALL',*opts,str(ROOT/'slurm/stage_confirmation.sh')],env=dict(os.environ,SC_MODE=mode),text=True,capture_output=True)
        item={'mode':mode,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'options':opts};p['submissions'].append(item);write(path,p)
        job=r.stdout.strip().split(';')[0]
        if r.returncode or not job.isdigit():raise RuntimeError('Submission failed/ambiguous; inspect ledger, do not repeat')
        p[mode+'_job_id']=job;write(path,p);return job
    job=sbatch('full',['--array=0-5%3','--partition=rocky','--constraint=type_e','--gres=gpu:a100:1','--cpus-per-task=4','--mem=0','--time=03:00:00','--job-name=target-confirm',f'--output={log}/target-confirm-%A_%a.out',f'--error={log}/target-confirm-%A_%a.err'])
    sbatch('summary',['--partition=rocky','--cpus-per-task=1','--mem=0','--time=00:15:00','--job-name=stage-close-summary',f'--dependency=afterany:{job}',f'--output={log}/summary-%j.out',f'--error={log}/summary-%j.err'])
    p['status']='submitted';write(path,p);print(path.read_text())

if __name__=='__main__':main()
