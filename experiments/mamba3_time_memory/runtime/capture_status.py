"""One compact read-only poll, rate limited to ten minutes. No background work."""
import argparse
import base64
import hashlib
import json
import subprocess
from datetime import datetime,timezone
from pathlib import Path
REMOTE=r'''
import base64,json,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
job,execution,attempt,submission=sys.argv[1:];root=Path('/home/daryumin/iberdov/diplom');here=root/'experiments/mamba3_time_memory';logs=here/('slurm_logs/attempt_'+attempt);runs=here/('runs/attempt_'+attempt)
def command(args):
 p=subprocess.run(args,cwd=root,capture_output=True,text=True);return dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
value=dict(job_id=job,execution_attempt=attempt,checked_at=datetime.now(timezone.utc).isoformat(),checkout=command(['git','rev-parse','HEAD']),tracked_status=command(['git','status','--porcelain','--untracked-files=no']),squeue=command(['squeue','-j',job,'-h','-o','%i|%T|%M|%N|%R|%S']),sacct=command(['sacct','-j',job,'--format=JobIDRaw,State,ExitCode,Elapsed,NodeList,Start,End','-P']),pipeline=None,records=[],gate=None,smoke=None,submission_files={})
for key,path in [('pipeline',logs/'pipeline_status.json'),('gate',runs/'targeted_gate.json'),('smoke',runs/'smoke.json')]:
 if path.exists():
  r=json.loads(path.read_text());value[key]={k:r.get(k) for k in ('status','started_at','finished_at','error','scientific_fits_started','scientific_fits_completed')}
  if key=='pipeline':value[key]['stages']=r.get('stages',[])
  if key=='gate':value[key]['cases']=[{k:x.get(k) for k in ('case_id','status','failed_keys','missing_keys')} for x in r.get('cases',[])]
  if key=='smoke':value[key]['rows']=[dict(memory_mode=x.get('memory_mode'),status=x.get('status'),steps=len(x.get('steps',[]))) for x in r.get('rows',[])]
for path in sorted(logs.glob('mamba3_time_memory_*/progress.json')):value['records'].append(json.loads(path.read_text()))
if submission=='yes':
 for name in ['login_verification.json','reservation.json','submission.json','train_coverage.json','cpu_preflight_'+execution+'.json','no_git_preflight_'+execution+'.json']:
  path=logs/name;value['submission_files'][name]=base64.b64encode(path.read_bytes()).decode()
print(json.dumps(value))
'''

def main():
    parser=argparse.ArgumentParser();parser.add_argument('job');parser.add_argument('execution');parser.add_argument('--attempt',choices=['001','002'],default='001');args=parser.parse_args()
    if not args.job.isdecimal() or len(args.execution)!=40 or any(x not in '0123456789abcdef' for x in args.execution):raise ValueError('Exact job/commit required')
    here=Path(__file__).resolve().parents[1];folder=here/'evidence'/('job'+args.job);latest=here/'runtime'/('status_'+args.job+'.json')
    if latest.exists() and (datetime.now(timezone.utc)-datetime.fromisoformat(json.loads(latest.read_text())['checked_at'])).total_seconds()<600:raise ValueError('Next compact poll is not due yet (ten-minute limit)')
    value=json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes','hse-karizma','python3','-',args.job,args.execution,args.attempt,'no' if (folder/'submission_preservation.json').exists() else 'yes'],input=REMOTE.encode()))
    folder.mkdir(parents=True,exist_ok=True);rows=[]
    for name,encoded in value.pop('submission_files').items():
        raw=base64.b64decode(encoded);path=folder/'submission'/name;path.parent.mkdir(exist_ok=True)
        if path.exists() and path.read_bytes()!=raw:raise ValueError('Immutable submission evidence changed')
        if not path.exists():path.write_bytes(raw)
        rows.append(dict(path=str(path.relative_to(here)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    if rows:(folder/'submission_preservation.json').write_text(json.dumps(dict(job_id=args.job,files=rows),indent=2)+'\n')
    raw=json.dumps(value,indent=2)+'\n';stamp=value['checked_at'].replace(':','').replace('+','_')
    with (folder/('status_'+stamp+'.json')).open('x') as stream:stream.write(raw)
    latest.write_text(raw);print(raw)

if __name__=='__main__':main()
