"""One explicit read-only scheduler poll; no loop or background process."""
import base64
import hashlib
import json
import subprocess
from pathlib import Path

REMOTE=r'''
import base64,json,subprocess
from pathlib import Path
from datetime import datetime,timezone
root=Path('/home/daryumin/iberdov/diplom');here=root/'experiments/mamba3_layer_temporal';logs=here/'slurm_logs/attempt_001'
job='4372822';execution='30640e2be36b45c6f89e31f91aae83f76f2d2254'
def command(args):
 p=subprocess.run(args,cwd=root,capture_output=True,text=True);return dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
value=dict(job_id=job,checked_at=datetime.now(timezone.utc).isoformat(),checkout=command(['git','rev-parse','HEAD']),tracked_status=command(['git','status','--porcelain','--untracked-files=no']),
 squeue=command(['squeue','-j',job,'-h','-o','%i|%T|%M|%N|%R|%S']),sacct=command(['sacct','-j',job,'--format=JobIDRaw,State,ExitCode,Elapsed,NodeList,Start,End','-P']),pipeline=None,records=[],gate=None,smoke=None,submission_files={})
for key,path in [('pipeline',logs/'pipeline_status.json'),('gate',here/'runs/attempt_001/targeted_gate.json'),('smoke',here/'runs/attempt_001/smoke.json')]:
 if path.exists():
  r=json.loads(path.read_text());value[key]={k:v for k,v in r.items() if k not in ('runtime','required_cases')}
for path in sorted((here/'runs/attempt_001').glob('mamba3_layer_temporal_*.json')):
 r=json.loads(path.read_text());value['records'].append({k:r.get(k) for k in ('run_id','temporal_sharing','status','stage','scientific_fit_started','actual_epochs','error')})
for name in ['login_verification.json','reservation.json','submission.json','cpu_preflight_'+execution+'.json','no_git_preflight_'+execution+'.json']:
 path=logs/name;raw=path.read_bytes();value['submission_files'][name]=base64.b64encode(raw).decode()
print(json.dumps(value))
'''

def main():
    here=Path(__file__).resolve().parents[1]
    value=json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes','hse-karizma','python3 -'],input=REMOTE.encode()))
    folder=here/'evidence/job4372822';folder.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name,encoded in value.pop('submission_files').items():
        raw=base64.b64decode(encoded);path=folder/'submission'/name;path.parent.mkdir(exist_ok=True)
        if path.exists() and path.read_bytes()!=raw:raise ValueError('Submission artifact changed '+name)
        if not path.exists():path.write_bytes(raw)
        rows.append(dict(path=str(path.relative_to(here)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    manifest=folder/'submission_preservation.json'
    if not manifest.exists():manifest.write_text(json.dumps(dict(job_id='4372822',files=rows),indent=2)+'\n')
    raw=json.dumps(value,indent=2)+'\n'
    stamp=value['checked_at'].replace(':','').replace('+','_')
    path=folder/('status_'+stamp+'.json')
    with path.open('x') as stream:stream.write(raw)
    (here/'runtime/status_4372822.json').write_text(raw)
    print(json.dumps({k:v for k,v in value.items() if k not in ('gate','smoke')},indent=2))
    for key in ('gate','smoke'):
        r=value[key]
        if r:print(key,r.get('status'),r.get('error'),[(x.get('case_id'),x.get('status'),x.get('failed_keys')) for x in r.get('cases',[])])

if __name__=='__main__':main()
