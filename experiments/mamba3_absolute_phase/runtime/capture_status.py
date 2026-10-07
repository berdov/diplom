"""One compact read-only poll, rate limited to ten minutes. No background work."""
import argparse
import base64
import hashlib
import json
import re
import subprocess
from datetime import datetime,timedelta,timezone
from pathlib import Path
REMOTE=r'''
import base64,json,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
job,execution,attempt,stage,submission=sys.argv[1:];root=Path('/home/daryumin/iberdov/diplom');here=root/'experiments/mamba3_absolute_phase';logs=here/'slurm_logs'/stage/('attempt_'+attempt);runs=here/'runs'/stage/('attempt_'+attempt)
def command(args):
 p=subprocess.run(args,cwd=root,capture_output=True,text=True);return dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
value=dict(job_id=job,execution_attempt=attempt,study_phase=stage,checked_at=datetime.now(timezone.utc).isoformat(),checkout=command(['git','rev-parse','HEAD']),tracked_status=command(['git','status','--porcelain','--untracked-files=no']),squeue=command(['squeue','-j',job,'-h','-o','%i|%T|%M|%N|%R|%S']),sacct=command(['sacct','-j',job,'--format=JobIDRaw,State,ExitCode,Elapsed,NodeList,Start,End','-P']),pipeline=None,records=[],gate=None,smoke=None,submission_files={})
for key,path in [('pipeline',logs/'pipeline_status.json'),('gate',runs/'targeted_gate.json'),('smoke',runs/'smoke.json')]:
 if path.exists():
  r=json.loads(path.read_text());value[key]={k:r.get(k) for k in ('status','started_at','finished_at','error','scientific_fits_started','scientific_fits_completed')}
  if key=='pipeline':value[key]['stages']=r.get('stages',[])
  if key=='gate':value[key]['cases']=[{k:x.get(k) for k in ('case_id','status','failed_keys','missing_keys')} for x in r.get('cases',[])]
  if key=='smoke':value[key]['rows']=[dict(phase_mode=x.get('phase_mode'),status=x.get('status'),steps=len(x.get('steps',[]))) for x in r.get('rows',[])]
for path in sorted(logs.glob('mamba3_absolute_phase_*/progress.json')):value['records'].append(json.loads(path.read_text()))
if submission=='yes':
 for name in ['login_verification.json','reservation.json','submission.json','train_coverage.json','cpu_preflight_'+execution+'.json','no_git_preflight_'+execution+'.json']:
  path=logs/name;value['submission_files'][name]=base64.b64encode(path.read_bytes()).decode()
print(json.dumps(value))
'''


def sync_handoff(value,runtime):
    """Update only operational state/prose from an already saved compact poll."""
    state_path,handoff_path=runtime/'state.json',runtime/'HANDOFF.md'
    if not state_path.exists() or not handoff_path.exists():return None
    state=json.loads(state_path.read_text());checked=datetime.fromisoformat(value['checked_at'])
    if str(state.get('job_id'))!=str(value['job_id']):raise ValueError('Handoff belongs to another job')
    if state.get('checked_at') and datetime.fromisoformat(state['checked_at'])>checked:raise ValueError('Older snapshot cannot replace handoff')
    scheduler={};sacct=value.get('sacct') or {};queue=value.get('squeue') or {}
    lines=sacct.get('stdout','').splitlines() if sacct.get('returncode')==0 else []
    if lines:
        scheduler=next((row for line in lines[1:] if (row:=dict(zip(lines[0].split('|'),line.split('|')))).get('JobIDRaw')==str(value['job_id'])),{})
    queued=next((line.split('|') for line in queue.get('stdout','').splitlines() if line.split('|')[0]==str(value['job_id'])),[]) if queue.get('returncode')==0 else []
    job_state=scheduler.get('State') or (queued[1] if len(queued)>1 else None)
    job_state=job_state.split()[0].rstrip('+') if job_state else None
    node=scheduler.get('NodeList') or (queued[3] if len(queued)>3 else None)
    if node in ('','None assigned','Unknown','N/A','(null)'):node=None
    elapsed=scheduler.get('Elapsed') or (queued[2] if len(queued)>2 else None)
    terminal=job_state in {'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','BOOT_FAIL','DEADLINE','PREEMPTED','REVOKED','SPECIAL_EXIT'}
    phase='AWAITING_TERMINAL_AUDIT' if terminal else 'SUBMITTED_PENDING' if job_state=='PENDING' else 'RUNNING' if job_state in ('RUNNING','CONFIGURING','COMPLETING','SUSPENDED') else 'STATUS_UNKNOWN'
    pipeline=value.get('pipeline') or {};records=value.get('records') or []
    records=[r for r in records if str(r.get('job_id'))==str(value['job_id']) and r.get('phase_mode') in ('baseline_dual','relative_phase','absolute_phase')]
    counters={}
    for key,observed in (('scientific_fits_started',sum(r.get('scientific_fit_started') is True for r in records)),('scientific_fits_completed',sum(r.get('status')=='PASS' for r in records))):
        candidates=[pipeline[key]] if type(pipeline.get(key)) is int and pipeline[key]>=0 else []
        if observed or len({(r['phase_mode'],r['seed']) for r in records})==(3 if value['study_phase']=='pilot' else 12):candidates.append(observed)
        counters[key]=max(candidates) if candidates else None
    due=(checked+timedelta(seconds=600)).isoformat() if not terminal else None
    prior=state.get('prior_scientific_fits_completed',0)
    totals={key:(number+prior if number is not None else None) for key,number in counters.items()}
    state.update(totals,current_phase_fits_started=counters['scientific_fits_started'],current_phase_fits_completed=counters['scientific_fits_completed'],phase=phase,job_state=job_state,node=node,elapsed=elapsed,
                 reason=queued[4].strip('()') if job_state=='PENDING' and len(queued)>4 else None,
                 checked_at=value['checked_at'],updated_at=value['checked_at'],next_poll_not_before=due,terminal_status_verified=terminal,
                 gpu_gate=(value.get('gate') or {}).get('status'),smoke=(value.get('smoke') or {}).get('status'),
                 pipeline_status=pipeline.get('status'),fit_counters_source='Observed pipeline/progress; null means not recorded in this snapshot',
                 next_step='Preserve terminal evidence and perform independent audit; no automatic poll, resubmit or refit' if terminal else 'One explicit compact poll no earlier than next_poll_not_before; no duplicate submission')
    show=lambda x:'UNKNOWN' if x is None else str(x)
    msk=lambda x:datetime.fromisoformat(x).astimezone(timezone(timedelta(hours=3))).strftime('%Y-%m-%d %H:%M:%S MSK')
    paragraph=(f"Phase: {phase}. **{msk(value['checked_at'])}:** {show(job_state)}"
        +(f" ({state['reason']})" if state['reason'] else '')
        +f", node {show(node)}, elapsed {show(elapsed)}. Scientific fits started {show(counters['scientific_fits_started'])}, completed {show(counters['scientific_fits_completed'])}/{show(state.get('current_phase_max_fits'))}. "
        +f"GPU gate {show(state['gpu_gate'])}; smoke {show(state['smoke'])}. "
        +(f"Next explicit poll no earlier than {msk(due)} ({due}). " if due else 'Terminal evidence awaits preservation and independent audit. ')
        +'No background process.\n')
    def replace_status(match):
        previous=match.group()
        if re.search(r'scancel|cancellation|отмен',previous,re.I):
            notes=previous.split('\n',1)[1] if '\n' in previous else previous
            return paragraph+'\nCancellation request notes (retained):\n'+notes
        return paragraph.rstrip('\n')+('\n' if previous.endswith('\n') else '')
    handoff,count=re.subn(r'^Phase:.*?(?=^\[Exact snapshot\]|\n\n|\Z)',replace_status,handoff_path.read_text(),count=1,flags=re.M|re.S)
    if count!=1:raise ValueError('Expected one existing handoff status paragraph')
    for path,text in ((state_path,json.dumps(state,indent=2)+'\n'),(handoff_path,handoff)):
        temporary=path.with_name(path.name+'.capture.tmp');temporary.write_text(text);temporary.replace(path)
    return state


def main():
    parser=argparse.ArgumentParser();parser.add_argument('job');parser.add_argument('execution');parser.add_argument('--attempt',choices=['001','002'],default='001');parser.add_argument('--stage',choices=['pilot','confirmation'],default='pilot');parser.add_argument('--ssh-identity');args=parser.parse_args()
    if not args.job.isdecimal() or len(args.execution)!=40 or any(x not in '0123456789abcdef' for x in args.execution):raise ValueError('Exact job/commit required')
    here=Path(__file__).resolve().parents[1];folder=here/'evidence'/('job'+args.job);latest=here/'runtime'/('status_'+args.job+'.json')
    if latest.exists() and (datetime.now(timezone.utc)-datetime.fromisoformat(json.loads(latest.read_text())['checked_at'])).total_seconds()<600:raise ValueError('Next compact poll is not due yet (ten-minute limit)')
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15']
    if args.ssh_identity:ssh+=['-i',str(Path(args.ssh_identity).expanduser())]
    value=json.loads(subprocess.check_output(ssh+['hse-karizma','python3','-',args.job,args.execution,args.attempt,args.stage,'no' if (folder/'submission_preservation.json').exists() else 'yes'],input=REMOTE.encode()))
    folder.mkdir(parents=True,exist_ok=True);rows=[]
    for name,encoded in value.pop('submission_files').items():
        raw=base64.b64decode(encoded);path=folder/'submission'/name;path.parent.mkdir(exist_ok=True)
        if path.exists() and path.read_bytes()!=raw:raise ValueError('Immutable submission evidence changed')
        if not path.exists():path.write_bytes(raw)
        rows.append(dict(path=str(path.relative_to(here)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    if rows:(folder/'submission_preservation.json').write_text(json.dumps(dict(job_id=args.job,files=rows),indent=2)+'\n')
    raw=json.dumps(value,indent=2)+'\n';stamp=value['checked_at'].replace(':','').replace('+','_')
    with (folder/('status_'+stamp+'.json')).open('x') as stream:stream.write(raw)
    latest.write_text(raw);sync_handoff(value,here/'runtime');print(raw)

if __name__=='__main__':main()
