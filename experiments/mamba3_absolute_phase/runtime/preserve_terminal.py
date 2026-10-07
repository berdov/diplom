"""Read compact terminal evidence and stream checkpoint hashes; no weights copied."""
import argparse
import io
import inspect
import json
import subprocess
import tarfile
from pathlib import Path


def preservation_documents(stage, attempt, read_relative):
    """Resolve the byte-bound confirmation lineage without loading code/models."""
    if stage not in ('pilot', 'confirmation') or attempt not in ('001', '002'):
        raise ValueError('Invalid preservation phase/attempt')
    manifest = 'source_manifest'+('_confirmation' if stage == 'confirmation' else '')+('_002' if attempt == '002' else '')+'.json'
    names = ['study_plan.json', manifest, 'DESIGN.md', 'NEW_PLAN.md']
    if stage == 'confirmation':
        names.extend(['source_manifest.json', 'runtime/confirmation_decision.json'])
        decision = read_relative('runtime/confirmation_decision.json')
        if 'source_lineage_path' in decision:
            prefix = Path('experiments/mamba3_absolute_phase')
            def package_relative(value):
                path = Path(value)
                if path.is_absolute() or '..' in path.parts or not path.is_relative_to(prefix):
                    raise ValueError('Unsafe lineage dependency')
                return str(path.relative_to(prefix))
            name = package_relative(decision['source_lineage_path'])
            names.append(name)
            lineage = read_relative(name)
            names.extend(package_relative(lineage[key]) for key in
                         ('pilot_manifest_path', 'confirmation_manifest_path', 'failure_evidence_path', 'review_path'))
    if attempt == '002':
        names.append('runtime/retry_review_'+stage+'.json')
    return manifest, list(dict.fromkeys(names))


REMOTE=inspect.getsource(preservation_documents)+r'''
import hashlib,io,json,subprocess,sys,tarfile
from pathlib import Path
from datetime import datetime,timezone
root=Path('/home/daryumin/iberdov/diplom');here=root/'experiments/mamba3_absolute_phase'
job,execution,attempt,stage=sys.argv[1:]
def run(args):return subprocess.check_output(args,cwd=root,text=True)
assert run(['git','rev-parse','HEAD']).strip()==execution
assert not run(['git','status','--porcelain','--untracked-files=no']).strip()
raw=run(['sacct','-j',job,'--format=JobIDRaw,State,ExitCode,Start,End,Elapsed,NodeList','-P'])
rows=[dict(zip(raw.splitlines()[0].split('|'),line.split('|'))) for line in raw.splitlines()[1:] if line]
main=next(r for r in rows if r['JobIDRaw']==job)
assert main['State'].split()[0].rstrip('+') in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','BOOT_FAIL','DEADLINE','PREEMPTED','REVOKED','SPECIAL_EXIT')
blobs={};files=[];weights={}
for base in (here/'runs'/stage/('attempt_'+attempt),here/'slurm_logs'/stage/('attempt_'+attempt)):
 for path in sorted(base.rglob('*')):
  if not path.is_file() or any(x in path.parts for x in ('__pycache__','tilelang_cache','tensorboard')):continue
  rel=str(path.relative_to(here))
  if path.name=='best_state_dict.pth':
   h=hashlib.sha256();size=0
   with path.open('rb') as stream:
    for block in iter(lambda:stream.read(1024*1024),b''):h.update(block);size+=len(block)
   weights[rel]=dict(path=str(path),bytes=size,sha256=h.hexdigest())
  elif path.suffix in ('.json','.log','.out','.err','.md','.lock'):
   content=path.read_bytes()
   if len(content)>=100_000_000:raise ValueError('Compact evidence exceeds 100 MB: '+rel)
   blobs['files/'+rel]=content;files.append(dict(path=rel,cluster_path=str(path),bytes=len(content),sha256=hashlib.sha256(content).hexdigest()))
manifest_name,documents=preservation_documents(stage,attempt,lambda name:json.loads((here/name).read_bytes()))
for name in documents:
 path=here/name
 if path.is_symlink() or not path.resolve().is_relative_to(here.resolve()):raise ValueError('Unsafe preserved dependency: '+name)
 content=path.read_bytes()
 if len(content)>=100_000_000:raise ValueError('Compact evidence exceeds 100 MB: '+name)
 if 'files/'+name in blobs:
  if blobs['files/'+name]!=content:raise ValueError('Dependency changed during preservation: '+name)
  continue
 blobs['files/'+name]=content
 files.append(dict(path=name,cluster_path=str(path),bytes=len(content),sha256=hashlib.sha256(content).hexdigest()))
scheduler=dict(checked_at=datetime.now(timezone.utc).isoformat(),timezone='Europe/Moscow',sacct_raw=raw,steps=rows,job=main)
manifest=dict(job_id=job,execution_attempt=attempt,study_phase=stage,execution_commit=execution,source_hash=json.loads((here/manifest_name).read_text())['source_hash'],
 preserved_at=scheduler['checked_at'],files=files,checkpoints=weights,checkpoint_loading=False,weights_copied=False)
for name,value in [('scheduler_terminal.json',scheduler),('preservation_manifest.json',manifest)]:blobs[name]=(json.dumps(value,indent=2)+'\n').encode()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
 for name,content in blobs.items():
  info=tarfile.TarInfo(name);info.size=len(content);archive.addfile(info,io.BytesIO(content))
'''

def main():
    parser=argparse.ArgumentParser();parser.add_argument('job');parser.add_argument('execution');parser.add_argument('--attempt',choices=['001','002'],default='001');parser.add_argument('--stage',choices=['pilot','confirmation'],default='pilot');parser.add_argument('--ssh-identity');args=parser.parse_args()
    if not args.job.isdecimal() or len(args.execution)!=40 or any(x not in '0123456789abcdef' for x in args.execution):raise ValueError('Exact job/commit required')
    folder=Path(__file__).resolve().parents[1]/'evidence'/('job'+args.job)
    if (folder/'preservation_manifest.json').exists():raise ValueError('Already preserved')
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15']
    if args.ssh_identity:ssh+=['-i',str(Path(args.ssh_identity).expanduser())]
    raw=subprocess.check_output(ssh+['hse-karizma','python3','-',args.job,args.execution,args.attempt,args.stage],input=REMOTE.encode())
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
        for member in archive:
            path=folder/member.name
            if not member.isfile() or not path.resolve().is_relative_to(folder.resolve()):raise ValueError('Unexpected archive path')
            content=archive.extractfile(member).read()
            if path.exists() and path.read_bytes()!=content:raise ValueError('Refusing to replace existing evidence')
            path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():path.write_bytes(content)
    print(folder)

if __name__=='__main__':main()
