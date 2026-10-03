"""Read compact terminal evidence and stream checkpoint hashes; no weights copied."""
import argparse
import io
import json
import subprocess
import tarfile
from pathlib import Path

REMOTE=r'''
import hashlib,io,json,subprocess,sys,tarfile
from pathlib import Path
from datetime import datetime,timezone
root=Path('/home/daryumin/iberdov/diplom');here=root/'experiments/mamba3_layer_temporal'
job,execution=sys.argv[1:]
def run(args):return subprocess.check_output(args,cwd=root,text=True)
assert run(['git','rev-parse','HEAD']).strip()==execution
assert not run(['git','status','--porcelain','--untracked-files=no']).strip()
raw=run(['sacct','-j',job,'--format=JobIDRaw,State,ExitCode,Start,End,Elapsed,NodeList','-P'])
rows=[dict(zip(raw.splitlines()[0].split('|'),line.split('|'))) for line in raw.splitlines()[1:] if line]
main=next(r for r in rows if r['JobIDRaw']==job)
assert main['State'] not in ('RUNNING','PENDING','COMPLETING','CONFIGURING')
blobs={};files=[];weights={}
for base in (here/'runs',here/'slurm_logs'):
 for path in sorted(base.rglob('*')):
  if not path.is_file() or any(x in path.parts for x in ('__pycache__','tilelang_cache','tensorboard')):continue
  rel=str(path.relative_to(here))
  if path.name=='best_state_dict.pth':
   h=hashlib.sha256();size=0
   with path.open('rb') as stream:
    for block in iter(lambda:stream.read(1024*1024),b''):h.update(block);size+=len(block)
   weights[rel]=dict(path=str(path),bytes=size,sha256=h.hexdigest())
  elif path.suffix in ('.json','.log','.out','.err','.md','.lock'):
   content=path.read_bytes();assert len(content)<20_000_000
   blobs['files/'+rel]=content;files.append(dict(path=rel,cluster_path=str(path),bytes=len(content),sha256=hashlib.sha256(content).hexdigest()))
for name in ('study_plan.json','source_manifest.json','DESIGN.md'):
 path=here/name;content=path.read_bytes();blobs['files/'+name]=content
 files.append(dict(path=name,cluster_path=str(path),bytes=len(content),sha256=hashlib.sha256(content).hexdigest()))
scheduler=dict(checked_at=datetime.now(timezone.utc).isoformat(),timezone='Europe/Moscow',sacct_raw=raw,steps=rows,job=main)
manifest=dict(job_id=job,execution_commit=execution,source_hash=json.loads((here/'source_manifest.json').read_text())['source_hash'],
 preserved_at=scheduler['checked_at'],files=files,checkpoints=weights,checkpoint_loading=False,weights_copied=False)
for name,value in [('scheduler_terminal.json',scheduler),('preservation_manifest.json',manifest)]:blobs[name]=(json.dumps(value,indent=2)+'\n').encode()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
 for name,content in blobs.items():
  info=tarfile.TarInfo(name);info.size=len(content);archive.addfile(info,io.BytesIO(content))
'''

def main():
    parser=argparse.ArgumentParser();parser.add_argument('job');parser.add_argument('execution');args=parser.parse_args()
    if not args.job.isdecimal() or len(args.execution)!=40 or any(x not in '0123456789abcdef' for x in args.execution):raise ValueError('Exact job/commit required')
    folder=Path(__file__).resolve().parents[1]/'evidence'/('job'+args.job)
    if (folder/'preservation_manifest.json').exists():raise ValueError('Already preserved')
    raw=subprocess.check_output(['ssh','-o','BatchMode=yes','hse-karizma','python3','-',args.job,args.execution],input=REMOTE.encode())
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
