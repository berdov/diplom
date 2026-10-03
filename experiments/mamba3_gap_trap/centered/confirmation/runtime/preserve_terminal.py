"""Read compact terminal evidence and stream weight hashes from the canonical cluster."""
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

REMOTE = r'''
import hashlib, io, json, subprocess, sys, tarfile
from datetime import datetime, timezone
from pathlib import Path
root=Path('/home/daryumin/iberdov/diplom')
here=root/'experiments/mamba3_gap_trap/centered/confirmation'
job='4372023'
def run(args):return subprocess.check_output(args,cwd=root,text=True)
commit=run(['git','rev-parse','HEAD']).strip()
assert commit=='23468e74389aed841aab0ac16b370c5ce376e328'
assert not run(['git','status','--porcelain','--untracked-files=no']).strip()
sacct=run(['sacct','-j',job,'--format=JobIDRaw,State,ExitCode,Start,End,Elapsed,NodeList','-P'])
rows=[dict(zip(sacct.splitlines()[0].split('|'),line.split('|'))) for line in sacct.splitlines()[1:] if line]
main=next(r for r in rows if r['JobIDRaw']==job)
assert main['State']=='COMPLETED' and main['ExitCode']=='0:0'
pipeline=json.loads((here/'slurm_logs/attempt_001/pipeline_status.json').read_text())
assert pipeline['status']=='PASS' and pipeline['scientific_fits_completed']==8 and pipeline['complete_pairs']==4
checked=datetime.now(timezone.utc).isoformat()
scheduler={'checked_at':checked,'timezone':'Europe/Moscow','sacct_raw':sacct,'jobs':{job:{'state':main['State'],'exit_code':main['ExitCode'],'start':main['Start'],'end':main['End'],'elapsed':main['Elapsed'],'node':main['NodeList']}},'steps':rows}
paths=[]
for base in (here/'runs',here/'slurm_logs'):
 for path in base.rglob('*'):
  if not path.is_file() or any(x in path.parts for x in ('tilelang_cache','tensorboard','__pycache__')):continue
  if path.suffix in ('.json','.log','.out','.err','.lock','.md'):paths.append(path)
paths.extend(here/name for name in ('study_plan.json','source_index.json','source_manifest.json','DESIGN.md'))
blobs={};files=[]
for path in sorted(set(paths)):
 rel=str(path.relative_to(here));raw=path.read_bytes()
 assert len(raw)<20_000_000
 blobs['files/'+rel]=raw
 files.append({'path':rel,'cluster_path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
checkpoints={}
for path in sorted((here/'slurm_logs/fits').glob('*/checkpoints/best_state_dict.pth')):
 digest=hashlib.sha256();count=0
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block);count+=len(block)
 checkpoints[str(path.relative_to(here))]={'path':str(path),'bytes':count,'sha256':digest.hexdigest()}
assert len(checkpoints)==8
manifest={'job_id':job,'preserved_at':checked,'checkout_commit':commit,'tracked_clean':True,'source_hash':pipeline['source_hash'],'files':files,'checkpoints':checkpoints,'checkpoint_loading':False,'weights_copied':False}
for name,value in [('scheduler_terminal.json',scheduler),('preservation_manifest.json',manifest)]:blobs[name]=(json.dumps(value,indent=2,ensure_ascii=False)+'\n').encode()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 for name,raw in blobs.items():
  info=tarfile.TarInfo(name);info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
'''

def main():
    folder=Path(__file__).resolve().parents[1]/'evidence/job4372023'
    if (folder/'preservation_manifest.json').exists():raise ValueError('Already preserved; do not replace terminal evidence')
    raw=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','hse-karizma','python3 -'],input=REMOTE.encode())
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as tar:
        for member in tar:
            path=folder/member.name
            if not member.isfile() or not path.resolve().is_relative_to(folder.resolve()):raise ValueError('Unexpected archive entry')
            data=tar.extractfile(member).read()
            if path.exists() and path.read_bytes()!=data:raise ValueError('Refusing overwrite '+str(path))
            path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():path.write_bytes(data)
    manifest=json.loads((folder/'preservation_manifest.json').read_text())
    for row in manifest['files']:
        data=(folder/'files'/row['path']).read_bytes()
        assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
    print(json.dumps({'files':len(manifest['files']),'bytes':sum(x['bytes'] for x in manifest['files']),'checkpoints':len(manifest['checkpoints']),'folder':str(folder)}))

if __name__=='__main__':main()
