"""Frozen protocol, provenance and overwrite protection; no ML imports."""
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TARGETS = ('is_click','long_view','is_like','is_profile_enter')
METRICS = tuple(f'{m}@{k}' for m in ('HR','NDCG','Recall') for k in (5,10,20,50))

def read(path):
    return json.loads(Path(path).read_text())

def config():
    return yaml.safe_load((HERE/'config.yaml').read_text())

def combinations():
    return yaml.safe_load((HERE/'combinations.yaml').read_text())['combinations']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def write(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(text);tmp.replace(path)

def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()

def weights(active):
    if len(active)!=len(set(active)) or set(active)-set(TARGETS): raise ValueError('Invalid targets')
    return {t:1/len(active) for t in active}

def check_manifest(cells):
    expected={frozenset(s) for n in range(5) for s in itertools.combinations(TARGETS,n)}
    assert len(cells)==16
    assert {frozenset(c['active_targets']) for c in cells}==expected
    assert [c['array_index'] for c in cells]==list(range(16))
    for key in ['run_id','combination_id']: assert len({c[key] for c in cells})==16
    for c in cells:
        assert c['n_aux_targets']==len(c['active_targets'])
        assert c['combination_id']==''.join('1' if t in c['active_targets'] else '0' for t in TARGETS)
        weights(c['active_targets'])

def frozen_checks():
    cfg=config();check_manifest(combinations())
    assert cfg['evaluation_split']=='validation' and cfg['test_usage']=='forbidden' and cfg['test_evaluation_count']==0
    assert cfg['loss_weight_mode']=='uniform_normalized_aux'
    best=yaml.safe_load((ROOT/cfg['source']['best_params']).read_text())
    for k,v in cfg['optimization'].items():
        assert v==(best[k] if k=='effective_pos_weights' else best['params'][k]),k
    assert cfg['training']['seed']==2026 and cfg['training']['max_epochs']==80
    source=yaml.safe_load((ROOT/cfg['source']['validation_only_config']).read_text())
    assert source['recbole_overrides']['benchmark_filename']==['train','valid']
    assert source['recbole_overrides']['eval_args']['split']=={'LS':'valid_only'}
    assert source['recbole_overrides']['eval_args']['mode']=='full'
    assert all(not v for v in source['test_policy'].values())
    assert source['protocol']['identity_hash']==cfg['fingerprint']
    hashes=read(HERE/'historical_hashes.json')
    assert all(sha(ROOT/f)==h for f,h in hashes.items()),'Historical files changed'
    return {'historical_files_checked':len(hashes),'test_evaluation_count':0}

def source_digest():
    paths=list(HERE.glob('*.py'))+list(HERE.glob('*.yaml'))+list((HERE/'tests').glob('*.py'))+[HERE/'historical_hashes.json']+list((ROOT/'slurm').glob('*target_combinations*.sh'))
    return hashlib.sha256('\n'.join(f'{p.relative_to(ROOT)} {sha(p)}' for p in sorted(paths)).encode()).hexdigest()

def provenance():
    frozen_checks();head=git('rev-parse','HEAD')
    assert os.environ.get('TC_COMMIT')==head,'Exact published TC_COMMIT required'
    assert git('branch','--show-current')==config()['branch']
    assert not git('status','--porcelain','--untracked-files=no'),'Dirty tracked checkout'
    assert git('rev-parse','origin/'+config()['branch'])==head
    cert=read(HERE/'verification.json')
    assert cert['status']=='passed' and cert['source_digest']==source_digest()
    return head

def reserve(run_id):
    output=HERE/'runs'/f'{run_id}.json';artifact=HERE/'artifacts'/run_id
    if output.exists(): raise FileExistsError(output)
    artifact.parent.mkdir(parents=True,exist_ok=True)
    artifact.mkdir()  # atomic, also prevents retries after SIGKILL
    return output,artifact
