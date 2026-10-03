"""New routing evidence with inherited, unchanged MIMO kernel admission."""
import hashlib
import json
import math
from . import config as c
from experiments.mamba3_gap_trap import provenance as parent
from experiments.mamba3_gap_trap.centered.reuse import bind
from experiments.mamba3_mimo_time.confirmation.provenance import inherited
from experiments.mamba3_mimo_time.records import read,sha,digest,create,now


def freeze():
    c.unused();inherited();c.plan()
    files=dict(read(c.PARENT_MANIFEST)['files'])
    for path in [*c.HERE.glob('*.py'),*(c.HERE/'tests').glob('*.py'),c.HERE/'DESIGN.md',c.HERE/'study_plan.json',c.LAUNCHER,c.PARENT_MANIFEST,c.PILOT]:
        files[str(path.relative_to(c.ROOT))]=sha(path)
    value=dict(files=files,source_hash=digest(files),publication_commit=c.PUBLICATION,core_hash=c.CORE)
    create(c.MANIFEST,value);return value


def validate_smoke(r):
    if (r.get('batch'),r.get('history_length'),r.get('kernel_length'),r.get('steps_per_mode'))!=(2048,50,56,3):raise ValueError('Smoke shape/steps')
    if [row.get('temporal_sharing') for row in r.get('rows',[])]!=list(c.MODES):raise ValueError('Smoke variants')
    for row in r['rows']:
        names=set(c.parameter_keys(row['temporal_sharing']))
        if row['status']!='PASS' or len(row['steps'])!=3:raise ValueError('Smoke incomplete')
        for i,step in enumerate(row['steps']):
            g=step['gradient_norms']
            if step['step']!=i or not step['finite_loss'] or not math.isfinite(step['loss']) or set(g)!=names or any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in g.values()):raise ValueError('Smoke loss/gradients')
        if any(not any(s['gradient_norms'][n]>0 for s in row['steps'][1:]) for n in names if n.startswith(('times.','layer1_times.'))):raise ValueError('Temporal learning not demonstrated')
        if not row['roundtrip_passed'] or row['roundtrip']!='weights_only=True':raise ValueError('Smoke roundtrip')
        if row['temporal_sharing']=='layer_specific' and not row.get('diverged_parameters'):raise ValueError('No layer parameter divergence')


_engine=bind(parent,dict(c=c,inherited=inherited,validate_smoke=validate_smoke),__package__)
verify,runtime,imported_sources,bindings,identity,validate_ownership,require_stage=(_engine[k] for k in
    ('verify','runtime','imported_sources','bindings','identity','validate_ownership','require_stage'))


def login_verify():
    import subprocess
    def git(*args):return subprocess.check_output(['git',*args],cwd=c.ROOT,text=True).strip()
    m=verify();commit=git('rev-parse','HEAD')
    if git('branch','--show-current')!=c.BRANCH or git('status','--porcelain','--untracked-files=no'):raise ValueError('Clean canonical branch required')
    if git('rev-parse','origin/'+c.BRANCH)!=commit:raise ValueError('Exact execution not published')
    subprocess.run(['git','merge-base','--is-ancestor',c.PUBLICATION,commit],cwd=c.ROOT,check=True)
    for path,expected in m['files'].items():
        if hashlib.sha256(subprocess.check_output(['git','show',commit+':'+path],cwd=c.ROOT)).hexdigest()!=expected:raise ValueError('Published source blob '+path)
    old=read(c.ROOT/'experiments/mamba3_mimo_time/source_manifest.json')
    for path,expected in old['files'].items():
        if hashlib.sha256(subprocess.check_output(['git','show',c.PILOT_COMMIT+':'+path],cwd=c.ROOT)).hexdigest()!=expected:raise ValueError('Historical implementation blob '+path)
    return dict(**bindings(commit,m),status='PASS',tracked_clean=True,published_commit=commit,source_blobs_verified=True,verified_at=now(),runtime=runtime(False))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
    result=freeze() if args.freeze else verify();print(json.dumps(dict(source_hash=result['source_hash'],files=len(result['files']))))
