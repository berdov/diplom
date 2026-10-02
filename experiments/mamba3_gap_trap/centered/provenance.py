"""Centered source closure plus unchanged one-sided/kernel evidence lineage."""
import hashlib
import subprocess
from .. import provenance as parent
from .. import config as old
from . import config as c
from .reuse import bind
from experiments.mamba3_mimo_time.records import read,sha,digest,create,accepted_cases


def inherited():
    previous = parent.inherited()
    manifest = parent.verify()
    spec = old.plan()['required_cases']
    gate,smoke = read(old.GATE),read(old.SMOKE)
    expected = dict(status='PASS',execution_commit=c.PARENT_EXECUTION,source_hash=manifest['source_hash'],job_id='4370162',TEST='NOT_RUN',test_evaluation_count=0)
    for record in (gate,smoke):
        if any(record.get(k)!=v for k,v in expected.items()):
            raise ValueError('One-sided inherited evidence binding')
    if not accepted_cases(gate['cases'],spec):raise ValueError('One-sided gate incomplete')
    parent.validate_smoke(smoke)
    if smoke['targeted_gate_sha256']!=sha(old.GATE):raise ValueError('One-sided smoke binding')
    return dict(status='INHERITED_PASS',previous=previous,one_sided=dict(
        job_id='4370162',execution_commit=c.PARENT_EXECUTION,source_hash=manifest['source_hash'],
        manifest_sha256=sha(old.MANIFEST),gate_sha256=sha(old.GATE),smoke_sha256=sha(old.SMOKE),
        cases=len(spec),checks=sum(len(x['required_keys']) for x in spec),repeated_gpu_checks=0))


def previous_attempt():
    folder=c.HERE/'evidence/job4371302'
    saved=read(folder/'preservation_manifest.json')
    audit=read(folder/'failure_audit.json')
    scheduler=read(folder/'scheduler_terminal.json')
    if (saved['checkout_commit']!='4347ecbd3bfa98b075dbf0dd8f227bfff3590e50'
        or scheduler['job_id']!='4371302' or scheduler['state']!='FAILED' or scheduler['exit_code']!='1:0'
        or audit['audit_status']!='PASS' or audit['scientific_fits_started']!=0
        or audit['scientific_fits_completed']!=0 or audit['test_evaluation_count']!=0 or saved['checkpoints']):
        raise ValueError('Previous attempt is not the audited zero-fit failure')
    for row in saved['files']:
        path=folder/'files'/row['path']
        if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:
            raise ValueError('Previous attempt evidence changed')
    return dict(job_id='4371302',execution_commit=saved['checkout_commit'],scientific_fits_started=0,
                failure_audit_sha256=sha(folder/'failure_audit.json'))


def freeze():
    c.unused();inherited();previous_attempt()
    files = dict(read(old.MANIFEST)['files'])
    sources = [*c.HERE.glob('*.py'),*(c.HERE/'tests').glob('*.py'),c.HERE/'DESIGN.md',
               c.PLAN,c.HERE/'study_plan.json',c.HERE/'source_manifest.json',c.HERE/'reuse_bindings.json',c.LAUNCHER,old.MANIFEST,old.GATE,old.SMOKE]
    sources += [old.paths(v)['result'] for v in old.MODES]
    folder=c.HERE/'evidence/job4371302'
    sources += [folder/name for name in ('preservation_manifest.json','failure_audit.json','scheduler_terminal.json')]
    sources += [folder/'files'/row['path'] for row in read(folder/'preservation_manifest.json')['files']]
    for path in sources:files[str(path.relative_to(c.ROOT))]=sha(path)
    result = dict(files=files,source_hash=digest(files),publication_commit=c.PUBLICATION,core_hash=c.CORE,
                  parent_execution=c.PARENT_EXECUTION,parent_manifest_sha256=sha(old.MANIFEST),execution_attempt=c.EXECUTION_ATTEMPT,previous_attempt=previous_attempt())
    create(c.MANIFEST,result)
    return result


_engine = bind(parent, {'c':c,'inherited':inherited}, __package__)
_base_bindings=_engine['bindings']
_base_verify=_engine['verify']


def bindings(commit,manifest):
    value=_base_bindings(commit,manifest)
    value.update(execution_attempt=c.EXECUTION_ATTEMPT,plan_sha256=sha(c.PLAN),
                 previous_job_id='4371302',previous_failure_audit_sha256=sha(c.HERE/'evidence/job4371302/failure_audit.json'))
    return value


def verify():
    manifest=_base_verify()
    if manifest.get('execution_attempt')!=c.EXECUTION_ATTEMPT or manifest.get('previous_attempt')!=previous_attempt():
        raise ValueError('Attempt002 provenance mismatch')
    return manifest


_engine.update(bindings=bindings,verify=verify)
imported_sources,runtime,identity,require_stage,validate_smoke,validate_ownership = (
    _engine[k] for k in ('imported_sources','runtime','identity','require_stage','validate_smoke','validate_ownership'))


def login_verify():
    result = _engine['login_verify']()
    for name,expected in read(old.MANIFEST)['files'].items():
        actual = hashlib.sha256(subprocess.check_output(['git','show',c.PARENT_EXECUTION+':'+name],cwd=c.ROOT)).hexdigest()
        if actual!=expected:raise ValueError('One-sided execution blob mismatch: '+name)
    for name,expected in read(c.HERE/'source_manifest.json')['files'].items():
        actual=hashlib.sha256(subprocess.check_output(['git','show','4347ecbd3bfa98b075dbf0dd8f227bfff3590e50:'+name],cwd=c.ROOT)).hexdigest()
        if actual!=expected:raise ValueError('Attempt001 execution blob mismatch: '+name)
    result['one_sided_execution_blobs_verified'] = True
    result['previous_attempt_execution_blobs_verified'] = True
    return result


if __name__ == '__main__':
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');args=p.parse_args()
    value=freeze() if args.freeze else verify()
    print(json.dumps(dict(source_hash=value['source_hash'],files=len(value['files']))))
