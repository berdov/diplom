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


def freeze():
    c.unused();inherited()
    files = dict(read(old.MANIFEST)['files'])
    sources = [*c.HERE.glob('*.py'),*(c.HERE/'tests').glob('*.py'),c.HERE/'DESIGN.md',
               c.HERE/'study_plan.json',c.HERE/'reuse_bindings.json',c.LAUNCHER,old.MANIFEST,old.GATE,old.SMOKE]
    sources += [old.paths(v)['result'] for v in old.MODES]
    for path in sources:files[str(path.relative_to(c.ROOT))]=sha(path)
    result = dict(files=files,source_hash=digest(files),publication_commit=c.PUBLICATION,core_hash=c.CORE,
                  parent_execution=c.PARENT_EXECUTION,parent_manifest_sha256=sha(old.MANIFEST))
    create(c.MANIFEST,result)
    return result


_engine = bind(parent, {'c':c,'inherited':inherited}, __package__)
verify, imported_sources, runtime, bindings, identity, require_stage, validate_smoke, validate_ownership = (
    _engine[k] for k in ('verify','imported_sources','runtime','bindings','identity','require_stage','validate_smoke','validate_ownership'))


def login_verify():
    result = _engine['login_verify']()
    for name,expected in read(old.MANIFEST)['files'].items():
        actual = hashlib.sha256(subprocess.check_output(['git','show',c.PARENT_EXECUTION+':'+name],cwd=c.ROOT)).hexdigest()
        if actual!=expected:raise ValueError('One-sided execution blob mismatch: '+name)
    result['one_sided_execution_blobs_verified'] = True
    return result


if __name__ == '__main__':
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');args=p.parse_args()
    value=freeze() if args.freeze else verify()
    print(json.dumps(dict(source_hash=value['source_hash'],files=len(value['files']))))
