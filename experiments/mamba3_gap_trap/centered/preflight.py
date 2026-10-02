"""Inherited CPU preflight, explicit centered imports; optional Git-denial audit."""
import argparse
import importlib
import json
from pathlib import Path
from .. import preflight as parent
from . import config as c
from .reuse import bind
from .provenance import verify,runtime,imported_sources

def stamp(record):
    record['execution_attempt']=c.EXECUTION_ATTEMPT
    return record


def create_evidence(path,record):
    return parent.create(path,stamp(record))


def update_evidence(path,record):
    return parent.update(path,stamp(record))


_engine=bind(parent,dict(c=c,verify=verify,runtime=runtime,imported_sources=imported_sources,
                        create=create_evidence,update=update_evidence),__package__)


def run(commit,evidence=None):
    for name in ('runner','pipeline','report','state','gate','smoke','submit'):
        importlib.import_module(__package__+'.'+name)
    return stamp(_engine['run'](commit,evidence))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execution-commit',required=True)
    p.add_argument('--evidence',type=Path);p.add_argument('--deny-git',action='store_true');args=p.parse_args()
    if args.deny_git:
        import subprocess
        original=subprocess.Popen
        class NoGitPopen(original):
            def __init__(self,command,*a,**kw):
                if isinstance(command,(list,tuple)) and Path(str(command[0])).name=='git':
                    raise RuntimeError('Git forbidden in compute/content verification')
                super().__init__(command,*a,**kw)
        subprocess.Popen=NoGitPopen
    result=run(args.execution_commit,args.evidence)
    print(json.dumps(result,indent=2));raise SystemExit(0 if result['status']=='PASS' else 1)
