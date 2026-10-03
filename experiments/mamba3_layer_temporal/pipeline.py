"""One allocation and exactly two ordered fresh fits; stop on any failure."""
from experiments.mamba3_gap_trap import pipeline as parent
from experiments.mamba3_gap_trap.centered.reuse import bind
from experiments.mamba3_mimo_time.records import read,create,sha,now
from . import config as c
from .process_env import child_environment
from .provenance import identity,inherited,runtime,require_stage


def child(stage,module,args,deadline,directory):
    expected='experiments.mamba3_gap_trap.'+('runner' if stage in c.MODES else stage)
    if module!=expected:raise ValueError('Unexpected child module')
    return _child(stage,'experiments.mamba3_layer_temporal.'+('runner' if stage in c.MODES else stage),args,deadline,directory)


def run():
    code=_run();status=read(c.PIPELINE)
    checks={}
    for v in c.MODES:
        path=c.paths(v)['checkpoint']
        if path.exists():checks[v]=dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))
    terminal={k:v for k,v in status.items() if k not in ('stages','runtime')}
    terminal.update(pipeline_sha256=sha(c.PIPELINE),checkpoints=checks,checkpoint_loading=False,finished_at=now())
    create(c.RUNS/'terminal_metadata.json',terminal);return code


_engine=bind(parent,dict(c=c,child_environment=child_environment,identity=identity,inherited=inherited,runtime=runtime,require_stage=require_stage),__package__)
_child,_run=(_engine[k] for k in ('child','run'));_engine.update(child=child,run=run)
main=_engine['main']
if __name__=='__main__':raise SystemExit(main())
