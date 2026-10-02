"""One inherited allocation pipeline; strict new child namespace and final hashes."""
from .. import pipeline as parent
from . import config as c
from .reuse import bind
from .process_env import child_environment
from .provenance import identity,inherited,runtime,require_stage
from experiments.mamba3_mimo_time.records import read,create,sha,now


def child(stage,module,args,deadline,directory):
    expected='experiments.mamba3_gap_trap.'+('runner' if stage in c.MODES else stage)
    if stage not in ('gate','smoke',*c.MODES) or module!=expected or args!=(['--variant',stage] if stage in c.MODES else []):
        raise ValueError('Unexpected child routing')
    target=module.replace('experiments.mamba3_gap_trap.','experiments.mamba3_gap_trap.centered.',1)
    return _child(stage,target,args,deadline,directory)


def run():
    code=_run()
    validated=read(c.PIPELINE)
    terminal=dict(status='PASS' if code==0 else 'INCOMPLETE',finished_at=now(),
        pipeline_sha256=sha(c.PIPELINE),scientific_fits_started=validated['scientific_fits_started'],
        scientific_fits_completed=validated['scientific_fits_completed'],TEST='NOT_RUN',test_evaluation_count=0,
        checkpoints={v:sha(c.paths(v)['checkpoint']) for v in c.MODES if c.paths(v)['checkpoint'].exists()},
        scope='In-allocation streaming hashes; sacct and process-log audit required after terminal Slurm state')
    create(c.RUNS/'terminal_metadata.json',terminal)
    return code


_engine=bind(parent,dict(c=c,child_environment=child_environment,identity=identity,inherited=inherited,
                        runtime=runtime,require_stage=require_stage),__package__)
_child,_run=(_engine[k] for k in ('child','run'))
_engine.update(child=child,run=run)
main=_engine['main']
if __name__=='__main__':raise SystemExit(main())
