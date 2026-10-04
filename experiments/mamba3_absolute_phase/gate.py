"""One frozen targeted admission, with durable fail-closed case callbacks."""
import traceback
from experiments.mamba3_mimo_time.records import (
    Registry, accepted_cases, create, update, now,
)
from experiments.mamba3_head_timescales.progress import progress_callback


def run_required_case(registry, spec, function):
    """Use the real Registry; a later callback/final value cannot erase a leaf."""
    def guarded(save):
        latest = {}
        persist = progress_callback(save, latest)
        persist({}, dict(stage='case_started'))
        def callback(value):
            persist(value.get('checks', {}), dict(stage='case_progress'))
            # Persist numeric fixtures as well as the immutable check history.
            evidence = {k: v for k, v in value.items()
                        if k not in ('checks', 'required_keys', 'progress_history')}
            if evidence:
                save(evidence)
        result = function(callback)
        persist(result.get('checks', {}), dict(stage='case_finished'))
        return result
    registry.run(spec, guarded)


def main():
    import torch
    from . import config as c
    from .checks import required_cases, dispatch
    from .provenance import identity, runtime, require_stage
    base = identity()
    require_stage(c.INHERITED, base)
    result = dict(**base, status='RUNNING', scientific_fits=0)
    create(c.GATE, result)
    specs = c.plan()['required_cases']
    registry = Registry(c.GATE, result, specs)
    try:
        if specs != required_cases(c.plan()['common_parameter_keys']):
            raise ValueError('Frozen case IDs/leaves differ from production gate')
        torch.backends.cuda.matmul.allow_tf32 = False
        result['runtime'] = runtime(True)
        update(c.GATE, result)
        for spec in specs:
            run_required_case(registry, spec, lambda save, spec=spec: dispatch(spec, save))
            torch.cuda.empty_cache()
        if not accepted_cases(result['cases'], specs):
            raise ValueError('Incomplete targeted gate')
        result['status'] = 'PASS'
    except BaseException as exc:
        result.update(status='FAIL', error=repr(exc), traceback=traceback.format_exc())
    result['finished_at'] = now()
    update(c.GATE, result)
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
