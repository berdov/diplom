"""One targeted selector/reader/integration admission; mathematical kernels inherited."""
import traceback
import torch
from . import config as c
from .checks import parity, selector_reader, model_edges, causal, negative_controls, required_cases
from .provenance import identity, runtime, require_stage
from experiments.mamba3_mimo_time.records import create, update, Registry, accepted_cases, now


def main():
    base = identity(); require_stage(c.INHERITED,base)
    result = dict(**base,status='RUNNING',scientific_fits=0)
    create(c.GATE,result)
    specs = c.plan()['required_cases']; registry = Registry(c.GATE,result,specs)
    try:
        if specs != required_cases(c.plan()['common_parameter_keys']):
            raise ValueError('Frozen case IDs/leaves differ from production gate')
        torch.backends.cuda.matmul.allow_tf32 = False
        result['runtime'] = runtime(True); update(c.GATE,result)
        functions = [lambda save,mode=mode: parity(mode,save) for mode in c.MODES]
        functions += [selector_reader]
        functions += [lambda save,mode=mode: model_edges(mode,save) for mode in ('index_memory','time_memory')]
        functions += [lambda save,mode=mode,prefix=prefix: causal(mode,prefix,save)
                      for mode in ('index_memory','time_memory') for prefix in (8,9)]
        functions += [negative_controls]
        if len(functions) != len(specs): raise ValueError('Case count drift')
        for spec,function in zip(specs,functions):
            registry.run(spec,function); torch.cuda.empty_cache()
        if not accepted_cases(result['cases'],specs): raise ValueError('Incomplete targeted gate')
        result['status'] = 'PASS'
    except BaseException as exc:
        result.update(status='FAIL',error=repr(exc),traceback=traceback.format_exc())
    result['finished_at'] = now(); update(c.GATE,result)
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__': raise SystemExit(main())
