"""Scoped backward instrumentation; no mutation of upstream globals/autotuners."""

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import types

_capture = ContextVar("three_time_drift_capture", default=None)
_events = ContextVar("three_time_drift_events", default=None)
OUTPUTS = ("dQ_mid", "dK_mid", "dV", "dADT", "dQK_dot", "dD", "d_input_state")
FIXED_LAUNCH = dict(num_warps=4, num_stages=2, maxnreg=128)


@contextmanager
def capturing(rows, events=None):
    token = _capture.set(rows)
    event_token = _events.set(events)
    try:
        yield
    finally:
        _capture.reset(token)
        _events.reset(event_token)


def scalar(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): scalar(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scalar(v) for v in value]
    return str(value)


class LaunchRecorder:
    def __init__(self, autotuner, *, matched=False):
        self.autotuner, self.matched, self.records = autotuner, matched, []

    def __getitem__(self, grid):
        def run(*args, **kwargs):
            import triton
            target = triton.runtime.driver.active.get_current_target()
            if self.matched:
                configs = self.autotuner.configs
                if not any(all(getattr(c, k, None) == v for k, v in FIXED_LAUNCH.items()) for c in configs):
                    raise RuntimeError("Pre-frozen matched config is not in the supported config list")
                compiled = self.autotuner.fn[grid](*args, **kwargs, **FIXED_LAUNCH)
                config = dict(FIXED_LAUNCH)
            else:
                compiled = self.autotuner[grid](*args, **kwargs)
                best = getattr(self.autotuner, "best_config", None)
                config = {k: getattr(best, k, None) for k in FIXED_LAUNCH}
                config["kwargs"] = getattr(best, "kwargs", None)
            jit = self.autotuner.fn
            meta = getattr(compiled, "metadata", None)
            self.records.append(dict(config=scalar(config), target=str(target),
                compile_flags=scalar(kwargs), grid=scalar(grid), matched=self.matched,
                jit_name=getattr(jit, "__name__", None), jit_cache_key=str(getattr(jit, "cache_key", None)),
                jit_source_sha256=hashlib.sha256(jit.src.encode()).hexdigest(),
                compiled_name=getattr(compiled, "name", None), compiled_hash=getattr(compiled, "hash", None),
                metadata=scalar(meta._asdict() if hasattr(meta, "_asdict") else meta),
                compiled_metadata_available=meta is not None))
            return compiled
        return run


def invoke(function, inputs, *, matched=False):
    name = "mamba3_siso_bwd_kernel_dqkv"
    proxy = LaunchRecorder(function.__globals__[name], matched=matched)
    namespace = dict(function.__globals__, **{name: proxy})
    local = types.FunctionType(function.__code__, namespace, function.__name__, function.__defaults__)
    local.__kwdefaults__ = function.__kwdefaults__
    outputs = local(**inputs)
    return outputs, proxy.records


def dqkv_call(function, variant, **inputs):
    sink = _capture.get()
    hybrid = variant == "diagnostic_hybrid"
    if sink is None and not hybrid:
        return function(**inputs)
    outputs, launches = invoke(function, inputs)
    if hybrid:
        from .stable_adt import compute_dqkv
        stable, second = invoke(compute_dqkv, inputs)
        outputs = (*outputs[:3], stable[3], *outputs[4:])
        launches += second
    if sink is not None:
        # The wrappers do not mutate these inputs. Keep exact storage/strides for replay.
        sink.append(dict(inputs={k:v.detach() if hasattr(v, "detach") else v for k,v in inputs.items()},
            outputs={k:v.detach().clone() for k,v in zip(OUTPUTS, outputs) if v is not None},
            launches=launches, backend=variant, hybrid_double_compute=hybrid, stages={}))
        if _events.get() is not None:
            _events.get().append((f"reverse_layer{len(sink)-1}/dqkv", sink[-1]["outputs"]))
    return outputs


def capture_stages(**values):
    sink = _capture.get()
    if sink is not None:
        sink[-1]["stages"] = {k:v.detach().clone() for k,v in values.items() if v is not None}
        if _events.get() is not None:
            _events.get().append((f"reverse_layer{len(sink)-1}/rotary_write_phase",sink[-1]["stages"]))
