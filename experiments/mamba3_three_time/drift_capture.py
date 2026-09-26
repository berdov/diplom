"""Scoped backward instrumentation; no mutation of upstream globals/autotuners."""

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import types
import threading
import copy

_capture = ContextVar("three_time_drift_capture", default=None)
OUTPUTS = ("dQ_mid", "dK_mid", "dV", "dADT", "dQK_dot", "dD", "d_input_state")
FIXED_LAUNCH = dict(num_warps=4, num_stages=2, maxnreg=128)


@contextmanager
def capturing(rows, events=None):
    token = _capture.set(Collector(rows, events))
    try:
        yield
    finally:
        _capture.reset(token)


class Collector:
    def __init__(self, rows, events):
        self.rows, self.events = rows, events
        self.lock = threading.Lock()
        self.forward_count = 0

    def bind(self):
        with self.lock:
            index = self.forward_count
            self.forward_count += 1
        return dict(collector=self, invocation_id=f"forward_{index}",
                    collector_identity=id(self), forward_thread=threading.get_ident(),
                    forward_calls=1, backward_calls=0, record=None)


def forward_handle():
    """Called only from forward; the handle is explicitly carried by autograd ctx."""
    collector = _capture.get()
    return collector.bind() if collector is not None else None


def snapshot_inputs(inputs):
    import torch
    snapshots, layouts = {}, {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            snapshots[key] = value
            continue
        original_stride = list(value.stride())
        try:
            copy = torch.empty_strided(value.shape, value.stride(), device=value.device, dtype=value.dtype)
            copy.copy_(value.detach())
        except RuntimeError:
            copy = value.detach().clone(memory_format=torch.contiguous_format)
        snapshots[key] = copy
        layouts[key] = dict(shape=list(value.shape), original_stride=original_stride,
            snapshot_stride=list(copy.stride()), original_storage_offset=value.storage_offset(),
            snapshot_storage_offset=copy.storage_offset(),
            storage_offset_preserved=copy.storage_offset() == value.storage_offset(),
            layout_preserved=list(copy.stride()) == original_stride,
            storage_independent=copy.untyped_storage().data_ptr() != value.untyped_storage().data_ptr())
    return snapshots, layouts


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
        self.autotuner = copy.copy(autotuner)
        if hasattr(autotuner, "cache"):
            self.autotuner.cache = dict(autotuner.cache)
        self.matched, self.records = matched, []

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


def dqkv_call(function, variant, *, invocation=None, **inputs):
    hybrid = variant == "diagnostic_hybrid"
    if invocation is None and not hybrid:
        return function(**inputs)
    saved, layouts = snapshot_inputs(inputs) if invocation is not None else ({}, {})
    outputs, launches = invoke(function, inputs)
    if hybrid:
        from .stable_adt import compute_dqkv
        stable, second = invoke(compute_dqkv, inputs)
        outputs = (*outputs[:3], stable[3], *outputs[4:])
        launches += second
    if invocation is not None:
        collector = invocation["collector"]
        invocation["backward_calls"] += 1
        record = dict(inputs=saved, input_layouts=layouts,
            outputs={k:v.detach().clone() for k,v in zip(OUTPUTS, outputs) if v is not None},
            launches=launches, backend=variant, hybrid_double_compute=hybrid, stages={},
            invocation_id=invocation["invocation_id"], collector_identity=invocation["collector_identity"],
            forward_thread=invocation["forward_thread"], backward_thread=threading.get_ident(),
            forward_calls=invocation["forward_calls"], backward_calls=invocation["backward_calls"])
        invocation["record"] = record
        with collector.lock:
            collector.rows.append(record)
            if collector.events is not None:
                collector.events.append((record["invocation_id"]+"/dqkv", record["outputs"]))
    return outputs


def capture_stages(*, invocation=None, **values):
    if invocation is not None:
        record = invocation["record"]
        if record is None:
            raise RuntimeError("Missing invocation-owned dqkv record")
        record["stages"] = {k:v.detach().clone() for k,v in values.items() if v is not None}
        collector = invocation["collector"]
        with collector.lock:
            if collector.events is not None:
                collector.events.append((record["invocation_id"]+"/rotary_write_phase", record["stages"]))
