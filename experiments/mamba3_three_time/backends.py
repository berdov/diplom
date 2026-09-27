"""Explicit, process-local diagnostic selection; never patches upstream globals."""

from contextlib import contextmanager
from contextvars import ContextVar

VARIANTS = ("upstream", "stable_angle", "stable_adt", "stable_scan")
_variant = ContextVar("three_time_backend", default="upstream")
_trace = ContextVar("three_time_backward_trace", default=None)


def current():
    return _variant.get()


def trace_sink():
    return _trace.get()


@contextmanager
def selected(name, trace=None):
    if name not in (*VARIANTS, "diagnostic_hybrid"):
        raise ValueError("Unknown arithmetic variant: " + name)
    token, sink = _variant.set(name), _trace.set(trace)
    try:
        yield
    finally:
        _trace.reset(sink)
        _variant.reset(token)


def record_trace(sink, chunk, variant, **tensors):
    if sink is not None:
        sink.append(dict(chunk=chunk, backend=variant,
                         tensors={k:v.detach().clone() for k,v in tensors.items() if v is not None}))
