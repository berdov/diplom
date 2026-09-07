"""FERERO projected dual subproblem with a homogeneous loss-ray preference.

Independent implementation of the paper's equality-constrained subproblem;
tested against the pinned official NumPy solver, which accepts arbitrary B_h.
"""
import numpy as np


def simplex(x):
    ordered = np.sort(x)[::-1]
    offsets = (np.cumsum(ordered) - 1) / np.arange(1, len(x) + 1)
    active = np.flatnonzero(ordered > offsets)[-1]
    return np.maximum(x - offsets[active], 0)


def ray_matrix(preference):
    r = np.asarray(preference, dtype=np.float64)
    if r.ndim != 1 or len(r) < 2 or not np.all(np.isfinite(r) & (r > 0)):
        raise ValueError("Strictly positive finite preference required")
    v = 1 / r
    v /= np.linalg.norm(v)
    b = np.zeros((len(r) - 1, len(r)))
    b[:, 0] = v[1:]
    b[np.arange(len(r) - 1), np.arange(1, len(r))] = -v[0]
    return b


def solve(losses, gram, preference, *, step_size=0.1, max_iter=200, tolerance=1e-3,
          spectral_cap=0.9):
    f = np.asarray(losses, dtype=np.float64)
    q = np.asarray(gram, dtype=np.float64)
    if q.shape != (len(f), len(f)) or not np.isfinite(q).all() or not np.isfinite(f).all():
        raise ValueError("Invalid FERERO losses/Gram matrix")
    b = ray_matrix(preference)
    h = b @ f
    transform = np.vstack((np.eye(len(f)), b))
    lipschitz = max(float(np.linalg.eigvalsh(transform @ q @ transform.T)[-1]), 0.)
    gamma = min(step_size, spectral_cap / max(lipschitz, 1e-15))
    dual_f = np.full(len(f), 1 / len(f))
    dual_h = np.full(len(h), 1 / len(h))
    coeff = dual_f + b.T @ dual_h
    for iteration in range(max_iter):
        direction_dot = q @ coeff
        next_f = simplex(dual_f - gamma * direction_dot)
        next_h = dual_h - gamma * (b @ direction_dot - h)
        updated = next_f + b.T @ next_h
        change = np.abs(updated - coeff).sum()
        dual_f, dual_h, coeff = next_f, next_h, updated
        if change < tolerance:
            break
    if not np.isfinite(coeff).all():
        raise FloatingPointError("FERERO dual diverged; no scalarization fallback")
    return coeff, {"dual_f": dual_f.tolist(), "dual_h": dual_h.tolist(),
                   "constraint_state": h.tolist(), "B_h": b.tolist(),
                   "dual_objective": float(.5 * coeff @ q @ coeff - dual_h @ h),
                   "effective_step_size": gamma, "iterations": iteration + 1,
                   "coefficient_sum": float(coeff.sum()), "coefficients": coeff.tolist()}
