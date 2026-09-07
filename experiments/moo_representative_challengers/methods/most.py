"""MosT transport / conditional MGDA core for five tasks and three solutions.

No preference argument: ranking preferences belong exclusively to evaluation.
Official one-iteration Frank-Wolfe behavior is deliberately retained for parity.
"""
import numpy as np


def ipot(a, b, cost, *, beta=.01, iterations=200):
    a, b, cost = map(lambda x: np.asarray(x, dtype=np.float64), (a, b, cost))
    kernel = np.exp(-cost / beta)
    plan = np.full(cost.shape, 1 / cost.size)
    v = np.ones(len(b))
    for _ in range(iterations):
        q = kernel * plan
        u = a / (q @ v)
        v = b / (q.T @ u)
        plan = u[:, None] * q * v[None, :]
    if not np.isfinite(plan).all():
        raise FloatingPointError("Non-finite IPOT transport; no fallback")
    return plan


def frank_wolfe_reference_step(gram):
    n = len(gram)
    a = np.full(n, 1 / n)
    if n == 1:
        return a
    vertex = int(np.argmin(a @ gram))
    aa, av, vv = a @ gram @ a, a @ gram[vertex], gram[vertex, vertex]
    gamma = 1. if aa <= av else (0. if vv <= av else (vv - av) / (aa + vv - 2 * av))
    a *= gamma
    a[vertex] += 1 - gamma
    return a


class MosT:
    def __init__(self, *, beta=.01, ipot_iterations=200, transport_iterations=20,
                 mask_threshold=1e-8, topk_model_ratio=.1, ema_decay=.95):
        self.beta, self.ipot_iterations = beta, ipot_iterations
        self.transport_iterations = transport_iterations
        self.mask_threshold, self.topk_model_ratio = mask_threshold, topk_model_ratio
        self.ema_decay, self.average_objective = ema_decay, 0.
        self.steps = 0

    def solve(self, losses, grams):
        f = np.asarray(losses, dtype=np.float64)  # tasks x solutions
        grams = np.asarray(grams, dtype=np.float64)  # solutions x tasks x tasks
        n, m = f.shape
        if grams.shape != (m, n, n) or not np.isfinite(f).all() or not np.isfinite(grams).all():
            raise ValueError("Invalid MosT losses/Grams")
        cost = np.maximum(f, 0)
        if cost.sum() <= 0:
            raise FloatingPointError("MosT zero total cost")
        cost /= cost.sum()
        a = np.full(n, m / n)
        top = np.sort(cost, axis=0)[:max(int(self.topk_model_ratio * n), 1)].sum(axis=0)
        if top.sum() <= 0:
            raise FloatingPointError("MosT zero marginal denominator")
        b = m * top / top.sum()  # official adjust_ab=1, set_obj_stepsize=1

        def transport(c):
            t = ipot(a, b, c, beta=self.beta, iterations=self.ipot_iterations)
            # Official torch float cast precedes gradient combinations.
            return (m * t / t.sum()).astype(np.float32).astype(np.float64)

        plan = transport(cost)
        objective = float((plan * f).sum())
        mgda = objective < self.average_objective
        coefficients = plan.copy()
        if mgda:
            for _ in range(self.transport_iterations):
                shifted = f.copy()
                for k in range(m):
                    active = np.flatnonzero(plan[:, k] > self.mask_threshold)
                    if not len(active):
                        raise FloatingPointError("Empty MosT assignment")
                    w = plan[active, k]
                    g = grams[k][np.ix_(active, active)] * np.outer(w, w)
                    coefficients[:, k] = 0
                    coefficients[active, k] = frank_wolfe_reference_step(g)
                    c = coefficients[:, k]
                    # Official return uses alpha @ raw gradients, without a second OT factor.
                    shifted[:, k] += np.max(grams[k] @ c) + .5 * np.sqrt(max(float(c @ grams[k] @ c), 0.))
                cost = np.maximum(shifted, 0)
                cost /= cost.sum()
                plan = transport(cost)
            objective = float((plan * f).sum())
        self.average_objective = (objective if self.average_objective == 0 else
                                  self.ema_decay * self.average_objective + (1 - self.ema_decay) * objective)
        self.steps += 1
        return coefficients, {"transport": plan.tolist(), "coefficients": coefficients.tolist(),
                              "set_objective": objective, "mgda_active": bool(mgda),
                              "row_marginals": a.tolist(), "column_marginals": b.tolist(),
                              "row_residual_max": float(np.max(np.abs(plan.sum(1) - a))),
                              "column_residual_max": float(np.max(np.abs(plan.sum(0) - b))),
                              "average_objective": self.average_objective, "steps": self.steps}
