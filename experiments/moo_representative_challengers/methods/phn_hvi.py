"""Independent, exact small-set hypervolume geometry in any objective dimension."""
import itertools
import numpy as np
import torch


def hypervolume(points, reference):
    points, reference = np.asarray(points), np.asarray(reference)
    total = 0.
    for size in range(1, len(points) + 1):
        for indices in itertools.combinations(range(len(points)), size):
            total += (-1.) ** (size + 1) * np.prod(np.maximum(reference - points[list(indices)].max(0), 0))
    return float(total)


def front_indices(points):
    remaining = list(range(len(points)))
    while remaining:
        front = [i for i in remaining if not any(
            np.all(points[j] <= points[i]) and np.any(points[j] < points[i]) for j in remaining)]
        yield front
        remaining = [i for i in remaining if i not in front]


def hv_weights(points, base_reference):
    """Return positive -dHV/dloss coefficients, unit norm per solution/front.

    Duplicate points get the same gradient, as in the reference. Partially tied
    coordinates are nondifferentiable: match the reference's zero-gradient rule.
    """
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or not np.isfinite(p).all():
        raise ValueError("Expected finite solutions x objectives")
    ref = np.maximum(np.asarray(base_reference), 1.1 * p.max(0))
    weights = np.zeros_like(p)
    ranks = np.zeros(len(p), dtype=int)
    for rank, indices in enumerate(front_indices(p)):
        unique, inverse = np.unique(p[indices], axis=0, return_inverse=True)
        g = np.zeros_like(unique)
        shared = np.any(unique[:, None, :] == unique[None, :, :], axis=2)
        np.fill_diagonal(shared, False)
        active = np.flatnonzero(~shared.any(1))
        for k in range(p.shape[1]):
            dims = np.arange(p.shape[1]) != k
            for i in active:
                # Exclusive projected rectangle against points better in coordinate k.
                before = [j for j in active if unique[j, k] < unique[i, k]]
                clipped = np.maximum(unique[before][:, dims], unique[i, dims])
                rectangle = np.prod(np.maximum(ref[dims] - unique[i, dims], 0))
                g[i, k] = max(float(rectangle - hypervolume(clipped, ref[dims])), 0.)
        norm = np.linalg.norm(g, axis=1, keepdims=True)
        norm[np.isclose(norm, 0)] = 1.
        weights[indices] = (g / norm)[inverse]
        ranks[indices] = rank
    return weights.astype(np.float32), {"dynamic_reference": ref.tolist(), "front_ranks": ranks.tolist(),
                                       "hypervolume": hypervolume(p, ref)}


def surrogate(losses, preferences, weights, cosine_weight=.001):
    w = torch.as_tensor(weights, device=losses.device, dtype=losses.dtype).detach()
    r = torch.as_tensor(preferences, device=losses.device, dtype=losses.dtype)
    return (w * losses).sum() - cosine_weight * torch.nn.functional.cosine_similarity(losses, r, dim=-1).sum()


def preference_samples(rng, count=8, alpha=.2, tasks=5):
    return rng.dirichlet(np.full(tasks, alpha), size=count).astype(np.float32)


def inverse_priority_rays(preferences):
    p = np.asarray(preferences, dtype=np.float64)
    if not np.isfinite(p).all() or np.any(p <= 0):
        raise ValueError('Strictly positive finite priority preferences required')
    rays = p.min(axis=-1, keepdims=True) / p
    return (rays / np.linalg.norm(rays, axis=-1, keepdims=True)).astype(np.float32)


def replay_backward(model, preferences, loss_fn, base_reference, cosine_weight=.001,
                    inverse_priority=False):
    """Two passes reproduce joint backward with one live backbone graph.

    The first pass determines interacting HV coefficients. Replay restores each
    forward's CPU/CUDA RNG; the outward RNG state equals eight ordinary forwards.
    TiM4Rec has no training-time running-statistic buffers.
    """
    devices = [next(model.parameters()).device.index] if next(model.parameters()).is_cuda else []
    states, values = [], []
    with torch.no_grad():
        for r in preferences:
            states.append((torch.get_rng_state(), [torch.cuda.get_rng_state(d) for d in devices]))
            model.set_preference(r)
            values.append(loss_fn().detach())
    points = torch.stack(values)
    weights, info = hv_weights(points.cpu().numpy(), base_reference)
    loss_rays = inverse_priority_rays(preferences) if inverse_priority else preferences
    total = 0.
    with torch.random.fork_rng(devices=devices):
        for i, r in enumerate(preferences):
            torch.set_rng_state(states[i][0])
            for d, state in zip(devices, states[i][1]):
                torch.cuda.set_rng_state(state, d)
            model.set_preference(r)
            losses = loss_fn()
            if not torch.allclose(losses.detach(), points[i], rtol=1e-5, atol=1e-6):
                raise RuntimeError("HVI replay changed objective values")
            scalar = surrogate(losses[None], loss_rays[i:i+1], weights[i:i+1], cosine_weight)
            scalar.backward()
            total += float(scalar.detach())
    return total, points.cpu().numpy(), info | {"weights": weights.tolist(), "preferences": preferences.tolist(),
                                              "cosine_loss_rays": loss_rays.tolist()}
