"""Constant-gap intervention before the existing historical-input model path."""
import torch


def constant_timestamps(items, lengths, reference=838393.0):
    positions = torch.arange(items.shape[1], device=items.device)
    valid = items != 0
    if (lengths < 1).any() or not torch.equal(valid, positions[None, :] < lengths[:, None]):
        raise ValueError('Expected nonempty right-padded histories')
    times = positions.to(torch.float64) * reference
    return torch.where(valid, times[None, :], 0.)
