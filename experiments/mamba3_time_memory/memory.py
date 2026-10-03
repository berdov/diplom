"""Causal addresses and a projection-free reader over the current encoder output.

The address choice is discrete metadata processing. Values and queries remain
in the scientific autograd graph; no representation survives this call.
"""
import math

import torch

from experiments.mamba3_timeaware.time_inputs import history_gaps

REFERENCE_MS = 838393
ANCHORS = (1, 4, 16, 32)
K = len(ANCHORS)
MEMORY_MODES = ('index_memory', 'time_memory')


def select(timestamps, valid, mode):
    """Select at most four unique j<t, with latest-index exact-distance ties.

``indices/mask/ages/event_lags`` are sorted by event position for the reader.
``anchor_*`` retain greedy anchor order. Out-of-range refers to the remaining
candidate age range at that anchor's selection step, not the original range.
Both reported age arrays always contain physical elapsed time divided by R0,
including in index mode; ``address_ages`` use that mode's addressing clock.
"""
    if mode not in MEMORY_MODES:
        raise ValueError('Expected index_memory or time_memory')
    if timestamps.ndim != 2 or timestamps.shape != valid.shape:
        raise ValueError('Expected timestamps and valid [B,L]')
    if valid.dtype != torch.bool or valid.device != timestamps.device:
        raise ValueError('Expected boolean valid on timestamp device')
    if not timestamps.is_floating_point() or timestamps.shape[1] < 1:
        raise ValueError('Expected floating timestamps and a nonempty sequence axis')
    # No absolute timestamp is rounded to fp32. The inherited function also
    # explicitly rejects nonfinite valid values and adjacent-gap overflow.
    with torch.no_grad():
        gaps, _ = history_gaps(timestamps, valid)
        clock = gaps.cumsum(dim=1)
        if not torch.isfinite(clock).all():
            raise ValueError('Cumulative history clock overflow')
        batch, length = valid.shape
        positions = torch.arange(length, device=valid.device)
        lags = positions[:, None] - positions[None, :]
        physical_ages = (clock[:, :, None] - clock[:, None, :]).clamp_min(0) / REFERENCE_MS
        ages = (lags.clamp_min(0).double()[None].expand(batch, -1, -1)
                if mode == 'index_memory' else physical_ages)
        candidates = (valid[:, :, None] & valid[:, None, :]
                      & (lags[None] > 0))
        available = candidates.clone()
        log_ages = torch.log1p(ages)
        chosen, present, outside = [], [], []
        for anchor in ANCHORS:
            exists = available.any(dim=-1)
            low = ages.masked_fill(~available, float('inf')).amin(dim=-1)
            high = ages.masked_fill(~available, -float('inf')).amax(dim=-1)
            outside.append(exists & ((anchor < low) | (anchor > high)))
            distance = (log_ages - math.log1p(anchor)).abs().masked_fill(~available, float('inf'))
            minimum = distance.amin(dim=-1, keepdim=True)
            ties = available & (distance == minimum)
            # Explicit lexicographic rule: distance first, latest index second.
            index = torch.where(ties, positions[None, None, :], -1).amax(dim=-1)
            chosen.append(index)
            present.append(exists)
            available = available & (positions[None, None, :] != index[:, :, None])
        anchor_indices = torch.stack(chosen, dim=-1)
        anchor_mask = torch.stack(present, dim=-1)
        anchor_outside = torch.stack(outside, dim=-1)
        safe_anchor = anchor_indices.clamp_min(0)
        anchor_ages = torch.where(anchor_mask, physical_ages.gather(2, safe_anchor), 0.)
        anchor_address_ages = torch.where(anchor_mask, ages.gather(2, safe_anchor), 0.)
        anchor_lags = torch.where(anchor_mask, positions[None, :, None] - safe_anchor, 0)
        # Invalid slots follow all valid slots; -1 is never used as a gather index.
        order = anchor_indices.masked_fill(~anchor_mask, length).argsort(dim=-1, stable=True)
        indices = anchor_indices.gather(2, order)
        mask = anchor_mask.gather(2, order)
        return dict(indices=indices, mask=mask,
                    ages=anchor_ages.gather(2, order),
                    address_ages=anchor_address_ages.gather(2, order),
                    event_lags=anchor_lags.gather(2, order),
                    anchor_indices=anchor_indices, anchor_mask=anchor_mask,
                    anchor_ages=anchor_ages, anchor_address_ages=anchor_address_ages,
                    anchor_event_lags=anchor_lags, anchor_out_of_range=anchor_outside,
                    query_valid=valid, clock_ms=clock)


def _read_memory(hidden, selection, beta, compute_dtype):
    if hidden.ndim != 3 or not hidden.is_floating_point() or hidden.shape[-1] < 1:
        raise ValueError('Expected floating H [B,L,D]')
    indices, mask = selection['indices'], selection['mask']
    if indices.shape != (*hidden.shape[:2], K) or mask.shape != indices.shape:
        raise ValueError('Expected four indices/masks per query')
    if indices.dtype != torch.long or mask.dtype != torch.bool:
        raise ValueError('Expected int64 indices and boolean slot mask')
    if indices.device != hidden.device or mask.device != hidden.device:
        raise ValueError('Reader inputs must be on the same device')
    if beta.ndim != 0 or not beta.is_floating_point() or beta.device != hidden.device:
        raise ValueError('Expected one floating scalar beta on H device')
    if not torch.isfinite(hidden).all() or not torch.isfinite(beta):
        raise ValueError('Reader inputs must be finite')
    length = hidden.shape[1]
    query_pos = torch.arange(length, device=hidden.device)[None, :, None]
    if ((indices[mask] < 0).any() or (indices[mask] >= length).any()
            or ((indices >= query_pos) & mask).any()
            or (indices[~mask] != -1).any()):
        raise ValueError('Invalid or noncausal reader address')
    query_valid = selection.get('query_valid')
    if query_valid is not None:
        if query_valid.shape != hidden.shape[:2] or query_valid.dtype != torch.bool:
            raise ValueError('Expected boolean query_valid [B,L]')
        selected_valid = query_valid.gather(1, indices.clamp_min(0).reshape(hidden.shape[0], -1)).reshape_as(mask)
        if (mask & (~query_valid[:, :, None] | ~selected_valid)).any():
            raise ValueError('Padding cannot query or supply memory')
    safe = indices.clamp_min(0)
    batch_index = torch.arange(hidden.shape[0], device=hidden.device)[:, None, None]
    # Gather only K states: never construct [B,L,L,D]. Original H is the bank.
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        query = hidden.to(compute_dtype)
        values = hidden[batch_index, safe].to(compute_dtype)
        values = torch.where(mask[..., None], values, torch.zeros_like(values))
        scores = (query[:, :, None, :] * values).sum(dim=-1) / math.sqrt(hidden.shape[-1])
        if not torch.isfinite(scores).all():
            raise ValueError('Reader score overflow')
        any_slot = mask.any(dim=-1, keepdim=True)
        masked_scores = scores.masked_fill(~mask, -float('inf'))
        # Replace all-empty rows BEFORE softmax; no hidden NaN backward branch.
        safe_scores = torch.where(any_slot, masked_scores, torch.zeros_like(masked_scores))
        weights = torch.where(mask, torch.softmax(safe_scores, dim=-1), torch.zeros_like(scores))
        readout = (weights[..., None] * values).sum(dim=-2)
        strength = torch.tanh(beta.to(compute_dtype))
        output = hidden + (strength * readout).to(hidden.dtype)
    return output, dict(weights=weights, readout=readout, values=values,
                        scores=scores, memory_shape=list(values.shape))


def read_memory(hidden, selection, beta):
    """Production reader: fp32 scores/readout, signed scalar residual gate."""
    if beta.dtype != torch.float32:
        raise ValueError('Production beta must be fp32')
    return _read_memory(hidden, selection, beta, torch.float32)


def read_memory_reference(hidden, selection, beta):
    """Same fixed-index algebra in fp64, only for reader derivative checks."""
    if hidden.dtype != torch.float64 or beta.dtype != torch.float64:
        raise ValueError('Reference reader requires fp64 H and beta')
    return _read_memory(hidden, selection, beta, torch.float64)
