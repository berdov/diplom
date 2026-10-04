"""Detached phase evidence from existing VALID forwards and fixed analytic grids.

No dataset/model replay, hooks on gradients, or RNG calls occur here. Native
increment diagnostics evaluate pi*tanh(raw)*DT at observed pre-kernel inputs;
they do not re-run angle_dt accumulation or the recurrence.
"""
import math

import torch

PERIODS_MS = (21600000, 86400000)
ABSOLUTE_HOURS = tuple(range(24))
RELATIVE_HOURS = (0, 1, 3, 6, 12, 24, 48)
CONTENT_DT_FIXTURES = ((-2.0, .05), (-.5, .25), (0.0, .5), (.5, 1.0), (2.0, 2.0))
SATURATION_ABS_DELTA = .99


class ScalarStats:
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.squared_total = 0.0
        self.minimum = self.maximum = None
        self.zeros = 0

    def update(self, value):
        value = value.detach().double().reshape(-1)
        if not value.numel():
            return
        if not torch.isfinite(value).all():
            raise ValueError('Nonfinite phase diagnostic')
        self.count += value.numel()
        self.total += value.sum().item()
        self.squared_total += value.square().sum().item()
        low, high = value.min().item(), value.max().item()
        self.minimum = low if self.minimum is None else min(self.minimum, low)
        self.maximum = high if self.maximum is None else max(self.maximum, high)
        self.zeros += int((value == 0).sum())

    def result(self):
        return dict(count=self.count, sum=self.total, squared_sum=self.squared_total,
                    mean=self.total / self.count if self.count else None,
                    rms=math.sqrt(self.squared_total / self.count) if self.count else None,
                    min=self.minimum, max=self.maximum,
                    abs_max=max(abs(self.minimum), abs(self.maximum)) if self.count else None,
                    zero_count=self.zeros, zero_fraction=self.zeros / self.count if self.count else None)


def _stats(value):
    stat = ScalarStats()
    stat.update(value)
    return stat.result()


def _features(hours):
    clock = torch.tensor(hours, dtype=torch.float64) * 3600000
    pieces = []
    for period in PERIODS_MS:
        theta = clock.remainder(period) * (2 * math.pi / period)
        pieces.extend((theta.sin(), theta.cos()))
    return torch.stack(pieces, dim=-1).div(math.sqrt(2)).float()


def _analytic_grid(weight, hours):
    phi = _features(hours)
    pair6, pair24 = phi[:, :2] @ weight[:, :2].T, phi[:, 2:] @ weight[:, 2:].T
    preactivation = phi @ weight.T
    delta = preactivation.tanh()
    fixtures = []
    for content, dt in CONTENT_DT_FIXTURES:
        before = torch.full_like(delta, content)
        corrected = before + delta
        base_inc = math.pi * before.tanh() * dt
        inc = math.pi * corrected.tanh() * dt
        bf_base = before.bfloat16().float()
        bf_corrected = corrected.bfloat16().float()
        bf_delta = bf_corrected - bf_base
        bf_difference = math.pi * (bf_corrected.tanh() - bf_base.tanh()) * dt
        fixtures.append(dict(content_raw_angle=content, dt_phase=dt,
                             sampled_angle_indices=list(range(min(4, weight.shape[0]))),
                             samples_scope='all grid clocks, first four angle coordinates; stats cover every angle',
                             native_increment_difference=(inc - base_inc)[:, :4].tolist(),
                             native_increment_difference_stats=_stats(inc - base_inc),
                             counterfactual_bf16_effective_raw_correction=bf_delta[:, :4].tolist(),
                             counterfactual_bf16_increment_difference=bf_difference[:, :4].tolist(),
                             counterfactual_bf16_increment_difference_stats=_stats(bf_difference),
                             counterfactual_bf16_rounding_increment_error_stats=_stats(bf_difference - (inc - base_inc)),
                             counterfactual_nonzero_fp32_corrections_lost_in_bf16=int(((delta != 0) & (bf_delta == 0)).sum())))
    return dict(hours=list(hours), clock_ms=[h * 3600000 for h in hours],
                features=phi.tolist(), feature_dtype='float32 after float64 remainder/sin/cos',
                frequency_6h_linear_contribution=pair6.tolist(),
                frequency_24h_linear_contribution=pair24.tolist(),
                frequency_6h_stats=_stats(pair6), frequency_24h_stats=_stats(pair24),
                linear_preactivation=preactivation.tolist(), raw_angle_correction=delta.tolist(),
                raw_angle_correction_stats=_stats(delta),
                abs_delta_near_saturation_count=int((delta.abs() >= SATURATION_ABS_DELTA).sum()),
                abs_delta_near_saturation_fraction=float((delta.abs() >= SATURATION_ABS_DELTA).double().mean()),
                content_dt_fixtures=fixtures)


@torch.no_grad()
def parameter_snapshot(phase_adapter, mode, train_coverage_summary=None):
    """Epoch-boundary W and last available gradient; no model forward.

    Both fixed grids are saved for every modified variant. They evaluate the
    periodic basis directly; every grid point is treated as an active observed
    event. In particular hour0 is not an artificial first-event special case.
    """
    if mode not in ('baseline_dual', 'relative_phase', 'absolute_phase'):
        raise ValueError('Unknown phase mode')
    if mode == 'baseline_dual':
        if phase_adapter is not None:
            raise ValueError('Baseline cannot own a phase adapter')
        return dict(status='NOT_APPLICABLE', phase_mode=mode, extra_parameters=0,
                    W='NOT_APPLICABLE', gradients='NOT_APPLICABLE',
                    train_coverage_summary=train_coverage_summary if train_coverage_summary is not None else 'NOT_RECORDED')
    if phase_adapter is None or not hasattr(phase_adapter, 'W'):
        raise ValueError('Modified model is missing W')
    source = phase_adapter.W
    if source.dtype != torch.float32 or source.ndim != 2 or source.shape[1] != 4:
        raise ValueError('Expected shared FP32 W[A,4]')
    weight = source.detach().cpu().clone()
    if not torch.isfinite(weight).all():
        raise ValueError('Nonfinite W at VALID boundary')
    gradient = source.grad
    gradient_info = dict(status='NOT_RECORDED', scope='last available TRAIN backward at VALID boundary; not all optimizer steps')
    if gradient is not None:
        grad = gradient.detach().cpu()
        finite = bool(torch.isfinite(grad).all())
        gradient_info.update(status='MEASURED', finite=finite,
                             l2_norm=grad.double().norm().item() if finite else None,
                             nonzero_count=int((grad != 0).sum()))
        if not finite:
            raise ValueError('Nonfinite W gradient at VALID boundary')
    return dict(status='MEASURED', phase_mode=mode, extra_parameters=weight.numel(),
                W=weight.tolist(), W_shape=list(weight.shape), W_dtype=str(weight.dtype),
                sharing='one W shared across both layers and both heads',
                W_l2_norm=weight.double().norm().item(), W_abs_max=weight.abs().max().item(),
                frequency_pair_l2_norms={'6h': weight[:, :2].double().norm().item(),
                                         '24h': weight[:, 2:].double().norm().item()},
                gradients=gradient_info,
                saturation_definition=f'abs(tanh(W phi)) >= {SATURATION_ABS_DELTA}; descriptive counter, not acceptance threshold',
                grids_scope='fixed analytic inputs only; no dataset or model evaluation; absolute grid is Unix day0 in UTC, not user local time',
                grid_active_policy='all grid entries active, including hour0; separate gate verifies first/padding neutrality',
                raw_angle_dtype='native MIMO FP32; BF16 raw-angle comparisons below are counterfactual only',
                frequency_contribution_definition='W_pair phi_pair before shared tanh; post-tanh contributions are not additive',
                absolute_24h_grid=_analytic_grid(weight, ABSOLUTE_HOURS),
                relative_grid=_analytic_grid(weight, RELATIVE_HOURS),
                train_coverage_summary=train_coverage_summary if train_coverage_summary is not None else 'NOT_RECORDED',
                interpretation='W norm or a nonzero frequency contribution does not establish importance, behavioural periodicity or personal local-time habits.')


class PhaseDiagnostics:
    """One collector per existing VALID epoch; last active query per history.

    Layer labels describe observed downstream inputs, not separate W matrices.
    Hypothetical BF16 raw-angle casting is compared on detached selected
    values. Neither angle_dt nor a recurrence/model is executed for diagnostics.
    """
    KEYS = ('raw_angle_correction', 'raw_angles_before', 'raw_angles_corrected',
            'native_increment_before_fp32', 'native_increment_after_fp32',
            'native_increment_difference_fp32', 'counterfactual_effective_raw_correction_bf16',
            'counterfactual_native_increment_difference_bf16', 'counterfactual_bf16_correction_rounding_error',
            'counterfactual_bf16_increment_rounding_error')

    def __init__(self, mode):
        if mode not in ('relative_phase', 'absolute_phase'):
            raise ValueError('Only modified variants have phase observations')
        self.mode = mode
        self.layers = {}

    @torch.no_grad()
    def observe(self, layer_index, payload):
        if type(layer_index) is not int or layer_index not in (0, 1):
            raise ValueError('Expected one of two layer indices')
        active = payload['phase_active'].detach()
        before = payload['raw_angles_before'].detach()
        corrected = payload['raw_angles_corrected'].detach()
        delta = payload['raw_angle_correction'].detach()
        dp = payload['dp'].detach()
        features = payload['phase_features'].detach()
        if active.ndim != 2 or active.dtype != torch.bool:
            raise ValueError('Expected phase active mask [B,L]')
        b, length = active.shape
        if before.ndim != 4 or before.shape[:2] != (b, length) or corrected.shape != before.shape:
            raise ValueError('Expected raw angles [B,L,H,A]')
        if delta.shape != (b, length, before.shape[-1]) or dp.shape != (b, before.shape[2], length):
            raise ValueError('Phase correction/DT layout mismatch')
        if features.shape != (b, length, 4):
            raise ValueError('Periodic feature layout mismatch')
        # Account for every input but retain only the last active observed event.
        positions = torch.arange(length, device=active.device)[None].expand_as(active)
        last = torch.where(active, positions, -1).amax(1)
        rows = torch.nonzero(last >= 0, as_tuple=False).flatten()
        positions = last[rows]
        base = before[rows, positions].float()
        after = corrected[rows, positions].float()
        correction = delta[rows, positions].float()
        dt = dp.permute(0, 2, 1)[rows, positions].float().unsqueeze(-1)
        chosen_features = features[rows, positions].float()
        if not torch.equal(after, base + correction[:, None, :]):
            raise ValueError('Observed phase correction does not reconstruct pre-boundary raw angles')
        base_inc = math.pi * base.tanh() * dt
        after_inc = math.pi * after.tanh() * dt
        cast_base, cast_after = base.bfloat16().float(), after.bfloat16().float()
        effective = cast_after - cast_base
        fp32_difference = after_inc - base_inc
        bf16_difference = math.pi * (cast_after.tanh() - cast_base.tanh()) * dt
        expanded_correction = correction[:, None, :].expand_as(base)
        values = dict(raw_angle_correction=correction, raw_angles_before=base,
                      raw_angles_corrected=after, native_increment_before_fp32=base_inc,
                      native_increment_after_fp32=after_inc, native_increment_difference_fp32=fp32_difference,
                      counterfactual_effective_raw_correction_bf16=effective,
                      counterfactual_native_increment_difference_bf16=bf16_difference,
                      counterfactual_bf16_correction_rounding_error=effective - expanded_correction,
                      counterfactual_bf16_increment_rounding_error=bf16_difference - fp32_difference)
        layer = self.layers.setdefault(layer_index, dict(
            calls=0, histories=0, histories_without_active_events=0, observed_active_positions=0,
            selected_active_queries=0, near_saturation=0, correction_elements=0,
            bf16_lost_nonzero=0, expanded_nonzero=0, stats={key: ScalarStats() for key in self.KEYS},
            examples=[]))
        for key, value in values.items():
            layer['stats'][key].update(value)
        layer['calls'] += 1
        layer['histories'] += b
        layer['histories_without_active_events'] += b - len(rows)
        layer['observed_active_positions'] += int(active.sum())
        layer['selected_active_queries'] += len(rows)
        layer['near_saturation'] += int((correction.abs() >= SATURATION_ABS_DELTA).sum())
        layer['correction_elements'] += correction.numel()
        layer['bf16_lost_nonzero'] += int(((expanded_correction != 0) & (effective == 0)).sum())
        layer['expanded_nonzero'] += int((expanded_correction != 0).sum())
        # Fixed first eight stream examples; no random sampling and no full H.
        for i in range(min(len(rows), 8 - len(layer['examples']))):
            layer['examples'].append(dict(query_position=int(positions[i]),
                                           features=chosen_features[i].cpu().tolist(),
                                           raw_angle_correction=correction[i].cpu().tolist(),
                                           raw_angles_before=base[i].cpu().tolist(),
                                           raw_angles_corrected=after[i].cpu().tolist(),
                                           dt_phase=dt[i, :, 0].cpu().tolist(),
                                           counterfactual_effective_raw_correction_bf16=effective[i].cpu().tolist(),
                                           counterfactual_native_increment_difference_bf16=bf16_difference[i].cpu().tolist()))

    def result(self):
        layers = {}
        for index, layer in sorted(self.layers.items()):
            count = layer['correction_elements']
            denominator = layer['expanded_nonzero']
            layers[str(index)] = dict(
                forward_calls=layer['calls'], histories=layer['histories'],
                histories_without_active_events=layer['histories_without_active_events'],
                observed_active_positions=layer['observed_active_positions'],
                selected_active_queries=layer['selected_active_queries'],
                stats={key: stat.result() for key, stat in layer['stats'].items()},
                abs_delta_near_saturation_count=layer['near_saturation'],
                abs_delta_near_saturation_fraction=layer['near_saturation'] / count if count else None,
                counterfactual_nonzero_fp32_corrections_lost_in_bf16=layer['bf16_lost_nonzero'],
                nonzero_expanded_corrections=denominator,
                counterfactual_bf16_lost_nonzero_fraction=layer['bf16_lost_nonzero'] / denominator if denominator else None,
                examples=layer['examples'])
        return dict(status='MEASURED' if layers else 'NOT_RECORDED', phase_mode=self.mode,
                    scope='existing VALID forwards only; last active observed event per input history, separately for each layer',
                    phase_statistics='instantaneous pre-accumulation increments; not accumulated theta or counterfactual model outputs',
                    bf16_scope='COUNTERFACTUAL raw-angle BF16 cast only; native MIMO angles stay FP32. Actual Q/K rotation casting is downstream and not measured by these counters',
                    raw_correction_scope='shared across heads; correction stats count each angle once, native increments count each head',
                    saturation_abs_delta=SATURATION_ABS_DELTA, layers=layers,
                    sampling='first eight active histories per layer; no randomness; exact aggregate counters cover all selected last queries',
                    training_rng_consumed=False, additional_dataset_forwards=0,
                    accumulated_theta='NOT_RECORDED', full_history_hidden_states='NOT_RECORDED',
                    interpretation='frequency contribution, correction size and saturation are diagnostics, not causal feature importance')
