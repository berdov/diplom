"""One positional-gradient capture, then hook-free forward interventions."""
import torch
from .numerics import compare, finite, residual
from experiments.mamba3_three_time.fixtures import scalar_loss
from experiments.mamba3_three_time.evidence import tensor_records


def check_prefix(net, data, prefix, multiplier, oracle, interventions, save):
    checks, captured = {}, []
    phase = dict(stages=[], capture_count=0, hook_removed=False)

    def persist(stage, **details):
        phase['capture_count'] = len(captured)
        phase['stages'].append(dict(stage=stage, capture_count=len(captured),
                                    grad_enabled=torch.is_grad_enabled(), **details))
        save(dict(checks), phase)

    expected_shape = (*data[0].shape, net.item_embedding.embedding_dim)

    def retain(_module, args, value):
        if not torch.is_grad_enabled() or not value.requires_grad:
            raise ValueError('Gradient phase requires differentiable embedding output')
        if args[0] is not data[0] or tuple(value.shape) != expected_shape or captured:
            raise ValueError('Expected exactly one positional embedding tensor')
        value.retain_grad()
        captured.append(value)

    hook = net.item_embedding.register_forward_hook(retain)
    try:
        persist('gradient_forward_started')
        if not torch.is_grad_enabled():
            raise ValueError('Gradient phase requires grad enabled')
        net.zero_grad(set_to_none=True)
        output = net.encode_sequence(*data, oracle=oracle)
        persist('gradient_forward_completed', output_requires_grad=output.requires_grad)
        if len(captured) != 1 or not output.requires_grad:
            raise ValueError('Missing/detached positional gradient path')
        (scalar_loss(output[0, :prefix]) * multiplier).backward()
        gradient = captured[0].grad
        if gradient is None or tuple(gradient.shape) != expected_shape or not bool(torch.isfinite(gradient).all()):
            raise ValueError('Missing/nonfinite/wrong-shape positional gradient')
        # Independent snapshots survive hook removal and subsequent no_grad forwards.
        gradient = gradient.detach().clone()
        baseline = output.detach().clone()
        phase['snapshots'] = tensor_records(dict(output=baseline, positional_gradient=gradient))
        persist('positional_gradient_received', gradient_finite=True)
        checks['residual'] = residual(gradient[:1], prefix)
        checks['cross_user_gradient'] = compare(gradient[1:], torch.zeros_like(gradient[1:]))
        checks['finite_output'] = finite(baseline)
        persist('gradient_checks_completed')
    finally:
        hook.remove()
        phase['hook_removed'] = True
        persist('hook_removed')

    with torch.no_grad():
        for index, changed in enumerate(interventions):
            persist('intervention_started', intervention=index)
            altered = net.encode_sequence(*changed, oracle=oracle)
            if altered.requires_grad or len(captured) != 1:
                raise ValueError('Interventions must not create/capture a gradient path')
            checks[f'prefix_intervention{index}'] = compare(altered[:, :prefix], baseline[:, :prefix])
            checks[f'cross_user_intervention{index}'] = compare(altered[1], baseline[1])
            persist('intervention_completed', intervention=index, output_requires_grad=altered.requires_grad)
    return checks
