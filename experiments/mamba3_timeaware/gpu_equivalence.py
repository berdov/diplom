"""Opt-in synthetic official-kernel comparison. No datasets or training loop."""

import copy
import importlib.metadata
import json

import torch

from .time_mamba3 import time_mamba3_forward


PIN = 'e9594ce1c732d97440f0332fdc43170a2294dbfa'


def main():
    if not torch.cuda.is_available():
        raise SystemExit('NOT RUN: CUDA required; no CPU substitute')
    direct = json.loads(importlib.metadata.distribution('mamba-ssm').read_text('direct_url.json') or '{}')
    if direct.get('vcs_info', {}).get('commit_id') != PIN:
        raise SystemExit('FAIL: installed mamba-ssm does not confirm pinned commit')
    print(json.dumps(dict(pinned_commit=PIN, torch_version=torch.__version__,
                          gpu=torch.cuda.get_device_name(0))), flush=True)
    from mamba_ssm import Mamba3

    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = False
    passed = True
    for training in (False, True):
        for length in (50, 64):
            vanilla = Mamba3(d_model=64, d_state=128, expand=2, headdim=64,
                             ngroups=1, rope_fraction=0.5, chunk_size=64,
                             is_mimo=False, mimo_rank=4, is_outproj_norm=False,
                             device='cuda', dtype=torch.bfloat16).train(training)
            temporal = copy.deepcopy(vanilla)
            x = torch.randn(2, length, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
            xt = x.detach().clone().requires_grad_(True)
            expected = vanilla(x)
            actual = time_mamba3_forward(temporal, xt, torch.ones(2, length, 2, device='cuda'))
            expected.float().square().mean().backward()
            actual.float().square().mean().backward()
            for name, lhs, rhs in [('output', expected, actual), ('input_gradient', x.grad, xt.grad)]:
                error = (lhs.float() - rhs.float()).abs()
                # Identical operation order at scale=1; no broad bf16 tolerance.
                ok = torch.allclose(lhs, rhs, atol=1e-6, rtol=1e-5)
                passed &= ok
                print(json.dumps(dict(mode='train' if training else 'eval', length=length,
                                      tensor=name, max_abs_error=error.max().item(),
                                      mean_abs_error=error.mean().item(), atol=1e-6, rtol=1e-5,
                                      status='PASS' if ok else 'FAIL')))
    if not passed:
        raise SystemExit('FAIL: numerical equivalence; stop before experiments')
    print('PASS')


if __name__ == '__main__':
    main()
