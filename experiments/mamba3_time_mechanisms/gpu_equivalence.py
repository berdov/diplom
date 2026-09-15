"""Opt-in synthetic GPU evidence. No dataset, optimizer, or scientific evaluation."""

import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
import torch
from .provenance import PIN, fingerprint
from .time_mamba3 import mechanism_mamba3_forward


def compare(lhs, rhs):
    if lhs is None or rhs is None:
        return dict(passed=False, reason='missing gradient')
    delta = (lhs.float() - rhs.float()).abs()
    return dict(passed=bool(torch.allclose(lhs, rhs, atol=1e-6, rtol=1e-5)),
                max_abs_error=delta.max().item(), mean_abs_error=delta.mean().item())


def suite_a(training, length):
    from mamba_ssm import Mamba3
    ref = Mamba3(d_model=64,d_state=128,expand=2,headdim=64,ngroups=1,rope_fraction=.5,
                 chunk_size=64,is_mimo=False,mimo_rank=4,is_outproj_norm=False,
                 device='cuda',dtype=torch.bfloat16).train(training)
    new = copy.deepcopy(ref)
    x = torch.randn(2,length,64,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    xn = x.detach().clone().requires_grad_(True)
    one = torch.ones(2,length,2,device='cuda')
    y = ref(x)
    yn = mechanism_mamba3_forward(new,xn,one,one)
    y.float().square().mean().backward()
    yn.float().square().mean().backward()
    return dict(output=compare(y,yn), input_gradient=compare(x.grad,xn.grad))


class SyntheticCatalog:
    def num(self, field):
        return 7112 if field == 'item_id' else 23952


def evaluate_side(model, items, lengths, times, targets):
    captured = []
    def capture(module, args, output):
        output.retain_grad()
        captured.append(output)
    handle = model.item_embedding.register_forward_hook(capture)
    model.zero_grad(set_to_none=True)
    torch.manual_seed(2026)
    output = model(items,lengths,times)
    loss = model.loss_fct(output @ model.item_embedding.weight.T, targets)
    loss.backward()
    handle.remove()
    gradients = {n:p.grad.detach().clone() if p.grad is not None else None for n,p in model.named_parameters()}
    return output.detach(), loss.detach(), captured[0].grad, gradients


def suite_b(training, length):
    from recbole.config import Config
    from experiments.mamba3_timeaware.model import TimeAwareMamba3Rec
    from experiments.mamba3_baseline.model import Mamba3Rec
    from .model import MechanismMamba3Rec
    from .config import load_config
    config = Config(model=MechanismMamba3Rec, config_dict=load_config('shared'))
    ref = TimeAwareMamba3Rec(config,SyntheticCatalog()).cuda().train(training)
    # Nontrivial conditioning is essential: zero-init alone cannot detect mixed paths.
    with torch.no_grad():
        ref.time_calibrator.last.weight.normal_(0,.03)
        ref.time_calibrator.last.bias.fill_(.07)
    new = MechanismMamba3Rec(config,SyntheticCatalog()).cuda().train(training)
    mapped = {n.replace('time_calibrator.', 'mechanisms.calibrators.shared.'):v for n,v in ref.state_dict().items()}
    new.load_state_dict(mapped, strict=True)
    items = torch.randint(1,7112,(2,length),device='cuda')
    lengths = torch.tensor([length,length-7],device='cuda')
    items[1,length-7:] = 0
    times = torch.randint(1,2000000,(2,length),device='cuda').double().cumsum(1) + 1.6e12
    targets = torch.tensor([3,7],device='cuda')
    y,loss,g,grads = evaluate_side(ref,items,lengths,times,targets)
    yn,lossn,gn,gradsn = evaluate_side(new,items,lengths,times,targets)
    metrics = dict(output=compare(y,yn), loss=compare(loss,lossn), input_gradient=compare(g,gn))
    for name,value in grads.items():
        metrics['gradient:'+name] = compare(value,gradsn[name.replace('time_calibrator.','mechanisms.calibrators.shared.')])
    # All five complete wrappers must equal the same backbone at zero initialization.
    state = {n:v for n,v in ref.state_dict().items() if not n.startswith('time_calibrator.')}
    for mode in ('vanilla','decay_only','scan_only','shared','separate'):
        cfg = Config(model=MechanismMamba3Rec,config_dict=load_config(mode))
        initial = MechanismMamba3Rec(cfg,SyntheticCatalog()).cuda().train(training)
        initial.load_state_dict({**initial.state_dict(),**state}, strict=True)
        torch.manual_seed(2026)
        expected = Mamba3Rec.forward(initial,items,lengths)
        torch.manual_seed(2026)
        actual = initial(items,lengths,times)
        metrics['zero_init:'+mode] = compare(expected,actual)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print(json.dumps(dict(status='SKIP', reason='CUDA unavailable; GPU equivalence NOT RUN')))
        raise SystemExit(2)
    if args.output.exists():
        raise FileExistsError('Evidence exists; no overwrite')
    direct = json.loads(importlib.metadata.distribution('mamba-ssm').read_text('direct_url.json'))
    if direct['vcs_info']['commit_id'] != PIN:
        raise RuntimeError('Pinned Mamba mismatch')
    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = False
    result = dict(status='FAIL',pinned_commit=PIN,source_fingerprint=fingerprint(),
                  gpu=torch.cuda.get_device_name(),atol=1e-6,rtol=1e-5,cases=[])
    try:
        for suite, function in (('A',suite_a),('B',suite_b)):
            for training in (False,True):
                for length in (50,64):
                    metrics = function(training,length)
                    result['cases'].append(dict(suite=suite,training=training,length=length,
                        metrics=metrics,passed=all(v['passed'] for v in metrics.values())))
        if all(c['passed'] for c in result['cases']):
            result['status'] = 'PASS'
    except Exception as exc:
        result['error'] = repr(exc)
        raise
    finally:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as handle:
            json.dump(result,handle,indent=2,allow_nan=False)
        print(json.dumps(result,allow_nan=False))
    if result['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
