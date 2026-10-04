"""Three synthetic full-shape Adam steps per mode; no recommendation loader."""
import io
import traceback
import torch
from . import config as c
from .checks import fresh, histories, SYNTHETIC_SEED
from .provenance import identity, require_stage, runtime
from experiments.mamba3_mimo_time.records import create, update, now
from experiments.mamba3_mimo_time.provenance import assert_upstream


def main():
    base = identity()
    record = dict(**base, status='RUNNING', targeted_gate_sha256=require_stage(c.GATE, base),
                  scientific_fits=0, fixture_seed=SYNTHETIC_SEED,
                  batch=2048, history_length=50, kernel_length=56, steps_per_mode=3, rows=[])
    create(c.SMOKE, record)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        record['runtime'] = runtime(True)
        data = histories(50, batch=2048)
        target = torch.arange(2048, device='cuda') % 7111+1
        for mode in c.MODES:
            net = fresh(mode, 'cuda', seed=SYNTHETIC_SEED).train()
            assert_upstream(net)
            optimizer = torch.optim.Adam(net.parameters(), lr=.001)
            row = dict(phase_mode=mode, status='RUNNING', steps=[])
            record['rows'].append(row)
            update(c.SMOKE, record)
            torch.cuda.reset_peak_memory_stats()
            for step in range(3):
                optimizer.zero_grad(set_to_none=True)
                observed = []
                def observer(index, values):
                    correction = values['raw_angle_correction']
                    active = values['phase_active']
                    observed.append(dict(layer=index, correction_shape=list(correction.shape),
                        finite_correction=bool(torch.isfinite(correction).all()),
                        first_padding_neutral=bool((correction[~active] == 0).all()),
                        native_angle_dtype=str(values['angles'].dtype)))
                net.phase_observer = observer
                before = None if net.phase_adapter is None else net.phase_adapter.W.detach().clone()
                try:
                    sequence = net.encode_sequence(*data, observer=observer)
                    output = net.gather_indexes(sequence, data[1]-1)
                    scores = output @ net.item_embedding.weight.T
                    loss = torch.nn.functional.cross_entropy(scores, target)
                    loss.backward()
                finally:
                    net.phase_observer = None
                norms = {name: (p.grad.double().norm().item()
                         if p.grad is not None and bool(torch.isfinite(p.grad).all()) else None)
                         for name, p in net.named_parameters()}
                entry = dict(step=step, loss=loss.item() if torch.isfinite(loss) else None,
                             finite_loss=bool(torch.isfinite(loss)), gradient_norms=norms,
                             observer_removed=net.phase_observer is None, phase_layers=observed,
                             W_before_l2=None if before is None else before.double().norm().item())
                row['steps'].append(entry)
                update(c.SMOKE, record)
                if not torch.isfinite(loss) or any(value is None for value in norms.values()):
                    raise ValueError('Nonfinite/missing smoke gradient')
                if len(observed) != 2 or any(v['correction_shape'] != [2048, 50, 32] or
                    not v['finite_correction'] or not v['first_padding_neutral'] or
                    v['native_angle_dtype'] != 'torch.float32' for v in observed):
                    raise ValueError('Phase shape/mask/native dtype smoke failure')
                optimizer.step()
                if before is not None:
                    entry.update(W_after_l2=net.phase_adapter.W.detach().double().norm().item(),
                                 W_update_l2=(net.phase_adapter.W.detach()-before).double().norm().item())
                update(c.SMOKE, record)
            stream = io.BytesIO()
            torch.save(net.state_dict(), stream)
            stream.seek(0)
            saved = torch.load(stream, map_location='cpu', weights_only=True)
            row['roundtrip_passed'] = set(saved) == set(net.state_dict()) and all(
                torch.equal(value.cpu(), saved[key]) for key, value in net.state_dict().items())
            update(c.SMOKE, record)
            if not row['roundtrip_passed']:
                raise ValueError('State dict roundtrip failed')
            net.load_state_dict(saved, strict=True)
            row.update(status='PASS', peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(), roundtrip='weights_only=True')
            update(c.SMOKE, record)
            del net, optimizer, output, scores, loss, saved, stream, before
            torch.cuda.empty_cache()
        record['status'] = 'PASS'
    except BaseException as exc:
        record.update(status='BLOCKED_OOM' if isinstance(exc, torch.cuda.OutOfMemoryError) else 'FAIL',
                      error=repr(exc), traceback=traceback.format_exc())
    record['finished_at'] = now()
    update(c.SMOKE, record)
    return 0 if record['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
