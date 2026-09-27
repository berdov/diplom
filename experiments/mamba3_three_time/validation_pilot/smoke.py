"""Synthetic capacity/optimizer/state_dict check, isolated from scientific RNG."""
import json
import time
import traceback
import torch
from recbole.config import Config
from recbole.data.interaction import Interaction
from experiments.mamba3_three_time.config import SyntheticCatalog
from experiments.mamba3_three_time.model import ThreeTimeMamba3Rec
from experiments.mamba3_three_time.initialization import seed_all
from experiments.mamba3_three_time.evidence import compare
from experiments.mamba3_three_time.records_003 import required_pass
from .config import MODES, LOGS, SMOKE, settings
from .gate import histories
from .provenance import (identity, require_gate, create_record, atomic_json, runtime, backend_guard,
                         save_state_dict, tensor_hash, now)


def measure(mode):
    config = Config(model=ThreeTimeMamba3Rec, config_dict=settings(mode))
    seed_all(2026)
    model = ThreeTimeMamba3Rec(config, SyntheticCatalog()).cuda().train()
    guard = backend_guard(model)
    items, lengths, timestamps = histories(50, batch=2048)
    target = (items[:, -1] + 41) % 7111 + 1
    data = Interaction({model.ITEM_SEQ: items, model.ITEM_SEQ_LEN: lengths,
                        model.time_sequence_field: timestamps, model.POS_ITEM_ID: target})
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    initial = tensor_hash(model.state_dict())
    initial_calibrators = {k: tensor_hash(v.state_dict()) for k, v in model.times.calibrators.items()}
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        loss = model.calculate_loss(data)
        loss.backward()
        checks = dict(loss_finite=dict(passed=bool(torch.isfinite(loss))))
        for name, p in model.named_parameters():
            checks['gradient:' + name] = dict(passed=p.grad is not None and bool(torch.isfinite(p.grad).all()))
        for name, calibrator in model.times.calibrators.items():
            g = calibrator.last.weight.grad
            checks['calibrator_signal:' + name] = dict(passed=g is not None and bool(torch.isfinite(g).all()) and bool((g != 0).any()))
        if all(v['passed'] for v in checks.values()):
            clip = config['clip_grad_norm']
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), **clip)
            optimizer.step()
        checks['optimizer_update'] = dict(passed=tensor_hash(model.state_dict()) != initial)
        for name, calibrator in model.times.calibrators.items():
            checks['calibrator_update:' + name] = dict(passed=tensor_hash(calibrator.state_dict()) != initial_calibrators[name])
        torch.cuda.synchronize()
        row = dict(mode=mode, batch_size=2048, length=50, loss=float(loss.detach()) if torch.isfinite(loss) else None,
                   seconds=time.perf_counter()-start, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(), checks=checks)
        path = LOGS / 'smoke' / (mode + '_state_dict.pth')
        if path.exists():
            raise FileExistsError(path)
        row.update(checkpoint_path=str(path), checkpoint_sha256=save_state_dict(model, path))
        restored = ThreeTimeMamba3Rec(config, SyntheticCatalog()).cuda().eval()
        restored.load_state_dict(torch.load(path, map_location='cuda', weights_only=True), strict=True)
        restored_guard = backend_guard(restored)
        try:
            checks['state_dict_roundtrip'] = dict(passed=tensor_hash(restored.state_dict()) == tensor_hash(model.state_dict()))
            small = (items[:2], lengths[:2], timestamps[:2])
            model.eval()
            with torch.no_grad():
                checks['roundtrip_output'] = compare(model(*small), restored(*small), atol=0, rtol=0)
        finally:
            restored_guard.remove()
        row['passed'] = required_pass(checks)
        return row
    finally:
        guard.remove()


def main():
    result = dict(**identity(), admission_sha256=require_gate(), status='RUNNING', cases=[],
                  scientific_fits=0, started_at=now())
    create_record(SMOKE, result)
    try:
        result['runtime'] = runtime(require_cuda=True)
        for mode in MODES:
            row = measure(mode)
            result['cases'].append(row)
            atomic_json(SMOKE, result)
            if not row['passed']:
                raise ValueError('Synthetic smoke failed: ' + mode)
        result['status'] = 'PASS'
    except BaseException as exc:
        result.update(status='BLOCKED_OOM' if isinstance(exc, torch.cuda.OutOfMemoryError) else 'FAIL',
                      error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        result['finished_at'] = now()
        atomic_json(SMOKE, result)


if __name__ == '__main__':
    from experiments.mamba3_three_time.backends import selected
    with selected('upstream'):
        main()
