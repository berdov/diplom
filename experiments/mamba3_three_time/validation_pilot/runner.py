"""One scratch SISO TRAIN->VALID fit in a fresh process, no TEST loader."""
import argparse
import hashlib
import json
import os
import signal
import traceback
import torch
from .config import MODES, COUNTS, paths, settings, plan, SMOKE
from .provenance import (identity, require_gate, create_record, atomic_json, sha, now, runtime,
                         effective_check, backbone_hash, tensor_hash, rng_hash, backend_guard, assert_upstream)


def train(mode, record, p):
    from recbole.config import Config
    from recbole.utils import init_seed, init_logger
    from recbole.data.dataloader import TrainDataLoader, FullSortEvalDataLoader
    from recbole.trainer import Trainer
    from experiments.mamba3_timeaware.dataset import PreciseHistoryDataset
    from experiments.mamba3_timeaware.run import verify_protocol, verify_history_stats
    from experiments.mamba3_time_confirmation.config import STATS, STATS_SHA, MANIFEST_SHA
    from experiments.mamba3_three_time.model import ThreeTimeMamba3Rec
    from .trainer import trainer_class

    record['runtime'] = runtime(require_cuda=True)
    values = settings(mode)
    config = Config(model=ThreeTimeMamba3Rec, config_dict=values)
    record['effective_config_parity'] = effective_check(config, mode)
    record.update(config=values, effective_config=dict(config.final_config_dict),
                  config_sha256=hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest())
    init_seed(config['seed'] + config['local_rank'], config['reproducibility'])
    init_logger(config)
    protocol = verify_protocol(config, check_sha=True)
    stats = json.loads(STATS.read_text())
    dataset = PreciseHistoryDataset(config)
    train_ds, valid_ds, reserved = dataset.build()
    del reserved
    if (len(train_ds), len(valid_ds), train_ds.item_num) != (1062567, 23951, 7112):
        raise ValueError('Frozen chronological Protocol B mismatch')
    verified_stats = verify_history_stats(train_ds, stats, config)
    train_data = TrainDataLoader(config, train_ds, None, shuffle=config['shuffle'])
    valid_data = FullSortEvalDataLoader(config, valid_ds, None, shuffle=False)
    model = ThreeTimeMamba3Rec(config, train_ds).to(config['device'])
    handle = backend_guard(model)
    record.update(protocol=protocol, manifest_sha256=MANIFEST_SHA, train_time_stats_sha256=STATS_SHA,
                  verified_history_stats=verified_stats, parameter_count=sum(p.numel() for p in model.parameters()),
                  initial_backbone_sha256=backbone_hash(model),
                  initial_calibrator_hashes={k: tensor_hash(v.state_dict()) for k, v in model.times.calibrators.items()},
                  history=[], actual_epochs=0)
    if record['parameter_count'] != COUNTS[mode]:
        raise ValueError('Scientific parameter count mismatch')
    dual = json.loads(paths('dual')['result'].read_text()) if mode == 'triple' else None
    if dual is not None:
        if dual['status'] != 'PASS':
            raise ValueError('Completed same-pilot dual required')
        for key in ('execution_commit', 'source_hash', 'policy_sha256', 'admission_sha256', 'job_id',
                    'initial_backbone_sha256', 'protocol', 'verified_history_stats', 'manifest_sha256', 'train_time_stats_sha256'):
            if dual[key] != record[key]:
                raise ValueError('Paired scientific provenance mismatch: ' + key)
        if (record['initial_calibrator_hashes']['decay'] != dual['initial_calibrator_hashes']['decay'] or
                any(record['initial_calibrator_hashes'][k] != dual['initial_calibrator_hashes']['scan'] for k in ('write', 'phase'))):
            raise ValueError('Paired calibrator initialization mismatch')
    cls = trainer_class(Trainer, valid_data, record, p, None if dual is None else dual['first_train_batch_sha256'])
    trainer = cls(config, model)
    trainer.saved_model_file = str(p['checkpoint'])
    record.update(rng_before_fit_sha256=rng_hash(), checkpoint_path=str(p['checkpoint']),
                  checkpoint_metadata_path=str(p['metadata']), tensorboard_path=str(trainer.tensorboard.log_dir))
    try:
        if dual is not None and record['rng_before_fit_sha256'] != dual['rng_before_fit_sha256']:
            raise ValueError('Python/NumPy/CPU/CUDA training RNG differs')
        require_gate()
        assert_upstream(model)
        record.update(stage='TRAIN_VALID', scientific_fit_started=True)
        atomic_json(p['result'], record)
        score, metrics = trainer.fit(train_data, valid_data, saved=True, show_progress=False)
        if record['best_valid_score'] != float(score) or record['best_valid_metrics'] != dict(metrics):
            raise ValueError('Best checkpoint differs from unchanged Trainer result')
        if len(trainer.train_loss_dict) != len(record['history']) or [r['epoch'] for r in record['history']] != list(range(record['actual_epochs'])):
            raise ValueError('Incomplete epoch history')
        if sha(p['checkpoint']) != record['checkpoint_sha256']:
            raise ValueError('Checkpoint SHA mismatch')
        rows = record['history']
        record.update(status='PASS', stage='COMPLETED', first27_best_ndcg10=max(r['valid_ndcg10'] for r in rows[:27]),
            train_seconds=sum(r['train_seconds'] for r in rows), valid_seconds=sum(r['valid_seconds'] for r in rows),
            peak_gpu_allocated_bytes=max(max(r['train_peak_allocated_bytes'], r['valid_peak_allocated_bytes']) for r in rows),
            peak_gpu_reserved_bytes=max(max(r['train_peak_reserved_bytes'], r['valid_peak_reserved_bytes']) for r in rows))
    finally:
        handle.remove()
        trainer.tensorboard.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=MODES, required=True)
    mode = parser.parse_args().mode
    base = identity()
    admission = require_gate()
    smoke = json.loads(SMOKE.read_text())
    if smoke['status'] != 'PASS' or any(smoke.get(k) != v for k, v in base.items()) or smoke['admission_sha256'] != admission:
        raise ValueError('Same-source admission and smoke required')
    p = paths(mode)
    if any(p[k].exists() for k in ('result', 'lock', 'checkpoint', 'metadata')):
        raise FileExistsError('Scientific run already owned; no retry')
    record = dict(**base, **next(t for t in plan()['tasks'] if t['mode'] == mode),
                  admission_sha256=admission, smoke_sha256=sha(SMOKE), status='RUNNING', stage='SETUP',
                  scientific_fit_started=False, started_at=now(), epoch_indexing='zero-based',
                  selection_split='VALID', evaluation_mode='full-ranking')
    create_record(p['lock'], record)
    create_record(p['result'], record)
    def terminate(signum, frame):
        raise TimeoutError(f'Allocation interrupted: {signum}')
    signal.signal(signal.SIGTERM, terminate)
    try:
        os.chdir(p['runtime'])
        train(mode, record, p)
    except BaseException as exc:
        status = 'BLOCKED_OOM' if isinstance(exc, torch.cuda.OutOfMemoryError) else 'INCOMPLETE' if isinstance(exc, TimeoutError) else 'FAIL'
        record.update(status=status, error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        record['finished_at'] = now()
        atomic_json(p['result'], record)


if __name__ == '__main__':
    from experiments.mamba3_three_time.backends import selected
    with selected('upstream'):
        main()
