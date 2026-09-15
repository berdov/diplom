"""One smoke or TRAIN->VALID run; never construct or evaluate a TEST loader."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from recbole.config import Config
from recbole.data.dataloader import TrainDataLoader, FullSortEvalDataLoader
from recbole.trainer import Trainer
from recbole.utils import init_logger, init_seed

from .config import load_config
from .compute_train_time_stats import sha256, summarize
from .dataset import PreciseHistoryDataset
from .gpu_equivalence import PIN
from .model import TimeAwareMamba3Rec

# Frozen run.py uses a script-local model import; reuse its verifier unchanged.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/mamba3_baseline'))
from experiments.mamba3_baseline.run import verify_protocol, runtime_info
sys.path.pop(0)
HERE = Path(__file__).resolve().parent
STATS = HERE / 'runs/train_time_stats_001.json'


def save_json(path, result):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2, default=str) + '\n')
    temporary.replace(path)


def verify_history_stats(dataset, stats, config):
    counts = Counter()
    field = config['TIME_FIELD'] + config['LIST_SUFFIX']
    time = dataset.inter_feat[field]
    lengths = dataset.inter_feat[config['ITEM_LIST_LENGTH_FIELD']]
    if time.dtype != torch.float64:
        raise ValueError('History timestamp precision was lost')
    for start in range(0, len(time), 4096):
        values = time[start:start + 4096].numpy()
        mask = np.arange(1, values.shape[1])[None, :] < lengths[start:start + 4096].numpy()[:, None]
        gaps = np.diff(values, axis=1)[mask]
        if (gaps < 0).any():
            raise ValueError('History order differs from chronological Protocol B')
        unique, weights = np.unique(gaps.astype(np.int64), return_counts=True)
        counts.update(dict(zip(unique.tolist(), weights.tolist())))
    actual = summarize(counts)
    for key in ('count_total_gaps', 'count_positive_gaps', 'median', 'p25', 'p75', 'p99'):
        if actual[key] != stats[key]:
            raise ValueError(f'TRAIN history statistic mismatch: {key}: {actual[key]} != {stats[key]}')
    return actual


def smoke(trainer, train_data, valid_data):
    model = trainer.model
    model.train()
    losses = []
    before = model.time_calibrator.last.weight.detach().clone()
    for index, batch in enumerate(train_data):
        batch = batch.to(trainer.device)
        trainer.optimizer.zero_grad(set_to_none=True)
        loss = model.calculate_loss(batch)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite smoke loss')
        loss.backward()
        if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise ValueError('Nonfinite smoke gradient')
        if model.time_calibrator.last.weight.grad.abs().sum() == 0:
            raise ValueError('No calibrator gradient')
        trainer.optimizer.step()
        losses.append(loss.item())
        if index == 1:
            break
    if len(losses) != 2 or torch.equal(before, model.time_calibrator.last.weight):
        raise ValueError('Smoke optimizer did not update calibrator')
    trainer.eval_collector.data_collect(train_data)
    metrics = trainer.evaluate(valid_data, load_best_model=False)
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError('Nonfinite smoke VALID metric')
    trainer._save_checkpoint(0)
    saved = torch.load(trainer.saved_model_file, map_location=trainer.device, weights_only=False)
    model.load_state_dict(saved['state_dict'])
    model.load_other_parameter(saved.get('other_parameter'))
    if not all(torch.equal(value, model.state_dict()[key]) for key, value in saved['state_dict'].items()):
        raise ValueError('Checkpoint roundtrip failed')
    return dict(smoke_train_batches=2, losses=losses, validation_scope='first 64 VALID histories; full-ranking',
                smoke_valid_metrics=dict(metrics), checkpoint_roundtrip='PASS', finite_gradients=True,
                calibrator_updated=True, history_dtype='float64')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('smoke', 'train'), required=True)
    args = parser.parse_args()
    if args.mode == 'train':
        gate = json.loads((HERE / 'runs/mamba3_timeaware_smoke_001.json').read_text())
        if gate['status'] != 'PASS' or gate['test_evaluation_count'] != 0:
            raise RuntimeError('Full run requires smoke PASS without TEST')
        if gate['train_time_stats_sha256'] != sha256(STATS):
            raise RuntimeError('TRAIN statistics changed since smoke')
    run_id = 'mamba3_timeaware_smoke_001' if args.mode == 'smoke' else 'mamba3_timeaware_validation_001'
    path = HERE / 'runs' / (run_id + '.json')
    if path.exists():
        raise RuntimeError('Run JSON already exists; no automatic retry')
    lock = HERE / 'slurm_logs' / (run_id + '.lock')
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('x') as handle:
        handle.write(os.environ.get('SLURM_JOB_ID', 'local'))
    result = dict(run_id=run_id, mode=args.mode, status='running', test_evaluation_count=0,
                  selection_split='VALID', evaluation_mode='full-ranking',
                  git_commit=os.environ['RUN_COMMIT'], job_id=os.environ.get('SLURM_JOB_ID'),
                  started_at=datetime.now(timezone.utc).isoformat())
    save_json(path, result)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required')
        direct = json.loads(importlib.metadata.distribution('mamba-ssm').read_text('direct_url.json'))
        if direct['vcs_info']['commit_id'] != PIN:
            raise RuntimeError('Mamba pin mismatch')
        settings = load_config()
        settings['checkpoint_dir'] = str(HERE / 'slurm_logs' / run_id / 'checkpoints')
        config = Config(model=TimeAwareMamba3Rec, config_dict=settings)
        init_seed(config['seed'] + config['local_rank'], config['reproducibility'])
        init_logger(config)
        protocol = verify_protocol(config, check_sha=True)
        stats = json.loads(STATS.read_text())
        if stats['split'] != 'TRAIN_ONLY' or stats['time_scale_reference'] != config['time_scale_reference']:
            raise ValueError('Frozen TRAIN reference mismatch')
        if stats['manifest_sha256'] != sha256(protocol['manifest_path']):
            raise ValueError('Stats manifest mismatch')
        dataset = PreciseHistoryDataset(config)
        train_ds, valid_ds, heldout_ds = dataset.build()
        del heldout_ds  # Reserved split is discarded; no TEST loader is constructed.
        if (len(train_ds), len(valid_ds)) != (1086518 - 23951, 23951):
            raise ValueError('Protocol B sequential split sizes mismatch')
        history_stats = verify_history_stats(train_ds, stats, config)
        train_data = TrainDataLoader(config, train_ds, None, shuffle=config['shuffle'])
        if args.mode == 'smoke':
            valid_ds = valid_ds.copy(valid_ds.inter_feat[:64])
        valid_data = FullSortEvalDataLoader(config, valid_ds, None, shuffle=False)
        model = TimeAwareMamba3Rec(config, train_ds).to(config['device'])
        trainer = Trainer(config, model)
        result.update(protocol=protocol, train_time_stats_path=str(STATS), train_time_stats_sha256=sha256(STATS),
                      verified_history_stats=history_stats, time_scale_reference=config['time_scale_reference'],
                      max_log_scale=config['max_log_scale'], runtime=runtime_info(), pinned_mamba_commit=PIN,
                      model_params=sum(p.numel() for p in model.parameters()),
                      config=settings, checkpoint_path=trainer.saved_model_file)
        if args.mode == 'smoke':
            result.update(smoke(trainer, train_data, valid_data))
        else:
            def progress(epoch, score):
                result.update(last_epoch=epoch, last_valid_ndcg10=float(score), actual_epochs=epoch + 1)
                save_json(path, result)
            score, metrics = trainer.fit(train_data, valid_data, saved=True, show_progress=False, callback_fn=progress)
            saved = torch.load(trainer.saved_model_file, map_location='cpu', weights_only=False)
            result.update(best_epoch=int(saved['epoch']), epoch_indexing='zero-based',
                          actual_epochs=len(trainer.train_loss_dict), best_valid_score=float(score),
                          best_valid_metrics=dict(metrics))
        result.update(status='PASS', finished_at=datetime.now(timezone.utc).isoformat())
        save_json(path, result)
        print(json.dumps(result, indent=2, default=str))
    except Exception as exc:
        result.update(status='FAIL', error=repr(exc))
        save_json(path, result)
        raise


if __name__ == '__main__':
    main()
