"""Prepared TRAIN->VALID runner. Shared/vanilla scientific reruns forbidden."""

import argparse
import importlib.metadata
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .provenance import HERE, PIN, require_equivalence

RUN_IDS = dict(decay_only='mamba3_decay_only_validation_001',
               scan_only='mamba3_scan_only_validation_001',
               separate='mamba3_separate_time_validation_001')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=tuple(RUN_IDS), required=True)
    parser.add_argument('--equivalence-json', type=Path, required=True)
    args = parser.parse_args()
    evidence = require_equivalence(args.equivalence_json)
    import torch
    from recbole.config import Config
    from recbole.data.dataloader import TrainDataLoader, FullSortEvalDataLoader
    from recbole.trainer import Trainer
    from recbole.utils import init_seed, init_logger
    from experiments.mamba3_timeaware.run import verify_protocol, runtime_info, verify_history_stats, save_json, STATS, sha256
    from experiments.mamba3_timeaware.dataset import PreciseHistoryDataset
    from .config import load_config
    from .model import MechanismMamba3Rec
    from .diagnostics import TemporalDiagnostics

    class DiagnosticTrainer(Trainer):
        def _valid_epoch(self, valid_data, show_progress=False):
            collector = TemporalDiagnostics(args.mode)
            self.model.diagnostic_collector = collector
            try:
                result = super()._valid_epoch(valid_data, show_progress=show_progress)
                self.last_diagnostics = collector.result()
                return result
            finally:
                self.model.diagnostic_collector = None

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no scientific CPU fallback')
    direct = json.loads(importlib.metadata.distribution('mamba-ssm').read_text('direct_url.json'))
    if direct['vcs_info']['commit_id'] != PIN:
        raise ValueError('Mamba pin mismatch')
    run_id = RUN_IDS[args.mode]
    path = HERE / 'runs' / (run_id + '.json')
    if path.exists():
        raise FileExistsError('Scientific result exists; no automatic retry')
    logs = HERE / 'slurm_logs' / run_id
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / 'run.lock').open('x') as handle:
        handle.write(os.environ.get('SLURM_JOB_ID', 'local'))
    result = dict(run_id=run_id, status='running', mode=args.mode, TEST='NOT_RUN', test_evaluation_count=0,
                  git_commit=os.environ['RUN_COMMIT'], job_id=os.environ.get('SLURM_JOB_ID'),
                  evaluation_mode='full-ranking', selection_metric='VALID NDCG@10',
                  started_at=datetime.now(timezone.utc).isoformat(),
                  equivalence_fingerprint=evidence['source_fingerprint'])
    save_json(path, result)
    try:
        settings = load_config(args.mode)
        settings['checkpoint_dir'] = str(logs / 'checkpoints')
        config = Config(model=MechanismMamba3Rec, config_dict=settings)
        init_seed(config['seed'] + config['local_rank'], config['reproducibility'])
        init_logger(config)
        protocol = verify_protocol(config, check_sha=True)
        stats = json.loads(STATS.read_text())
        if stats['split'] != 'TRAIN_ONLY' or stats['time_scale_reference'] != 838393 or stats['manifest_sha256'] != sha256(protocol['manifest_path']):
            raise ValueError('Frozen TRAIN reference mismatch')
        dataset = PreciseHistoryDataset(config)
        train_ds, valid_ds, reserved = dataset.build()
        del reserved  # No TEST loader is constructed.
        if (len(train_ds),len(valid_ds),train_ds.item_num) != (1062567,23951,7112):
            raise ValueError('Protocol B split mismatch')
        history_stats = verify_history_stats(train_ds, stats, config)
        train_data = TrainDataLoader(config, train_ds, None, shuffle=config['shuffle'])
        valid_data = FullSortEvalDataLoader(config, valid_ds, None, shuffle=False)
        model = MechanismMamba3Rec(config, train_ds).to(config['device'])
        trainer = DiagnosticTrainer(config, model)
        result.update(config=settings, protocol=protocol, runtime=runtime_info(), pinned_mamba_commit=PIN,
                      verified_history_stats=history_stats, train_time_stats_sha256=sha256(STATS),
                      parameter_count=sum(p.numel() for p in model.parameters()),
                      checkpoint_path=trainer.saved_model_file, diagnostics_history=[])
        def progress(epoch, score):
            diag = trainer.last_diagnostics
            result['diagnostics_history'].append(dict(epoch=epoch, **diag))
            if score > result.get('best_valid_score', -float('inf')):
                result.update(best_valid_score=float(score), best_diagnostics=diag)
            result.update(last_epoch=epoch, actual_epochs=epoch+1)
            save_json(path, result)
        score, metrics = trainer.fit(train_data, valid_data, saved=True, show_progress=False, callback_fn=progress)
        saved = torch.load(trainer.saved_model_file, map_location='cpu', weights_only=False)
        result.update(status='PASS', best_epoch=int(saved['epoch']), epoch_indexing='zero-based',
                      actual_epochs=len(trainer.train_loss_dict), best_valid_score=float(score),
                      best_valid_metrics=dict(metrics), finished_at=datetime.now(timezone.utc).isoformat())
        trainer.tensorboard.close()
        save_json(path, result)
    except Exception as exc:
        result.update(status='FAIL', error=repr(exc))
        save_json(path, result)
        raise


if __name__ == '__main__':
    main()
