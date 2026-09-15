"""One final TEST of the frozen best-VALID checkpoint; no optimizer or fit."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path

import torch
from recbole.config import Config
from recbole.data.dataloader import FullSortEvalDataLoader
from recbole.trainer import Trainer
from recbole.utils import init_logger, init_seed

from .compute_train_time_stats import sha256
from .config import load_config
from .dataset import PreciseHistoryDataset
from .gpu_equivalence import PIN
from .model import TimeAwareMamba3Rec
from .one_shot import OneShotResult
from .run import verify_protocol, runtime_info


HERE = Path(__file__).resolve().parent
SELECTION = HERE / 'runs/mamba3_timeaware_validation_001.json'
RESULT = HERE / 'runs/mamba3_timeaware_final_test_001.json'
LOCK = HERE / 'slurm_logs/mamba3_timeaware_final_test_001.lock'
CHECKPOINT = Path('/home/daryumin/iberdov/diplom/experiments/mamba3_timeaware/slurm_logs/'
                  'mamba3_timeaware_validation_001/checkpoints/TimeAwareMamba3Rec-Sep-14-2026_11-38-41.pth')
CHECKPOINT_SHA = 'd8960963f8c94baa1229f803eec50fb194e645ef8d209c5fd88b4f6148f9a93d'
SELECTION_SHA = 'eec574c17527e1bb690705172618a2ff40a90d435a7a5cbec39fabd5c2056539'
INTER_SHA = 'e275ded0b330c2827b49ccf567d6784452d6dcbf8cd719dc3009d36eadc2e2cc'


class TestOnlyEvaluator(Trainer):
    def _build_optimizer(self, **kwargs):
        return None

    def fit(self, *args, **kwargs):
        raise RuntimeError('Training forbidden in final TEST')

    def _train_epoch(self, *args, **kwargs):
        raise RuntimeError('Training forbidden in final TEST')


def preflight():
    if RESULT.exists() or LOCK.exists():
        raise FileExistsError('Existing TEST result/lock; no retry permitted')
    if sha256(SELECTION) != SELECTION_SHA:
        raise ValueError('Frozen selection JSON changed')
    selection = json.loads(SELECTION.read_text())
    if (selection['status'], selection['best_epoch'], selection['best_valid_score'],
        selection['test_evaluation_count'], selection['checkpoint_path']) != (
            'PASS', 15, 0.0605, 0, str(CHECKPOINT)):
        raise ValueError('Frozen selection mismatch')
    if not CHECKPOINT.is_file() or sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError('Frozen checkpoint missing or changed')
    direct = json.loads(importlib.metadata.distribution('mamba-ssm').read_text('direct_url.json'))
    if direct['vcs_info']['commit_id'] != PIN or selection['pinned_mamba_commit'] != PIN:
        raise ValueError('Mamba commit mismatch')
    settings = load_config()
    for key, value in settings.items():
        if key == 'checkpoint_dir':
            continue
        frozen_value = selection['config'][key]
        if (str(value) != str(frozen_value)) if key == 'gpu_id' else (value != frozen_value):
            raise ValueError(f'Config changed: {key}')
    config = Config(model=TimeAwareMamba3Rec, config_dict=selection['config'])
    protocol = verify_protocol(config, check_sha=True)
    if protocol['recbole_inter_sha256'] != INTER_SHA:
        raise ValueError('Protocol B SHA mismatch')
    if set(config['metrics']) != {'Hit', 'Recall', 'NDCG'} or list(config['topk']) != [5, 10, 20, 50]:
        raise ValueError('Metric protocol mismatch')
    if config['eval_args']['mode']['test'] != 'full':
        raise ValueError('Full-ranking required')
    stats = HERE / 'runs/train_time_stats_001.json'
    if sha256(stats) != selection['train_time_stats_sha256']:
        raise ValueError('Frozen TRAIN reference evidence changed')
    saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    if saved['epoch'] != 15 or float(saved['best_valid_score']) != 0.0605:
        raise ValueError('Checkpoint epoch/selection score mismatch')
    for key in settings:
        if key not in ('checkpoint_dir', 'data_path', 'eval_args', 'train_neg_sample_args', 'gpu_id'):
            if saved['config'][key] != config[key]:
                raise ValueError(f'Checkpoint config mismatch: {key}')
    return config, protocol, saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true', help='Check frozen inputs only; no dataset/evaluation')
    args = parser.parse_args()
    config, protocol, saved = preflight()
    print(json.dumps(dict(preflight='PASS', checkpoint_sha256=CHECKPOINT_SHA,
                          pinned_mamba_commit=PIN, protocol_sha256=INTER_SHA,
                          selected_epoch=15, selection_value=0.0605)), flush=True)
    if args.preflight:
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    commit = os.environ['RUN_COMMIT']
    guard = OneShotResult(RESULT, LOCK)
    started = datetime.now(timezone.utc).isoformat()
    phase = 'building_test_loader'
    try:
        init_seed(config['seed'] + config['local_rank'], config['reproducibility'])
        init_logger(config)
        dataset = PreciseHistoryDataset(config)
        unused_train, unused_valid, test_dataset = dataset.build()
        del unused_train, unused_valid
        if len(test_dataset) != 23951 or test_dataset.item_num - 1 != 7111:
            raise ValueError('TEST count/catalog mismatch')
        test_data = FullSortEvalDataLoader(config, test_dataset, None, shuffle=False)
        model = TimeAwareMamba3Rec(config, test_dataset).to(config['device'])
        model.load_state_dict(saved['state_dict'], strict=True)
        model.load_other_parameter(saved.get('other_parameter'))
        if float(model.time_calibrator.reference) != float(config['time_scale_reference']):
            raise ValueError('Checkpoint temporal reference mismatch')
        model.requires_grad_(False)
        model.eval()
        evaluator = TestOnlyEvaluator(config, model)
        if evaluator.optimizer is not None:
            raise RuntimeError('Optimizer creation forbidden')
        phase = 'test_evaluation_started'
        print(phase, flush=True)
        metrics = evaluator.evaluate(test_data, load_best_model=False, show_progress=False)
        phase = 'test_evaluation_finished'
        expected = {f'{metric}@{k}' for metric in ('hit', 'recall', 'ndcg') for k in (5, 10, 20, 50)}
        if set(metrics) != expected or not all(math.isfinite(float(v)) for v in metrics.values()):
            raise ValueError('Incomplete/nonfinite TEST metrics')
        if sha256(CHECKPOINT) != CHECKPOINT_SHA:
            raise ValueError('Checkpoint changed during evaluation')
        result = dict(run_id='mamba3_timeaware_final_test_001', status='PASS', mode='final_test',
                      split='TEST', selection_checkpoint=str(CHECKPOINT), checkpoint_path=str(CHECKPOINT),
                      checkpoint_sha256=CHECKPOINT_SHA, selection_json_sha256=SELECTION_SHA,
                      selected_epoch=15, selection_metric='VALID NDCG@10', selection_value=0.0605,
                      checkpoint_selected_only_by_valid=True, test_evaluation_count=1,
                      protocol=protocol, catalog_size=7111, evaluation_mode='full-ranking',
                      loader_class=type(test_data).__name__, git_commit=commit,
                      pinned_mamba_commit=PIN, runtime=runtime_info(),
                      job_id=os.environ.get('SLURM_JOB_ID'), node=os.environ.get('SLURM_JOB_NODELIST'),
                      test_result={key: float(value) for key, value in metrics.items()},
                      started_at=started, finished_at=datetime.now(timezone.utc).isoformat(),
                      no_post_test_tuning=True, no_training_during_final_test=True,
                      training_count=0, valid_evaluation_count=0)
        guard.publish(result)
        print('FINAL_TEST_PASS', flush=True)
        print(json.dumps(result, indent=2))
    except Exception as exc:
        failure = LOCK.with_suffix('.failure.json')
        with failure.open('x') as handle:
            json.dump(dict(phase=phase, error=repr(exc), automatic_retry=False), handle, indent=2)
        raise


if __name__ == '__main__':
    main()
