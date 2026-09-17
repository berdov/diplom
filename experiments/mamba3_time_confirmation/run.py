"""One array task: synthetic smoke, fresh TRAIN -> VALID, never TEST."""
import json
import math
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .config import (CORE, COUNTS, HERE, MANIFEST, MANIFEST_SHA, ROOT, STATS,
                     STATS_SHA, STUDY_ID, paths, plan, settings)
from .provenance import acquire, atomic_json, sha, verify


def now():
    return datetime.now(timezone.utc).isoformat()


def backbone_hash(model):
    import hashlib
    import torch
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        if key.startswith(('time_calibrator.', 'mechanisms.')):
            continue
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def synthetic_smoke(task, config, task_paths):
    import torch
    from recbole.config import Config
    from recbole.utils import init_seed
    from .model import model_class

    class SyntheticDataset:
        def num(self, field):
            return 7112 if field == config['ITEM_ID_FIELD'] else 23952

    dataset = SyntheticDataset()
    signatures = []
    for mode in ('shared', 'separate'):
        pair_task = dict(task, mode=mode)
        pair_config = Config(model=model_class(mode), config_dict=settings(pair_task))
        init_seed(task['seed'], config['reproducibility'])
        pair = model_class(mode)(pair_config, dataset)
        signatures.append(backbone_hash(pair))
        del pair
    if signatures[0] != signatures[1]:
        raise ValueError('Paired backbone initialization differs')

    init_seed(task['seed'], config['reproducibility'])
    cls = model_class(task['mode'])
    model = cls(config, dataset).to(config['device'])
    count = sum(p.numel() for p in model.parameters())
    if count != COUNTS[task['mode']]:
        raise ValueError('Parameter count mismatch')
    device = config['device']
    items = torch.arange(1, 51, device=device).repeat(2, 1)
    lengths = torch.tensor([50, 32], device=device)
    items[1, 32:] = 0
    timestamps = torch.arange(50, device=device, dtype=torch.float64)[None, :].repeat(2, 1)*838393 + 1e12
    timestamps[:, 1] = timestamps[:, 0]  # Real active zero gaps must not bypass the control.
    targets = torch.tensor([250, 251], device=device)
    calibrators = ([model.time_calibrator] if task['mode'] == 'shared'
                   else list(model.mechanisms.calibrators.values()))
    before = [c.last.weight.detach().clone() for c in calibrators]
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    model.train()
    output = model(items, lengths, timestamps)
    loss = model.loss_fct(output @ model.item_embedding.weight.T, targets)
    if not torch.isfinite(output).all() or not torch.isfinite(loss):
        raise ValueError('Nonfinite smoke output/loss')
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError('Nonfinite smoke gradients')
    if any(c.last.weight.grad is None or not c.last.weight.grad.abs().sum() for c in calibrators):
        raise ValueError('Missing calibrator training signal')
    optimizer.step()
    if any(torch.equal(old, c.last.weight) for old, c in zip(before, calibrators)):
        raise ValueError('Calibrator did not update')
    model.eval()
    with torch.no_grad():
        expected = model(items, lengths, timestamps)
        if task['mode'] == 'separate_constant_gap':
            changed = model(items, lengths, timestamps*3 + 987654321)
            torch.testing.assert_close(expected, changed, atol=0, rtol=0)
    if task_paths['smoke_checkpoint'].exists():
        raise FileExistsError('Smoke checkpoint exists')
    torch.save(model.state_dict(), task_paths['smoke_checkpoint'])
    clone = cls(config, dataset).to(device)
    clone.load_state_dict(torch.load(task_paths['smoke_checkpoint'], map_location=device, weights_only=True))
    clone.eval()
    with torch.no_grad():
        torch.testing.assert_close(clone(items, lengths, timestamps), expected, atol=1e-6, rtol=1e-5)
    result = dict(status='PASS', synthetic_only=True, parameter_count=count,
                  finite_forward_backward=True, calibrators_updated=len(calibrators),
                  constant_output_invariant=task['mode'] == 'separate_constant_gap',
                  paired_backbone_initialization_sha256=signatures[0], checkpoint_roundtrip='PASS',
                  checkpoint=str(task_paths['smoke_checkpoint']), checkpoint_sha256=sha(task_paths['smoke_checkpoint']))
    del clone, model, optimizer, output, loss, calibrators
    torch.cuda.empty_cache()
    return result


def train(task, result, task_paths):
    import importlib.metadata
    import torch
    from recbole.config import Config
    from recbole.data.dataloader import TrainDataLoader, FullSortEvalDataLoader
    from recbole.trainer import Trainer
    from recbole.utils import init_logger, init_seed
    from experiments.mamba3_time_mechanisms.diagnostics import TemporalDiagnostics
    from experiments.mamba3_time_mechanisms.provenance import PIN
    from experiments.mamba3_timeaware.dataset import PreciseHistoryDataset
    from experiments.mamba3_timeaware.run import verify_protocol, verify_history_stats, runtime_info
    from .model import model_class

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no scientific CPU fallback')
    direct = json.loads(importlib.metadata.distribution('mamba-ssm').read_text('direct_url.json'))
    if direct['vcs_info']['commit_id'] != PIN:
        raise ValueError('Pinned Mamba mismatch')
    cls = model_class(task['mode'])
    config = Config(model=cls, config_dict=settings(task))
    result.update(stage='synthetic_smoke', runtime=runtime_info(), pinned_mamba_commit=PIN,
                  config=settings(task), effective_config=dict(config.final_config_dict))
    reference = json.loads((ROOT / plan()['reuse'][0]['source_json']).read_text())
    for key in ('torch', 'recbole', 'mamba_ssm', 'triton', 'numpy'):
        if result['runtime'][key] != reference['runtime'][key]:
            raise ValueError(f'Frozen runtime drift: {key}')
    atomic_json(task_paths['result'], result)
    result['synthetic_smoke'] = synthetic_smoke(task, config, task_paths)

    # Start from scratch in the same order as the reference runners. Smoke RNG is discarded.
    init_seed(config['seed'] + config['local_rank'], config['reproducibility'])
    init_logger(config)
    protocol = verify_protocol(config, check_sha=True)
    stats = json.loads(STATS.read_text())
    dataset = PreciseHistoryDataset(config)
    train_ds, valid_ds, reserved = dataset.build()
    del reserved  # Never construct a TEST loader.
    if (len(train_ds), len(valid_ds), train_ds.item_num) != (1062567, 23951, 7112):
        raise ValueError('Frozen sequential split mismatch')
    history_stats = verify_history_stats(train_ds, stats, config)
    train_data = TrainDataLoader(config, train_ds, None, shuffle=config['shuffle'])
    valid_data = FullSortEvalDataLoader(config, valid_ds, None, shuffle=False)
    model = cls(config, train_ds).to(config['device'])
    result.update(stage='TRAIN_VALID', protocol=protocol, verified_history_stats=history_stats,
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  initial_backbone_sha256=backbone_hash(model), history=[])
    if result['parameter_count'] != COUNTS[task['mode']]:
        raise ValueError('Scientific model parameter count mismatch')

    class ValidationTrainer(Trainer):
        def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
            self.current_epoch = epoch_idx
            start = time.perf_counter()
            loss = super()._train_epoch(train_data, epoch_idx, loss_func, show_progress)
            if not math.isfinite(float(loss)):
                raise ValueError('Nonfinite training loss')
            self.current_training = dict(train_loss=float(loss), train_seconds=time.perf_counter()-start)
            return loss

        def evaluate(self, eval_data, *args, **kwargs):
            if eval_data is not valid_data:
                raise ValueError('Only the exact VALID loader is permitted')
            return super().evaluate(eval_data, *args, **kwargs)

        def _valid_epoch(self, valid_data, show_progress=False):
            collector = TemporalDiagnostics('shared' if task['mode']=='shared' else 'separate')
            hook = None
            if task['mode'] == 'shared':
                hook = self.model.time_calibrator.register_forward_hook(
                    lambda module, args, scale: collector.update(scale, scale, args[1]))
            else:
                self.model.diagnostic_collector = collector
            start = time.perf_counter()
            try:
                score, metrics = super()._valid_epoch(valid_data, show_progress=show_progress)
            finally:
                if hook is not None:
                    hook.remove()
                else:
                    self.model.diagnostic_collector = None
            if not all(math.isfinite(float(v)) for v in metrics.values()):
                raise ValueError('Nonfinite VALID metrics')
            result['history'].append(dict(epoch=self.current_epoch, valid_ndcg10=float(score),
                valid_metrics=dict(metrics), valid_seconds=time.perf_counter()-start,
                diagnostics=collector.result(), **self.current_training))
            result.update(actual_epochs=len(result['history']), last_epoch=self.current_epoch)
            atomic_json(task_paths['result'], result)
            return score, metrics

    trainer = ValidationTrainer(config, model)
    result['tensorboard_path'] = str(Path(trainer.tensorboard.log_dir).resolve())
    result['checkpoint_path'] = trainer.saved_model_file
    atomic_json(task_paths['result'], result)
    try:
        score, metrics = trainer.fit(train_data, valid_data, saved=True, show_progress=False)
        saved = torch.load(trainer.saved_model_file, map_location='cpu', weights_only=False)
        epoch = int(saved['epoch'])
        best = next(row for row in result['history'] if row['epoch'] == epoch)
        if best['valid_ndcg10'] != float(score) or best['valid_metrics'] != dict(metrics):
            raise ValueError('Best checkpoint/VALID history mismatch')
        result.update(status='PASS', stage='COMPLETED', best_epoch=epoch,
            epoch_indexing='zero-based', actual_epochs=len(trainer.train_loss_dict),
            best_valid_score=float(score), best_valid_metrics=dict(metrics),
            best_diagnostics=best['diagnostics'], checkpoint_sha256=sha(trainer.saved_model_file),
            finished_at=now())
        if result['actual_epochs'] != len(result['history']):
            raise ValueError('Incomplete epoch history')
        atomic_json(task_paths['result'], result)
    finally:
        trainer.tensorboard.close()


def main():
    manifest = verify()
    if os.environ.get('EXPECTED_CORE_HASH') != CORE or os.environ.get('EXPECTED_STUDY_HASH') != manifest['source_hash']:
        raise ValueError('Submission hash mismatch')
    index = int(os.environ['SLURM_ARRAY_TASK_ID'])
    if not 0 <= index < 9:
        raise ValueError('Array task index outside frozen plan')
    task = plan()['tasks'][index]
    task_paths = paths(task)
    result = dict(**task, study_id=STUDY_ID, status='RUNNING', stage='PREFLIGHT',
                  selection_split='VALID', TEST='NOT_RUN', test_evaluation_count=0,
                  evaluation_mode='full-ranking', submission_commit=os.environ['RUN_COMMIT'],
                  core_fingerprint=CORE, study_source_hash=manifest['source_hash'],
                  manifest_sha256=MANIFEST_SHA, train_time_stats_sha256=STATS_SHA,
                  job_id=os.environ['SLURM_JOB_ID'], array_job_id=os.environ['SLURM_ARRAY_JOB_ID'],
                  array_task_id=int(os.environ['SLURM_ARRAY_TASK_ID']), started_at=now(),
                  paths={k:str(v) for k,v in task_paths.items()})
    acquire(task_paths, result)
    atomic_json(task_paths['result'], result)
    try:
        os.chdir(task_paths['runtime'])  # Isolate RecBole logger/TensorBoard default relative paths.
        train(task, result, task_paths)
    except BaseException as exc:
        result.update(status='FAIL', error=repr(exc), traceback=traceback.format_exc(), finished_at=now())
        atomic_json(task_paths['result'], result)
        raise


if __name__ == '__main__':
    main()
