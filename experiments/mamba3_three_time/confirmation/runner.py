"""Pilot orchestration adapted to explicit task paths; trainer/fit remain frozen."""
import argparse
import hashlib
import json
import os
import signal
import traceback
from .config import task, settings, effective_check, paths, COUNTS, BATCH, INIT
from .provenance import identity, gates, atomic_json, create_record, sha, now, imported_sources
from .state import initial, paired, precision, rng_record, canonical_optimizer_settings


def prepare(mode, seed, record, p, expected_batch=None):
    from recbole.config import Config
    from recbole.utils import init_seed, init_logger
    from recbole.data.dataloader import TrainDataLoader, FullSortEvalDataLoader
    from recbole.trainer import Trainer
    from experiments.mamba3_timeaware.dataset import PreciseHistoryDataset
    from experiments.mamba3_timeaware.run import verify_protocol, verify_history_stats
    from experiments.mamba3_time_confirmation.config import STATS, STATS_SHA, MANIFEST_SHA
    from experiments.mamba3_three_time.model import ThreeTimeMamba3Rec
    from experiments.mamba3_three_time.validation_pilot.trainer import trainer_class
    from experiments.mamba3_three_time.validation_pilot.provenance import runtime, backend_guard

    record['runtime'] = runtime(require_cuda=True)
    if record['runtime']['matmul_allow_tf32'] is not False or record['runtime']['cudnn_allow_tf32'] is not True:
        raise ValueError('Pilot TF32 flags changed')
    values = settings(mode, seed, p['checkpoint'].parent)
    config = Config(model=ThreeTimeMamba3Rec, config_dict=values)
    record['effective_config_parity'] = effective_check(config, mode, seed)
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
    record['rng_before_constructor'] = rng_record(train_data)
    model = ThreeTimeMamba3Rec(config, train_ds).to(config['device'])
    record['rng_after_constructor'] = rng_record(train_data)
    handle = backend_guard(model)
    record.update(protocol=protocol, manifest_sha256=MANIFEST_SHA, train_time_stats_sha256=STATS_SHA,
                  verified_history_stats=verified_stats, parameter_count=sum(p.numel() for p in model.parameters()),
                  history=[], actual_epochs=0)
    if record['parameter_count'] != COUNTS[mode]:
        raise ValueError('Scientific parameter count mismatch')
    if mode == 'triple':
        a, b = model.times.calibrators['write'], model.times.calibrators['phase']
        if {v.data_ptr() for v in a.parameters()} & {v.data_ptr() for v in b.parameters()}:
            raise ValueError('Triple calibrators must be independent tensors')
    record['calibrators_independent'] = True
    cls = trainer_class(Trainer, valid_data, record, p, expected_batch)
    trainer = cls(config, model)
    trainer.saved_model_file = str(p['checkpoint'])
    record.update(initial(model, train_data))
    record['optimizer_settings'] = canonical_optimizer_settings(
        [{k: v for k, v in group.items() if k != 'params'} for group in trainer.optimizer.param_groups])
    record.update(precision=precision(), checkpoint_path=str(p['checkpoint']),
                  checkpoint_metadata_path=str(p['metadata']), tensorboard_path=str(trainer.tensorboard.log_dir),
                  timing_note='Includes JIT; not steady-state overhead', imported_sources=imported_sources())
    return trainer, train_data, valid_data, handle


def main():
    global identity, gates, paths
    import torch
    from experiments.mamba3_three_time.validation_pilot.provenance import assert_upstream
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.resume:
        from .resume_provenance import identity, inherited_gates as gates
        from .resume_config import paths, execution_task
        execution_task(args.mode, args.seed)
    t = task(args.mode, args.seed)
    base = identity()
    init = gates(base)
    p = paths(args.mode, args.seed)
    if any(p[k].exists() or p[k].is_symlink() for k in ('result', 'lock', 'checkpoint', 'metadata')):
        raise FileExistsError('Owned scientific run; no retry')
    record = dict(**base, **t, status='RUNNING', stage='SETUP', scientific_fit_started=False,
                  started_at=now(), epoch_indexing='zero-based', selection_split='VALID', evaluation_mode='full-ranking',
                  one_batch_sha256=sha(BATCH), initialization_sha256=sha(INIT))
    create_record(p['lock'], record)
    create_record(p['result'], record)
    def terminate(signum, frame):
        raise TimeoutError(f'Allocation interrupted: {signum}')
    signal.signal(signal.SIGTERM, terminate)
    trainer = handle = None
    try:
        os.chdir(p['runtime'])
        if args.resume and args.mode == 'triple':
            from .resume_provenance import resolve
            dual = resolve('dual', args.seed, base)
        else:
            dual = json.loads(paths('dual', args.seed)['result'].read_text()) if args.mode == 'triple' else None
        if dual is not None and (dual['status'] != 'PASS' or
                (not args.resume and any(dual.get(k) != v for k, v in base.items()))):
            raise ValueError('Same-allocation completed dual required')
        trainer, train_data, valid_data, handle = prepare(args.mode, args.seed, record, p,
            None if dual is None else dual['first_train_batch_sha256'])
        expected = next(r for r in init['rows'] if r['mode'] == args.mode and r['seed'] == args.seed)
        for k in ('initial_backbone_sha256', 'initial_calibrator_hashes', 'rng_components', 'rng_before_fit_sha256'):
            if record[k] != expected[k]:
                raise ValueError('Pre-fit initialization differs from gate: ' + k)
        if dual is not None:
            paired(dual, record)
        gates(base)
        assert_upstream(trainer.model)
        record.update(stage='TRAIN_VALID', scientific_fit_started=True)
        atomic_json(p['result'], record)
        score, metrics = trainer.fit(train_data, valid_data, saved=True, show_progress=False)
        if record['best_valid_score'] != float(score) or record['best_valid_metrics'] != dict(metrics):
            raise ValueError('Best checkpoint differs from unchanged Trainer result')
        if len(trainer.train_loss_dict) != len(record['history']) or [r['epoch'] for r in record['history']] != list(range(record['actual_epochs'])):
            raise ValueError('Incomplete epoch history')
        if sha(p['checkpoint']) != record['checkpoint_sha256']:
            raise ValueError('Checkpoint SHA mismatch')
        if dual is not None:
            paired(dual, record, first_batch=True)
        rows = record['history']
        record.update(status='PASS', stage='COMPLETED', first27_best_ndcg10=max(r['valid_ndcg10'] for r in rows[:27]),
            first27_complete=len(rows) >= 27,
            train_seconds=sum(r['train_seconds'] for r in rows), valid_seconds=sum(r['valid_seconds'] for r in rows),
            peak_gpu_allocated_bytes=max(max(r['train_peak_allocated_bytes'], r['valid_peak_allocated_bytes']) for r in rows),
            peak_gpu_reserved_bytes=max(max(r['train_peak_reserved_bytes'], r['valid_peak_reserved_bytes']) for r in rows))
    except BaseException as exc:
        record.update(status='BLOCKED_OOM' if isinstance(exc, torch.cuda.OutOfMemoryError) else
                      'INCOMPLETE' if isinstance(exc, TimeoutError) else 'FAIL', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        if handle is not None:
            handle.remove()
        if trainer is not None:
            trainer.tensorboard.close()
        record['finished_at'] = now()
        atomic_json(p['result'], record)


if __name__ == '__main__':
    from experiments.mamba3_three_time.backends import selected
    with selected('upstream'):
        main()
