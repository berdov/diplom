"""Exactly one scratch TRAIN->VALID fit, never a TEST loader or checkpoint selection."""
import argparse
import hashlib
import json
import os
import signal
import traceback

from .config import CORE, GPU_EVIDENCE, ROOT, STUDY_ID, paths, plan, settings
from .provenance import acquire, atomic_json, backbone_hash, now, require_submission, rng_hash, sha, tensor_hash


def train(task, record, task_paths):
    from recbole.config import Config
    from recbole.data.dataloader import TrainDataLoader, FullSortEvalDataLoader
    from recbole.trainer import Trainer
    from recbole.utils import init_seed, init_logger
    from experiments.mamba3_timeaware.dataset import PreciseHistoryDataset
    from experiments.mamba3_timeaware.run import verify_protocol, verify_history_stats
    from experiments.mamba3_time_confirmation.config import STATS, STATS_SHA, MANIFEST_SHA
    from .model import ContextMamba3Rec
    from .temporal import COUNTS
    from .preflight import runtime_check, effective_settings_check
    from .trainer import trainer_class

    record['runtime'] = runtime_check(require_cuda=True)
    data = settings(task)
    config = Config(model=ContextMamba3Rec, config_dict=data)
    record['effective_config_parity'] = effective_settings_check(config, task)
    record.update(config=data, effective_config=dict(config.final_config_dict),
                  config_sha256=hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest())
    init_seed(config['seed'] + config['local_rank'], config['reproducibility'])
    init_logger(config)
    protocol = verify_protocol(config, check_sha=True)
    stats = json.loads(STATS.read_text())
    dataset = PreciseHistoryDataset(config)
    train_ds, valid_ds, reserved = dataset.build()
    del reserved
    if (len(train_ds), len(valid_ds), train_ds.item_num) != (1062567, 23951, 7112):
        raise ValueError('Frozen chronological split mismatch')
    history_stats = verify_history_stats(train_ds, stats, config)
    train_data = TrainDataLoader(config, train_ds, None, shuffle=config['shuffle'])
    valid_data = FullSortEvalDataLoader(config, valid_ds, None, shuffle=False)
    model = ContextMamba3Rec(config, train_ds).to(config['device'])
    record.update(protocol=protocol, train_time_stats_sha256=STATS_SHA, manifest_sha256=MANIFEST_SHA,
                  verified_history_stats=history_stats, initial_backbone_sha256=backbone_hash(model),
                  parameter_count=sum(p.numel() for p in model.parameters()), history=[])
    if record['parameter_count'] != COUNTS[task['mode']]:
        raise ValueError('Scientific model count mismatch')
    if task['mode'] in ('uniform', 'routed'):
        record['expert_bank_initialization_sha256'] = tensor_hash(model.mechanisms.bank.state_dict())
    reference = json.loads((ROOT / plan()['historical_reference']['source_json']).read_text())
    if protocol['recbole_inter_sha256'] != reference['protocol']['recbole_inter_sha256']:
        raise ValueError('Historical dataset drift')
    replay = None
    if task['mode'] != 'separate_replay':
        replay = json.loads(paths(plan()['tasks'][0])['result'].read_text())
        if replay['status'] != 'PASS' or replay['source_hash'] != record['source_hash']:
            raise ValueError('Successful same-source replay required')
        for key in ('initial_backbone_sha256', 'protocol', 'train_time_stats_sha256', 'verified_history_stats'):
            if record[key] != replay[key]:
                raise ValueError(f'Replay provenance drift: {key}')
    if task['mode'] == 'routed':
        uniform = json.loads(paths(plan()['tasks'][3])['result'].read_text())
        if uniform['status'] != 'PASS' or record['expert_bank_initialization_sha256'] != uniform['expert_bank_initialization_sha256']:
            raise ValueError('Uniform/routed initial banks differ')
    cls = trainer_class(Trainer, valid_data, record, task_paths,
                        None if replay is None else replay['first_train_batch_sha256'])
    trainer = cls(config, model)
    trainer.saved_model_file = str(task_paths['checkpoint'])
    record.update(stage='TRAIN_VALID', rng_before_fit_sha256=rng_hash(),
                  tensorboard_path=str(trainer.tensorboard.log_dir), checkpoint_path=str(task_paths['checkpoint']),
                  epoch_indexing='zero-based', historical_rng_evidence='Historical JSON lacks pre-fit RNG/initialization hashes; procedure and synthetic parity checked')
    try:
        if replay is not None and replay['rng_before_fit_sha256'] != record['rng_before_fit_sha256']:
            raise ValueError('Pre-fit CPU/CUDA/Python/NumPy RNG drift')
        atomic_json(task_paths['result'], record)
        score, metrics = trainer.fit(train_data, valid_data, saved=True, show_progress=False)
        if record['best_valid_score'] != float(score) or record['best_valid_metrics'] != dict(metrics):
            raise ValueError('Saved best checkpoint/history differs from Trainer result')
        if len(trainer.train_loss_dict) != len(record['history']) or record['actual_epochs'] != len(record['history']):
            raise ValueError('Incomplete training history')
        if sha(task_paths['checkpoint']) != record['checkpoint_sha256']:
            raise ValueError('Checkpoint changed')
        record.update(status='PASS', stage='COMPLETED', finished_at=now())
    finally:
        trainer.tensorboard.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=[t['mode'] for t in plan()['tasks']], required=True)
    args = parser.parse_args()
    manifest = require_submission()
    evidence = json.loads(GPU_EVIDENCE.read_text())
    from .gpu_checks import all_pass
    from .temporal import COUNTS
    expected_checks = {'identity:' + mode for mode in COUNTS} | {
        'separate_wrapper','one_expert_reduction','uniform_router_reduction','router_signal',
        'finite_backward','expert_update','safe_roundtrip'}
    if (evidence['status'] != 'PASS' or evidence['source_hash'] != manifest['source_hash'] or
            evidence['execution_commit'] != os.environ['RUN_COMMIT'] or evidence['job_id'] != os.environ['SLURM_JOB_ID'] or
            (evidence['atol'], evidence['rtol']) != (1e-6,1e-5) or
            len(evidence['cases']) != 4 or
            {(r['training'],r['length']) for r in evidence['cases']} != {(t,l) for t in (False,True) for l in (50,64)} or
            any(not r['passed'] or set(r['checks']) != expected_checks or not all_pass(r['checks']) or r['counts'] != COUNTS
                for r in evidence['cases'])):
        raise ValueError('Current-study GPU gate PASS required')
    task = next(t for t in plan()['tasks'] if t['mode'] == args.mode)
    task_paths = paths(task)
    record = dict(**task, study_id=STUDY_ID, status='RUNNING', stage='PREFLIGHT',
                  execution_commit=os.environ['RUN_COMMIT'], job_id=os.environ['SLURM_JOB_ID'],
                  node=os.environ.get('SLURMD_NODENAME'),
                  core_hash=CORE, source_hash=manifest['source_hash'], gpu_evidence_sha256=sha(GPU_EVIDENCE),
                  started_at=now(), selection_split='VALID', evaluation_mode='full-ranking',
                  test_evaluation_count=0, paths={k: str(v) for k,v in task_paths.items()})
    acquire(task_paths, record)
    atomic_json(task_paths['result'], record)
    def interrupted(signum, frame):
        raise TimeoutError(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        os.chdir(task_paths['runtime'])
        train(task, record, task_paths)
    except BaseException as exc:
        record.update(status='FAIL', error=repr(exc), traceback=traceback.format_exc(), finished_at=now())
        raise
    finally:
        atomic_json(task_paths['result'], record)


if __name__ == '__main__':
    main()
