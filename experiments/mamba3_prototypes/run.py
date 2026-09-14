"""One allocation: frozen preflight -> TRAIN KMeans -> smoke -> TRAIN/VALID."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from recbole.config import Config
from recbole.data.dataset import SequentialDataset
from recbole.data.dataloader import TrainDataLoader, FullSortEvalDataLoader
from recbole.trainer import Trainer
from recbole.utils import init_logger, init_seed

from .config import load_config
from .diagnostics import PrototypeDiagnostics
from .model import ProtoMamba3Rec
from experiments.mamba3_baseline.model import Mamba3Rec

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'experiments/mamba3_baseline'))
from experiments.mamba3_baseline.run import verify_protocol, runtime_info
sys.path.pop(0)

PIN = 'e9594ce1c732d97440f0332fdc43170a2294dbfa'
CHECKPOINT = Path('/home/daryumin/iberdov/diplom_exp_mamba3_baseline/experiments/'
                  'mamba3_baseline/checkpoints/Mamba3Rec-Sep-12-2026_17-55-38.pth')
CHECKPOINT_SHA = 'd0bc3bb504daf5df068b6fd5bd635d6454aa223da01c006c63c78da193bb9dbe'
RUN_ID = 'mamba3_prototypes_validation_001'
INIT = HERE / 'runs/prototype_init_001.json'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    with temporary.open('w') as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def preflight(mode='kmeans'):
    if mode == 'random':
        settings = load_config(mode)
        config = Config(model=ProtoMamba3Rec, config_dict=settings)
        direct = json.loads(importlib.metadata.distribution('mamba-ssm').read_text('direct_url.json'))
        if direct['vcs_info']['commit_id'] != PIN:
            raise ValueError('Pinned Mamba mismatch')
        protocol = verify_protocol(config, check_sha=True)
        if protocol['recbole_inter_sha256'] != 'e275ded0b330c2827b49ccf567d6784452d6dcbf8cd719dc3009d36eadc2e2cc':
            raise ValueError('Protocol B mismatch')
        return config, protocol, None
    from sklearn.cluster import MiniBatchKMeans
    del MiniBatchKMeans
    source = ROOT / 'experiments/mamba3_baseline/runs/mamba3_validation_001.json'
    if sha256(source) != '22678554b5c915a4827c4805ac7bd204d5320cba468785c028715e0549767ea6':
        raise ValueError('Frozen vanilla VALID JSON changed')
    reference = json.loads(source.read_text())
    if reference['status'] != 'ok' or reference['best_valid_score'] != 0.0584:
        raise ValueError('Vanilla VALID mismatch')
    if not CHECKPOINT.is_file() or sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError('Exact vanilla checkpoint missing or changed; no retraining allowed')
    direct = json.loads(importlib.metadata.distribution('mamba-ssm').read_text('direct_url.json'))
    if direct['vcs_info']['commit_id'] != PIN or reference['upstream']['pinned_commit'] != PIN:
        raise ValueError('Pinned Mamba mismatch')
    settings = load_config()
    if (settings['prototype_k'], settings['prototype_temperature'], settings['prototype_random_state'],
        settings['prototype_normalize_init'], settings['prototype_kmeans_batch_size']) != (8, 1.0, 2026, True, 2048):
        raise ValueError('Fixed prototype initialization settings changed')
    if settings['backbone_initialization'] != 'scratch':
        raise ValueError('This run uses scratch backbone; frozen encoder is for KMeans only')
    config = Config(model=ProtoMamba3Rec, config_dict=settings)
    architecture = dict(hidden_size=64, num_layers=2, dropout_prob=.2, use_ffn=True,
                        ffn_inner_size=256, mamba3_d_state=128, mamba3_expand=2,
                        mamba3_headdim=64, mamba3_ngroups=1, mamba3_rope_fraction=.5,
                        mamba3_chunk_size=64, mamba3_is_mimo=False, mamba3_mimo_rank=4,
                        mamba3_is_outproj_norm=False, mamba3_dtype='bfloat16')
    saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    for key, value in architecture.items():
        if config[key] != value or reference['model'][key] != value or saved['config'][key] != value:
            raise ValueError(f'Frozen architecture mismatch: {key}')
    for key in ('seed', 'epochs', 'stopping_step', 'eval_step', 'learner', 'learning_rate',
                'train_batch_size', 'eval_batch_size', 'loss_type', 'MAX_ITEM_LIST_LENGTH', 'weight_decay'):
        if config[key] != saved['config'][key]:
            raise ValueError(f'Frozen training setting mismatch: {key}')
    if saved['epoch'] != 15 or float(saved['best_valid_score']) != .0584:
        raise ValueError('Frozen checkpoint selection mismatch')
    protocol = verify_protocol(config, check_sha=True)
    if protocol['recbole_inter_sha256'] != 'e275ded0b330c2827b49ccf567d6784452d6dcbf8cd719dc3009d36eadc2e2cc':
        raise ValueError('Protocol B mismatch')
    return config, protocol, saved


class DiagnosticTrainer(Trainer):
    def _valid_epoch(self, valid_data, show_progress=False):
        collector = PrototypeDiagnostics()
        self.model.prototypes.diagnostic_collector = collector
        try:
            result = super()._valid_epoch(valid_data, show_progress=show_progress)
            self.last_diagnostics = collector.result(self.model.prototypes)
            return result
        finally:
            self.model.prototypes.diagnostic_collector = None


def fresh_model(config, train_dataset, centroids):
    init_seed(config['seed'] + config['local_rank'], config['reproducibility'])
    model = ProtoMamba3Rec(config, train_dataset).to(config['device'])
    model.prototypes.set_centroids(centroids)
    return model


def smoke(config, train_dataset, valid_dataset, centroids):
    model = fresh_model(config, train_dataset, centroids)
    train_data = TrainDataLoader(config, train_dataset, None, shuffle=config['shuffle'])
    valid_subset = valid_dataset.copy(valid_dataset.inter_feat[:64])
    valid_data = FullSortEvalDataLoader(config, valid_subset, None, shuffle=False)
    batch = train_dataset.inter_feat[:8].to(config['device'])
    model.eval()
    with torch.no_grad():
        h = Mamba3Rec.forward(model, batch[model.ITEM_SEQ], batch[model.ITEM_SEQ_LEN])
        actual = model(batch[model.ITEM_SEQ], batch[model.ITEM_SEQ_LEN])
        if not torch.equal(h, actual):
            raise ValueError('Exact identity gate failed before smoke')
    trainer = DiagnosticTrainer(config, model)
    smoke_name = 'random_smoke_checkpoint.pth' if config['prototype_initialization'] == 'random' else 'smoke_checkpoint.pth'
    trainer.saved_model_file = str(HERE / 'slurm_logs' / smoke_name)
    original = model.prototypes.P.detach().clone()
    losses = []
    model.train()
    for index, interaction in enumerate(train_data):
        trainer.optimizer.zero_grad(set_to_none=True)
        loss = model.calculate_loss(interaction.to(config['device']))
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite smoke loss')
        loss.backward()
        if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise ValueError('Nonfinite smoke gradient')
        trainer.optimizer.step()
        losses.append(loss.item())
        if index == 1:
            break
    if len(losses) != 2 or torch.equal(original, model.prototypes.P):
        raise ValueError('Prototypes did not learn in smoke')
    trainer.eval_collector.data_collect(train_data)
    _, metrics = trainer._valid_epoch(valid_data)
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError('Nonfinite smoke VALID metrics')
    trainer._save_checkpoint(0)
    checkpoint = torch.load(trainer.saved_model_file, map_location=config['device'], weights_only=False)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.load_other_parameter(checkpoint.get('other_parameter'))
    if not all(torch.equal(value, model.state_dict()[key]) for key, value in checkpoint['state_dict'].items()):
        raise ValueError('Smoke checkpoint roundtrip failed')
    trainer.tensorboard.close()
    return dict(status='PASS', train_batches=2, losses=losses, exact_identity=True,
                prototype_update=True, finite_gradients=True, checkpoint_roundtrip='PASS',
                validation_scope='64 VALID histories, full-ranking', test_evaluation_count=0)


def main(mode='kmeans'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    config, protocol, saved = preflight(mode)
    print(json.dumps(dict(preflight='PASS', initialization=mode, pinned_mamba_commit=PIN)), flush=True)
    if args.preflight_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    run_id = 'mamba3_prototypes_random_validation_001' if mode == 'random' else RUN_ID
    init_path = HERE / 'runs/prototype_random_init_001.json' if mode == 'random' else INIT
    result_path = HERE / 'runs' / (run_id + '.json')
    if result_path.exists() or init_path.exists():
        raise FileExistsError('Existing scientific/init artifact; no automatic retry')
    logdir = HERE / 'slurm_logs'
    logdir.mkdir(parents=True, exist_ok=True)
    with (logdir / (run_id + '.lock')).open('x') as handle:
        handle.write(os.environ.get('SLURM_JOB_ID', 'local'))
    commit = os.environ['RUN_COMMIT']
    result = dict(run_id=run_id, status='running', stage='prototype_initialization', git_commit=commit,
                  job_id=os.environ.get('SLURM_JOB_ID'), test_evaluation_count=0, TEST='NOT_RUN',
                  protocol=protocol, K=8, temperature=1.0,
                  backbone_initialization='scratch', started_at=datetime.now(timezone.utc).isoformat(),
                  pinned_mamba_commit=PIN)
    if mode == 'random':
        from .random_init import metadata
        result.update(metadata())
    else:
        result.update(prototype_initialization='kmeans', initialization='TRAIN-only MiniBatchKMeans',
                      vanilla_checkpoint_path=str(CHECKPOINT), vanilla_checkpoint_sha256=CHECKPOINT_SHA)
    save_json(result_path, result)
    try:
        init_seed(config['seed'] + config['local_rank'], config['reproducibility'])
        init_logger(config)
        dataset = SequentialDataset(config)
        train_dataset, valid_dataset, reserved = dataset.build()
        del reserved  # Never create a TEST loader.
        if (len(train_dataset), len(valid_dataset), train_dataset.item_num - 1) != (1062567, 23951, 7111):
            raise ValueError('Protocol B sequential split mismatch')
        if mode == 'random':
            from .random_init import random_prototypes, metadata
            centroids = random_prototypes(config['prototype_random_state'], config['prototype_random_init_std'])
            init_artifact = dict(**metadata(), prototypes=centroids.tolist(), git_commit=commit)
        else:
            from .prototype_init import initialize_from_train
            vanilla = Mamba3Rec(config, train_dataset).to(config['device'])
            vanilla.load_state_dict(saved['state_dict'], strict=True)
            vanilla.load_other_parameter(saved.get('other_parameter'))
            centroids, counts = initialize_from_train(vanilla, train_dataset, config)
            if counts['train_histories'] != len(train_dataset):
                raise ValueError('Incomplete TRAIN-only initialization')
            del vanilla, saved
            init_artifact = dict(split='TRAIN_ONLY', valid_examples_used=False, test_examples_used=False,
                                 target_inputs_used=False, normalized_hidden_states=True,
                                 K=8, temperature=1.0, random_state=2026, n_init=3,
                                 batch_size=2048, reassignment_ratio=.01, passes=1,
                                 source_checkpoint=str(CHECKPOINT), source_checkpoint_sha256=CHECKPOINT_SHA,
                                 protocol=protocol, git_commit=commit, centroids=centroids.tolist(),
                                 sklearn_version=importlib.metadata.version('scikit-learn'), **counts)
        save_json(init_path, init_artifact)
        result.update(prototype_init_artifact=str(init_path), prototype_init_sha256=sha256(init_path), stage='smoke')
        save_json(result_path, result)
        smoke_result = smoke(config, train_dataset, valid_dataset, centroids)
        smoke_name = 'prototype_random_smoke_001.json' if mode == 'random' else 'prototype_smoke_001.json'
        save_json(HERE / 'runs' / smoke_name, smoke_result)
        result.update(smoke=smoke_result, stage='TRAIN_VALID')
        # Discard all smoke updates. Reset both model RNG and loader generators.
        model = fresh_model(config, train_dataset, centroids)
        parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if parameter_count != 610440 + 8769:
            raise ValueError(f'Parameter parity failed: {parameter_count}')
        result.update(parameter_parity=True, expected_trainable_parameters=619209)
        train_data = TrainDataLoader(config, train_dataset, None, shuffle=config['shuffle'])
        valid_data = FullSortEvalDataLoader(config, valid_dataset, None, shuffle=False)
        trainer = DiagnosticTrainer(config, model)
        result.update(parameter_count=sum(p.numel() for p in model.parameters()), runtime=runtime_info(),
                      evaluation_mode='full-ranking', selection_metric='VALID NDCG@10',
                      checkpoint_path=trainer.saved_model_file, prototype_diagnostics_history=[])
        def progress(epoch, score):
            diagnostics = trainer.last_diagnostics
            result['prototype_diagnostics_history'].append(dict(epoch=epoch, **diagnostics))
            if score > result.get('best_valid_score', -float('inf')):
                result.update(best_valid_score=float(score), best_prototype_diagnostics=diagnostics)
            result.update(last_epoch=epoch, actual_epochs=epoch + 1, final_prototype_diagnostics=diagnostics)
            save_json(result_path, result)
        score, metrics = trainer.fit(train_data, valid_data, saved=True, show_progress=False, callback_fn=progress)
        best = torch.load(trainer.saved_model_file, map_location='cpu', weights_only=False)
        result.update(status='PASS', stage='finished', best_epoch=int(best['epoch']), epoch_indexing='zero-based',
                      actual_epochs=len(trainer.train_loss_dict), best_valid_score=float(score),
                      best_valid_metrics=dict(metrics), finished_at=datetime.now(timezone.utc).isoformat())
        if mode == 'kmeans' and sha256(CHECKPOINT) != CHECKPOINT_SHA:
            raise ValueError('Frozen vanilla checkpoint changed')
        save_json(result_path, result)
        print(json.dumps(result, indent=2))
    except Exception as exc:
        result.update(status='FAIL', error=repr(exc))
        save_json(result_path, result)
        raise


if __name__ == '__main__':
    main()
