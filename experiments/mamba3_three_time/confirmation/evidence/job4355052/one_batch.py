"""Exactly one TRAIN batch, one forward/backward/Adam step per B/C model."""
import json
import traceback
import torch
from .config import BATCH, LOGS, settings, effective_check, B_COMMIT, PILOT_COMMIT
from .provenance import identity, create_record, atomic_json, now, imported_sources
from .state import (capture_rng, restore_rng, rng_record, precision, strict_transfer,
                    normalized, compare_named, ATOL, RTOL)
from experiments.mamba3_context_time.provenance import tensor_hash


def tensors(model):
    return {k: v.detach().cpu().clone() for k, v in normalized(model.state_dict()).items()}


def optimizer_tensors(optimizer, model):
    result = {}
    for name, parameter in normalized(dict(model.named_parameters())).items():
        for key, value in optimizer.state[parameter].items():
            if not torch.is_tensor(value):
                value = torch.tensor(value)
            result[name + '/' + key] = value.detach().cpu().clone()
    return result


def execute(record, save):
    from recbole.config import Config
    from recbole.utils import init_seed, init_logger
    from recbole.data.dataloader import TrainDataLoader
    from recbole.trainer import Trainer
    from experiments.mamba3_context_time.model import ContextMamba3Rec
    from experiments.mamba3_three_time.model import ThreeTimeMamba3Rec
    from experiments.mamba3_timeaware.dataset import PreciseHistoryDataset
    from experiments.mamba3_timeaware.run import verify_protocol, verify_history_stats
    from experiments.mamba3_time_confirmation.config import STATS, STATS_SHA, MANIFEST_SHA
    from experiments.mamba3_three_time.validation_pilot.provenance import runtime, assert_upstream

    record['runtime'] = runtime(True)
    if record['runtime']['matmul_allow_tf32'] is not False or record['runtime']['cudnn_allow_tf32'] is not True:
        raise ValueError('Frozen pilot precision flags required')
    config = Config(model=ThreeTimeMamba3Rec, config_dict=settings('dual', 2026, LOGS/'one_batch/C'))
    effective_check(config, 'dual', 2026)
    init_seed(config['seed'] + config['local_rank'], config['reproducibility'])
    init_logger(config)
    record.update(protocol=verify_protocol(config, check_sha=True), manifest_sha256=MANIFEST_SHA,
                  train_time_stats_sha256=STATS_SHA, precision=precision(),
                  effective_config=dict(config.final_config_dict))
    dataset = PreciseHistoryDataset(config)
    train_ds, unused_valid, reserved = dataset.build()
    del unused_valid, reserved
    if (len(train_ds), train_ds.item_num) != (1062567, 7112):
        raise ValueError('Frozen TRAIN split mismatch')
    record['verified_history_stats'] = verify_history_stats(train_ds, json.loads(STATS.read_text()), config)
    loader = TrainDataLoader(config, train_ds, None, shuffle=config['shuffle'])
    before = capture_rng()
    record['before_constructors'] = rng_record(loader)
    values_b = settings('dual', 2026, LOGS/'one_batch/B')
    values_b.pop('three_time_mode')
    values_b.update(time_mechanism_mode='separate', context_time_mode='separate_replay')
    cfg_b = Config(model=ContextMamba3Rec, config_dict=values_b)
    bcfg = json.loads(json.dumps(cfg_b.final_config_dict, default=str))
    ccfg = json.loads(json.dumps(config.final_config_dict, default=str))
    ignore = {'model', 'checkpoint_dir', 'three_time_mode', 'time_mechanism_mode', 'context_time_mode'}
    if {k: v for k, v in bcfg.items() if k not in ignore} != {k: v for k, v in ccfg.items() if k not in ignore}:
        raise ValueError('Diagnostic B/C config mismatch')
    left = ContextMamba3Rec(cfg_b, train_ds).to(config['device'])
    record['after_B_constructor'] = rng_record(loader)
    restore_rng(before)
    right = ThreeTimeMamba3Rec(config, train_ds).to(config['device'])
    record['after_C_constructor'] = rng_record(loader)
    record['initial_states_before_transfer'] = {}
    for name, model in (('B', left), ('C', right)):
        state = tensors(model)
        record['initial_states_before_transfer'][name] = dict(
            backbone_sha256=tensor_hash({k: v for k, v in state.items() if not k.startswith('times.')}),
            calibrators={k: dict(dtype=str(v.dtype), shape=list(v.shape), values=v.tolist())
                         for k, v in state.items() if k.startswith('times.calibrators.')})
    record['state_mapping'] = strict_transfer(left, right)
    initial = tensors(left)
    record['initial_mapped_sha256'] = tensor_hash(initial)
    record['imported_sources'] = imported_sources()
    tb, tc = Trainer(cfg_b, left), Trainer(config, right)
    try:
        if not isinstance(tb.optimizer, torch.optim.Adam) or not isinstance(tc.optimizer, torch.optim.Adam):
            raise ValueError('Adam required')
        options = lambda opt: [{k: v for k, v in group.items() if k != 'params'} for group in opt.param_groups]
        if tb.optimizer.state or tc.optimizer.state or options(tb.optimizer) != options(tc.optimizer):
            raise ValueError('Diagnostic optimizer initial state/options mismatch')
        record['optimizer_settings'] = options(tb.optimizer)
        record['loader_before_batch'] = rng_record(loader)
        batch = next(iter(loader)).to(config['device'])
        record['diagnostic_batches'] = 1
        if tuple(batch[left.ITEM_SEQ].shape) != (2048, 50):
            raise ValueError('Diagnostic batch must be [2048,50]')
        record['batch_sha256'] = tensor_hash(batch.interaction)
        record['rng_before_forward'] = rng_record(loader)
        common_rng = capture_rng()
        results = {}
        for name, model, optimizer in (('B', left, tb.optimizer), ('C', right, tc.optimizer)):
            record['stage'] = name + '_forward_backward_step'
            save()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            restore_rng(common_rng)
            if rng_record(loader) != record['rng_before_forward']:
                raise ValueError('RNG restore mismatch')
            assert_upstream()
            encoded = model._encode(batch)
            logits = encoded @ model.item_embedding.weight.T
            loss = model.loss_fct(logits, batch[model.POS_ITEM_ID])
            loss.backward()
            record['forward_backward_calls'] += 1
            gradients = {k: None if p.grad is None else p.grad.detach().cpu().clone()
                         for k, p in normalized(dict(model.named_parameters())).items()}
            outputs = dict(encoded=encoded.detach().cpu(), logits=logits.detach().cpu(), ce=loss.detach().cpu())
            optimizer.step()
            record['optimizer_steps'] += 1
            after = tensors(model)
            results[name] = dict(outputs=outputs, gradients=gradients, parameters=after,
                                 updates={k: after[k].double()-initial[k].double() for k in initial},
                                 optimizer=optimizer_tensors(optimizer, model))
            record.setdefault('rng_after_steps', {})[name] = rng_record(loader)
            del encoded, logits, loss
            save()
        record['checks'] = {k: compare_named(results['C'][k], results['B'][k])
                            for k in ('outputs', 'gradients', 'updates', 'parameters', 'optimizer')}
        record['checks_passed'] = all(v['passed'] for v in record['checks'].values())
        record['failed_stages'] = [k for k, v in record['checks'].items() if not v['passed']]
        save()
        if not record['checks_passed']:
            raise ValueError('Controlled step mismatch: ' + repr(record['failed_stages']))
    finally:
        tb.tensorboard.close()
        tc.tensorboard.close()


def main():
    row = dict(**identity(), status='RUNNING', stage='SETUP', started_at=now(), diagnostic_seed=2026,
               comparison_commits={'B': B_COMMIT, 'C': PILOT_COMMIT},
               historical_initialization_reconstructed=False, historical_metric_cause='UNKNOWN',
               diagnostic_batches=0, max_diagnostic_batches=1, forward_backward_calls=0, optimizer_steps=0, scientific_fits=0,
               structural_tolerance=dict(atol=ATOL, rtol=RTOL))
    create_record(BATCH, row)
    try:
        execute(row, lambda: atomic_json(BATCH, row))
        row.update(status='PASS', stage='COMPLETED')
    except BaseException as exc:
        row.update(status='FAIL', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        row['finished_at'] = now()
        atomic_json(BATCH, row)


if __name__ == '__main__':
    from experiments.mamba3_three_time.backends import selected
    with selected('upstream'):
        main()
