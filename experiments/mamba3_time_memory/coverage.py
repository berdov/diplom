"""Frozen TRAIN-only address coverage, on CPU before any GPU scientific job.

Dataset.build() describes the existing three splits; only TRAIN input fields are
read. No loaders, recommender instances, model forwards or target fields occur.
"""
import hashlib
import json
import os
import traceback

from experiments.mamba3_mimo_time.records import create, update, now, digest

MAX_EXAMPLES = 10000
BATCH_SIZE = 256
INDEX_RULE = 'floor(i*(N-1)/(n-1)) for i=0..n-1; n=min(N,10000); [0] when n=1'
UNINFORMATIVE = 'UNINFORMATIVE_ON_AUDITED_TRAIN_SAMPLE'


def fixed_indices(population, maximum=MAX_EXAMPLES):
    if type(population) is not int or population < 1:
        raise ValueError('Positive integer TRAIN population required')
    if type(maximum) is not int or not 1 <= maximum <= MAX_EXAMPLES:
        raise ValueError('Coverage is limited to 10000 fixed TRAIN examples')
    size = min(population, maximum)
    return [0] if size == 1 else [i * (population - 1) // (size - 1) for i in range(size)]


def tensor_fingerprint(value):
    """Explicit dtype/shape plus row-major bytes, only selected input fields."""
    value = value.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(json.dumps(dict(shape=list(value.shape), dtype=str(value.dtype)),
                        sort_keys=True, separators=(',', ':')).encode())
    h.update(b'\x00')
    h.update(value.numpy().tobytes(order='C'))
    return h.hexdigest()


def audit_histories(items, lengths, timestamps, indices, batch_size=BATCH_SIZE):
    import torch
    from .diagnostics import AddressDiagnostics
    if items.device.type != 'cpu' or lengths.device.type != 'cpu' or timestamps.device.type != 'cpu':
        raise ValueError('Coverage must stay on CPU')
    if timestamps.dtype != torch.float64 or items.shape != timestamps.shape or items.ndim != 2:
        raise ValueError('Expected precise TRAIN input histories [B,L]')
    if lengths.shape != (items.shape[0],) or len(indices) != items.shape[0]:
        raise ValueError('TRAIN sample length/index mismatch')
    if items.shape[1] > 50 or not 1 <= len(indices) <= MAX_EXAMPLES or batch_size < 1:
        raise ValueError('Coverage scope/batch drift')
    valid = items != 0
    expected = torch.arange(items.shape[1])[None] < lengths[:, None]
    if (lengths < 1).any() or not torch.equal(valid, expected):
        raise ValueError('Expected nonempty right-padded TRAIN input histories')
    collector = AddressDiagnostics('TRAIN input histories only, CPU metadata-only audit')
    for start in range(0, len(indices), batch_size):
        stop = min(start + batch_size, len(indices))
        collector.update(timestamps[start:stop], valid[start:stop],
                         example_ids=indices[start:stop])
    result = collector.result()
    result['status'] = UNINFORMATIVE if result['selected_sets_equal_count'] == len(indices) else 'PASS'
    result['decision_rule'] = 'all audited selected sets identical => UNINFORMATIVE; otherwise record actual overlap; no tuned percentage threshold'
    result['selected_input_hashes'] = dict(item_history=tensor_fingerprint(items),
                                           history_length=tensor_fingerprint(lengths),
                                           precise_history_timestamps=tensor_fingerprint(timestamps))
    result['selected_records_sha256'] = digest(result['selected_input_hashes'])
    return result


def run(base):
    import torch
    from . import config as c
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_available() or torch.cuda.is_initialized():
        raise RuntimeError('Coverage requires its own subprocess with CUDA hidden before Torch import')
    c.plan()
    result = dict(base)
    result.update(status='RUNNING', stage='TRAIN_ADDRESS_COVERAGE', started_at=now(),
                  scientific_fit_started=False, scientific_fits_started=0,
                  model_instances_created=0, model_forward_count=0,
                  train_loaders_created=0, valid_loaders_created=0, test_loaders_created=0,
                  test_evaluation_count=0, target_fields_read=False,
                  split='TRAIN', anchors=[1, 4, 16, 32], reference_ms=838393,
                  max_history=50, max_examples=MAX_EXAMPLES, batch_size=BATCH_SIZE,
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  cuda_initialized_before=False, cuda_available=False)
    create(c.COVERAGE, result)
    try:
        from recbole.config import Config
        from recbole.utils import init_seed
        from experiments.mamba3_three_time.model import ThreeTimeMamba3Rec
        from experiments.mamba3_timeaware.dataset import PreciseHistoryDataset
        from experiments.mamba3_timeaware.run import verify_protocol
        from experiments.mamba3_time_confirmation.config import MANIFEST_SHA, STATS_SHA
        torch.set_num_threads(4)
        values = c.settings('no_memory', device='cpu')
        cfg = Config(model=ThreeTimeMamba3Rec, config_dict=values)
        if torch.cuda.is_initialized() or cfg['device'].type != 'cpu' or cfg['use_gpu']:
            raise RuntimeError('CPU coverage Config reopened CUDA')
        init_seed(cfg['seed'] + cfg['local_rank'], cfg['reproducibility'])
        protocol = verify_protocol(cfg, check_sha=True)
        dataset = PreciseHistoryDataset(cfg)
        train_ds, reserved_valid, reserved_test = dataset.build()
        del reserved_valid, reserved_test
        population = len(train_ds)
        if population != 1062567 or train_ds.item_num != 7112:
            raise ValueError('Frozen TRAIN data count/mapping changed')
        indices = fixed_indices(population)
        tensor_indices = torch.tensor(indices, dtype=torch.long)
        item_field = cfg['ITEM_ID_FIELD'] + cfg['LIST_SUFFIX']
        time_field = cfg['TIME_FIELD'] + cfg['LIST_SUFFIX']
        length_field = cfg['ITEM_LIST_LENGTH_FIELD']
        items = train_ds.inter_feat[item_field][tensor_indices]
        timestamps = train_ds.inter_feat[time_field][tensor_indices]
        lengths = train_ds.inter_feat[length_field][tensor_indices]
        # Input history only: no scalar item target, scalar timestamp, or IDs read.
        audit = audit_histories(items, lengths, timestamps, indices)
        result.update(audit, train_population=population, sampled_examples=len(indices),
                      index_rule=INDEX_RULE, selected_train_indices=indices,
                      selected_indices_sha256=digest(indices),
                      input_field_names_read=[item_field, time_field, length_field],
                      protocol=protocol, manifest_sha256=MANIFEST_SHA,
                      frozen_train_time_stats_sha256=STATS_SHA,
                      config_label='ThreeTimeMamba3Rec, CPU data preparation only',
                      split_preparation='existing dataset.build describes all splits; only TRAIN input histories are inspected',
                      cuda_initialized_after=torch.cuda.is_initialized())
        if result['cuda_initialized_after']:
            raise RuntimeError('Coverage initialized CUDA')
    except BaseException as exc:
        result.update(status='FAIL', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        result['finished_at'] = now()
        update(c.COVERAGE, result)
    return result


def main():
    from . import provenance as p
    commit = os.environ.get('RUN_COMMIT', '')
    if len(commit) != 40 or any(x not in '0123456789abcdef' for x in commit):
        raise ValueError('Coverage requires frozen exact RUN_COMMIT')
    result = run(p.bindings(commit, p.verify()))
    print(json.dumps(dict(status=result['status'], sampled_examples=result['sampled_examples'],
                          selected_sets_equal_fraction=result['selected_sets_equal_fraction'])))
    if result['status'] != 'PASS':
        raise SystemExit('TRAIN coverage is uninformative; GPU submission is blocked')


if __name__ == '__main__':
    main()
