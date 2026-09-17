"""Read-only aggregation of complete matched VALID seeds; no evaluation."""
import json
import math
import statistics

from .config import (CORE, COUNTS, HERE, MANIFEST_SHA, ROOT, STATS_SHA, STUDY_ID,
                     plan, scientific_settings, settings)
from .provenance import sha, verify

METRICS = [f'{family}@{k}' for family in ('hit', 'ndcg', 'recall') for k in (5,10,20,50)]


def summary(values):
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                sample_std=statistics.stdev(values) if len(values)>1 else None)


def paired(shared, separate):
    delta = {str(seed):separate[seed]-shared[seed] for seed in sorted(shared)}
    return dict(shared=summary(list(shared.values())), separate=summary(list(separate.values())),
                paired_deltas=delta, paired_delta=summary(list(delta.values())),
                positive=sum(v>0 for v in delta.values()), negative=sum(v<0 for v in delta.values()),
                zero=sum(v==0 for v in delta.values()))


def validate(row, task, source_hash=None, reused=False):
    if row.get('status') != 'PASS':
        raise ValueError('Only completed PASS rows may be aggregated')
    if row.get('test_evaluation_count') != 0 or row.get('evaluation_mode') != 'full-ranking':
        raise ValueError('Wrong evaluation scope')
    if row.get('selection_split', 'VALID' if reused else None) != 'VALID' or row.get('split','validation') not in ('VALID','validation'):
        raise ValueError('VALID/TEST mixing forbidden')
    if row['run_id'] != task['run_id'] or row['config']['seed'] != task['seed']:
        raise ValueError('Unexpected run/seed')
    if scientific_settings(row['config']) != scientific_settings(settings(task)):
        raise ValueError('Training setting drift')
    if not reused:
        if (row['mode'],row['seed'],row['study_id'],row['core_fingerprint'],row['study_source_hash']) != (
                task['mode'],task['seed'],STUDY_ID,CORE,source_hash):
            raise ValueError('Study provenance drift')
        if (row['manifest_sha256'],row['train_time_stats_sha256'],row['parameter_count'],row['array_task_id']) != (
                MANIFEST_SHA,STATS_SHA,COUNTS[task['mode']],task['task_index']):
            raise ValueError('Data/parameter/array provenance drift')
        if [h['epoch'] for h in row['history']] != list(range(row['actual_epochs'])):
            raise ValueError('Incomplete successful epoch history')
        best=row['history'][row['best_epoch']]
        if best['valid_metrics']!=row['best_valid_metrics'] or best['diagnostics']!=row['best_diagnostics']:
            raise ValueError('Best-epoch diagnostics/metrics mismatch')
    metrics = row['best_valid_metrics']
    if set(metrics) != set(METRICS) or not all(math.isfinite(float(v)) for v in metrics.values()):
        raise ValueError('Missing/nonfinite metrics')
    if not 0 <= row['best_epoch'] < row['actual_epochs']:
        raise ValueError('Invalid epoch provenance')


def aggregate(records, histories, expected_seeds=(2026,2027,2028,2029,2030)):
    by_mode = {'shared':{}, 'separate':{}}
    control = None
    for row in records:
        if row.get('test_evaluation_count') != 0 or row.get('selection_split') != 'VALID':
            raise ValueError('No TEST rows allowed')
        mode, seed = row['mode'], row['seed']
        if row['status'] != 'PASS':
            raise ValueError('Incomplete result supplied as successful')
        if mode == 'separate_constant_gap':
            if seed != 2026 or control is not None:
                raise ValueError('Unexpected/duplicate exploratory control')
            control = dict(run_id=row['run_id'],seed=seed,metrics=row['best_valid_metrics'],
                           aggregation='one exploratory run; not part of paired study')
            continue
        if mode not in by_mode or seed not in expected_seeds or seed in by_mode[mode]:
            raise ValueError('Unexpected/duplicate matched seed')
        if not all(math.isfinite(float(v)) for v in row['best_valid_metrics'].values()):
            raise ValueError('Nonfinite metric')
        by_mode[mode][seed] = row
    seeds = sorted(set(by_mode['shared']) & set(by_mode['separate']))
    result = dict(study_id=STUDY_ID,selection_split='VALID',test_evaluation_count=0,
                  n_expected=len(expected_seeds), n_available=len(seeds),
                  incomplete_study=len(seeds)!=len(expected_seeds), matched_seeds=seeds,
                  missing={mode:[s for s in expected_seeds if s not in rows] for mode,rows in by_mode.items()},
                  full_horizon={}, first27=dict(available=False), exploratory_control=control)
    for metric in METRICS:
        result['full_horizon'][metric] = paired(
            {s:by_mode['shared'][s]['best_valid_metrics'][metric] for s in seeds},
            {s:by_mode['separate'][s]['best_valid_metrics'][metric] for s in seeds})
    prefix = {mode:{} for mode in by_mode}
    for mode in by_mode:
        for seed in seeds:
            row=by_mode[mode][seed]; history=histories.get(row['run_id'], [])
            if ([h['epoch'] for h in history] != list(range(row['actual_epochs']))
                    or not all(math.isfinite(h['valid_ndcg10']) for h in history)):
                return result
            prefix[mode][seed] = max(h['valid_ndcg10'] for h in history if h['epoch']<27)
    if seeds:
        result['first27'] = dict(available=True, metric='ndcg@10', **paired(prefix['shared'],prefix['separate']))
    return result


def main():
    manifest = verify()
    records, histories, unavailable = [], {}, []
    original_histories = json.loads((ROOT/'reports/assets/time_mechanisms/extracted.json').read_text())['runs']
    for task in plan()['reuse']:
        row=json.loads((ROOT/task['source_json']).read_text())
        validate(row, task, reused=True)
        row.update(mode=task['mode'],seed=task['seed'],selection_split='VALID')  # In memory only.
        records.append(row)
        histories[row['run_id']] = original_histories[task['mode']]['history']
    for task in plan()['tasks']:
        path=HERE/'runs'/(task['run_id']+'.json')
        if not path.exists():
            unavailable.append(dict(run_id=task['run_id'],status='MISSING'))
            continue
        row=json.loads(path.read_text())
        if row.get('status') != 'PASS':
            unavailable.append(dict(run_id=task['run_id'],status=row.get('status','UNKNOWN')))
            continue
        validate(row,task,source_hash=manifest['source_hash'])
        records.append(row); histories[row['run_id']] = row['history']
    result=aggregate(records,histories)
    result.update(unavailable=unavailable,study_source_hash=manifest['source_hash'])
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
