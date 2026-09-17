"""Read-only publication calculations from saved JSON; no model imports."""

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def read(path):
    return json.loads(path.read_text())


def records():
    old = read(ROOT / 'reports/assets/time_mechanisms/extracted.json')['runs']
    result = []
    for source in read(HERE / 'sources.json')['sources']:
        path = ROOT / source['source_json']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source['sha256'], str(path)
        row = read(path)
        assert row['status'] == 'PASS' and row['test_evaluation_count'] == 0
        assert row['run_id'] == source['run_id'] and row['config']['seed'] == source['seed']
        assert row.get('submission_commit', row.get('git_commit')) == source['execution_commit']
        history = row['history'] if source['kind'] == 'new_validation' else old[source['mode']]['history']
        assert [h['epoch'] for h in history] == list(range(row['actual_epochs']))
        assert all(math.isfinite(h['valid_ndcg10']) for h in history)
        assert row['best_valid_score'] == max(h['valid_ndcg10'] for h in history)
        assert history[row['best_epoch']]['valid_ndcg10'] == row['best_valid_metrics']['ndcg@10']
        result.append(dict(source=source, raw=row, history=history,
                           first27=max(h['valid_ndcg10'] for h in history if h['epoch'] < 27)))
    assert len(result) == 11 and len({r['source']['run_id'] for r in result}) == 11
    return result


def paired(rows, prefix=False):
    values = {mode: [r[mode + ('_first27' if prefix else '')] for r in rows]
              for mode in ('shared', 'separate')}
    delta = [b - a for a, b in zip(values['shared'], values['separate'])]
    stats = lambda x: dict(mean=statistics.mean(x), sample_std=statistics.stdev(x))
    return dict(shared=stats(values['shared']), separate=stats(values['separate']),
                paired_delta=stats(delta), positive=sum(d > 0 for d in delta), n=len(rows),
                relative_pct=100 * (statistics.mean(values['separate']) / statistics.mean(values['shared']) - 1))


def build():
    data = records()
    by_key = {(r['source']['mode'], r['source']['seed']): r for r in data}
    rows = []
    for seed in range(2026, 2031):
        item = dict(seed=seed)
        for mode in ('shared', 'separate'):
            r = by_key[mode, seed]
            item[mode] = r['raw']['best_valid_metrics']['ndcg@10']
            item[mode + '_first27'] = r['first27']
        item['delta'] = item['separate'] - item['shared']
        rows.append(item)
    result = dict(pairs=rows, full_horizon=paired(rows), new_seeds=paired(rows[1:]), first27=paired(rows, True))
    result['constant_gap'] = [dict(mode=mode, seed=2026, first27=by_key[mode, 2026]['first27'],
                                 full=by_key[mode, 2026]['raw']['best_valid_score'])
                              for mode in ('separate', 'separate_constant_gap')]
    saved = read(HERE / 'summary.json')
    assert saved['n_available'] == 5 and not saved['incomplete_study'] and not saved['unavailable']
    for key, original in [('full_horizon', saved['full_horizon']['ndcg@10']), ('first27', saved['first27'])]:
        for mode in ('shared', 'separate', 'paired_delta'):
            for stat in ('mean', 'sample_std'):
                assert math.isclose(result[key][mode][stat], original[mode][stat], abs_tol=1e-15)
        assert result[key]['positive'] == original['positive'] == 5
    # User-supplied control values detect wrong sources; they never generate results.
    assert [(r['shared'], r['separate']) for r in rows] == [
        (.0605, .0633), (.0620, .0628), (.0629, .0635), (.0617, .0623), (.0614, .0625)]
    for group, expected in [('full_horizon', (.061700, .062880, .001180)),
                            ('new_seeds', (.062000, .062775, .000775)),
                            ('first27', (.061000, .062060, .001060))]:
        for mode, value in zip(('shared', 'separate', 'paired_delta'), expected):
            assert math.isclose(result[group][mode]['mean'], value, abs_tol=1e-15)
    assert [(r['first27'], r['full']) for r in result['constant_gap']] == [(.0614, .0633), (.0586, .0586)]
    return result


def csv_rows():
    with (ROOT / 'experiments/results.csv').open(newline='') as f:
        header = next(csv.reader(f))
    writer = csv.DictWriter(sys.stdout, fieldnames=header, lineterminator='\n')
    for item in records():
        source, raw = item['source'], item['raw']
        if source['kind'] != 'new_validation':
            continue
        row = dict(record_type='experiment', source='ours', run_id=raw['run_id'],
                   model=raw['effective_config']['model'], model_variant=raw['mode'], dataset='KuaiRand',
                   protocol='B', split='validation', evaluation='full_7111_items', status='completed',
                   seed=raw['seed'], train_candidates='full_softmax', item_universe=raw['protocol']['items'],
                   best_epoch=raw['best_epoch'], actual_epochs=raw['actual_epochs'],
                   validation_ndcg10=raw['best_valid_metrics']['ndcg@10'], test_evaluation_count=0,
                   git_commit=raw['submission_commit'], notes_path='reports/MAMBA3_TIME_MECHANISMS_RESULTS.md',
                   test_used='no', source_json=source['source_json'])
        for family, raw_family in [('HR', 'hit'), ('Recall', 'recall'), ('NDCG', 'ndcg')]:
            for k in (5, 10, 20, 50):
                row[f'{family}@{k}'] = raw['best_valid_metrics'][f'{raw_family}@{k}']
        writer.writerow(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv-rows', action='store_true', help='Print the nine registry rows; do not write files')
    args = parser.parse_args()
    summary = build()
    if args.csv_rows:
        csv_rows()
    else:
        print(json.dumps(summary, indent=2, allow_nan=False))
