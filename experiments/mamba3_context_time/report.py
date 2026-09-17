"""Read-only validation and aggregation; no fit/evaluate/model imports."""
import json
import math
from pathlib import Path

from .config import CORE, HERE, ROOT, STUDY_ID, SUMMARY, paths, plan, scientific_settings, settings
from .provenance import atomic_json, sha, verify
from experiments.mamba3_time_confirmation.config import MANIFEST_SHA, STATS_SHA

METRICS = {f'{family}@{k}' for family in ('hit', 'ndcg', 'recall') for k in (5,10,20,50)}


def validate(row, task, source_hash):
    expected = {**task, 'study_id': STUDY_ID, 'core_hash': CORE, 'source_hash': source_hash,
                'selection_split': 'VALID', 'evaluation_mode': 'full-ranking', 'test_evaluation_count': 0}
    if any(row.get(k) != v for k, v in expected.items()) or row.get('status') != 'PASS':
        raise ValueError('Invalid completed run identity/scope')
    if scientific_settings(row['config']) != scientific_settings(settings(task)):
        raise ValueError('Configuration drift in result')
    reference=json.loads((ROOT/plan()['historical_reference']['source_json']).read_text())
    if (row['manifest_sha256'] != MANIFEST_SHA or row['train_time_stats_sha256'] != STATS_SHA or
            row['protocol']['recbole_inter_sha256'] != reference['protocol']['recbole_inter_sha256']):
        raise ValueError('Dataset provenance drift')
    counts = dict(separate_replay=610572, dense11=611214, dense12=611284, uniform=610968, routed=611232)
    if row['parameter_count'] != counts[task['mode']]:
        raise ValueError('Parameter count differs')
    history = row['history']
    if not history or not 1 <= row['actual_epochs'] <= 300 or [h['epoch'] for h in history] != list(range(row['actual_epochs'])):
        raise ValueError('Incomplete history')
    for h in history:
        if set(h['valid_metrics']) != METRICS or not all(math.isfinite(float(x)) for x in h['valid_metrics'].values()):
            raise ValueError('Missing/nonfinite VALID metrics')
        if not math.isfinite(h['train_loss']) or not math.isfinite(h['valid_ndcg10']):
            raise ValueError('Nonfinite history')
        if h['valid_ndcg10'] != h['valid_metrics']['ndcg@10'] or any(not math.isfinite(h[k]) or h[k]<0 for k in ('train_seconds','valid_seconds')):
            raise ValueError('Inconsistent score/timing')
    maximum = max(h['valid_ndcg10'] for h in history)
    best_index = max(h['epoch'] for h in history if h['valid_ndcg10'] == maximum)
    if row['best_epoch'] != best_index or row['best_valid_score'] != maximum:
        raise ValueError('Last-equal-maximum epoch mismatch')
    best = history[best_index]
    if best['valid_metrics'] != row['best_valid_metrics'] or best['diagnostics'] != row['best_diagnostics']:
        raise ValueError('Best metrics/diagnostics mismatch')


def summarize(records, source_hash):
    rows = []
    replay = records.get('separate_replay', {})
    for task in plan()['tasks']:
        row = records.get(task['mode'])
        view = dict(**task, status='NOT_RUN' if row is None else row.get('status', 'INVALID'))
        if row is not None and row.get('status') == 'PASS':
            validate(row, task, source_hash)
            history = row['history']
            view.update(parameters=row['parameter_count'], ndcg10=row['best_valid_score'],
                hr10=row['best_valid_metrics']['hit@10'], best_epoch=row['best_epoch'], actual_epochs=row['actual_epochs'],
                first27=max(h['valid_ndcg10'] for h in history[:27]),
                train_seconds=sum(h['train_seconds'] for h in history), valid_seconds=sum(h['valid_seconds'] for h in history),
                delta_replay=(row['best_valid_score']-replay['best_valid_score'] if replay.get('status') == 'PASS' else None))
        elif row is not None:
            view['error'] = row.get('error', 'Stage did not complete')
        rows.append(view)
    complete = all(r['status'] == 'PASS' for r in rows)
    interpretation = ['Неполные/ошибочные runs: победитель не выбирается.']
    comparisons = {}
    if complete:
        v = {r['mode']: r['ndcg10'] for r in rows}
        comparisons = {f'routed_minus_{k}': v['routed']-v[k] for k in ('separate_replay','dense11','dense12','uniform')}
        comparisons.update({f'{k}_minus_separate_replay': v[k]-v['separate_replay'] for k in ('dense11','dense12')})
        interpretation = []
        if v['routed'] > max(v[k] for k in ('dense11','dense12','uniform')):
            interpretation.append('Предварительный аргумент в пользу условной экспертной факторизации на одном seed.')
        if max(v['dense11'], v['dense12']) >= v['routed']:
            interpretation.append('Польза MoE относительно dense controls не показана.')
        if v['uniform'] >= v['routed']:
            interpretation.append('Польза адаптивного выбора экспертов не подтверждена.')
        if all(v[k] < v['separate_replay'] for k in ('dense11','dense12','uniform','routed')):
            interpretation.append('Все новые режимы ниже separate: автоматически не усложнять модель.')
    return dict(study_id=STUDY_ID, status='COMPLETE' if complete else 'INCOMPLETE', source_hash=source_hash,
                selection_split='VALID', test_evaluation_count=0, rows=rows, comparisons=comparisons,
                interpretation=interpretation, limitations='Один seed, exploratory. Replay не независимый seed; без significance/CI и сравнения с внешними paper TEST.')


def render(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = result['rows']
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, row in enumerate(rows):
        if row['status'] == 'PASS':
            ax.scatter(i, row['ndcg10'], color='#167f6c', marker='o', label='Full horizon' if i==0 else None)
            ax.scatter(i, row['first27'], color='#b34f58', marker='x', label='First 27 epochs' if i==0 else None)
        else:
            ax.text(i, .5, row['status'], transform=ax.get_xaxis_transform(), ha='center', fontsize=8, rotation=90)
    ax.axhline(result['historical']['ndcg10'], color='#555555', linestyle=':', label='Historical separate')
    ax.set_xticks(range(len(rows)), [r['mode'] for r in rows], rotation=15)
    ax.set_ylabel('VALID NDCG@10')
    ax.set_title('Context-time pilot: seed 2026')
    ax.grid(axis='y', alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, metadata={'Date': None})
    plt.close(fig)


def main():
    manifest = verify()
    records, sources = {}, {}
    for task in plan()['tasks']:
        path = paths(task)['result']
        if path.exists():
            row = json.loads(path.read_text())
            if row.get('status') == 'PASS':
                validate(row, task, manifest['source_hash'])
                if sha(paths(task)['checkpoint']) != row['checkpoint_sha256']:
                    raise ValueError('Completed checkpoint hash mismatch')
            records[task['mode']] = row
            sources[task['run_id']] = dict(path=str(path), sha256=sha(path))
    successes = [r for r in records.values() if r.get('status') == 'PASS']
    if len({(r['execution_commit'], r['job_id']) for r in successes}) > 1:
        raise ValueError('Mixed executions in pilot')
    for key in ('initial_backbone_sha256','rng_before_fit_sha256','first_train_batch_sha256'):
        if len({r[key] for r in successes}) > 1:
            raise ValueError(f'Paired procedure mismatch: {key}')
    result = summarize(records, manifest['source_hash'])
    ref = plan()['historical_reference']
    old = json.loads((ROOT / ref['source_json']).read_text())
    history = json.loads((ROOT / ref['history_json']).read_text())['runs']['separate']['history']
    result.update(sources=sources, source_files=manifest['files'], historical=dict(
        source=ref, ndcg10=old['best_valid_score'], first27=max(h['valid_ndcg10'] for h in history[:27]),
        best_epoch=old['best_epoch'], actual_epochs=old['actual_epochs'], role='same seed2026, historical only'))
    if SUMMARY.exists():
        raise FileExistsError('Summary already exists; no overwrite')
    atomic_json(SUMMARY, result)
    def fmt(value):
        return '-' if value is None else f'{value:.4f}'
    lines = ['# Контекстная временная калибровка Mamba3', '', result['limitations'], '',
        '| mode | seed | parameters | VALID NDCG@10 | HR@10 | delta replay | best epoch (0-based) | epochs | TRAIN/VALID sec | status |',
        '|---|---|---|---|---|---|---|---|---|---|']
    for r in result['rows']:
        lines.append(f"| {r['mode']} | 2026 | {r.get('parameters','-')} | {fmt(r.get('ndcg10'))} | {fmt(r.get('hr10'))} | {fmt(r.get('delta_replay'))} | {r.get('best_epoch','-')} | {r.get('actual_epochs','-')} | {round(r.get('train_seconds',0))}/{round(r.get('valid_seconds',0))} | {r['status']} |")
    lines += ['', '| mode | первые 27 эпох | полный запуск |', '|---|---|---|']
    lines += [f"| {r['mode']} | {fmt(r.get('first27'))} | {fmt(r.get('ndcg10'))} |" for r in result['rows']]
    lines += [f"| historical separate2026 | {fmt(result['historical']['first27'])} | {fmt(old['best_valid_score'])} |", '',
        *result['interpretation'], '', 'TEST=0. Численные различия не доказывают семантические режимы или новизну.', '',
        '![VALID NDCG@10](pilot_summary.svg)', '', '[JSON и hashes](pilot_summary.json)', '']
    lines += [f"- [{run_id}]({Path(v['path']).name}), SHA256 `{v['sha256']}`" for run_id,v in sources.items()]
    (HERE / 'runs/pilot_summary.md').write_text('\n'.join(lines) + '\n')
    render(result, HERE / 'runs/pilot_summary.svg')


if __name__ == '__main__':
    main()
