"""Read owned run records and checkpoints as bytes; no model/evaluation call."""
import json
import math
from .config import MODES, COUNTS, SUMMARY, paths, plan
from .provenance import identity, create_record, sha, now


def validate_run(row, mode):
    if row['mode'] != mode or row['TEST'] != 'NOT_RUN' or row['test_evaluation_count'] != 0:
        raise ValueError('Run identity/TEST contract mismatch')
    if row['status'] != 'PASS':
        return
    if row['parameter_count'] != COUNTS[mode] or row['seed'] != 2026:
        raise ValueError('Run configuration mismatch')
    history = row['history']
    if not history or [r['epoch'] for r in history] != list(range(row['actual_epochs'])):
        raise ValueError('Incomplete history')
    scores = [r['valid_ndcg10'] for r in history]
    if not all(math.isfinite(s) for s in scores):
        raise ValueError('Nonfinite history')
    best = max(i for i, s in enumerate(scores) if s == max(scores))
    if (row['best_epoch'] != best or row['best_valid_score'] != scores[best] or
            row['best_valid_metrics'] != history[best]['valid_metrics'] or
            row['best_diagnostics'] != history[best]['diagnostics'] or
            row['first27_best_ndcg10'] != max(scores[:27])):
        raise ValueError('Best/last-equal checkpoint history mismatch')
    p = paths(mode)
    meta = json.loads(p['metadata'].read_text())
    if meta['epoch'] != best or meta['metrics'] != row['best_valid_metrics']:
        raise ValueError('Checkpoint metadata mismatch')
    if sha(p['checkpoint']) != row['checkpoint_sha256'] or meta['checkpoint_sha256'] != row['checkpoint_sha256']:
        raise ValueError('Checkpoint digest mismatch')


def build(rows, base):
    if len(rows) != 2:
        raise ValueError('Exactly two scientific records required')
    for row, mode in zip(rows, MODES):
        validate_run(row, mode)
        if any(row.get(k) != v for k, v in base.items()):
            raise ValueError('Run source/allocation identity differs')
    done = all(r['status'] == 'PASS' for r in rows)
    result = dict(**base, status='PASS' if done else 'INCOMPLETE', generated_at=now(),
                  scientific_fits=sum(r.get('scientific_fit_started', False) for r in rows),
                  historical_reference=plan()['historical_reference'], exploratory=True,
                  primary_comparator='new dual from this allocation', independent_seed_count=1, rows=[])
    lines = ['# SISO Dual / Triple Pilot', '',
             'KuaiRand Protocol B, VALID full-ranking. TEST NOT RUN. Один seed, exploratory; статистическая значимость не установлена.', '',
             '| mode | parameters | VALID NDCG@10 | HR@10 | first27 best | best_epoch (0-based) | actual_epochs | TRAIN/VALID seconds | status |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        entry = dict(mode=r['mode'], parameters=r.get('parameter_count'), status=r['status'],
                     valid_metrics=r.get('best_valid_metrics'), first27_best=r.get('first27_best_ndcg10'),
                     best_epoch=r.get('best_epoch'), actual_epochs=r.get('actual_epochs', 0),
                     train_seconds=r.get('train_seconds'), valid_seconds=r.get('valid_seconds'),
                     peak_gpu_allocated_bytes=r.get('peak_gpu_allocated_bytes'),
                     source_json=str(paths(r['mode'])['result']), source_sha256=sha(paths(r['mode'])['result']))
        result['rows'].append(entry)
        metrics = entry['valid_metrics'] or {}
        values = [entry['mode'], entry['parameters'], metrics.get('ndcg@10'), metrics.get('hit@10'),
                  entry['first27_best'], entry['best_epoch'], entry['actual_epochs'],
                  f"{entry['train_seconds']} / {entry['valid_seconds']}", entry['status']]
        lines.append('| ' + ' | '.join('нет данных' if v is None else str(v) for v in values) + ' |')
    if done:
        for key in ('initial_backbone_sha256', 'rng_before_fit_sha256', 'first_train_batch_sha256', 'admission_sha256'):
            if rows[0][key] != rows[1][key]:
                raise ValueError('Scientific pair mismatch: ' + key)
        dual, triple = (r['best_valid_metrics']['ndcg@10'] for r in rows)
        result['triple_minus_dual'] = dict(absolute=triple-dual, relative=(triple-dual)/dual if dual else None)
        lines += ['', 'triple - dual NDCG@10: ' + json.dumps(result['triple_minus_dual'])]
    else:
        lines += ['', 'Пилот не завершён; победитель не выбирается.']
    lines += ['', 'Исторический dual seed2026: VALID NDCG@10 = 0.0633, только контекст; это не дополнительный независимый seed.',
              '', 'Старые exact-zero FAIL остаются FAIL. Допуск ограничен текущим SISO-пилотом; MIMO не авторизован.']
    return result, '\n'.join(lines) + '\n'


def main():
    if SUMMARY.exists() or SUMMARY.with_suffix('.md').exists():
        raise FileExistsError('Summary already exists')
    result, markdown = build([json.loads(paths(m)['result'].read_text()) for m in MODES], identity())
    create_record(SUMMARY, result)
    with SUMMARY.with_suffix('.md').open('x') as f:
        f.write(markdown)


if __name__ == '__main__':
    main()
