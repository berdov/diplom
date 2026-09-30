"""Publish saved VALID records; no model imports or evaluation."""
import csv
import hashlib
import io
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PACKAGE = ROOT / 'experiments/mamba3_head_timescales'
REPORT = ROOT / 'reports/MAMBA3_TIME_MECHANISMS_RESULTS.md'
OLD_MAIN = 'ce6da46099fcc15f74f61c28ed946f78210ac053'
MODES = ('fixed', 'shared_tau', 'head_tau')


def read(path):
    return json.loads(Path(path).read_text())


def pilot_records():
    return [read(PACKAGE / f'runs/attempt_002/mamba3_headtime_{v}_seed2026_001.json') for v in MODES]


def append_registry(records, sources, anchor):
    path = ROOT / 'experiments/results.csv'
    frozen = subprocess.check_output(['git', 'show', OLD_MAIN + ':experiments/results.csv'], cwd=ROOT)
    before = path.read_bytes()
    assert before.startswith(frozen) and before.endswith(b'\n')
    reader = csv.DictReader(io.StringIO(before.decode()))
    header, existing = reader.fieldnames, list(reader)
    ids = [r['run_id'] for r in existing if r['source'] == 'ours']
    assert len(ids) == len(set(ids))
    additions = []
    for r, source in zip(records, sources, strict=True):
        assert r['status'] == 'PASS' and r['scientific_fit_started'] is True
        assert r['TEST'] == 'NOT_RUN' and r['test_evaluation_count'] == 0
        row = {key: '' for key in header}
        row.update(record_type='experiment', source='ours', run_id=r['run_id'],
                   model='HeadTimescaleMamba3Rec', model_variant='MIMO_dual_' + r['time_scale_mode'],
                   dataset='KuaiRand', protocol='B', split='validation', evaluation='full_7111_items',
                   status='completed', seed=str(r['seed']), train_candidates='full_softmax',
                   item_universe='7111', best_epoch=str(r['best_epoch']), actual_epochs=str(r['actual_epochs']),
                   validation_ndcg10=str(r['best_valid_score']), test_evaluation_count='0',
                   git_commit=r['execution_commit'], test_used='no', source_json=source,
                   notes_path='reports/MAMBA3_TIME_MECHANISMS_RESULTS.md#' + anchor)
        for label, metric in [('HR', 'hit'), ('Recall', 'recall'), ('NDCG', 'ndcg')]:
            for k in (5, 10, 20, 50):
                row[f'{label}@{k}'] = str(r['best_valid_metrics'][f'{metric}@{k}'])
        found = [e for e in existing if e['source'] == 'ours' and e['run_id'] == row['run_id']]
        if found:
            assert found == [row], r['run_id']
        else:
            additions.append(row)
    stream = io.StringIO(newline='')
    csv.DictWriter(stream, fieldnames=header, lineterminator='\n').writerows(additions)
    path.write_bytes(before + stream.getvalue().encode())
    assert path.read_bytes().startswith(before)
    return dict(before=len(existing), added=len(additions), after=len(existing) + len(additions),
                prior_bytes=len(before), prior_sha256=hashlib.sha256(before).hexdigest())


def pilot_section(rows):
    lines = ['<a id="head-timescales-pilot"></a>', '## Обучаемые временные масштабы: пилот', '',
             'Завершены три новых TRAIN→VALID запуска на seed2026, job4362620. Основа — MIMO dual, '
             'rank4/chunk8, два слоя и две temporal heads. Fixed сохраняет TRAIN reference '
             'R₀=838393 мс; shared_tau обучает один R на механизм decay/scan; head_tau — отдельный R '
             'для каждой головы каждого механизма. Параметры R общие для пользователей и слоёв.', '',
             '`R=R₀·exp(log(4)·tanh(α))`, bounds `[R₀/4,4R₀]`, output scales `[0.5,2]`. '
             'Конфигурация, данные, начальный backbone, общие MLP и RNG совпадают. '
             'Gate 9/9 cases, 228/228 checks; smoke 3×3 шага. TEST не выполнялся.', '',
             '| Variant | Parameters | VALID NDCG@10 | HR@10 | Best epoch, с нуля | Epochs | Best first27 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        window = r['history'][:27]
        first = f"{max(x['valid_ndcg10'] for x in window):.4f}" if len(window) == 27 else 'неполное окно'
        source = f"../experiments/mamba3_head_timescales/runs/attempt_002/{r['run_id']}.json"
        lines.append(f"| [{r['time_scale_mode']}]({source}) | {r['parameter_count']} | {r['best_valid_score']:.4f} | "
                     f"{r['best_valid_metrics']['hit@10']:.4f} | {r['best_epoch']} | {r['actual_epochs']} | {first} |")
    by = {r['time_scale_mode']: r for r in rows}
    lines += ['', '| Контраст | Δ VALID NDCG@10 | Относительно контроля |', '|---|---:|---:|']
    for left, right in [('head_tau', 'shared_tau'), ('head_tau', 'fixed'), ('shared_tau', 'fixed')]:
        a, b = by[left]['best_valid_score'], by[right]['best_valid_score']
        lines.append(f'| {left} − {right} | {a-b:+.4f} | {100*(a-b)/b:+.3f}% |')
    lines += ['', 'Общая шкала в этом пилоте не улучшила основную метрику; индивидуальные шкалы дали '
              'положительную разницу. Head достиг лучшего результата позже. First27 — реальные эпохи '
              '0–26 из тех же histories, не независимая репликация и не равный GPU-бюджет.', '',
              '| Variant | Mechanism | Head/shared | α | R/R₀ | R, мс |', '|---|---|---|---:|---:|---:|']
    for r in rows:
        for mechanism, d in r['best_diagnostics']['head_timescales']['mechanisms'].items():
            for i, ratio in enumerate(d['reference_ratio']):
                label = 'оба heads' if d['alpha'] is None else 'shared' if len(d['alpha']) == 1 else f'h{i}'
                alpha = '—' if d['alpha'] is None else f"{d['alpha'][i]:.6f}"
                lines.append(f"| {r['time_scale_mode']} | {mechanism} | {label} | {alpha} | {ratio:.6f} | {d['reference_ms'][i]:.0f} |")
    lines += ['', 'Все обучаемые α изменились; near-reference-bound flags в histories false. '
              'Сохранённые scale(gap) не сводятся к одному R: decay второй головы почти насыщен '
              'на верхнем output bound. R — масштаб нормировки, не итоговый коэффициент затухания '
              'и не доказанный период интересов. Разные R не устанавливают специализацию голов.', '',
              'Новый fixed точно воспроизвёл 39 эпох, метрики и checkpoint SHA исторического MIMO dual '
              'seed2026. В реестре это отдельный выполненный run; в прежнюю MIMO-статистику он '
              'не добавлен как независимый seed. Совпадение метрики не было условием technical PASS.', '',
              'Один exploratory seed не устанавливает устойчивость или статистическую значимость. '
              'Сравнения нашего VALID с опубликованным TEST и утверждения о новом SOTA здесь нет.', '',
              '[Raw summary](../experiments/mamba3_head_timescales/runs/attempt_002/pilot_summary.json), '
              '[preservation и SHA](../experiments/mamba3_head_timescales/evidence/job4362620/preservation_manifest.json). '
              'Execution `4724392c88a2298e57fa662cba33baa3ab9ecdbb`; исходный failed job4361071 сохранён отдельно.', '']
    return '\n'.join(lines)


def publish_pilot():
    rows = pilot_records()
    sources = [f"experiments/mamba3_head_timescales/runs/attempt_002/{r['run_id']}.json" for r in rows]
    registry = append_registry(rows, sources, 'head-timescales-pilot')
    text = REPORT.read_text()
    section = pilot_section(rows)
    marker = '<!-- head-timescales:pilot:start -->'
    end = '<!-- head-timescales:pilot:end -->'
    if marker in text:
        a, rest = text.split(marker)
        _, b = rest.split(end)
        text = a + marker + '\n' + section + end + b
    else:
        text += '\n' + marker + '\n' + section + end + '\n'
    text = text.replace('Следующий пункт — временные масштабы\nголов — пока не реализован и не запущен.',
                        'Для временных масштабов голов завершён [пилот](#head-timescales-pilot).')
    REPORT.write_text(text)
    overview = ROOT / 'reports/RESULTS.md'
    value = overview.read_text()
    start = '**Обучаемые масштабы MIMO dual, пилот seed2026:**'
    paragraph = (start + ' fixed/shared_tau/head_tau получили VALID NDCG@10 **' +
                 ' / '.join(f"{r['best_valid_score']:.4f}" for r in rows) +
                 '**. Общая шкала не улучшила NDCG, индивидуальные дали небольшой выигрыш на одном seed. '
                 'TEST не выполнялся; устойчивость не установлена. '
                 '[Таблица, первые 27 эпох и выученные масштабы](MAMBA3_TIME_MECHANISMS_RESULTS.md#head-timescales-pilot).')
    if start not in value:
        value = value.replace('## Mamba3: VALID\n', '## Mamba3: VALID\n\n' + paragraph + '\n', 1)
    overview.write_text(value)
    print(json.dumps(registry))


if __name__ == '__main__':
    publish_pilot()
