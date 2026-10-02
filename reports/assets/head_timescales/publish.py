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


def confirmation_records(attempt):
    from experiments.mamba3_head_timescales.confirmation import config as config,report as numeric
    audit=read(config.RUNTIME/'audit.json')
    if audit.get('status')!='PASS' or audit.get('fits_verified')!=12:
        raise ValueError('Independent terminal audit required before publication')
    entries=config.index(attempt)['entries'];rows=[];sources=[]
    for entry in entries:
        p=ROOT/entry['result'];raw=p.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=audit['result_sha256'][entry['run_id']]:raise ValueError('Raw changed after audit')
        r=json.loads(raw);numeric.validate_record(r,entry)
        if r['status']!='PASS':raise ValueError('Only complete successful confirmation can be published')
        rows.append(r);sources.append(entry['result'])
    summary=numeric.summarize({(r['seed'],r['time_scale_mode']):r for r in rows},{v:r for v,r in zip(MODES,pilot_records(),strict=True)})
    if summary['status']!='PASS' or summary['complete_new_triples']!=4:raise ValueError('Incomplete or inconsistent series')
    return rows,sources,summary


def pm(values,signed=False):
    if values['mean'] is None:return '—'
    value=0.0 if round(values['mean'],6)==0 else values['mean']
    mean=format(value,'+.6f' if signed else '.6f')
    return mean+' ± '+format(values['sample_std'],'.6f') if values['sample_std'] is not None else mean+' (n=1)'


def confirmation_section(summary):
    rows=summary['rows'];by={(r['seed'],r['variant']):r for r in rows}
    lines=['<a id="head-timescales-confirmation"></a>','## Обучаемые временные масштабы: подтверждение','',
        'Завершены12 новых fresh fits: fixed/shared_tau/head_tau на заранее выбранных seeds2027–2030. '
        'Все три контроля сохранены независимо от качества. Основной контраст — head_tau−shared_tau. '
        'Пилот2026 показан отдельно; исторический MIMO dual2026 повторно в статистику не включён. '
        'Только VALID, TEST=0.','',
        '| Seed | Fixed | Shared τ | Head τ | Δ head−shared | Δ head−fixed | Δ shared−fixed |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for seed in range(2027,2031):
        f,s,h=[by[seed,v]['ndcg10'] for v in MODES]
        cells=[f"[{by[seed,v]['ndcg10']:.4f}](../experiments/mamba3_head_timescales/confirmation/runs/attempt_001/{by[seed,v]['run_id']}.json)" for v in MODES]
        lines.append(f'| {seed} | '+ ' | '.join(cells)+f' | {h-s:+.4f} | {h-f:+.4f} | {s-f:+.4f} |')
    lines+=['','Mean ± sample std, ddof=1. Относительные разницы посчитаны по средним на одинаковом наборе seeds.','',
            '| Набор | n | Fixed | Shared τ | Head τ |','|---|---:|---:|---:|---:|']
    for key,label in [('new4_full','Четыре новых seed'),('all5_full','Пять с exploratory pilot')]:
        g=summary['cohorts'][key]
        lines.append('| '+label+' | '+str(len(g['complete_triples']))+' | '+' | '.join(pm(g['models_same_complete_triples'][v]) for v in MODES)+' |')
    lines+=['','| Набор | Контраст | Δ mean ± std | + / − / 0 | Relative % |','|---|---|---:|---:|---:|']
    for key,label in [('new4_full','Новые4'),('all5_full','Все5')]:
        for name,d in summary['cohorts'][key]['contrasts'].items():
            lines.append(f"| {label} | {name} | {pm(d,True)} | {d['positive']}/{d['negative']}/{d['zero']} | {d['relative_percent']:+.3f}% |")
    g=summary['cohorts']['new4_full'];primary=g['contrasts']['head_tau-shared_tau'];fixed=g['contrasts']['head_tau-fixed']
    first=summary['cohorts']['new4_first27']['contrasts']['head_tau-shared_tau']
    lines+=['',f"На четырёх новых seeds head_tau превосходит shared_tau в среднем на **{primary['relative_percent']:+.3f}%**, "
        f"в {primary['positive']}/{primary['n_available']} пар. Относительно fixed разница составляет лишь "
        f"**{fixed['relative_percent']:+.3f}%**, положительны {fixed['positive']}/{fixed['n_available']} пар. "
        f"В first27 средняя разница head−shared равна **{first['mean']:.6f}**. "
        'Устойчивое практически значимое преимущество отдельных reference scales над fixed не подтверждено. '
        'Они не выбираются как обязательное усложнение backbone; рабочим контролем остаётся MIMO dual с fixed reference. '
        'Это не доказательство эквивалентности моделей или отсутствия эффекта вообще. '
        'Пункт 2 завершён в текущем KuaiRand/VALID-протоколе.','']
    lines+=['','![Парные разницы head_tau минус shared_tau, VALID NDCG@10](assets/head_timescales/paired_delta.svg)','',
            '### Первые27 эпох','',
            '| Seed | Fixed | Shared τ | Head τ |','|---|---:|---:|---:|']
    for seed in range(2026,2031):
        values=[]
        for v in MODES:
            r=by[seed,v]
            values.append(f"{r['first27_ndcg10']:.4f}" if r['first27_complete'] else f"неполное ({r['actual_epochs']} эпох)")
        lines.append('| '+str(seed)+(' (пилот)' if seed==2026 else '')+' | '+' | '.join(values)+' |')
    lines+=['','| Набор | Парный контраст first27 | Доступно / ожидается | Seeds | Δ mean ± std | + / − / 0 | Relative % |',
            '|---|---|---:|---|---:|---:|---:|']
    for key,label in [('new4_first27','Новые4'),('all5_first27','Все5')]:
        for name,d in summary['cohorts'][key]['contrasts'].items():
            relative='—' if d['relative_percent'] is None else f"{d['relative_percent']:+.3f}%"
            lines.append(f"| {label} | {name} | {d['n_available']}/{d['n_expected']} | {d['seeds']} | {pm(d,True)} | {d['positive']}/{d['negative']}/{d['zero']} | {relative} |")
    lines+=['','First27 использует только реальные полные окна0–26. Пары выбираются независимо: '
            'короткий третий run не исключает полную пару. Это срез тех же histories, '
            'не независимая репликация и не строго равный GPU-бюджет.','',
            '<details>','<summary>Эпохи, HR, время, память и выученные масштабы</summary>','',
            '| Seed | Variant | HR@10 | Best epoch, с нуля | Epochs | TRAIN / VALID, s | Peak allocated / reserved, GiB |',
            '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['seed']} | {r['variant']} | {r['hr10']:.4f} | {r['best_epoch']} | {r['actual_epochs']} | "
                     f"{r['train_seconds']:.1f} / {r['valid_seconds']:.1f} | {r['peak_allocated_bytes']/2**30:.3f} / {r['peak_reserved_bytes']/2**30:.3f} |")
    lines+=['','| Seed | Variant | Mechanism | Head/shared | α | R/R₀ | R, мс | Near bound |','|---|---|---|---|---:|---:|---:|---|']
    for r in rows:
        for name,d in r['best_diagnostics']['head_timescales']['mechanisms'].items():
            for i,ratio in enumerate(d['reference_ratio']):
                alpha='—' if d['alpha'] is None else f"{d['alpha'][i]:.6f}"
                head='оба' if d['alpha'] is None else 'shared' if len(d['alpha'])==1 else f'h{i}'
                lines.append(f"| {r['seed']} | {r['variant']} | {name} | {head} | {alpha} | {ratio:.6f} | {d['reference_ms'][i]:.0f} | {d['near_reference_bound'][i]} |")
    lines+=['','Срез сохранённых эффективных функций на `gap=R₀` (H0 / H1), best epoch. '
            'Новых forward для этой таблицы нет.','',
            '| Seed | Variant | Decay scale, H0 / H1 | Scan scale, H0 / H1 |','|---|---|---:|---:|']
    for r in rows:
        curves=[]
        for name in ('decay','scan'):
            d=r['best_diagnostics']['head_timescales']['mechanisms'][name]
            values=d['scale_by_gap_and_head'][d['gaps_over_R0'].index(1)]
            curves.append(' / '.join(f'{v:.4f}' for v in values))
        lines.append(f"| {r['seed']} | {r['variant']} | "+' | '.join(curves)+' |')
    lines+=['','</details>','',
        'Настройки и математическая реализация совпадают с пилотом: MIMO dual rank4/chunk8, '
        'две temporal heads, два слоя, batch2048, history50/padding56, Adam0.001, epochs300, '
        'stopping_step10 и прежняя last-tie семантика. Counts715020/715022/715024. '
        'Внутри каждого seed проверены общий backbone/MLP/buffers, RNG, precision, Adam и первый '
        'фактически потреблённый batch. Время включает JIT/cache и не служит сравнением warm-kernel latency.','',
        'R — глобальные параметры модели, не персональные периоды пользователей. Bounds '
        '[R₀/4,4R₀] при R₀=838393 мс и output bounds[0.5,2] неизменны. Сохранённые '
        'scale(gap) и histories рассматриваются вместе с R; одинаковые или разные R '
        'сами по себе не доказывают специализацию голов или причину разницы качества. '
        'На seeds2027–2029 у head_tau R(decay,h0)>R(decay,h1), а R(scan,h0)<R(scan,h1); '
        'на seed2030 оба направления меняются. Индексы heads не имеют стабильной short/long семантики. '
        'Reference bounds не достигнуты; при этом на seeds2028–2029 выходной decay-scale второй головы '
        'у shared_tau и head_tau близок к верхней границе.','',
        'Четыре новых seed дают ограниченную оценку разброса на одном датасете. Новые p-values '
        'не подбирались, статистическая значимость и эквивалентность не установлены. '
        'Наш VALID не сравнивается с опубликованным TEST как доказанный апгрейд; '
        'TEST-модель автоматически не выбрана.','',
        '[Числовая сводка](assets/head_timescales/confirmation_summary.json) · '
        '[Markdown](assets/head_timescales/confirmation_summary.md) · '
        '[Индекс источников](assets/head_timescales/sources.json) · '
        '[TeX VALID](assets/head_timescales/valid_table.tex) · '
        '[Сохранённые artifacts и SHA](../experiments/mamba3_head_timescales/confirmation/evidence/job4365206/preservation_manifest.json) · '
        '[Независимый аудит](../experiments/mamba3_head_timescales/confirmation/evidence/job4365206/independent_audit.json).','']
    return '\n'.join(lines)


def confirmation_tex(summary):
    lines=[r'\begin{table}[t]',r'\centering',r'\small',
           r'\caption{Learned global normalization scales, KuaiRand VALID NDCG@10 on seeds 2027--2030. Mean $\pm$ sample standard deviation ($ddof=1$). Exploratory pilot 2026 is excluded; TEST was not evaluated.}',
           r'\label{tab:head-timescales-valid}',r'\begin{tabular}{lcc}',r'\toprule',
           r'Variant & Seeds & VALID NDCG@10 \\',r'\midrule']
    for v in MODES:
        d=summary['cohorts']['new4_full']['models_same_complete_triples'][v]
        lines.append(v.replace('_',r'\_')+' & '+str(d['n_available'])+' & $'+pm(d).replace('±',r'\pm')+r'$ \\')
    return '\n'.join(lines+[r'\bottomrule',r'\end{tabular}',r'\end{table}'])+'\n'


def confirmation_markdown(summary):
    from experiments.mamba3_head_timescales.confirmation import report as numeric
    lines=numeric.markdown(summary).splitlines();cohort=None
    for i,line in enumerate(lines):
        if line.startswith('## '):cohort=line[3:]
        if cohort in summary['cohorts']:
            for name,d in summary['cohorts'][cohort]['contrasts'].items():
                if line.startswith('| '+name+' |'):
                    cells=line.split('|');cells[5]=f" {d['positive']}/{d['negative']}/{d['zero']} "
                    lines[i]='|'.join(cells)
        lines[i]=lines[i].replace('+ / 0 / −','+ / − / 0')
    return '\n'.join(lines)+'\n'


def source_entry(record,path,role):
    package=PACKAGE if role=='pilot' else PACKAGE/'confirmation'
    remote=Path('/home/daryumin/iberdov/diplom')/package.relative_to(ROOT)
    metadata=package/f"evidence/job{record['job_id']}/files"/Path(record['checkpoint_metadata_path']).relative_to(remote)
    meta=read(metadata)
    if meta['checkpoint_sha256']!=record['checkpoint_sha256']:raise ValueError('Preserved checkpoint metadata differs')
    return dict(run_id=record['run_id'],seed=record['seed'],variant=record['time_scale_mode'],role=role,
                path=path,sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),
                metadata_path=str(metadata.relative_to(ROOT)),metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(),
                execution_commit=record['execution_commit'],source_hash=record['source_hash'],job_id=record['job_id'],
                physical_attempt=record['execution_attempt'],checkpoint_path=record['checkpoint_path'],checkpoint_sha256=record['checkpoint_sha256'])


def publish_confirmation(attempt):
    from experiments.mamba3_head_timescales.confirmation import report as numeric
    rows,sources,summary=confirmation_records(attempt)
    registry=append_registry(rows,sources,'head-timescales-confirmation')
    (HERE/'confirmation_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    (HERE/'confirmation_summary.md').write_text(confirmation_markdown(summary))
    entries=[source_entry(r,p,'confirmation') for r,p in zip(rows,sources,strict=True)]
    entries += [source_entry(r,f"experiments/mamba3_head_timescales/runs/attempt_002/{r['run_id']}.json",'pilot') for r in pilot_records()]
    (HERE/'sources.json').write_text(json.dumps(dict(sources=entries),indent=2)+'\n')
    (HERE/'valid_table.tex').write_text(confirmation_tex(summary))
    text=REPORT.read_text();start='<!-- head-timescales:confirmation:start -->';end='<!-- head-timescales:confirmation:end -->'
    section=confirmation_section(summary)
    if start in text:
        a,rest=text.split(start);_,b=rest.split(end);text=a+start+'\n'+section+end+b
    else:text+='\n'+start+'\n'+section+end+'\n'
    pilot_link='[Завершённое подтверждение на четырёх новых seeds](#head-timescales-confirmation).\n\n'
    text=text.replace('## Обучаемые временные масштабы: пилот\n\n','## Обучаемые временные масштабы: пилот\n\n'+pilot_link,1) if pilot_link not in text else text
    text=text.replace('2. Обучаемые временные масштабы отдельно по heads: не реализованы и не проверены как следующий самостоятельный механизм.',
        '2. Обучаемые временные масштабы heads: [пилот и подтверждение завершены](#head-timescales-confirmation). Head-specific reference scales не выбраны как обязательное усложнение; контроль — MIMO dual fixed.')
    text=text.replace('Для временных масштабов голов завершён [пилот](#head-timescales-pilot).',
        'Для временных масштабов голов завершены [пилот](#head-timescales-pilot) и [подтверждение](#head-timescales-confirmation).')
    REPORT.write_text(text)
    g=summary['cohorts']['new4_full'];d=g['contrasts']['head_tau-shared_tau']
    paragraph=('**Обучаемые масштабы MIMO dual, подтверждение2027–2030:** fixed/shared_tau/head_tau — VALID NDCG@10 **'+
        ' / '.join(f"{g['models_same_complete_triples'][v]['mean']:.6f}" for v in MODES)+
        f"**. Head−shared: **{d['mean']:+.6f} ({d['relative_percent']:+.3f}%)**, пары +/−/0: {d['positive']}/{d['negative']}/{d['zero']}. "
        f"Head−fixed: {g['contrasts']['head_tau-fixed']['relative_percent']:+.3f}%, "
        f"{g['contrasts']['head_tau-fixed']['positive']}/4 wins; first27 head−shared: "
        f"{summary['cohorts']['new4_first27']['contrasts']['head_tau-shared_tau']['mean']:.6f}. "
        'Пункт 2 завершён: head-specific reference scales не выбираются как обязательное усложнение, контроль остаётся fixed. '
        'Пять seeds с exploratory pilot показаны отдельно. TEST=0; эквивалентность и отсутствие эффекта вообще не установлены. '
        '[Полный результат, график и TeX](MAMBA3_TIME_MECHANISMS_RESULTS.md#head-timescales-confirmation).')
    p=ROOT/'reports/RESULTS.md';overview=p.read_text()
    if paragraph not in overview:overview=overview.replace('## Mamba3: VALID\n','## Mamba3: VALID\n\n'+paragraph+'\n',1)
    p.write_text(overview)
    print(json.dumps(registry))


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--confirmation',action='store_true');parser.add_argument('--attempt',choices=['001','002'],default='001')
    args=parser.parse_args()
    if args.confirmation:publish_confirmation(args.attempt)
    else:publish_pilot()
