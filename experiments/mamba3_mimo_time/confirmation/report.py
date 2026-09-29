"""Pure saved-record aggregation: paired seeds, ddof=1, explicit missing rows."""
import statistics
from . import config as c
from .records import read, create, sha
from experiments.mamba3_mimo_time.report import summarize as pilot_summarize


def stats(values):
    return dict(n=len(values),mean=statistics.mean(values) if values else None,
                sample_std=statistics.stdev(values) if len(values)>1 else None,ddof=1)


def subset(records,seeds):
    rows = []
    for seed in seeds:
        for mode in c.MODES:
            r = records.get((seed,mode))
            if r is not None and (r.get('seed') != seed or r.get('mode') != mode or r.get('parameter_count',c.COUNTS[mode]) != c.COUNTS[mode]):
                raise ValueError('Summary seed/mode/count mismatch')
        summary = pilot_summarize({m:records[(seed,m)] for m in c.MODES if (seed,m) in records})
        for row in summary['rows']:
            row.update(seed=seed,role='exploratory' if seed==2026 else 'confirmation',
                       first27_observed_epochs=min(27,row['actual_epochs']))
        rows += summary['rows']
    by = {(r['seed'],r['mode']):r for r in rows}
    def score(seed,mode):
        r = by[(seed,mode)]
        return r['metrics']['ndcg@10'] if r['status']=='PASS' else None
    complete = [s for s in seeds if all(score(s,m) is not None for m in c.MODES)]
    model_stats = {m:stats([score(s,m) for s in complete]) for m in c.MODES}
    contrasts = []
    for left,right in (('triple','dual'),('dual','base'),('triple','base')):
        available = [s for s in seeds if score(s,left) is not None and score(s,right) is not None]
        pairs = [dict(seed=s,left=score(s,left),right=score(s,right),delta=score(s,left)-score(s,right)) for s in available]
        deltas = [p['delta'] for p in pairs]
        a,b = stats([p['left'] for p in pairs]),stats([p['right'] for p in pairs])
        contrasts.append(dict(comparison=f'{left} - {right}',n_expected=len(seeds),n_available=len(available),
                              seeds=available,missing_seeds=[s for s in seeds if s not in available],
                              status='COMPLETE' if len(available)==len(seeds) else 'INCOMPLETE_SUBSET',pairs=pairs,
                              left=a,right=b,delta=stats(deltas),positive=sum(v>0 for v in deltas),
                              negative=sum(v<0 for v in deltas),zero=sum(v==0 for v in deltas),
                              relative_percent=100*(a['mean']/b['mean']-1) if pairs and b['mean'] else None))
    return dict(status='COMPLETE' if len(complete)==len(seeds) else 'INCOMPLETE',rows=rows,
                n_expected=len(seeds),n_available=len(complete),complete_triple_seeds=complete,
                model_stats_same_complete_triples=model_stats,contrasts=contrasts,
                first27_note='Observed epochs only; incomplete windows are not padded or aggregated as full windows')


def summarize(records,pilot):
    expected = {(s,m) for s in c.SEEDS for m in c.MODES}
    if not set(records) <= expected:
        raise ValueError('Unplanned scientific result')
    if set(pilot) != {(2026,m) for m in c.MODES}:
        raise ValueError('Exactly three exploratory pilot rows required')
    for (seed,mode),r in records.items():
        if r.get('run_id') != c.paths(mode,seed)['run_id']:
            raise ValueError('Run ID mismatch')
    a = subset(records,c.SEEDS)
    b = subset(pilot | records,(2026,*c.SEEDS))
    return dict(status='PASS' if a['status']=='COMPLETE' else 'INCOMPLETE',confirmatory=a,with_exploratory_pilot=b,
                scientific_fits_started=sum(bool(r.get('scientific_fit_started')) for r in records.values()),
                scientific_fits_completed=sum(r['status']=='PASS' for r in records.values()),
                scientific_fits_expected=12,selection_split='VALID',TEST='NOT_RUN',test_evaluation_count=0,
                pilot_role='Seed2026 informed continuation; not independent confirmation',
                interpretation='No automatic backbone selection or significance claim; VALID is not published TEST')


def markdown(summary):
    lines = ['# MIMO base / dual / triple: подтверждающая серия','',
             'KuaiRand: хронологический leave-one-out, полный каталог. Только VALID; TEST не использовался.',
             f"Статус: {summary['status']}; fits начато/завершено/план: {summary['scientific_fits_started']}/{summary['scientific_fits_completed']}/12.",
             'Seed2026 использован при решении продолжить исследование и не является независимым подтверждением.','']
    f = lambda v:'нет данных' if v is None else f'{v:.6f}'
    for key,title in [('confirmatory','Четыре новые тройки2027–2030'),('with_exploratory_pilot','Все пять с exploratory pilot2026')]:
        sub = summary[key]
        lines += ['## '+title,'',f"Полных троек: {sub['n_available']}/{sub['n_expected']}; {sub['status']}.",'',
                  '| seed | base | dual | triple | triple − dual | dual − base | triple − base |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        seeds = sorted({r['seed'] for r in sub['rows']})
        for seed in seeds:
            rows = {r['mode']:r for r in sub['rows'] if r['seed']==seed}
            scores = {m:(r['metrics']['ndcg@10'] if r['status']=='PASS' else None) for m,r in rows.items()}
            cells = [f(scores[m]) if scores[m] is not None else rows[m]['status'] for m in c.MODES]
            deltas = [scores[l]-scores[r] if scores[l] is not None and scores[r] is not None else None for l,r in [('triple','dual'),('dual','base'),('triple','base')]]
            lines.append('| '+str(seed)+' | '+' | '.join(cells+[f(v) for v in deltas])+' |')
        lines += ['',f"Средние моделей на одном наборе полных троек: {sub['complete_triple_seeds']}; sample std, ddof=1."]
        for mode,row in sub['model_stats_same_complete_triples'].items():
            lines.append(f"- {mode}: {f(row['mean'])} ± {f(row['sample_std'])}, n={row['n']}.")
        for r in sub['contrasts']:
            lines.append(f"- {r['comparison']}: mean Δ={f(r['delta']['mean'])}, std Δ={f(r['delta']['sample_std'])}; mean left/right={f(r['left']['mean'])}/{f(r['right']['mean'])}; relative={f(r['relative_percent'])}%; +/−/0={r['positive']}/{r['negative']}/{r['zero']}; n={r['n_available']}/{r['n_expected']}, seeds={r['seeds']}, {r['status']}.")
        lines += ['','| seed | mode | status | best epoch (с нуля) | epochs | best0–26 | наблюдения / окно |','|---|---|---|---:|---:|---:|---|']
        for r in sub['rows']:
            window = 'полное' if r.get('first27_complete') else 'неполное'
            lines.append(f"| {r['seed']} | {r['mode']} | {r['status']} | {r['best_epoch']} | {r['actual_epochs']} | {f(r['first27'])} | {r['first27_observed_epochs']}/27, {window} |")
        lines.append('')
    lines += ['Пропуски не являются нулями. Неполные поднаборы не подтверждают весь план. Отрицательные разницы сохранены.',
              'Основной критерий — весь штатный run с frozen selection rule; первые27 не дополняются.',
              'Знак среднего не доказывает статистическую значимость; автоматического выбора backbone нет.',
              'Время включает JIT/cache effects. VALID не сравнивается напрямую с TEST TiM4Rec.','']
    return '\n'.join(lines)


def write(base):
    records = {}
    for task in c.tasks():
        p = c.paths(task['mode'],task['seed'])
        if not p['result'].exists():
            continue
        r = read(p['result'])
        if any(r.get(k) != v for k,v in base.items()):
            raise ValueError('Result ownership mismatch')
        if r['status']=='PASS':
            meta = read(p['metadata'])
            if (meta['metrics'] != r['best_valid_metrics'] or meta['epoch'] != r['best_epoch']
                    or meta['run_id'] != r['run_id'] or meta['execution_commit'] != r['execution_commit']
                    or meta['checkpoint_sha256'] != r['checkpoint_sha256'] or sha(p['checkpoint']) != r['checkpoint_sha256']):
                raise ValueError('Checkpoint metadata/integrity')
        records[(task['seed'],task['mode'])] = r
    pilot = {(2026,m):read(c.pilot_path(m)) for m in c.MODES}
    value = dict(base,**summarize(records,pilot))
    create(c.SUMMARY,value)
    with c.SUMMARY.with_suffix('.md').open('x') as f:
        f.write(markdown(value))
    return value
