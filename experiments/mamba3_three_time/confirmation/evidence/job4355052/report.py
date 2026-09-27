"""Predeclared paired summaries; read-only inputs, no model or evaluation."""
import json
import math
import os
import statistics
from .config import MODES, SEEDS, SUMMARY, PILOT, paths, task, COUNTS
from .provenance import atomic_json, create_record, sha, now


def validate(row, mode, seed):
    if row['mode'] != mode or row['seed'] != seed or row.get('TEST') != 'NOT_RUN' or row.get('test_evaluation_count') != 0:
        raise ValueError('Run identity/TEST mismatch')
    if row['status'] != 'PASS':
        return
    h = row['history']
    if not h or [r['epoch'] for r in h] != list(range(row['actual_epochs'])) or row['parameter_count'] != COUNTS[mode]:
        raise ValueError('Incomplete history/incorrect count')
    scores = [r['valid_ndcg10'] for r in h]
    if not all(math.isfinite(v) for v in scores):
        raise ValueError('Nonfinite history')
    best = max(i for i, v in enumerate(scores) if v == max(scores))
    if (row['best_epoch'] != best or row['best_valid_score'] != scores[best] or
            row['best_valid_metrics'] != h[best]['valid_metrics'] or row['best_diagnostics'] != h[best]['diagnostics'] or
            row['first27_best_ndcg10'] != max(scores[:27])):
        raise ValueError('Best epoch/tie/diagnostics mismatch')


def stats(values):
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                sample_std_ddof1=statistics.stdev(values) if len(values) > 1 else None)


def summarize(records, seeds, horizon):
    rows, values, deltas = [], {m: [] for m in MODES}, []
    for seed in seeds:
        pair = dict(seed=seed, role='exploratory' if seed == 2026 else 'confirmation')
        for mode in MODES:
            r = records[(seed, mode)]
            reason = r.get('error')
            available = r['status'] == 'PASS'
            if horizon == 'first27' and r.get('actual_epochs', 0) < 27:
                available = False
                reason = reason or 'Fewer than 27 epochs; not an equal observation window'
            metric = (r['best_valid_score'] if horizon == 'full' else max(x['valid_ndcg10'] for x in r['history'][:27])) if available else None
            pair[mode] = dict(status=r['status'], ndcg10=metric, reason=reason, actual_epochs=r.get('actual_epochs', 0),
                              source_json=r.get('source_json'), source_sha256=r.get('source_sha256'))
            if metric is not None:
                values[mode].append(metric)
        a, b = (pair[m]['ndcg10'] for m in MODES)
        pair['paired_delta'] = None if a is None or b is None else b-a
        if pair['paired_delta'] is not None:
            deltas.append(pair['paired_delta'])
        rows.append(pair)
    mode_stats = {m: stats(v) for m, v in values.items()}
    complete = len(deltas) == len(seeds)
    relative = ((mode_stats['triple']['mean']/mode_stats['dual']['mean']-1)*100
                if complete and mode_stats['dual']['mean'] else None)
    return dict(horizon=horizon, rows=rows, n_expected=len(seeds), n_available=len(deltas),
                n_expected_runs=2*len(seeds), n_available_runs=sum(len(v) for v in values.values()),
                incomplete=not complete, mode_statistics_all_available=mode_stats, paired_delta=stats(deltas),
                positive=sum(d>0 for d in deltas), negative=sum(d<0 for d in deltas), zero=sum(d==0 for d in deltas),
                relative_gain_percent=relative,
                missingness_note='All successful rows retained. Paired deltas use complete pairs only; no relative gain for incomplete sets.')


def build(records):
    expected = {(s,m) for s in (2026,*SEEDS) for m in MODES}
    if set(records) != expected:
        raise ValueError('Exactly the pilot and four new pairs required')
    for (s,m), r in records.items():
        validate(r,m,s)
    return {label: {h: summarize(records,seeds,h) for h in ('full','first27')}
            for label,seeds in (('new_four_pairs',SEEDS), ('all_five_pairs',(2026,*SEEDS)))}


def atomic_text(path, content):
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('x') as out:
        out.write(content)
        out.flush()
        os.fsync(out.fileno())
    os.link(temp, path)
    temp.unlink()


def markdown(value):
    lines = ['# SISO dual/triple confirmation', '',
             'KuaiRand Protocol B, VALID full-ranking; TEST NOT RUN. Seed2026: exploratory pilot, использованный для решения продолжить серию.',
             'Seeds2027-2030 заранее зафиксированы. Нет заявлений о статистической значимости; эпохи не являются независимыми повторениями.',
             'Время runs включает JIT. Историческая separate=.0633 не входит в агрегаты.', '']
    for label, horizons in value['summaries'].items():
        for horizon, r in horizons.items():
            lines += [f'## {label}: {horizon}', '',
                      f"Пары: {r['n_available']}/{r['n_expected']}; incomplete={r['incomplete']}", '',
                      '| seed | dual | triple | paired delta | status / reason |', '|---|---:|---:|---:|---|']
            for pair in r['rows']:
                reason = '; '.join(f"{m}: {pair[m]['status']} {pair[m]['reason'] or ''}" for m in MODES)
                lines.append(f"| {pair['seed']} | {pair['dual']['ndcg10']} | {pair['triple']['ndcg10']} | {pair['paired_delta']} | {reason.replace('|','/')} |")
            lines += ['', 'Все доступные успешные runs: ' + json.dumps(r['mode_statistics_all_available']),
                      'Парные разности (только полные пары): ' + json.dumps(r['paired_delta']),
                      f"Положительные/отрицательные/нулевые: {r['positive']}/{r['negative']}/{r['zero']}; relative gain: {r['relative_gain_percent']}%.", '']
    return '\n'.join(lines)+'\n'


def svg(value):
    rows = value['summaries']['all_five_pairs']['full']['rows']
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="640" height="260" viewBox="0 0 640 260">',
             '<rect width="640" height="260" fill="white"/>',
             '<text x="20" y="25" font-size="16">VALID NDCG@10: paired triple - dual</text>',
             '<line x1="320" y1="45" x2="320" y2="220" stroke="#777"/>']
    cap = max([abs(r['paired_delta']) for r in rows if r['paired_delta'] is not None]+[.001])
    for i,r in enumerate(rows):
        y = 65+i*35
        d = r['paired_delta']
        parts.append(f'<text x="20" y="{y}" font-size="13">{r["seed"]}</text>')
        if d is None:
            parts.append(f'<text x="90" y="{y}" font-size="13">INCOMPLETE</text>')
        else:
            x=320+d/cap*170
            parts.append(f'<line x1="320" y1="{y-4}" x2="{x}" y2="{y-4}" stroke="#17806d" stroke-width="5"/>')
            parts.append(f'<text x="520" y="{y}" font-size="13">{d:+.6f}</text>')
    return '\n'.join(parts+['</svg>'])


def main():
    from .provenance import identity
    from .state import paired
    base=identity()
    records={}
    for seed in (2026,*SEEDS):
        for mode in MODES:
            p=(PILOT/f'runs/mamba3_three_time_siso_{mode}_seed2026_001.json' if seed==2026 else paths(mode,seed)['result'])
            r=json.loads(p.read_text())
            validate(r,mode,seed)
            if seed!=2026:
                if any(r.get(k)!=v for k,v in base.items()) or r['run_id']!=task(mode,seed)['run_id']:
                    raise ValueError('Run source/identity mismatch')
                if r['status']=='PASS':
                    meta=json.loads(paths(mode,seed)['metadata'].read_text())
                    if meta['epoch']!=r['best_epoch'] or meta['metrics']!=r['best_valid_metrics'] or meta['checkpoint_sha256']!=r['checkpoint_sha256']:
                        raise ValueError('Checkpoint metadata mismatch')
                    if sha(paths(mode,seed)['checkpoint'])!=r['checkpoint_sha256']:
                        raise ValueError('Checkpoint bytes mismatch')
            r.update(source_json=str(p), source_sha256=sha(p))
            records[(seed,mode)]=r
        if seed!=2026 and all(records[(seed,m)]['status']=='PASS' for m in MODES):
            paired(records[(seed,'dual')],records[(seed,'triple')],first_batch=True)
    value=dict(**base, generated_at=now(), summaries=build(records),
               exploratory_seed=2026, scientific_fits=sum(records[(s,m)].get('scientific_fit_started',False) for s in SEEDS for m in MODES))
    value['status']='INCOMPLETE' if any(v['full']['incomplete'] for v in value['summaries'].values()) else 'PASS'
    create_record(SUMMARY,value)
    atomic_text(SUMMARY.with_suffix('.md'),markdown(value))
    try:
        atomic_text(SUMMARY.with_suffix('.svg'),svg(value))
    except Exception as exc:
        value['svg_error']=repr(exc)
        atomic_json(SUMMARY,value)


if __name__=='__main__':
    main()
