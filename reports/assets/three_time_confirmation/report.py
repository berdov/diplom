"""Publish saved SISO results using only the standard library; no model imports."""

import argparse
import csv
import hashlib
import io
import json
import math
import re
import statistics
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = Path('experiments/mamba3_three_time/confirmation')
EXECUTION = '995c5cde6449ea429c1d80ca6ab276b9791041c0'
OLD_MAIN = 'ccc4849e6e1ea460a316ab9648f36c54bd832fd4'
REPORT = Path('reports/MAMBA3_TIME_MECHANISMS_RESULTS.md')
ANCHOR = 'siso-dual-triple-confirmation'
MODES = ('dual', 'triple')


def read(path):
    return json.loads((ROOT / path).read_text())


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def relative(path):
    return Path(path).relative_to('/home/daryumin/iberdov/diplom')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def select_sources(entries):
    keys = [(e['seed'], e['mode']) for e in entries]
    expected = {(s, m) for s in range(2026, 2031) for m in MODES}
    require(len(keys) == 10 and set(keys) == expected, 'Source index incomplete or duplicated')
    require(len({e['path'] for e in entries}) == 10, 'Duplicated physical source')
    return entries


def records():
    plan = read(BASE / 'resume_plan_003.json')
    preservation = read(BASE / 'evidence/job4355314/preservation_manifest.json')
    hashes = {r['destination']: r['sha256'] for r in preservation['records']}
    checkpoints = {r['run_id']: r for r in preservation['checkpoints']}
    reservation = read(BASE / 'slurm_logs/attempt_003/submission_003.json')
    require(reservation['job_id'] == '4355314' and reservation['execution_commit'] == EXECUTION,
            'Continuation reservation mismatch')
    for e in plan['inherited_gates']:
        require(sha(e['path']) == sha(e['archive_path']) == e['sha256'], 'Inherited gate bytes changed')
        require(read(e['path'])['status'] == 'PASS', 'Inherited gate not PASS')
    rows = []
    for e in select_sources(plan['source_index']):
        path = e['path']
        r = read(path)
        identity = e.get('identity', {k: reservation[k] for k in
                       ('job_id', 'execution_commit', 'source_hash', 'core_hash', 'policy_sha256')})
        require(all(r[k] == v for k, v in identity.items()), f'Identity mismatch: {path}')
        require(sha(path) == hashes[path] == e.get('sha256', hashes[path]), f'Raw SHA: {path}')
        require(r['seed'] == e['seed'] and r['mode'] == e['mode'], 'Mode/seed mismatch')
        prefix = 'mamba3_three_time_siso' if e['kind'] == 'PILOT' else 'mamba3_three_time_confirm_siso'
        require(r['run_id'] == f"{prefix}_{e['mode']}_seed{e['seed']}_001", 'Run ID mismatch')
        require(r['status'] == 'PASS' and r['scientific_fit_started'], 'Scientific run not completed')
        require(r['backend'] == 'upstream' and r['architecture'] == 'SISO', 'Backend mismatch')
        require(r['TEST'] == 'NOT_RUN' and r['test_evaluation_count'] == 0, 'TEST used')
        require(r['selection_split'] == 'VALID' and r['evaluation_mode'] == 'full-ranking', 'Evaluation mismatch')
        require(r['policy_version'] == 'siso_numeric_acceptance_v1', 'Numeric policy mismatch')
        require(r['pinned_commit'] == 'e9594ce1c732d97440f0332fdc43170a2294dbfa', 'Mamba pin mismatch')
        require(r['parameter_count'] == dict(dual=610572, triple=610638)[r['mode']], 'Parameter count')
        h = r['history']
        require([x['epoch'] for x in h] == list(range(r['actual_epochs'])), 'History gaps')
        require(len(h) >= 27 and r['epoch_indexing'] == 'zero-based', 'Epoch horizon/indexing')
        scores = [x['valid_ndcg10'] for x in h]
        require(all(math.isfinite(x) for x in scores), 'Nonfinite score')
        best = max(i for i, v in enumerate(scores) if v == max(scores))
        require(r['best_epoch'] == best and r['best_valid_score'] == scores[best], 'Best epoch/tie')
        require(r['best_valid_metrics'] == h[best]['valid_metrics'], 'Best metrics')
        require(r['best_diagnostics'] == h[best]['diagnostics'], 'Best diagnostics')
        require(r['first27_best_ndcg10'] == max(scores[:27]), 'First27 mismatch')
        for x in h:
            require(x['valid_metrics']['ndcg@10'] == x['valid_ndcg10'], 'History metric mismatch')
            require(len(x['valid_metrics']) == 12 and all(math.isfinite(v) for v in x['valid_metrics'].values()),
                    'Missing/nonfinite metric')
        meta_path = relative(r['checkpoint_metadata_path'])
        meta = read(meta_path)
        ck = checkpoints[r['run_id']]
        require(sha(meta_path) == ck['metadata_sha256'] == hashes[str(meta_path)], 'Metadata SHA')
        for key in ('run_id', 'mode', 'seed', 'execution_commit', 'config_sha256', 'source_hash', 'core_hash', 'checkpoint_sha256'):
            require(meta[key] == r[key], f'Metadata {key}')
        require(meta['metrics'] == r['best_valid_metrics'] and meta['epoch'] == best, 'Metadata epoch/metrics')
        require(ck['sha256'] == r['checkpoint_sha256'] and ck['path'] == r['checkpoint_path'], 'Checkpoint evidence')
        log = (ROOT / meta_path).parent.parent / 'stderr.log'
        text = log.read_text()
        require('Traceback (most recent call last)' not in text, 'Unexpected traceback')
        epochs = re.findall(r'epoch (\d+) evaluating \[time: .*?valid_score: ([\d.]+)\]', text)
        require([(int(i), float(s)) for i, s in epochs] == list(enumerate(scores)), 'Log epoch scores')
        metrics = [dict((k, float(v)) for k, v in re.findall(r'((?:hit|recall|ndcg)@\d+) : ([\d.]+)', line))
                   for line in text.splitlines() if line.startswith('hit@5 :')]
        require(metrics == [x['valid_metrics'] for x in h], 'Log full metrics')
        cfg = r['config']
        config_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
        require(config_hash == r['config_sha256'], 'Config hash mismatch')
        require(cfg['eval_args'] == dict(split={'LS': 'valid_and_test'}, order='TO', group_by='user', mode='full'), 'Dataset split')
        require(cfg['time_scale_reference'] == 838393 and cfg['time_scale_reference_source'] == 'TRAIN', 'Time reference')
        require(r['protocol']['items'] == 7111 and r['protocol']['validation'] == 23951, 'Dataset counts')
        rows.append(dict(entry=e, raw=r, metadata_path=str(meta_path), checkpoint=ck))
    require(len({x['raw']['run_id'] for x in rows}) == 10, 'Duplicated logical run')
    by = {(x['raw']['seed'], x['raw']['mode']): x['raw'] for x in rows}
    for r in by.values():
        pilot = by[2026, r['mode']]
        for name in ('config', 'effective_config'):
            strip = lambda v: {k: val for k, val in v.items() if k not in ('seed', 'checkpoint_dir')}
            require(strip(r[name]) == strip(pilot[name]), f'{name} differs from same-mode pilot')
        for name in ('protocol', 'manifest_sha256', 'train_time_stats_sha256', 'policy_sha256', 'runtime'):
            require(r[name] == pilot[name], f'Dataset/policy/runtime mismatch: {name}')
    for seed in range(2026, 2031):
        a, b = (by[seed, m] for m in MODES)
        for name in ('initial_backbone_sha256', 'rng_before_fit_sha256', 'first_train_batch_sha256'):
            require(a[name] == b[name], f'Pair parity {seed} {name}')
        ca, cb = a['initial_calibrator_hashes'], b['initial_calibrator_hashes']
        require(ca['decay'] == cb['decay'] and ca['scan'] == cb['write'] == cb['phase'], 'Calibrator initialization')
        if seed != 2026:
            for name in ('rng_components', 'optimizer_settings', 'precision'):
                require(a[name] == b[name], f'Pair parity {seed} {name}')
    return rows


def statistics_for(pairs, first27=False):
    suffix = '_first27' if first27 else ''
    dual = [p['dual' + suffix] for p in pairs]
    triple = [p['triple' + suffix] for p in pairs]
    delta = [b - a for a, b in zip(dual, triple)]
    stats = lambda xs: dict(mean=statistics.mean(xs), sample_std=statistics.stdev(xs))
    return dict(dual=stats(dual), triple=stats(triple), delta=stats(delta), n=len(delta),
                positive=sum(d > 0 for d in delta), negative=sum(d < 0 for d in delta),
                zero=sum(d == 0 for d in delta), relative_percent=100 * (statistics.mean(triple) / statistics.mean(dual) - 1))


def aggregate(rows):
    by = {(r['raw']['seed'], r['raw']['mode']): r['raw'] for r in rows}
    pairs = []
    for seed in range(2026, 2031):
        pair = dict(seed=seed)
        for mode in MODES:
            r = by[seed, mode]
            pair.update({mode: r['best_valid_score'], mode + '_first27': max(x['valid_ndcg10'] for x in r['history'][:27]),
                         mode + '_hr10': r['best_valid_metrics']['hit@10'], mode + '_best': r['best_epoch'],
                         mode + '_epochs': r['actual_epochs']})
        pair['delta'] = pair['triple'] - pair['dual']
        pairs.append(pair)
    diagnostics = []
    for seed in range(2027, 2031):
        d = by[seed, 'triple']['best_diagnostics']
        diagnostics.append(dict(seed=seed, mean_abs_log_difference=d['write_phase_log_comparison']['mean_abs_log_difference'],
                                log_correlation=d['write_phase_log_comparison']['log_correlation'],
                                max_head_write_below_bound=max(h[0] for h in d['near_bound_fractions'][1]),
                                max_head_phase_above_bound=max(h[1] for h in d['near_bound_fractions'][2])))
    result = dict(pairs=pairs, best_triple_diagnostics=diagnostics)
    saved = read(BASE / 'runs/attempt_003/confirmation_summary.json')
    pipeline = read(BASE / 'slurm_logs/attempt_003/pipeline_status.json')
    for record in (saved, pipeline):
        require(record['status'] == 'PASS' and record['scientific_fits_this_attempt'] == 7
                and record['total_completed_confirmation_fits'] == 8, 'Incomplete continuation')
        reservation = read(BASE / 'slurm_logs/attempt_003/submission_003.json')
        for key in ('job_id', 'execution_commit', 'source_hash', 'core_hash', 'policy_sha256',
                    'resume_plan_sha256', 'lineage_sha256', 'login_verification_sha256'):
            require(record[key] == reservation[key], 'Summary/pipeline identity: ' + key)
        require(record['TEST'] == 'NOT_RUN' and record['test_evaluation_count'] == 0, 'Summary/pipeline TEST')
    for label, subset in [('new_four_pairs', pairs[1:]), ('all_five_pairs', pairs)]:
        result[label] = {}
        for horizon in ('full', 'first27'):
            stats = statistics_for(subset, horizon == 'first27')
            old = saved['summaries'][label][horizon]
            require(not old['incomplete'] and old['n_available'] == len(subset), 'Raw summary incomplete')
            for mode in (*MODES, 'delta'):
                original = old['paired_delta'] if mode == 'delta' else old['mode_statistics_all_available'][mode]
                require(math.isclose(stats[mode]['mean'], original['mean'], abs_tol=1e-15), 'Raw summary mean')
                require(math.isclose(stats[mode]['sample_std'], original['sample_std_ddof1'], abs_tol=1e-15), 'Raw summary std')
            for sign in ('positive', 'negative', 'zero'):
                require(stats[sign] == old[sign], 'Raw summary signs')
            require(math.isclose(stats['relative_percent'], old['relative_gain_percent'], abs_tol=1e-12), 'Relative gain')
            result[label][horizon] = stats
    return result


def pm(stats, latex=False):
    middle = r' \pm ' if latex else ' ± '
    return f"{stats['mean']:.6f}{middle}{stats['sample_std']:.6f}"


def tables(summary):
    lines = ['| Seed | Dual NDCG@10 | Triple NDCG@10 | Triple − dual |', '|---|---:|---:|---:|']
    epochs = ['| Seed | HR@10 dual / triple | Лучшая эпоха dual / triple (с нуля) | Всего эпох dual / triple |', '|---|---:|---:|---:|']
    for p in summary['pairs']:
        label = str(p['seed']) + (' (пилот)' if p['seed'] == 2026 else '')
        lines.append(f"| {label} | {p['dual']:.4f} | {p['triple']:.4f} | {p['delta']:+.4f} |")
        epochs.append(f"| {p['seed']} | {p['dual_hr10']:.4f} / {p['triple_hr10']:.4f} | {p['dual_best']} / {p['triple_best']} | {p['dual_epochs']} / {p['triple_epochs']} |")
    aggregates = ['| Пары | Dual: mean ± std | Triple: mean ± std | Парная Δ: mean ± std | + / − / 0 | Прирост по средним |', '|---|---:|---:|---:|---:|---:|']
    for label, title in [('new_four_pairs', '2027–2030, подтверждение'), ('all_five_pairs', '2026–2030, включая пилот')]:
        s = summary[label]['full']
        aggregates.append(f"| {title} | {pm(s['dual'])} | {pm(s['triple'])} | {pm(s['delta'])} | {s['positive']} / {s['negative']} / {s['zero']} | +{s['relative_percent']:.3f}% |")
    first = ['| Seed | Dual, первые 27 | Triple, первые 27 | Δ |', '|---|---:|---:|---:|']
    for p in summary['pairs']:
        first.append(f"| {p['seed']} | {p['dual_first27']:.4f} | {p['triple_first27']:.4f} | {p['triple_first27']-p['dual_first27']:+.4f} |")
    diag = ['| Seed | Средняя абсолютная Δ log(write/phase) | Корреляция log-scales | max head: write < 0.51 | max head: phase > 1.99 |', '|---|---:|---:|---:|---:|']
    for d in summary['best_triple_diagnostics']:
        diag.append(f"| {d['seed']} | {d['mean_abs_log_difference']:.3f} | {d['log_correlation']:.3f} | {100*d['max_head_write_below_bound']:.1f}% | {100*d['max_head_phase_above_bound']:.1f}% |")
    return dict(pairs='\n'.join(lines), aggregates='\n'.join(aggregates), epochs='\n'.join(epochs), first27='\n'.join(first), diagnostics='\n'.join(diag))


def svg(summary):
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="920" height="444" viewBox="0 0 920 444" role="img" aria-labelledby="title desc">',
             '<title id="title">SISO: парная разница VALID NDCG@10</title>',
             '<desc id="desc">Triple минус dual. Пилот 2026 отделён; отрицательная пара 2027 сохранена. Без доверительных интервалов.</desc>',
             '<rect width="920" height="444" fill="white"/>',
             '<g font-family="Arial, sans-serif" fill="#202124">',
             '<text x="28" y="34" font-size="22">SISO: парная разница VALID NDCG@10</text>',
             '<text x="28" y="61" font-size="15">Δ = triple − dual; один seed — одна точка; TEST не использовался</text>']
    x = lambda value: 540 + value / .002 * 250
    for tick in (-.002, -.001, 0, .001, .002):
        tx = x(tick)
        parts += [f'<line x1="{tx}" y1="92" x2="{tx}" y2="355" stroke="{"#555" if tick == 0 else "#e4e6e8"}"/>',
                  f'<text x="{tx}" y="381" text-anchor="middle" font-size="13">{tick:+.3f}</text>']
    parts.append('<line x1="28" y1="147" x2="890" y2="147" stroke="#999" stroke-dasharray="5 5"/>')
    for p, y in zip(summary['pairs'], (117, 186, 235, 284, 333)):
        pilot = p['seed'] == 2026
        label = '2026 · exploratory pilot' if pilot else str(p['seed'])
        color = '#777777' if pilot else ('#a43647' if p['delta'] < 0 else '#16716b')
        parts += [f'<text x="28" y="{y+5}" font-size="15">{label}</text>',
                  f'<circle cx="{x(p["delta"]):.3f}" cy="{y}" r="7" fill="{"white" if pilot else color}" stroke="{color}" stroke-width="2"/>',
                  f'<text x="875" y="{y+5}" text-anchor="end" font-size="15">{p["delta"]:+.4f}</text>']
    parts += ['<text x="28" y="421" font-size="14">2027–2030: заранее выбранные подтверждающие пары; 3 улучшения, 1 снижение.</text>', '</g>', '</svg>']
    return '\n'.join(parts) + '\n'


def tex(summary):
    lines = [r'\begin{table}[t]', r'\centering', r'\small',
             r'\caption{SISO validation ablation on KuaiRand (chronological leave-one-out, full catalog), not a TEST benchmark.}',
             r'\label{tab:siso-dual-triple-validation}', r'\begin{tabular}{lrrr}', r'\toprule',
             r'Seed / group & Dual & Triple & $\Delta$ (triple--dual) \\', r'\midrule']
    for p in summary['pairs']:
        label = str(p['seed']) + (' (pilot)' if p['seed'] == 2026 else '')
        lines.append(f"{label} & {p['dual']:.4f} & {p['triple']:.4f} & ${p['delta']:+.4f}$ " + r'\\')
    for label, title in [('new_four_pairs', '2027--2030: four confirmation pairs'), ('all_five_pairs', '2026--2030: five pairs including exploratory pilot')]:
        s = summary[label]['full']
        lines += [r'\midrule', r'\multicolumn{4}{l}{' + title + r'} \\',
                  r'Mean $\pm$ std ' + ''.join(f"& ${pm(s[m], latex=True)}$ " for m in (*MODES, 'delta')) + r'\\']
    lines += [r'\bottomrule', r'\end{tabular}', r'\par\smallskip',
              r'\begin{minipage}{0.98\linewidth}\footnotesize',
              r'All scores are VALID NDCG@10. Sample standard deviation uses $n-1$; it is not a confidence interval. The last column summarizes paired differences, not differences of standard deviations. Seed 2026 informed the decision to continue; it is not independent confirmation.',
              r'\end{minipage}', r'\end{table}', '']
    return '\n'.join(lines)


def sources(rows):
    output = []
    for item in rows:
        e, r, ck = item['entry'], item['raw'], item['checkpoint']
        output.append(dict(path=e['path'], sha256=sha(e['path']), run_id=r['run_id'], mode=r['mode'], seed=r['seed'],
                           role={'PILOT': 'pilot', 'REUSED_COMPLETED': 'parent', 'NEW': 'new'}[e['kind']],
                           execution_commit=r['execution_commit'], job_id=r['job_id'], source_hash=r['source_hash'],
                           checkpoint_path=r['checkpoint_path'], checkpoint_sha256=r['checkpoint_sha256'],
                           metadata_path=item['metadata_path'], metadata_sha256=ck['metadata_sha256']))
    return dict(source_index=str(BASE / 'resume_plan_003.json'), source_index_sha256=sha(BASE / 'resume_plan_003.json'),
                preservation_manifest=str(BASE / 'evidence/job4355314/preservation_manifest.json'), sources=output)


def registry_rows(rows, header):
    output = []
    for item in rows:
        r = item['raw']
        row = dict(record_type='experiment', source='ours', run_id=r['run_id'], model='ThreeTimeMamba3Rec',
                   model_variant='SISO_' + r['mode'], dataset='KuaiRand', protocol='B', split='validation',
                   evaluation='full_7111_items', status='completed', seed=r['seed'], train_candidates='full_softmax',
                   item_universe=r['protocol']['items'], best_epoch=r['best_epoch'], actual_epochs=r['actual_epochs'],
                   validation_ndcg10=r['best_valid_score'], test_evaluation_count=0, git_commit=r['execution_commit'],
                   notes_path=str(REPORT) + '#' + ANCHOR, test_used='no', source_json=item['entry']['path'])
        for label, raw_name in [('HR', 'hit'), ('Recall', 'recall'), ('NDCG', 'ndcg')]:
            for k in (5, 10, 20, 50):
                row[f'{label}@{k}'] = r['best_valid_metrics'][f'{raw_name}@{k}']
        require(not set(row) - set(header), 'Unknown registry columns')
        output.append({k: str(row.get(k, '')) for k in header})
    return output


def registry(rows, append=False):
    path = ROOT / 'experiments/results.csv'
    original = subprocess.check_output(['git', 'show', f'{OLD_MAIN}:experiments/results.csv'], cwd=ROOT)
    current = path.read_bytes()
    require(current.startswith(original), 'Historical CSV bytes changed')
    existing = list(csv.DictReader(io.StringIO(current.decode())))
    header = next(csv.reader(io.StringIO(current.decode())))
    expected = registry_rows(rows, header)
    missing = []
    for r in expected:
        found = [x for x in existing if x['run_id'] == r['run_id']]
        require(len(found) <= 1 and (not found or found[0] == r), f"Registry conflict: {r['run_id']}")
        if not found:
            missing.append(r)
    if append and missing:
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=header, lineterminator='\n')
        writer.writerows(missing)
        require(current.endswith(b'\n'), 'Registry lacks final newline')
        with path.open('ab') as out:
            out.write(stream.getvalue().encode())
    else:
        require(not missing, 'Registry rows not yet appended')
    before = len(list(csv.DictReader(io.StringIO(original.decode()))))
    return dict(before=before, added=len(expected), after=len(existing)+len(missing), old_bytes_sha256=hashlib.sha256(original).hexdigest())


def audit_sources():
    manifest = read(BASE / 'evidence/job4355314/preservation_manifest.json')
    parent = read(BASE / 'evidence/job4355052/source_manifest.json')['files']
    final = read(BASE / 'source_manifest.json')['files']
    changes = read(BASE / 'resume_lineage_003.json')['changes']
    require({p for p in parent if parent[p] != final.get(p)} == set(changes), 'Undeclared parent source change')
    for path, change in changes.items():
        require(parent[path] == change['before_sha256'] and final[path] == change['after_sha256'], 'Lineage SHA: ' + path)
    reservation = read(BASE / 'slurm_logs/attempt_003/submission_003.json')
    for key, path in [('source_manifest_sha256', BASE / 'source_manifest.json'),
                      ('resume_plan_sha256', BASE / 'resume_plan_003.json'),
                      ('lineage_sha256', BASE / 'resume_lineage_003.json'),
                      ('login_verification_sha256', BASE / 'slurm_logs/attempt_003/login_verification.json')]:
        require(sha(path) == reservation[key], 'Reservation binding: ' + key)
    for record in manifest['records']:
        require(sha(record['destination']) == record['sha256'], 'Preserved file changed')
    changed_documents = []
    allowed = {'README.md', 'reports/RESULTS.md', str(REPORT), 'reports/PAPER_RESULTS.md', 'experiments/results.csv'}
    for snapshot in manifest['historical_manifests']:
        sources_manifest = read(snapshot['manifest'])
        for path, expected in sources_manifest['files'].items():
            data = subprocess.check_output(['git', 'show', snapshot['execution_commit'] + ':' + path], cwd=ROOT)
            require(hashlib.sha256(data).hexdigest() == expected, 'Historical Git snapshot mismatch: ' + path)
            if snapshot['execution_commit'] == EXECUTION and sha(path) != expected:
                require(path in allowed, 'Frozen source changed: ' + path)
                changed_documents.append(path)
    return dict(historical_snapshots='PASS', preserved_files=len(manifest['records']),
                checkpoint_hashes_streamed_on_cluster=len(manifest['checkpoints']),
                declared_parent_source_changes_verified=len(changes),
                mathematical_sources='UNCHANGED', current_documents_differ_from_execution=sorted(changed_documents),
                current_whole_tree_equals_old_manifest=not changed_documents)


def outputs(rows, summary):
    dump = lambda obj: json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    return {'paired_delta.svg': svg(summary), 'siso_dual_triple_table.tex': tex(summary),
            'sources.json': dump(sources(rows)), 'summary.json': dump(summary)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-derived', action='store_true', help='Write only derived assets and marked report tables')
    parser.add_argument('--append-registry', action='store_true')
    parser.add_argument('--audit', action='store_true', help='Check historical Git objects and frozen current sources')
    args = parser.parse_args()
    rows = records()
    summary = aggregate(rows)
    for name, content in outputs(rows, summary).items():
        path = HERE / name
        if args.write_derived:
            path.write_text(content)
        else:
            require(path.read_text() == content, 'Derived file stale: ' + name)
    report = (ROOT / REPORT).read_text()
    for name, table in tables(summary).items():
        start, end = f'<!-- siso:{name}:start -->', f'<!-- siso:{name}:end -->'
        require(report.count(start) == report.count(end) == 1, 'Missing/duplicated table markers')
        a, rest = report.split(start)
        middle, b = rest.split(end)
        if args.write_derived:
            report = a + start + '\n' + table + '\n' + end + b
        else:
            require(middle == '\n' + table + '\n', 'Report table stale: ' + name)
    if args.write_derived:
        (ROOT / REPORT).write_text(report)
    registry_status = registry(rows, args.append_registry)
    print(json.dumps(dict(runs_verified=len(rows), registry=registry_status,
                         audit=audit_sources() if args.audit else 'not requested'), indent=2))


if __name__ == '__main__':
    main()
