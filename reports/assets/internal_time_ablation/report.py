"""Read saved JSON only; reproduce SISO/MIMO publication assets and registry checks."""
import argparse
import csv
import hashlib
import io
import json
import math
import re
import runpy
import statistics
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
C = Path('experiments/mamba3_mimo_time/confirmation')
OLD_MAIN = '9a01d220efa64b9d0fe6b4a7e69b948ebbe16f07'
EXECUTION = '5670e898ed04924a929756e52f39d1d00eb79c5a'
REPORT = ROOT / 'reports/MAMBA3_TIME_MECHANISMS_RESULTS.md'
MODES = ('base', 'dual', 'triple')
CONTRASTS = (('triple', 'dual'), ('dual', 'base'), ('triple', 'base'))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads((ROOT / path).read_text())


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def dump(value):
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'


def relative(path):
    return str(Path(path).relative_to('/home/daryumin/iberdov/diplom'))


def last_best(history):
    scores = [x['valid_ndcg10'] for x in history]
    return max(i for i, score in enumerate(scores) if score == max(scores))


def first27(record):
    h = record['history']
    return max(x['valid_ndcg10'] for x in h[:27]) if len(h) >= 27 else None


def records():
    # The historical publisher is itself stdlib-only; never import experiment code.
    siso = runpy.run_path(str(HERE.parent / 'three_time_confirmation/report.py'))
    sr = siso['records']()
    sources = siso['sources'](sr)['sources']
    rows = [x['raw'] for x in sr]
    sources = [dict(e, architecture='SISO', split='VALID',
                    role='pilot' if e['seed'] == 2026 else 'confirmation') for e in sources]
    plan = read(C / 'study_plan.json')
    frozen = read(C / 'source_manifest.json')
    expected = {(s, m) for s in range(2027, 2031) for m in MODES}
    require(len(plan['tasks']) == 12 and {(t['seed'], t['mode']) for t in plan['tasks']} == expected,
            'Unexpected confirmation plan')
    preserved = read(C / 'evidence/job4358583/preservation_manifest.json')
    pilot_evidence = read('experiments/mamba3_mimo_time/evidence/job4358147/preservation_manifest.json')
    hashes = {}
    for manifest in (preserved, pilot_evidence):
        for e in manifest['files']:
            require(sha(e['destination']) == e['sha256'], 'Preserved bytes: ' + e['destination'])
            require((ROOT / e['destination']).stat().st_size == e['size'], 'Preserved size')
            hashes[e['destination']] = e['sha256']
    checkpoints = {relative(e['path']) if e['path'].startswith('/') else e['path']: e
                   for m in (preserved, pilot_evidence) for e in m['checkpoints']}
    paths = list(plan['config_sources'].values()) + [str(C / 'runs' / (t['run_id'] + '.json')) for t in plan['tasks']]
    for path in paths:
        r = read(path)
        require(r['architecture'] == 'MIMO' and r['backend'] == 'upstream', 'MIMO backend')
        require(r['status'] == 'PASS' and r['stage'] == 'COMPLETED' and r['scientific_fit_started'], 'Incomplete fit')
        require(r['TEST'] == 'NOT_RUN' and r['test_evaluation_count'] == 0, 'TEST used')
        require(r['selection_split'] == 'VALID' and r['evaluation_mode'] == 'full-ranking', 'Split')
        require(r['rank'] == 4 and r['chunk'] == 8 and r['history_length'] == 50
                and r['kernel_length_for_max_history'] == 56, 'Architecture configuration')
        pilot = r['seed'] == 2026
        require(r['execution_commit'] == (plan['pilot_execution'] if pilot else EXECUTION), 'Execution identity')
        require(r['job_id'] == ('4358147' if pilot else '4358583'), 'Job identity')
        prefix = 'mamba3_mimo' if pilot else 'mamba3_mimo_confirm'
        require(r['run_id'] == f"{prefix}_{r['mode']}_seed{r['seed']}_001", 'Run identity')
        require(r['policy_sha256'] == plan['policy_sha256'], 'Policy mismatch')
        if not pilot:
            require(r['source_hash'] == frozen['source_hash'] and r['source_manifest_sha256'] == sha(C / 'source_manifest.json')
                    and r['plan_sha256'] == sha(C / 'study_plan.json'), 'Frozen provenance')
            require(r['inherited_record_sha256'] == sha(C / 'runs/inherited_admission_001.json'), 'Inherited record SHA')
            require(r['reservation_sha256'] == sha(C / 'slurm_logs/reservation_001.json')
                    and r['login_verification_sha256'] == sha(C / 'slurm_logs/login_verification_001.json'), 'Reservation/login SHA')
        require(r['parameter_count'] == plan['parameter_counts'][r['mode']], 'Parameter count')
        for kind in ('admission', 'smoke'):
            key = kind + '_sha256' if pilot else 'inherited_' + kind + '_sha256'
            require(r[key] == plan['inherited_' + kind + '_sha256'], 'Gate lineage')
        h = r['history']
        require([x['epoch'] for x in h] == list(range(r['actual_epochs'])), 'History gap')
        require(all(len(x['valid_metrics']) == 12 and all(math.isfinite(v) for v in x['valid_metrics'].values())
                    and x['valid_ndcg10'] == x['valid_metrics']['ndcg@10'] for x in h), 'History metrics')
        best = last_best(h)
        require(r['best_epoch'] == best and r['best_valid_score'] == h[best]['valid_ndcg10'], 'Last-tie selection')
        require(r['best_valid_metrics'] == h[best]['valid_metrics'] and r['best_diagnostics'] == h[best]['diagnostics'], 'Best record')
        require(r['actual_epochs'] - best - 1 == 11, 'Early stopping semantics')
        require(r['first27_complete'] == (first27(r) is not None), 'First27 completeness')
        require(r['first27_best_ndcg10'] == max(x['valid_ndcg10'] for x in h[:27]), 'Observed first27')
        meta_path = relative(r['checkpoint_metadata_path'])
        meta = read(meta_path)
        ck = checkpoints[relative(r['checkpoint_path'])]
        require(sha(meta_path) == hashes[meta_path] == ck['metadata_sha256'], 'Metadata SHA')
        require(r['checkpoint_sha256'] == ck['sha256'] == meta['checkpoint_sha256'], 'Checkpoint SHA evidence')
        for key in ('run_id', 'mode', 'seed', 'execution_commit', 'source_hash', 'core_hash', 'config_sha256'):
            require(meta[key] == r[key], 'Metadata ' + key)
        require(meta['epoch'] == best and meta['metrics'] == r['best_valid_metrics'], 'Metadata best metrics')
        log = (ROOT / meta_path).parent.parent / 'process/stderr.log'
        if pilot:
            log = (ROOT / meta_path).parents[2] / r['mode'] / 'stderr.log'
        text = log.read_text()
        logged = [(int(i), float(v)) for i, v in re.findall(r'epoch (\d+) evaluating \[time: .*?valid_score: ([\d.]+)\]', text)]
        require(logged == [(x['epoch'], x['valid_ndcg10']) for x in h], 'Logged scores')
        require('Traceback (most recent call last)' not in text, 'Process traceback')
        metrics = [dict((k, float(v)) for k, v in re.findall(r'((?:hit|recall|ndcg)@\d+) : ([\d.]+)', line))
                   for line in text.splitlines() if line.startswith('hit@5 :')]
        require(metrics == [x['valid_metrics'] for x in h], 'Logged full metrics')
        require(hashlib.sha256(json.dumps(r['config'], sort_keys=True).encode()).hexdigest() == r['config_sha256'], 'Config SHA')
        require(r['config']['eval_args'] == dict(split={'LS': 'valid_and_test'}, order='TO', group_by='user', mode='full'), 'Dataset split')
        require(r['config']['time_scale_reference'] == 838393 and r['config']['time_scale_reference_source'] == 'TRAIN', 'Time reference')
        rows.append(r)
        sources.append(dict(architecture='MIMO', split='VALID', role='pilot' if pilot else 'confirmation',
                            path=path, sha256=hashes[path], **{k: r[k] for k in
                            ('run_id', 'mode', 'seed', 'execution_commit', 'job_id', 'source_hash', 'checkpoint_path', 'checkpoint_sha256')},
                            metadata_path=meta_path, metadata_sha256=sha(meta_path)))
    by = {(r['architecture'], r['seed'], r['mode']): r for r in rows}
    require(len(by) == len(rows) == len({r['run_id'] for r in rows}) == 25, '25 unique runs required')
    require(set(by) == {(a, s, m) for a, modes in [('SISO', ('dual', 'triple')), ('MIMO', MODES)]
                        for s in range(2026, 2031) for m in modes}, 'Exact 25-run source selection')
    for r in rows:
        if r['architecture'] != 'MIMO':
            continue
        pilot = by['MIMO', 2026, r['mode']]
        for k in ('config', 'effective_config'):
            strip = lambda d: {a: b for a, b in d.items() if a not in ('seed', 'checkpoint_dir')}
            require(strip(r[k]) == strip(pilot[k]), 'Same-mode config drift')
        for k in ('protocol', 'manifest_sha256', 'train_time_stats_sha256', 'core_hash', 'pinned_commit', 'policy_sha256'):
            require(r[k] == pilot[k], 'Data/core/policy drift: ' + k)
        runtime = lambda d: {k: v for k, v in d.items() if k != 'imported_sources'}
        require(runtime(r['runtime']) == runtime(pilot['runtime']), 'Runtime versions drift')
    for seed in range(2026, 2031):
        a, b, c = (by['MIMO', seed, m] for m in MODES)
        for k in ('initial_backbone_sha256', 'rng_components', 'first_train_batch_sha256', 'precision', 'optimizer_settings'):
            require(a[k] == b[k] == c[k], 'Within-seed parity: ' + k)
        bc, cc = b['initial_calibrator_hashes'], c['initial_calibrator_hashes']
        require(not a['initial_calibrator_hashes'] and bc['decay'] == cc['decay']
                and bc['scan'] == cc['write'] == cc['phase'] and c['independent_write_phase_storage'], 'Calibrator initialization')
    for kind in ('admission', 'smoke'):
        require(sha(f'experiments/mamba3_mimo_time/runs/attempt_003/{kind}_001.json') == plan['inherited_' + kind + '_sha256'], 'Inherited bytes')
    for p in ('runs/confirmation_summary.json', 'slurm_logs/pipeline_status.json'):
        r = read(C / p)
        require(r['status'] == 'PASS' and r['scientific_fits_completed'] == 12, 'Pipeline incomplete')
        require(r['execution_commit'] == EXECUTION and r['job_id'] == '4358583'
                and r['TEST'] == 'NOT_RUN' and r['test_evaluation_count'] == 0, 'Pipeline identity')
    return rows, dict(selection=dict(SISO='experiments/mamba3_three_time/confirmation/resume_plan_003.json',
                                    MIMO=str(C / 'study_plan.json')), sources=sources)


def stats(values):
    return dict(mean=statistics.mean(values), sample_std=statistics.stdev(values), ddof=1)


def cohort(rows, architecture, seeds, modes, horizon='full'):
    by = {(r['seed'], r['mode']): r for r in rows if r['architecture'] == architecture}
    selected = [s for s in seeds if all((s, m) in by and (horizon == 'full' or first27(by[s, m]) is not None) for m in modes)]
    require(len(selected) > 1, 'At least two complete seeds for sample std')
    value = lambda s, m: by[s, m]['best_valid_score'] if horizon == 'full' else first27(by[s, m])
    contrasts = {}
    for left, right in CONTRASTS:
        if left not in modes or right not in modes:
            continue
        delta = [value(s, left) - value(s, right) for s in selected]
        contrasts[left + '-' + right] = dict(**stats(delta), deltas=delta,
            positive=sum(d > 0 for d in delta), negative=sum(d < 0 for d in delta), zero=sum(d == 0 for d in delta),
            relative_percent=100 * (statistics.mean(value(s, left) for s in selected) / statistics.mean(value(s, right) for s in selected) - 1))
    return dict(architecture=architecture, horizon=horizon, seeds=selected, n_seeds=len(selected),
                models={m: stats([value(s, m) for s in selected]) for m in modes}, contrasts=contrasts)


def aggregate(rows):
    result = dict(metric='VALID NDCG@10', dataset='KuaiRand: хронологический leave-one-out, полный каталог',
                  std='sample standard deviation, ddof=1; not a confidence interval', cohorts={})
    for arch, modes in [('SISO', ('dual', 'triple')), ('MIMO', MODES)]:
        for label, seeds in [('new4', list(range(2027, 2031))), ('all5', list(range(2026, 2031)))]:
            result['cohorts'][arch + '_' + label] = cohort(rows, arch, seeds, modes)
            result['cohorts'][arch + '_' + label + '_first27'] = cohort(rows, arch, seeds, modes, 'first27')
    result['cohorts']['MIMO_dual_triple_new4_first27'] = cohort(rows, 'MIMO', list(range(2027, 2031)), ('dual', 'triple'), 'first27')
    result['runs'] = [dict(**{k: r[k] for k in ('run_id', 'architecture', 'mode', 'seed', 'best_valid_score', 'best_epoch', 'actual_epochs', 'train_seconds', 'valid_seconds', 'peak_gpu_allocated_bytes', 'peak_gpu_reserved_bytes')},
                           first27=first27(r), observed_first27=max(x['valid_ndcg10'] for x in r['history'][:27]),
                           observed_first27_epochs=min(27, len(r['history']))) for r in rows]
    return result


def pm(d, signed=False):
    return f"{d['mean']:+.6f}" + f" ± {d['sample_std']:.6f}" if signed else f"{d['mean']:.6f} ± {d['sample_std']:.6f}"


def tables(rows, summary):
    by = {(r['seed'], r['mode']): r for r in rows if r['architecture'] == 'MIMO'}
    pairs = ['| Seed | Base | Dual | Triple | Triple − dual | Dual − base | Triple − base |', '|---|---:|---:|---:|---:|---:|---:|']
    epochs = ['| Seed | HR@10 base / dual / triple | Best epoch (с нуля), B / D / T | Actual epochs, B / D / T |', '|---|---:|---:|---:|']
    windows = ['| Seed | Base, first27 | Dual, first27 | Triple, first27 |', '|---|---:|---:|---:|']
    diagnostics = ['| Seed | mean abs log(write/phase), H0 / H1 | phase > 1.99, H0 / H1 |', '|---|---:|---:|']
    for seed in range(2026, 2031):
        rs = [by[seed, m] for m in MODES]
        vals = [r['best_valid_score'] for r in rs]
        label = str(seed) + (' (пилот)' if seed == 2026 else '')
        links = [f"[{r['best_valid_score']:.4f}](../{relative(r['checkpoint_metadata_path']).split('/slurm_logs/')[0]}/runs/" +
                 ('attempt_003/' if seed == 2026 else '') + r['run_id'] + '.json)' for r in rs]
        pairs.append('| ' + label + ' | ' + ' | '.join(links + [f'{vals[2]-vals[1]:+.4f}', f'{vals[1]-vals[0]:+.4f}', f'{vals[2]-vals[0]:+.4f}']) + ' |')
        epochs.append('| ' + label + ' | ' + ' | '.join([' / '.join(f"{r['best_valid_metrics']['hit@10']:.4f}" for r in rs), ' / '.join(str(r['best_epoch']) for r in rs), ' / '.join(str(r['actual_epochs']) for r in rs)]) + ' |')
        windows.append('| ' + label + ' | ' + ' | '.join(f'{first27(r):.4f}' if first27(r) is not None else f"нет ({r['actual_epochs']} эпох)" for r in rs) + ' |')
        d = rs[2]['best_diagnostics']
        diagnostics.append('| ' + label + ' | ' + ' / '.join(f'{v:.3f}' for v in d['mean_abs_write_phase_log_difference_per_head']) + ' | ' + ' / '.join(f'{100*h[1]:.2f}%' for h in d['near_bound_fractions'][2]) + ' |')
    aggregates = ['| Seeds; n | Base: mean ± std | Dual: mean ± std | Triple: mean ± std |', '|---|---:|---:|---:|']
    contrasts = ['| Seeds; n | Контраст | Парная Δ: mean ± std | + / − / 0 | Прирост по средним |', '|---|---|---:|---:|---:|']
    first_stats = ['| Полные first27 окна; n | Base: mean ± std | Dual: mean ± std | Triple: mean ± std | Δ triple−dual: mean ± std |', '|---|---:|---:|---:|---:|']
    for label, name in [('2027–2030; 4', 'MIMO_new4'), ('2026–2030; 5, с пилотом', 'MIMO_all5')]:
        c = summary['cohorts'][name]
        aggregates.append('| ' + label + ' | ' + ' | '.join(pm(c['models'][m]) for m in MODES) + ' |')
        for contrast, d in c['contrasts'].items():
            contrasts.append(f"| {label} | {contrast} | {pm(d, True)} | {d['positive']} / {d['negative']} / {d['zero']} | {d['relative_percent']:+.3f}% |")
    for name in ('MIMO_new4_first27', 'MIMO_dual_triple_new4_first27'):
        c = summary['cohorts'][name]
        first_stats.append('| ' + ', '.join(map(str, c['seeds'])) + f"; {c['n_seeds']} | " + ' | '.join(pm(c['models'][m]) if m in c['models'] else '—' for m in MODES) + ' | ' + pm(c['contrasts']['triple-dual'], True) + ' |')
    return {k: '\n'.join(v) for k, v in dict(pairs=pairs, aggregates=aggregates, contrasts=contrasts, epochs=epochs, first27=windows, first27_aggregates=first_stats, diagnostics=diagnostics).items()}


def svg(summary):
    c = summary['cohorts']['MIMO_all5']
    points = c['contrasts']['triple-dual']['deltas']
    # Standalone SVG, with numeric positions calculated from the same aggregates.
    x = lambda v: 460 + v * 160000
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="450" viewBox="0 0 900 450" role="img" aria-labelledby="title desc">',
             '<title id="title">MIMO: paired VALID NDCG@10 differences</title>',
             '<desc id="desc">Triple minus dual for five seeds. Pilot 2026 and confirmation seed 2029 are negative. No confidence intervals.</desc>',
             '<rect width="900" height="450" fill="white"/>',
             '<g font-family="Arial, sans-serif" fill="#172936">',
             '<text x="40" y="38" font-size="23" font-weight="bold">MIMO: triple − dual</text>',
             '<text x="40" y="65" font-size="15">KuaiRand: хронологический leave-one-out, полный каталог · VALID</text>']
    for tick in (-.002, -.001, 0, .001, .002):
        xx = x(tick)
        lines += [f'<line x1="{xx}" x2="{xx}" y1="92" y2="350" stroke="{"#667984" if tick == 0 else "#e2e8ec"}" stroke-width="{2 if tick == 0 else 1}"/>',
                  f'<text x="{xx}" y="379" text-anchor="middle" font-size="14">{tick:+.3f}</text>']
    for i, (seed, delta) in enumerate(zip(c['seeds'], points)):
        xx, yy = x(delta), 115 + i * 51
        color = '#b65e21' if i == 0 else '#156d8c'
        lines.append(f'<text x="40" y="{yy+5}" font-size="15">{seed}{" pilot" if i == 0 else ""}</text>')
        if i == 0:
            lines.append(f'<path d="M {xx} {yy-8} L {xx+8} {yy} L {xx} {yy+8} L {xx-8} {yy} Z" fill="{color}"/>')
        else:
            lines.append(f'<circle cx="{xx}" cy="{yy}" r="6" fill="{color}"/>')
        lines.append(f'<text x="{xx+14}" y="{yy+5}" font-size="15" fill="{color}">{delta:+.4f}</text>')
    lines += ['<text x="460" y="412" text-anchor="middle" font-size="16">Δ VALID NDCG@10</text>',
              '<text x="40" y="440" font-size="13">Diamond: exploratory pilot. Circles: seeds 2027–2030. Each point is one paired seed.</text>', '</g></svg>']
    return '\n'.join(lines) + '\n'


def tex(summary):
    lines = [r'\begin{table}[t]', r'\centering', r'\small',
             r'\caption{Internal temporal ablations on KuaiRand VALID: chronological leave-one-out, full catalog. NDCG@10 mean $\pm$ sample standard deviation ($ddof=1$); $\Delta$ is the paired triple-minus-dual difference. Pilot seed 2026 is exploratory. No TEST evaluation for these series.}',
             r'\label{tab:internal-time-valid}', r'\begin{tabular}{lcc}', r'\toprule',
             r'Variant / contrast & $n_{\mathrm{seeds}}$ & VALID NDCG@10 \\', r'\midrule']
    for i, (name, title) in enumerate([('SISO_new4','SISO: new seeds 2027--2030'), ('SISO_all5','SISO: seeds 2026--2030, with pilot'), ('MIMO_new4','MIMO: new seeds 2027--2030'), ('MIMO_all5','MIMO: seeds 2026--2030, with pilot')]):
        if i:
            lines.append(r'\midrule')
        c = summary['cohorts'][name]
        lines.append(r'\multicolumn{3}{l}{\textit{' + title + r'}} \\')
        for mode, d in c['models'].items():
            lines.append(f"{mode.capitalize()} & {c['n_seeds']} & $" + pm(d).replace('±', r'\pm') + r'$ \\')
        lines.append(r'$\Delta$ triple--dual & ' + str(c['n_seeds']) + ' & $' + pm(c['contrasts']['triple-dual'], True).replace('±', r'\pm') + r'$ \\')
    return '\n'.join(lines + [r'\bottomrule', r'\end{tabular}', r'\end{table}']) + '\n'


def registry(rows, sources, append=False):
    path = ROOT / 'experiments/results.csv'
    old = subprocess.check_output(['git', 'show', OLD_MAIN + ':experiments/results.csv'], cwd=ROOT)
    current = path.read_bytes()
    require(current.startswith(old), 'Historical CSV bytes changed')
    old_rows = list(csv.DictReader(io.StringIO(old.decode())))
    reader = csv.DictReader(io.StringIO(current.decode()))
    header, existing = reader.fieldnames, list(reader)
    ids = [r['run_id'] for r in existing if r['source'] == 'ours']
    require(len(ids) == len(set(ids)), 'Duplicate ours run_id')
    index = {s['run_id']: s for s in sources['sources']}
    additions = []
    for r in rows:
        entry = index[r['run_id']]
        matched = [e for e in existing if e['source'] == 'ours' and e['run_id'] == r['run_id']]
        expected = dict(run_id=r['run_id'], record_type='experiment', source='ours', split='validation', status='completed', seed=str(r['seed']),
                        git_commit=r['execution_commit'], source_json=entry['path'], test_evaluation_count='0',
                        actual_epochs=str(r['actual_epochs']), best_epoch=str(r['best_epoch']), test_used='no',
                        validation_ndcg10=str(r['best_valid_score']))
        expected.update({column: str(r['best_valid_metrics'][metric]) for kind, key in [('HR','hit'),('Recall','recall'),('NDCG','ndcg')] for k in (5,10,20,50) for column, metric in [(f'{kind}@{k}',f'{key}@{k}')]})
        if matched:
            e = matched[0]
            for k, value in expected.items():
                require(float(e[k]) == float(value) if '@' in k else e[k] == value, 'Registry mismatch: ' + r['run_id'] + ' ' + k)
            continue
        require(r['architecture'] == 'MIMO' and r['seed'] in range(2027, 2031), 'Missing historical row')
        e = {k: '' for k in header}
        e.update(expected)
        e.update(model='ThreeTimeMamba3Rec', model_variant='MIMO_' + r['mode'], dataset='KuaiRand', protocol='B',
                 evaluation='full_7111_items', train_candidates='full_softmax', item_universe='7111', test_used='no',
                 validation_ndcg10=str(r['best_valid_score']), notes_path='reports/MAMBA3_TIME_MECHANISMS_RESULTS.md#mimo-time-confirmation')
        additions.append(e)
    if append and additions:
        buf = io.StringIO(newline='')
        writer = csv.DictWriter(buf, fieldnames=header, lineterminator='\n')
        writer.writerows(additions)
        require(current.endswith(b'\n'), 'CSV has no final newline')
        path.write_bytes(current + buf.getvalue().encode())
        return registry(rows, sources)
    require(not additions, 'Missing registry rows: run --append-registry')
    published = subprocess.check_output(['git', 'show', 'ce6da46099fcc15f74f61c28ed946f78210ac053:experiments/results.csv'], cwd=ROOT)
    require(len(old_rows) == 81 and len(existing) >= 93 and current.startswith(published), 'Published registry prefix changed')
    return dict(before=81, added=12, after=len(existing), later_appended=len(existing)-93,
                scientific_rows_verified=25, historical_bytes=len(old), historical_sha256=hashlib.sha256(old).hexdigest())


def audit_sources():
    manifest = read(C / 'source_manifest.json')
    digest = hashlib.sha256(json.dumps(manifest['files'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    require(digest == manifest['source_hash'], 'Manifest source hash')
    allowed = {'README.md', 'reports/RESULTS.md', 'reports/PAPER_RESULTS.md', 'reports/MAMBA3_TIME_MECHANISMS_RESULTS.md', 'experiments/results.csv'}
    for path, expected in manifest['files'].items():
        b = subprocess.check_output(['git', 'show', EXECUTION + ':' + path], cwd=ROOT)
        require(hashlib.sha256(b).hexdigest() == expected, 'Historical source: ' + path)
        require(sha(path) == expected or path in allowed, 'Mathematical source changed: ' + path)
    return dict(execution_commit=EXECUTION, frozen_files_verified=len(manifest['files']), source_hash=digest, mathematical_sources='UNCHANGED')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-derived', action='store_true')
    parser.add_argument('--append-registry', action='store_true')
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    rows, sources = records()
    summary = aggregate(rows)
    outputs = {'sources.json': dump(sources), 'summary.json': dump(summary), 'mimo_paired_delta.svg': svg(summary), 'siso_mimo_valid_table.tex': tex(summary)}
    for name, content in outputs.items():
        p = HERE / name
        if args.write_derived:
            p.write_text(content)
        else:
            require(p.read_text() == content, 'Stale derived file: ' + name)
    text = REPORT.read_text()
    for name, content in tables(rows, summary).items():
        start, end = f'<!-- mimo:{name}:start -->', f'<!-- mimo:{name}:end -->'
        require(text.count(start) == text.count(end) == 1, 'Table markers: ' + name)
        a, rest = text.split(start)
        middle, b = rest.split(end)
        if args.write_derived:
            text = a + start + '\n' + content + '\n' + end + b
        else:
            require(middle == '\n' + content + '\n', 'Stale report table: ' + name)
    if args.write_derived:
        REPORT.write_text(text)
    print(dump(dict(runs_verified=len(rows), registry=registry(rows, sources, args.append_registry),
                    audit=audit_sources() if args.audit else 'not requested')))


if __name__ == '__main__':
    main()
