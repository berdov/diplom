"""Partial-safe numeric summary; no model, dataset or checkpoint loading."""
import math
from . import config as c
from experiments.mamba3_mimo_time.records import read,create,now,sha,finite_tree


def validate_history(r,variant,seed):
    if r.get('status')!='PASS':return
    if not finite_tree(r):raise ValueError('Nonfinite scientific record')
    p=c.paths(variant,seed);h=r.get('history',[])
    if (r.get('run_id')!=p['run_id'] or r.get('phase_mode')!=variant or r.get('mode')!='dual' or r.get('seed')!=seed
        or r.get('parameter_count')!=c.COUNTS[variant] or r.get('TEST')!='NOT_RUN' or r.get('test_evaluation_count')!=0
        or r.get('scientific_fit_started') is not True or not h
        or [x['epoch'] for x in h]!=list(range(r.get('actual_epochs',-1)))):
        raise ValueError('Invalid successful run identity/history')
    scores=[x['valid_ndcg10'] for x in h]
    if not all(math.isfinite(v) for v in scores):raise ValueError('Nonfinite history')
    metric_keys={f'{kind}@{k}' for kind in ('hit','ndcg','recall') for k in (5,10,20,50)}
    for row in h:
        if set(row['valid_metrics'])!=metric_keys or row['valid_metrics']['ndcg@10']!=row['valid_ndcg10']:
            raise ValueError('Metric keys/selection mismatch')
    best=max(i for i,v in enumerate(scores) if v==max(scores))
    if (r.get('best_epoch')!=best or r.get('best_valid_score')!=scores[best]
        or r.get('best_valid_metrics')!=h[best]['valid_metrics'] or r.get('best_diagnostics')!=h[best]['diagnostics']):
        raise ValueError('Best/last-tie mismatch')
    best_so_far=-math.inf;stale=0;stop=None
    for i,score in enumerate(scores):
        if score>=best_so_far:best_so_far=score;stale=0
        else:stale+=1
        if stale>10 and stop is None:stop=i
    if (len(h)<300 and stop!=len(h)-1) or (stop is not None and stop!=len(h)-1) or len(h)>300:
        raise ValueError('Not a complete unchanged early-stopped fit')
    if r.get('first27_complete')!=(len(h)>=27) or r.get('first27_best_ndcg10')!=(max(scores[:27]) if len(h)>=27 else None):
        raise ValueError('First27 mismatch')


PAIRING = ('initial_backbone_sha256', 'initial_common_calibrator_hashes', 'rng_components',
           'protocol', 'manifest_sha256', 'train_time_stats_sha256', 'verified_history_stats',
           'precision', 'optimizer_settings', 'first_train_batch_sha256')


def owner_artifacts(paths):
    candidates = [paths[k] for k in ('lock', 'checkpoint', 'metadata')]
    candidates.append(paths['runtime'] / 'progress.json')
    return [p for p in candidates if p.exists() or p.is_symlink()]


def scientific_start(record, paths=None):
    if not isinstance(record, dict) or type(record.get('scientific_fit_started')) is not bool:
        return None
    started = record['scientific_fit_started']
    if not started:
        if record.get('status') == 'PASS' or record.get('actual_epochs', 0) != 0 or record.get('history'):
            return None
        if paths is not None and (any(paths[k].exists() or paths[k].is_symlink() for k in ('checkpoint', 'metadata'))
                                  or record.get('status') == 'NOT_RUN' and owner_artifacts(paths)):
            return None
    return started


def unknown_record(base, variant, seed, reason):
    return dict(base, phase_mode=variant, mode='dual', seed=seed, run_id=c.paths(variant, seed)['run_id'],
                status='UNKNOWN', scientific_fit_started=None, actual_epochs=None, validation_error=reason)


def replay_check(r):
    old = read(c.historical(r['seed']))
    keys = ('seed', 'best_epoch', 'best_valid_metrics', 'actual_epochs', 'first27_best_ndcg10',
            'first_train_batch_sha256', 'initial_backbone_sha256', 'rng_components', 'optimizer_settings',
            'precision', 'protocol', 'manifest_sha256', 'train_time_stats_sha256', 'verified_history_stats')
    differences = [k for k in keys if r.get(k) != old.get(k)]
    if r.get('initial_common_calibrator_hashes') != old['initial_calibrator_hashes']:
        differences.append('initial_calibrators')
    for key in ('config', 'effective_config'):
        a, b = r[key], old[key]
        if any(a.get(k) != b.get(k) for k in set(a) | set(b) if k not in ('checkpoint_dir', 'phase_mode')):
            differences.append(key)
    if len(r['history']) != len(old['history']):
        differences.append('history_length')
    for i, (a, b) in enumerate(zip(r['history'], old['history'])):
        for key in ('epoch', 'valid_ndcg10', 'valid_metrics', 'train_loss', 'diagnostics'):
            if a[key] != b[key]:
                differences.append(f'history[{i}].{key}')
    if differences:
        raise ValueError('Unplanned historical scientific drift; no refit: ' + repr(differences))
    same_bytes = r['checkpoint_sha256'] == old['checkpoint_sha256']
    return dict(status='PASS' if same_bytes else 'CHECKPOINT_IDENTITY_REVIEW_REQUIRED',
                scientific_history_exact=True, checkpoint_serialization_match=same_bytes,
                historical_checkpoint_sha256=old['checkpoint_sha256'],
                current_checkpoint_sha256=r['checkpoint_sha256'], historical_run_id=old['run_id'],
                timing_memory_excluded=True,
                note='Serialization mismatch alone is not mathematical failure; review scientific state identity before progression')


def validate_record(r, variant, seed=2026):
    if r.get('status') != 'PASS':
        return
    validate_history(r, variant, seed)
    old = read(c.historical(seed))
    for key in ('protocol', 'manifest_sha256', 'train_time_stats_sha256', 'verified_history_stats',
                'precision', 'optimizer_settings', 'first_train_batch_sha256', 'initial_backbone_sha256', 'rng_components'):
        if r[key] != old[key]:
            raise ValueError('Frozen state/data drift ' + key)
    if r['initial_common_calibrator_hashes'] != old['initial_calibrator_hashes']:
        raise ValueError('Initial common calibrators')
    params = r['initial_phase_parameters']
    if variant == 'baseline_dual':
        if params:
            raise ValueError('Baseline has additional parameters')
    elif set(params) != {'phase_adapter.W'} or not all(x['shape'] == [32, 4] and x['dtype'] == 'torch.float32' and x['exact_zero'] is True for x in params.values()):
        raise ValueError('Initial phase parameters')
    for key in ('config', 'effective_config'):
        a, b = r[key], old[key]
        if a.get('phase_mode') != variant or any(a.get(k) != b.get(k) for k in set(a) | set(b) if k not in ('checkpoint_dir', 'phase_mode')):
            raise ValueError('Frozen config drift')
    if variant == 'baseline_dual':
        replay = replay_check(r)
        if replay['status'] != 'PASS':
            raise ValueError('Checkpoint serialization differs; pause for identity review, not a mathematical FAIL')


def validate_checkpoint(r, variant, seed=2026):
    p = c.paths(variant, seed)
    meta = read(p['metadata'])
    if sha(p['checkpoint']) != r['checkpoint_sha256'] or meta['checkpoint_sha256'] != r['checkpoint_sha256']:
        raise ValueError('Checkpoint SHA mismatch')
    for k in ('run_id', 'mode', 'phase_mode', 'seed', 'execution_commit', 'config_sha256', 'source_hash', 'core_hash'):
        if meta[k] != r[k]:
            raise ValueError('Checkpoint owner ' + k)
    if meta['epoch'] != r['best_epoch'] or meta['metrics'] != r['best_valid_metrics'] or meta['phase'] != r['best_diagnostics'].get('phase'):
        raise ValueError('Checkpoint selection')


def summarize(records):
    rows = []
    for task in c.tasks():
        v, seed = task['variant'], task['seed']
        r = records.get((v, seed), dict(status='NOT_RUN', scientific_fit_started=False))
        if scientific_start(r) is None:
            r = unknown_record({}, v, seed, r.get('validation_error', 'Unknown start evidence'))
        error = r.get('validation_error')
        try:
            validate_record(r, v, seed)
        except (ValueError, KeyError, TypeError) as exc:
            error = str(exc)
            r = dict(r, status='INVALID')
        good = r.get('status') == 'PASS'
        rows.append(dict(**task, status=r.get('status', 'NOT_RUN'), validation_error=error,
                         parameters=c.COUNTS[v], scientific_fit_started=r.get('scientific_fit_started'),
                         actual_epochs=r.get('actual_epochs', 0), **{k: r.get(k) if good else None for k in (
                             'best_valid_metrics', 'best_epoch', 'first27_complete', 'first27_best_ndcg10',
                             'train_seconds', 'valid_seconds', 'peak_gpu_allocated_bytes', 'peak_gpu_reserved_bytes',
                             'checkpoint_sha256', 'best_diagnostics')}))
    contrasts = []
    by = {(r['variant'], r['seed']): r for r in rows}
    for seed in c.SEEDS:
        for left, right in c.plan()['contrasts']:
            a, b = by[left, seed], by[right, seed]
            complete = a['status'] == b['status'] == 'PASS'
            delta = relative = first = None
            if complete:
                x, y = a['best_valid_metrics']['ndcg@10'], b['best_valid_metrics']['ndcg@10']
                delta, relative = x-y, 100*(x-y)/y if y else None
                if a['first27_complete'] and b['first27_complete']:
                    first = a['first27_best_ndcg10']-b['first27_best_ndcg10']
            contrasts.append(dict(seed=seed, comparison=left+' - '+right,
                                  status='COMPLETE' if complete else 'NOT_AVAILABLE', delta=delta,
                                  relative_percent=relative, first27_delta=first))
    return dict(status='PASS' if all(r['status'] == 'PASS' for r in rows) else 'INCOMPLETE',
                rows=rows, contrasts=contrasts, primary_contrast='absolute_phase - relative_phase',
                scientific_fits_started=sum(r['scientific_fit_started'] is True for r in rows),
                scientific_fits_completed=sum(r['status'] == 'PASS' for r in rows),
                unknown_scientific_starts=sum(r['scientific_fit_started'] is None for r in rows),
                scientific_fits_expected=len(c.tasks()), seeds=list(c.SEEDS), TEST='NOT_RUN', test_evaluation_count=0)


def write(base, reason=None):
    records = {}
    for t in c.tasks():
        v, seed = t['variant'], t['seed']
        p = c.paths(v, seed)
        if not p['result'].exists() and not p['result'].is_symlink():
            if owner_artifacts(p):
                records[v, seed] = unknown_record(base, v, seed, 'Owner exists without result')
                continue
            create(p['result'], dict(base, phase_mode=v, mode='dual', seed=seed, run_id=p['run_id'],
                   status='NOT_RUN', scientific_fit_started=False, actual_epochs=0, history=[], reason=reason))
        try:
            r = read(p['result'])
        except (ValueError, OSError) as exc:
            records[v, seed] = unknown_record(base, v, seed, 'Unreadable result: ' + repr(exc))
            continue
        if not isinstance(r, dict):
            records[v, seed] = unknown_record(base, v, seed, 'Result is not an object')
            continue
        if any(r.get(k) != value for k, value in base.items()) or r.get('phase_mode') != v or r.get('seed') != seed:
            raise ValueError('Foreign run record')
        if scientific_start(r, p) is None:
            records[v, seed] = unknown_record(base, v, seed, 'Contradictory start evidence')
            continue
        if r.get('status') == 'PASS':
            try:
                validate_record(r, v, seed)
                validate_checkpoint(r, v, seed)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                r = dict(r, status='INVALID', validation_error=str(exc))
        records[v, seed] = r
    result = dict(base, **summarize(records), blocking_reason=reason)
    result['fresh_control_replays'] = []
    if result['status'] == 'PASS':
        for seed in c.SEEDS:
            for v in c.MODES[1:]:
                if any(records[v, seed][k] != records['baseline_dual', seed][k] for k in PAIRING):
                    raise ValueError('Within-triple pairing mismatch')
            result['fresh_control_replays'].append(replay_check(records['baseline_dual', seed]))
    create(c.SUMMARY, result)
    return result
