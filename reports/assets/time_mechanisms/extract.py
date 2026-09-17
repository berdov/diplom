"""Read existing trusted cluster artifacts; CPU calibrators only, no evaluation.

Run from the canonical repository with envs/mamba3/bin/python; stdout is JSON.
The checkpoints themselves and raw logs are never exported.
"""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path('/home/daryumin/iberdov/diplom')
RUNS = {
    'vanilla': ('mamba3_baseline', 'mamba3_validation_001',
                '/home/daryumin/iberdov/diplom_exp_mamba3_baseline/experiments/mamba3_baseline/slurm_logs/mamba3-base-4322324.err'),
    'shared': ('mamba3_timeaware', 'mamba3_timeaware_validation_001',
               'experiments/mamba3_timeaware/slurm_logs/rt-mamba3-valid-4326765.err'),
    'decay_only': ('mamba3_time_mechanisms', 'mamba3_decay_only_validation_001',
                   'experiments/mamba3_time_mechanisms/slurm_logs/m3-decay-val-4332918.err'),
    'scan_only': ('mamba3_time_mechanisms', 'mamba3_scan_only_validation_001',
                  'experiments/mamba3_time_mechanisms/slurm_logs/m3-scan-val-4332919.err'),
    'separate': ('mamba3_time_mechanisms', 'mamba3_separate_time_validation_001',
                 'experiments/mamba3_time_mechanisms/slurm_logs/m3-separate-val-4332920.err'),
}


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    import torch
    from experiments.mamba3_timeaware.time_inputs import TimeCalibrator
    stats_path = ROOT / 'experiments/mamba3_timeaware/runs/train_time_stats_001.json'
    stats = json.loads(stats_path.read_text())
    # Frozen TRAIN endpoints and reference; active zero is a separate observation.
    positive = torch.logspace(torch.tensor(stats['min_positive'], dtype=torch.float64).log10(),
                              torch.tensor(stats['max'], dtype=torch.float64).log10(), 121,
                              dtype=torch.float64)
    grid = sorted(set([0., stats['time_scale_reference']] + positive.tolist()))
    result = dict(schema_version=1, evaluation_runs=0, device='cpu',
                  grid_definition='0 plus 121 log-spaced TRAIN min_positive..max and frozen reference',
                  train_stats_path=str(stats_path), train_stats_sha256=sha(stats_path),
                  reference_ms=stats['time_scale_reference'], gap_ms=grid, runs={})
    for mode, (family, run_id, log_path) in RUNS.items():
        source = ROOT / 'experiments' / family / 'runs' / (run_id + '.json')
        run = json.loads(source.read_text())
        log = ROOT / log_path
        row = dict(run_id=run_id, source_json=str(source), source_sha256=sha(source),
                   source_log=str(log), log_exists=log.exists(),
                   best_epoch=run.get('best_epoch'), actual_epochs=run.get('actual_epochs'),
                   epoch_indexing='zero-based', git_commit=run.get('git_commit'),
                   started_at=run.get('started_at'), finished_at=run.get('finished_at'), history=[])
        if log.exists():
            row['log_sha256'] = sha(log)
            text = re.sub(r'\x1b\[[0-9;]*m', '', log.read_text())
            train = {int(e): dict(train_seconds=float(s), train_loss=float(loss)) for e,s,loss in
                     re.findall(r'epoch (\d+) training \[time: ([\d.]+)s, train loss: ([\d.e+-]+)\]', text)}
            for e, seconds, score in re.findall(r'epoch (\d+) evaluating \[time: ([\d.]+)s, valid_score: ([\d.e+-]+)\]', text):
                row['history'].append(dict(epoch=int(e), valid_ndcg10=float(score),
                                           valid_seconds=float(seconds), **train.get(int(e), {})))
            if row['history']:
                early = [r for r in row['history'] if r['epoch'] < 27]
                row['best_within_first27'] = max(early, key=lambda r: r['valid_ndcg10'])
                if mode == 'vanilla':
                    row['best_epoch'] = max(row['history'], key=lambda r: r['valid_ndcg10'])['epoch']
                    row['actual_epochs'] = len(row['history'])
                    row['epoch_source'] = 'existing training log, not validation JSON'
        if mode == 'vanilla':
            final = json.loads((source.parent / 'mamba3_final_test_001.json').read_text())
            checkpoint = Path(final['checkpoint']['path'])
            row['checkpoint_path_source'] = 'mamba3_final_test_001.json'
            row['best_epoch'] = final['checkpoint']['epoch']
            row['checkpoint_epoch'] = final['checkpoint']['epoch']
            row['epoch_source'] = 'final TEST provenance checkpoint.epoch; log earliest tied maximum is separate'
            row['started_at'] = run.get('created_at_utc')
            row['finished_at'] = run.get('finished_at_utc')
        else:
            checkpoint = Path(run['checkpoint_path'])
        row.update(checkpoint_path=str(checkpoint), checkpoint_exists=checkpoint.exists())
        if checkpoint.exists():
            row['checkpoint_sha256'] = sha(checkpoint)
            if mode != 'vanilla':
                saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
                row['checkpoint_epoch'] = int(saved['epoch'])
                assert row['checkpoint_epoch'] == run['best_epoch']
                prefixes = ['time_calibrator.'] if mode == 'shared' else sorted({
                    k.rsplit('.', 2)[0] + '.' for k in saved['state_dict']
                    if k.startswith('mechanisms.calibrators.') and k.endswith('first.weight')})
                row['calibrators'] = {}
                for prefix in prefixes:
                    state = {k[len(prefix):]: v for k,v in saved['state_dict'].items() if k.startswith(prefix)}
                    calibrator = TimeCalibrator(2, stats['time_scale_reference'])
                    calibrator.load_state_dict(state)
                    gaps = torch.tensor([grid], dtype=torch.float64)
                    with torch.no_grad():
                        scales = calibrator(gaps, torch.ones_like(gaps, dtype=torch.bool))[0].tolist()
                    row['calibrators'][prefix] = dict(state={k:v.tolist() for k,v in state.items()},
                        active_scales=scales, inactive_scale=[1.,1.])
                del saved
        result['runs'][mode] = row
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
