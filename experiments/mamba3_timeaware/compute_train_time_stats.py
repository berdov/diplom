"""Read only manifest-verified train.parquet; count gaps inside TRAIN inputs."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def gap_counts(times, max_length=50):
    """Multiplicity across prefixes [max(0,j-L):j], never including target j."""
    counts = Counter()
    for k in range(1, len(times) - 1):
        weight = min(len(times) - 1, k + max_length - 1) - k
        gap = int(times[k]) - int(times[k - 1])
        if gap < 0:
            raise ValueError('Nonchronological TRAIN history')
        counts[gap] += weight
    return counts


def summarize(counts):
    total = sum(counts.values())
    pairs = sorted((gap, count) for gap, count in counts.items() if gap > 0 and count > 0)
    if not pairs:
        raise ValueError('No positive TRAIN historical gaps')
    values = np.array([p[0] for p in pairs], dtype=np.float64)
    weights = np.array([p[1] for p in pairs], dtype=np.int64)
    cumulative = np.cumsum(weights)
    positive = int(cumulative[-1])

    def quantile(q):
        rank = (positive - 1) * q
        lo, hi = int(np.floor(rank)), int(np.ceil(rank))
        a = values[np.searchsorted(cumulative, lo, side='right')]
        b = values[np.searchsorted(cumulative, hi, side='right')]
        return float(a + (b - a) * (rank - lo))

    return dict(count_total_gaps=total, count_positive_gaps=positive,
                zero_gap_share=counts.get(0, 0) / total,
                min_positive=float(values[0]), max=float(values[-1]),
                mean_positive=float(np.dot(values, weights) / positive),
                mean=float(np.dot(values, weights) / total),
                **{name: quantile(q) for name, q in
                   [('p25', .25), ('median', .5), ('p75', .75), ('p90', .9), ('p95', .95), ('p99', .99)]})


def main():
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-parquet', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path('outputs/data/protocol_b_manifest.json'))
    parser.add_argument('--output', type=Path, default=Path('experiments/mamba3_timeaware/runs/train_time_stats_001.json'))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    expected = next(f for f in manifest['files'] if f['relative_path'] == 'train.parquet')
    actual_sha = sha256(args.train_parquet)
    if actual_sha != expected['sha256']:
        raise ValueError('TRAIN parquet SHA mismatch')
    frame = pq.read_table(args.train_parquet, columns=['user_id', 'timestamp', 'source_row_id'])
    frame = frame.sort_by([('user_id', 'ascending'), ('timestamp', 'ascending'), ('source_row_id', 'ascending')])
    if len(frame) != manifest['split_stats']['train']['interactions']:
        raise ValueError('TRAIN row count mismatch')
    counts = Counter()
    users = frame['user_id'].to_numpy()
    times = frame['timestamp'].to_numpy()
    boundaries = np.r_[0, np.flatnonzero(users[1:] != users[:-1]) + 1, len(users)]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        counts.update(gap_counts(times[start:end]))
    result = summarize(counts)
    result.update(split='TRAIN_ONLY', units='milliseconds', timestamp_field='timestamp',
                  valid_timestamps_used=False, test_timestamps_used=False, target_gaps_used=False,
                  history_policy='Each TRAIN prefix up to 50 events; adjacent gaps only, target excluded; occurrence weighted',
                  quantile_population='strictly positive historical gaps; linear interpolation',
                  train_rows=len(frame), train_users=len(boundaries) - 1,
                  train_parquet_sha256=actual_sha, source_path=expected['path'],
                  manifest_path=str(args.manifest), manifest_sha256=sha256(args.manifest),
                  protocol_b_inter_sha256=next(f['sha256'] for f in manifest['files'] if f['relative_path'].endswith('.inter')),
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  reference_statistic='median of strictly positive TRAIN adjacent historical gaps',
                  time_scale_reference=result['median'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
