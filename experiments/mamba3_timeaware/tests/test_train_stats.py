from collections import Counter

import numpy as np
import pytest

from experiments.mamba3_timeaware.compute_train_time_stats import gap_counts, summarize


def test_history_weights_match_explicit_prefixes():
    times = np.cumsum(np.arange(70))
    explicit = Counter()
    for target in range(1, len(times)):
        explicit.update(np.diff(times[max(0, target - 50):target]).tolist())
    assert gap_counts(times) == explicit


def test_last_train_target_timestamp_not_used():
    assert gap_counts([0, 10, 30, 100]) == gap_counts([0, 10, 30, 9999999])
    assert gap_counts([0, 10, 30, 100]) == Counter({10: 2, 20: 1})


def test_weighted_quantiles_match_numpy():
    counts = Counter({0: 3, 1: 5, 10: 2, 100: 4})
    summary = summarize(counts)
    positive = np.array([1] * 5 + [10] * 2 + [100] * 4)
    assert summary['median'] == np.median(positive)
    assert summary['p99'] == np.quantile(positive, .99)
    assert summary['count_total_gaps'] == 14


def test_zero_only_fails():
    with pytest.raises(ValueError):
        summarize(Counter({0: 10}))
