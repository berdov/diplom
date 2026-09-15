from types import SimpleNamespace

import pytest

from experiments.mamba3_timeaware.time_mamba3 import require_siso


@pytest.mark.parametrize('rank', [1, 4])
def test_siso_accepts_frozen_rank(rank):
    require_siso(SimpleNamespace(is_mimo=False, mimo_rank=rank))


@pytest.mark.parametrize('rank', [1, 4])
def test_mimo_still_rejected(rank):
    with pytest.raises(ValueError, match='Only SISO'):
        require_siso(SimpleNamespace(is_mimo=True, mimo_rank=rank))
