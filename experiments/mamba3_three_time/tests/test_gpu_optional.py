import pytest
import torch


def test_cuda_availability_is_not_evidence():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable on Mac; synthetic GPU suites NOT RUN")
    pytest.skip("GPU suites require the explicit guarded technical launcher, not pytest")
