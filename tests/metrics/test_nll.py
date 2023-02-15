# fmt:off

import pytest
import torch

from torch_uncertainty.metrics import NegativeLogLikelihood

# fmt:on


@pytest.fixture
def probs_zero() -> torch.Tensor:
    probs = torch.as_tensor([[1, 0.0], [0.0, 1.0]])
    return probs


@pytest.fixture
def targets_zero() -> torch.Tensor:
    probs = torch.as_tensor([0, 1])
    return probs


class TestNegativeLogLikelihood:
    """Testing the NegativeLogLikelihood metric class."""

    def test_compute_zero(
        self, probs_zero: torch.Tensor, targets_zero: torch.Tensor
    ):
        metric = NegativeLogLikelihood()
        metric.update(probs_zero, targets_zero)
        res = metric.compute()
        assert res == 0
