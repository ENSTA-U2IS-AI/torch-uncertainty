# fmt:off
import pytest
import torch

from torch_uncertainty.metrics import (
    GaussianNegativeLogLikelihood,
    NegativeLogLikelihood,
)


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
    ) -> None:
        metric = NegativeLogLikelihood()
        metric.update(probs_zero, targets_zero)
        res = metric.compute()
        assert res == 0


class TestGaussianNegativeLogLikelihood:
    """Testing the NegativeLogLikelihood metric class."""

    def test_compute_zero(self) -> None:
        metric = GaussianNegativeLogLikelihood()
        means = torch.as_tensor([1, 10]).float()
        vars = torch.as_tensor([1, 2]).float()
        targets = torch.as_tensor([1, 10]).float()
        metric.update(means, targets, vars)
        res = metric.compute()
        assert res == torch.log(vars).mean() / 2
