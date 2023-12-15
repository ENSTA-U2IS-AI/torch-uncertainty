import pytest
import torch

from torch_uncertainty.metrics import (
    GaussianNegativeLogLikelihood,
    NegativeLogLikelihood,
)


@pytest.fixture()
def probs_zero() -> torch.Tensor:
    return torch.as_tensor([[1, 0.0], [0.0, 1.0]])


@pytest.fixture()
def targets_zero() -> torch.Tensor:
    return torch.as_tensor([0, 1])


class TestNegativeLogLikelihood:
    """Testing the NegativeLogLikelihood metric class."""

    def test_compute_zero(
        self, probs_zero: torch.Tensor, targets_zero: torch.Tensor
    ) -> None:
        metric = NegativeLogLikelihood()
        metric.update(probs_zero, targets_zero)
        res = metric.compute()
        assert res == 0

        metric = NegativeLogLikelihood(reduction="none")
        metric.update(probs_zero, targets_zero)
        res_sum = metric.compute()
        assert torch.all(res_sum == torch.zeros(2))

    def test_bad_argument(self) -> None:
        with pytest.raises(Exception):
            _ = NegativeLogLikelihood(reduction="geometric_mean")


class TestGaussianNegativeLogLikelihood:
    """Testing the NegativeLogLikelihood metric class."""

    def test_compute_zero(self) -> None:
        metric = GaussianNegativeLogLikelihood()
        means = torch.as_tensor([1, 10]).float()
        variances = torch.as_tensor([1, 2]).float()
        targets = torch.as_tensor([1, 10]).float()
        metric.update(means, targets, variances)
        res_mean = metric.compute()
        assert res_mean == torch.log(variances).mean() / 2

        metric = GaussianNegativeLogLikelihood(reduction="sum")
        metric.update(means, targets, variances)
        res_sum = metric.compute()
        assert res_sum == torch.log(variances).sum() / 2

        metric = GaussianNegativeLogLikelihood(reduction="none")
        metric.update(means, targets, variances)
        res_sum = metric.compute()
        assert torch.all(res_sum == torch.log(variances) / 2)
