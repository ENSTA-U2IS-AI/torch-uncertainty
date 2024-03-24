import pytest
import torch
from torch.distributions import Normal

from torch_uncertainty.metrics import CategoricalNLL, DistributionNLL


class TestCategoricalNegativeLogLikelihood:
    """Testing the CategoricalNLL metric class."""

    def test_compute_zero(self) -> None:
        probs = torch.as_tensor([[1, 0.0], [0.0, 1.0]])
        targets = torch.as_tensor([0, 1])

        metric = CategoricalNLL()
        metric.update(probs, targets)
        res = metric.compute()
        assert res == 0

        metric = CategoricalNLL(reduction="none")
        metric.update(probs, targets)
        res_sum = metric.compute()
        assert torch.all(res_sum == torch.zeros(2))

        metric = CategoricalNLL(reduction="sum")
        metric.update(probs, targets)
        res_sum = metric.compute()
        assert torch.all(res_sum == torch.zeros(1))

    def test_bad_argument(self) -> None:
        with pytest.raises(ValueError):
            _ = CategoricalNLL(reduction="geometric_mean")


class TestDistributionNLL:
    """Testing the TestDistributionNLL metric class."""

    def test_compute_zero(self) -> None:
        metric = DistributionNLL(reduction="mean")
        means = torch.as_tensor([1, 10]).float()
        stds = torch.as_tensor([1, 2]).float()
        targets = torch.as_tensor([1, 10]).float()
        dist = Normal(means, stds)
        metric.update(dist, targets)
        res_mean = metric.compute()
        assert res_mean == torch.mean(torch.log(2 * torch.pi * (stds**2)) / 2)

        metric = DistributionNLL(reduction="sum")
        metric.update(dist, targets)
        res_sum = metric.compute()
        assert res_sum == torch.log(2 * torch.pi * (stds**2)).sum() / 2

        metric = DistributionNLL(reduction="none")
        metric.update(dist, targets)
        res_all = metric.compute()
        assert torch.all(res_all == torch.log(2 * torch.pi * (stds**2)) / 2)
