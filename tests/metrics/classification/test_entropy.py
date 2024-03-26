import math

import pytest
import torch

from torch_uncertainty.metrics import Entropy


@pytest.fixture()
def vec2d_max() -> torch.Tensor:
    vec = torch.as_tensor([0.5, 0.5])
    return vec.unsqueeze(0)


@pytest.fixture()
def vec2d_min() -> torch.Tensor:
    vec = torch.as_tensor([0.0, 1.0])
    return vec.unsqueeze(0)


@pytest.fixture()
def vec3d() -> torch.Tensor:
    """Return a torch tensor with a mean entropy of 0 and an entropy of
    the mean of ln(2) to test the `ensemble` parameter of `Entropy`.
    """
    vec = torch.as_tensor([[0.0, 1.0], [1.0, 0.0]])
    return vec.unsqueeze(0)


class TestEntropy:
    """Testing the Entropy metric class."""

    def test_compute(self, vec2d_min: torch.Tensor):
        metric = Entropy()
        metric.update(vec2d_min)
        res = metric.compute()
        assert res == 0

    def test_compute_max(self, vec2d_max: torch.Tensor):
        metric = Entropy(reduction="sum")
        metric.update(vec2d_max)
        res = metric.compute()
        assert res == math.log(2)

    def test_multiple_compute_sum(
        self, vec2d_min: torch.Tensor, vec2d_max: torch.Tensor
    ):
        metric = Entropy(reduction="sum")
        metric.update(vec2d_min)
        metric.update(vec2d_max)
        res = metric.compute()
        assert res == math.log(2)

    def test_multiple_compute_mean(
        self, vec2d_min: torch.Tensor, vec2d_max: torch.Tensor
    ):
        metric = Entropy(reduction="mean")
        metric.update(vec2d_min)
        metric.update(vec2d_max)
        res = metric.compute()
        assert res == math.log(2) / 2

    def test_multiple_compute_none(
        self, vec2d_min: torch.Tensor, vec2d_max: torch.Tensor
    ):
        metric = Entropy(reduction=None)
        metric.update(vec2d_min)
        metric.update(vec2d_max)
        res = metric.compute()
        assert all(res == torch.as_tensor([0, math.log(2)]))

    def test_compute_3d(self, vec3d: torch.Tensor):
        metric = Entropy(reduction="mean")
        metric.update(vec3d)
        res = metric.compute()
        assert res == 0

    def test_compute_3d_to_2d(self, vec3d: torch.Tensor):
        metric = Entropy(reduction="mean")
        vec3d = vec3d.mean(1)
        metric.update(vec3d)
        res = metric.compute()
        assert res == math.log(2)

    def test_bad_argument(self):
        with pytest.raises(ValueError):
            _ = Entropy("geometric_mean")
