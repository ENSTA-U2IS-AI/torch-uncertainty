# fmt:off

import math

import pytest
import torch

from torch_uncertainty.metrics import Entropy

# fmt:on


@pytest.fixture
def vec2D_max() -> torch.Tensor:
    vec = torch.as_tensor([0.5, 0.5])
    return vec.unsqueeze(0)


@pytest.fixture
def vec2D_min() -> torch.Tensor:
    vec = torch.as_tensor([0.0, 1.0])
    return vec.unsqueeze(0)


@pytest.fixture
def vec3D() -> torch.Tensor:
    """
    Return a torch tensor with a mean entropy of 0 and an entropy of
        the mean of ln(2) to test the `ensemble` parameter of `Entropy`.
    """
    vec = torch.as_tensor([[0.0, 1.0], [1.0, 0.0]])
    return vec.unsqueeze(0)


class TestEntropy:
    """Testing the Entropy metric class."""

    def test_init(self):
        self.entropy = Entropy()

    def test_update(self, vec2D_min: torch.Tensor):
        self.entropy = Entropy()
        self.entropy.update(vec2D_min)

    def test_multiple_updates(
        self, vec2D_min: torch.Tensor, vec2D_max: torch.Tensor
    ):
        self.entropy = Entropy()
        self.entropy.update(vec2D_min)
        self.entropy.update(vec2D_max)

    def test_compute(self, vec2D_min: torch.Tensor):
        self.entropy = Entropy()
        self.entropy.update(vec2D_min)
        res = self.entropy.compute()
        assert res == 0

    def test_compute_max(self, vec2D_max: torch.Tensor):
        self.entropy = Entropy(reduction="sum")
        self.entropy.update(vec2D_max)
        res = self.entropy.compute()
        assert res == math.log(2)

    def test_multiple_compute_sum(
        self, vec2D_min: torch.Tensor, vec2D_max: torch.Tensor
    ):
        self.entropy = Entropy(reduction="sum")
        self.entropy.update(vec2D_min)
        self.entropy.update(vec2D_max)
        res = self.entropy.compute()
        assert res == math.log(2)

    def test_multiple_compute_mean(
        self, vec2D_min: torch.Tensor, vec2D_max: torch.Tensor
    ):
        self.entropy = Entropy(reduction="mean")
        self.entropy.update(vec2D_min)
        self.entropy.update(vec2D_max)
        res = self.entropy.compute()
        assert res == math.log(2) / 2

    def test_multiple_compute_none(
        self, vec2D_min: torch.Tensor, vec2D_max: torch.Tensor
    ):
        self.entropy = Entropy(reduction=None)
        self.entropy.update(vec2D_min)
        self.entropy.update(vec2D_max)
        res = self.entropy.compute()
        assert all(res == torch.as_tensor([0, math.log(2)]))

    def test_compute_3D(self, vec3D: torch.Tensor):
        self.entropy = Entropy(reduction="mean")
        self.entropy.update(vec3D)
        res = self.entropy.compute()
        assert res == 0

    def test_compute_3D_to_2D(self, vec3D: torch.Tensor):
        self.entropy = Entropy(reduction="mean")
        vec3D = vec3D.mean(1)
        self.entropy.update(vec3D)
        res = self.entropy.compute()
        assert res == math.log(2)
