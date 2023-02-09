# fmt:off

import pytest
import torch

from torch_uncertainty.metrics import BrierScore

# fmt:on


@pytest.fixture
def vec2D_max() -> torch.Tensor:
    vec = torch.as_tensor([0.5, 0.5])
    return vec.unsqueeze(0)


@pytest.fixture
def vec2D_max_target() -> torch.Tensor:
    vec = torch.as_tensor([0, 1])
    return vec.unsqueeze(0)


@pytest.fixture
def vec2D_min() -> torch.Tensor:
    vec = torch.as_tensor([0.0, 1.0])
    return vec.unsqueeze(0)


@pytest.fixture
def vec2D_min_target() -> torch.Tensor:
    vec = torch.as_tensor([0, 1])
    return vec.unsqueeze(0)


@pytest.fixture
def vec3D() -> torch.Tensor:
    """
    Return a torch tensor with a mean BrierScore of 0 and an BrierScore of
        the mean of ln(2) to test the `ensemble` parameter of `BrierScore`.
    """
    vec = torch.as_tensor([[0.0, 1.0], [1.0, 0.0]])
    return vec.unsqueeze(0)


@pytest.fixture
def vec3D_target() -> torch.Tensor:
    """
    Return a torch tensor with a mean BrierScore of 0 and an BrierScore of
        the mean of ln(2) to test the `ensemble` parameter of `BrierScore`.
    """
    vec = torch.as_tensor([0, 1])
    return vec.unsqueeze(0)


class TestBrierScore:
    """Testing the BrierScore metric class."""

    def test_compute(
        self, vec2D_min: torch.Tensor, vec2D_min_target: torch.Tensor
    ):
        self.metric = BrierScore()
        self.metric.update(vec2D_min, vec2D_min_target)
        res = self.metric.compute()
        assert res == 0

    def test_compute_max(
        self, vec2D_max: torch.Tensor, vec2D_max_target: torch.Tensor
    ):
        self.metric = BrierScore(reduction="sum")
        self.metric.update(vec2D_max, vec2D_max_target)
        res = self.metric.compute()
        assert res == 0.5

    def test_multiple_compute_sum(
        self,
        vec2D_min: torch.Tensor,
        vec2D_max: torch.Tensor,
        vec2D_min_target: torch.Tensor,
        vec2D_max_target: torch.Tensor,
    ):
        self.metric = BrierScore(reduction="sum")
        self.metric.update(vec2D_min, vec2D_min_target)
        self.metric.update(vec2D_max, vec2D_max_target)
        res = self.metric.compute()
        assert res == 0.5

    def test_multiple_compute_mean(
        self,
        vec2D_min: torch.Tensor,
        vec2D_max: torch.Tensor,
        vec2D_min_target: torch.Tensor,
        vec2D_max_target: torch.Tensor,
    ):
        self.metric = BrierScore(reduction="mean")
        self.metric.update(vec2D_min, vec2D_min_target)
        self.metric.update(vec2D_max, vec2D_max_target)
        res = self.metric.compute()
        assert res == 0.5 / 2

    def test_multiple_compute_none(
        self,
        vec2D_min: torch.Tensor,
        vec2D_max: torch.Tensor,
        vec2D_min_target: torch.Tensor,
        vec2D_max_target: torch.Tensor,
    ):
        self.metric = BrierScore(reduction=None)
        self.metric.update(vec2D_min, vec2D_min_target)
        self.metric.update(vec2D_max, vec2D_max_target)
        res = self.metric.compute()
        assert all(res == torch.as_tensor([0, 0.5]))

    def test_compute_3D_mean(
        self, vec3D: torch.Tensor, vec3D_target: torch.Tensor
    ):
        self.metric = BrierScore(reduction="mean")
        self.metric.update(vec3D, vec3D_target)
        res = self.metric.compute()
        assert res == 0.5

    def test_compute_3D_sum(
        self, vec3D: torch.Tensor, vec3D_target: torch.Tensor
    ):
        self.metric = BrierScore(reduction="sum")
        self.metric.update(vec3D, vec3D_target)
        res = self.metric.compute()
        assert res == 1

    def test_compute_3D_to_2D(
        self, vec3D: torch.Tensor, vec3D_target: torch.Tensor
    ):
        self.metric = BrierScore(reduction="mean")
        vec3D = vec3D.mean(1)
        self.metric.update(vec3D, vec3D_target)
        res = self.metric.compute()
        assert res == 0.5

    def test_bad_argument(self):
        with pytest.raises(Exception):
            _ = BrierScore("geometric_mean")
