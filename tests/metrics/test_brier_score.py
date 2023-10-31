# fmt:off
import pytest
import torch

from torch_uncertainty.metrics import BrierScore



@pytest.fixture
def vec2D_max() -> torch.Tensor:
    vec = torch.as_tensor([0.5, 0.5])
    return vec.unsqueeze(0)


@pytest.fixture
def vec2D_max_target() -> torch.Tensor:
    vec = torch.as_tensor([0, 1])
    return vec.unsqueeze(0)


@pytest.fixture
def vec2D_max_target1D() -> torch.Tensor:
    vec = torch.as_tensor([1])
    return vec


@pytest.fixture
def vec2D_min() -> torch.Tensor:
    vec = torch.as_tensor([0.0, 1.0])
    return vec.unsqueeze(0)


@pytest.fixture
def vec2D_min_target() -> torch.Tensor:
    vec = torch.as_tensor([0, 1])
    return vec.unsqueeze(0)


@pytest.fixture
def vec2D_5classes() -> torch.Tensor:
    vec = torch.as_tensor(
        [[0.2, 0.6, 0.1, 0.05, 0.05], [0.05, 0.25, 0.1, 0.3, 0.3]]
    )
    return vec


@pytest.fixture
def vec2D_5classes_target() -> torch.Tensor:
    vec = torch.as_tensor([[0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
    return vec


@pytest.fixture
def vec2D_5classes_target1D() -> torch.Tensor:
    vec = torch.as_tensor([3, 4])
    return vec


@pytest.fixture
def vec3D() -> torch.Tensor:
    """
    Return a torch tensor with a mean BrierScore of 0 and a BrierScore of
        the mean of 0.5 to test the `ensemble` parameter of `BrierScore`.
    """
    vec = torch.as_tensor([[0.0, 1.0], [1.0, 0.0]])
    return vec.unsqueeze(0)


@pytest.fixture
def vec3D_target() -> torch.Tensor:
    vec = torch.as_tensor([0, 1])
    return vec.unsqueeze(0)


@pytest.fixture
def vec3D_target1D() -> torch.Tensor:
    vec = torch.as_tensor([1])
    return vec.unsqueeze(0)


class TestBrierScore:
    """Testing the BrierScore metric class."""

    def test_compute(
        self, vec2D_min: torch.Tensor, vec2D_min_target: torch.Tensor
    ):
        metric = BrierScore(num_classes=2)
        metric.update(vec2D_min, vec2D_min_target)
        res = metric.compute()
        assert res == 0

    def test_compute_max(
        self, vec2D_max: torch.Tensor, vec2D_max_target: torch.Tensor
    ):
        metric = BrierScore(num_classes=2, reduction="sum")
        metric.update(vec2D_max, vec2D_max_target)
        res = metric.compute()
        assert res == 0.5

    def test_compute_max_target1D(
        self, vec2D_max: torch.Tensor, vec2D_max_target1D: torch.Tensor
    ):
        metric = BrierScore(num_classes=2, reduction="sum")
        metric.update(vec2D_max, vec2D_max_target1D)
        res = metric.compute()
        assert res == 0.5

    def test_compute_5classes(
        self,
        vec2D_5classes: torch.Tensor,
        vec2D_5classes_target: torch.Tensor,
        vec2D_5classes_target1D: torch.Tensor,
    ):
        metric = BrierScore(num_classes=5, reduction="sum")
        metric.update(vec2D_5classes, vec2D_5classes_target)
        metric.update(vec2D_5classes, vec2D_5classes_target1D)
        res = metric.compute()
        assert (
            res / 2
            == 0.2**2
            + 0.6**2
            + 0.1**2 * 2
            + 0.95**2
            + 0.05**2 * 2
            + 0.25**2
            + 0.3**2
            + 0.7**2
        )

    def test_multiple_compute_sum(
        self,
        vec2D_min: torch.Tensor,
        vec2D_max: torch.Tensor,
        vec2D_min_target: torch.Tensor,
        vec2D_max_target: torch.Tensor,
    ):
        metric = BrierScore(num_classes=2, reduction="sum")
        metric.update(vec2D_min, vec2D_min_target)
        metric.update(vec2D_max, vec2D_max_target)
        res = metric.compute()
        assert res == 0.5

    def test_multiple_compute_mean(
        self,
        vec2D_min: torch.Tensor,
        vec2D_max: torch.Tensor,
        vec2D_min_target: torch.Tensor,
        vec2D_max_target: torch.Tensor,
    ):
        metric = BrierScore(num_classes=2, reduction="mean")
        metric.update(vec2D_min, vec2D_min_target)
        metric.update(vec2D_max, vec2D_max_target)
        res = metric.compute()
        assert res == 0.5 / 2

    def test_multiple_compute_none(
        self,
        vec2D_min: torch.Tensor,
        vec2D_max: torch.Tensor,
        vec2D_min_target: torch.Tensor,
        vec2D_max_target: torch.Tensor,
    ):
        metric = BrierScore(num_classes=2, reduction=None)
        metric.update(vec2D_min, vec2D_min_target)
        metric.update(vec2D_max, vec2D_max_target)
        res = metric.compute()
        assert all(res == torch.as_tensor([0, 0.5]))

    def test_compute_3D_mean(
        self, vec3D: torch.Tensor, vec3D_target: torch.Tensor
    ):
        """
        Test that the metric returns the mean of the BrierScore over
            the estimators.
        """
        metric = BrierScore(num_classes=2, reduction="mean")
        metric.update(vec3D, vec3D_target)
        res = metric.compute()
        assert res == 1

    def test_compute_3D_sum(
        self, vec3D: torch.Tensor, vec3D_target: torch.Tensor
    ):
        metric = BrierScore(num_classes=2, reduction="sum")
        metric.update(vec3D, vec3D_target)
        res = metric.compute()
        assert res == 1

    def test_compute_3D_sum_target1D(
        self, vec3D: torch.Tensor, vec3D_target1D: torch.Tensor
    ):
        metric = BrierScore(num_classes=2, reduction="sum")
        metric.update(vec3D, vec3D_target1D)
        res = metric.compute()
        assert res == 1

    def test_compute_3D_to_2D(
        self, vec3D: torch.Tensor, vec3D_target: torch.Tensor
    ):
        metric = BrierScore(num_classes=2, reduction="mean")
        vec3D = vec3D.mean(1)
        metric.update(vec3D, vec3D_target)
        res = metric.compute()
        assert res == 0.5

    def test_bad_input(self) -> None:
        with pytest.raises(Exception):
            metric = BrierScore(num_classes=2, reduction="none")
            metric.update(torch.ones(2, 2, 2, 2), torch.ones(2, 2, 2, 2))

    def test_bad_argument(self):
        with pytest.raises(Exception):
            _ = BrierScore(num_classes=2, reduction="geometric_mean")
