import pytest
import torch

from torch_uncertainty.metrics import BrierScore


@pytest.fixture()
def vec2d_max() -> torch.Tensor:
    vec = torch.as_tensor([0.5, 0.5])
    return vec.unsqueeze(0)


@pytest.fixture()
def vec2d_max_target() -> torch.Tensor:
    vec = torch.as_tensor([0, 1])
    return vec.unsqueeze(0)


@pytest.fixture()
def vec2d_max_target1d() -> torch.Tensor:
    return torch.as_tensor([1])


@pytest.fixture()
def vec2d_min() -> torch.Tensor:
    vec = torch.as_tensor([0.0, 1.0])
    return vec.unsqueeze(0)


@pytest.fixture()
def vec2d_min_target() -> torch.Tensor:
    vec = torch.as_tensor([0, 1])
    return vec.unsqueeze(0)


@pytest.fixture()
def vec2d_5classes() -> torch.Tensor:
    return torch.as_tensor(
        [[0.2, 0.6, 0.1, 0.05, 0.05], [0.05, 0.25, 0.1, 0.3, 0.3]]
    )


@pytest.fixture()
def vec2d_5classes_target() -> torch.Tensor:
    return torch.as_tensor([[0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])


@pytest.fixture()
def vec2d_5classes_target1d() -> torch.Tensor:
    return torch.as_tensor([3, 4])


@pytest.fixture()
def vec3d() -> torch.Tensor:
    """Return a torch tensor with a mean BrierScore of 0 and a BrierScore of
    the mean of 0.5 to test the `ensemble` parameter of `BrierScore`.
    """
    vec = torch.as_tensor([[0.0, 1.0], [1.0, 0.0]])
    return vec.unsqueeze(0)


@pytest.fixture()
def vec3d_target() -> torch.Tensor:
    vec = torch.as_tensor([0, 1])
    return vec.unsqueeze(0)


@pytest.fixture()
def vec3d_target1d() -> torch.Tensor:
    vec = torch.as_tensor([1])
    return vec.unsqueeze(0)


class TestBrierScore:
    """Testing the BrierScore metric class."""

    def test_compute(
        self, vec2d_min: torch.Tensor, vec2d_min_target: torch.Tensor
    ):
        metric = BrierScore(num_classes=2)
        metric.update(vec2d_min, vec2d_min_target)
        assert metric.compute() == 0

        metric = BrierScore(num_classes=2, top_class=True)
        metric.update(vec2d_min, vec2d_min_target)
        assert metric.compute() == 0

    def test_compute_max(
        self, vec2d_max: torch.Tensor, vec2d_max_target: torch.Tensor
    ):
        metric = BrierScore(num_classes=2, reduction="sum")
        metric.update(vec2d_max, vec2d_max_target)
        assert metric.compute() == 0.5

    def test_compute_max_target1d(
        self, vec2d_max: torch.Tensor, vec2d_max_target1d: torch.Tensor
    ):
        metric = BrierScore(num_classes=2, reduction="sum")
        metric.update(vec2d_max, vec2d_max_target1d)
        assert metric.compute() == 0.5

    def test_compute_5classes(
        self,
        vec2d_5classes: torch.Tensor,
        vec2d_5classes_target: torch.Tensor,
        vec2d_5classes_target1d: torch.Tensor,
    ):
        metric = BrierScore(num_classes=5, reduction="sum")
        metric.update(vec2d_5classes, vec2d_5classes_target)
        metric.update(vec2d_5classes, vec2d_5classes_target1d)
        assert (
            metric.compute() / 2
            == 0.2**2
            + 0.6**2
            + 0.1**2 * 2
            + 0.95**2
            + 0.05**2 * 2
            + 0.25**2
            + 0.3**2
            + 0.7**2
        )

        metric = BrierScore(num_classes=5, top_class=True, reduction="sum")
        metric.update(vec2d_5classes, vec2d_5classes_target)
        assert metric.compute() == pytest.approx(0.6**2 + 0.3**2)

    def test_multiple_compute_sum(
        self,
        vec2d_min: torch.Tensor,
        vec2d_max: torch.Tensor,
        vec2d_min_target: torch.Tensor,
        vec2d_max_target: torch.Tensor,
    ):
        metric = BrierScore(num_classes=2, reduction="sum")
        metric.update(vec2d_min, vec2d_min_target)
        metric.update(vec2d_max, vec2d_max_target)
        assert metric.compute() == 0.5

    def test_multiple_compute_mean(
        self,
        vec2d_min: torch.Tensor,
        vec2d_max: torch.Tensor,
        vec2d_min_target: torch.Tensor,
        vec2d_max_target: torch.Tensor,
    ):
        metric = BrierScore(num_classes=2, reduction="mean")
        metric.update(vec2d_min, vec2d_min_target)
        metric.update(vec2d_max, vec2d_max_target)
        assert metric.compute() == 0.5 / 2

    def test_multiple_compute_none(
        self,
        vec2d_min: torch.Tensor,
        vec2d_max: torch.Tensor,
        vec2d_min_target: torch.Tensor,
        vec2d_max_target: torch.Tensor,
    ):
        metric = BrierScore(num_classes=2, reduction=None)
        metric.update(vec2d_min, vec2d_min_target)
        metric.update(vec2d_max, vec2d_max_target)
        assert all(metric.compute() == torch.as_tensor([0, 0.5]))

    def test_compute_3d_mean(
        self, vec3d: torch.Tensor, vec3d_target: torch.Tensor
    ):
        """Test that the metric returns the mean of the BrierScore over
        the estimators.
        """
        metric = BrierScore(num_classes=2, reduction="mean")
        metric.update(vec3d, vec3d_target)
        assert metric.compute() == 1

    def test_compute_3d_sum(
        self, vec3d: torch.Tensor, vec3d_target: torch.Tensor
    ):
        metric = BrierScore(num_classes=2, reduction="sum")
        metric.update(vec3d, vec3d_target)
        assert metric.compute() == 1

    def test_compute_3d_sum_target1d(
        self, vec3d: torch.Tensor, vec3d_target1d: torch.Tensor
    ):
        metric = BrierScore(num_classes=2, reduction="sum")
        metric.update(vec3d, vec3d_target1d)
        assert metric.compute() == 1

    def test_compute_3d_to_2d(
        self, vec3d: torch.Tensor, vec3d_target: torch.Tensor
    ):
        metric = BrierScore(num_classes=2, reduction="mean")
        vec3d = vec3d.mean(1)
        metric.update(vec3d, vec3d_target)
        assert metric.compute() == 0.5

    def test_bad_input(self) -> None:
        with pytest.raises(ValueError):
            metric = BrierScore(num_classes=2, reduction="none")
            metric.update(torch.ones(2, 2, 2, 2), torch.ones(2, 2, 2, 2))

    def test_bad_argument(self):
        with pytest.raises(
            ValueError, match="Expected argument `reduction` to be one of"
        ):
            _ = BrierScore(num_classes=2, reduction="geometric_mean")
