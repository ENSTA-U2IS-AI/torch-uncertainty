import matplotlib.pyplot as plt
import pytest
import torch

from torch_uncertainty.metrics import CE


@pytest.fixture
def preds_binary() -> torch.Tensor:
    return torch.as_tensor([0.25, 0.25, 0.55, 0.75, 0.75])


@pytest.fixture
def targets_binary() -> torch.Tensor:
    return torch.as_tensor([0, 0, 1, 1, 1])


@pytest.fixture
def preds_multiclass() -> torch.Tensor:
    return torch.as_tensor(
        [
            [0.25, 0.20, 0.55],
            [0.55, 0.05, 0.40],
            [0.10, 0.30, 0.60],
            [0.90, 0.05, 0.05],
        ]
    )


@pytest.fixture
def targets_multiclass() -> torch.Tensor:
    return torch.as_tensor([0, 1, 2, 0])


class TestCE:
    """Testing the CE metric class."""

    def test_plot_binary(
        self, preds_binary: torch.Tensor, targets_binary: torch.Tensor
    ) -> None:
        metric = CE(task="binary", n_bins=2, norm="l1")
        metric.update(preds_binary, targets_binary)
        fig, ax = metric.plot()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        assert ax.get_xlabel() == "Top-class Confidence (%)"
        assert ax.get_ylabel() == "Success Rate (%)"
        plt.close(fig)

    def test_plot_multiclass(
        self, preds_multiclass: torch.Tensor, targets_multiclass: torch.Tensor
    ) -> None:
        metric = CE(task="multiclass", n_bins=3, norm="l1", num_classes=3)
        metric.update(preds_multiclass, targets_multiclass)
        fig, ax = metric.plot()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        assert ax.get_xlabel() == "Top-class Confidence (%)"
        assert ax.get_ylabel() == "Success Rate (%)"
        plt.close(fig)

    def test_bad_task_argument(self) -> None:
        with pytest.raises(ValueError):
            _ = CE(task="geometric_mean")

    def test_bad_num_classes_argument(self) -> None:
        with pytest.raises(ValueError):
            _ = CE(task="multiclass", num_classes=1.5)
