import pytest
import torch
import matplotlib.pyplot as plt

from torch_uncertainty.metrics import CE


@pytest.fixture
def preds_binary() -> torch.Tensor:
    preds = torch.as_tensor([0.25, 0.25, 0.55, 0.75, 0.75])
    return preds


@pytest.fixture
def targets_binary() -> torch.Tensor:
    targets = torch.as_tensor([0, 0, 1, 1, 1])
    return targets


@pytest.fixture
def preds_multiclass() -> torch.Tensor:
    preds = torch.as_tensor(
        [
            [0.25, 0.20, 0.55],
            [0.55, 0.05, 0.40],
            [0.10, 0.30, 0.60],
            [0.90, 0.05, 0.05],
        ]
    )
    return preds


@pytest.fixture
def targets_multiclass() -> torch.Tensor:
    targets = torch.as_tensor([0, 1, 2, 0])
    return targets


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
