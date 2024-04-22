import matplotlib.pyplot as plt
import pytest
import torch

from torch_uncertainty.metrics import CalibrationError


class TestCalibrationError:
    """Testing the CalibrationError metric class."""

    def test_plot_binary(self) -> None:
        metric = CalibrationError(task="binary", n_bins=2, norm="l1")
        metric.update(
            torch.as_tensor([0.25, 0.25, 0.55, 0.75, 0.75]),
            torch.as_tensor([0, 0, 1, 1, 1]),
        )
        fig, ax = metric.plot()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        assert ax.get_xlabel() == "Top-class Confidence (%)"
        assert ax.get_ylabel() == "Success Rate (%)"
        plt.close(fig)

    def test_plot_multiclass(
        self,
    ) -> None:
        metric = CalibrationError(
            task="multiclass", n_bins=3, norm="l1", num_classes=3
        )
        metric.update(
            torch.as_tensor(
                [
                    [0.25, 0.20, 0.55],
                    [0.55, 0.05, 0.40],
                    [0.10, 0.30, 0.60],
                    [0.90, 0.05, 0.05],
                ]
            ),
            torch.as_tensor([0, 1, 2, 0]),
        )
        fig, ax = metric.plot()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        assert ax.get_xlabel() == "Top-class Confidence (%)"
        assert ax.get_ylabel() == "Success Rate (%)"
        plt.close(fig)

    def test_errors(self) -> None:
        with pytest.raises(ValueError):
            _ = CalibrationError(task="geometric_mean")

        with pytest.raises(ValueError):
            _ = CalibrationError(task="multiclass", num_classes=1.5)
