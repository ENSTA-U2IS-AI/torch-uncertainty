import matplotlib.pyplot as plt
import pytest
import torch

from torch_uncertainty.metrics import AdaptiveCalibrationError, CalibrationError


class TestCalibrationError:
    """Testing the CalibrationError metric class."""

    def test_plot_binary(self) -> None:
        metric = CalibrationError(task="binary", n_bins=2, norm="l1")
        metric.update(
            torch.as_tensor([0.25, 0.25, 0.55, 0.75, 0.75]),
            torch.as_tensor([0, 0, 1, 1, 1]),
        )
        fig, ax = metric.plot()
        metric.plot(ax=ax)
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
        with pytest.raises(TypeError, match="is expected to be `int`"):
            CalibrationError(task="multiclass", num_classes=None)


class TestAdaptiveCalibrationError:
    """Testing the AdaptiveCalibrationError metric class."""

    def test_main(self) -> None:
        ace = AdaptiveCalibrationError(
            task="binary", num_bins=2, norm="l1", validate_args=True
        )
        ace = AdaptiveCalibrationError(
            task="binary", num_bins=2, norm="l1", validate_args=False
        )
        ece = CalibrationError(task="binary", num_bins=2, norm="l1")
        ace.update(
            torch.as_tensor([0.35, 0.35, 0.75, 0.75]),
            torch.as_tensor([0, 0, 1, 1]),
        )
        ece.update(
            torch.as_tensor([0.35, 0.35, 0.75, 0.75]),
            torch.as_tensor([0, 0, 1, 1]),
        )
        assert ace.compute().item() == ece.compute().item()

        ace.reset()
        ace.update(
            torch.as_tensor([0.3, 0.24, 0.25, 0.2, 0.8]),
            torch.as_tensor([0, 0, 0, 1, 1]),
        )
        assert ace.compute().item() == pytest.approx(
            3 / 5 * (1 - 1 / 3 * (0.7 + 0.76 + 0.75)) + 2 / 5 * (0.8 - 0.5)
        )

        ace = AdaptiveCalibrationError(
            task="multiclass",
            num_classes=2,
            num_bins=2,
            norm="l2",
            validate_args=True,
        )
        ace.update(
            torch.as_tensor(
                [[0.7, 0.3], [0.76, 0.24], [0.75, 0.25], [0.2, 0.8], [0.8, 0.2]]
            ),
            torch.as_tensor([0, 0, 0, 1, 1]),
        )
        assert ace.compute().item() ** 2 == pytest.approx(
            3 / 5 * (1 - 1 / 3 * (0.7 + 0.76 + 0.75)) ** 2
            + 2 / 5 * (0.8 - 0.5) ** 2
        )

        ace = AdaptiveCalibrationError(
            task="multiclass",
            num_classes=2,
            num_bins=2,
            norm="max",
            validate_args=False,
        )
        ace.update(
            torch.as_tensor(
                [[0.7, 0.3], [0.76, 0.24], [0.75, 0.25], [0.2, 0.8], [0.8, 0.2]]
            ),
            torch.as_tensor([0, 0, 0, 1, 1]),
        )
        assert ace.compute().item() ** 2 == pytest.approx((0.8 - 0.5) ** 2)

    def test_errors(self) -> None:
        with pytest.raises(TypeError, match="is expected to be `int`"):
            AdaptiveCalibrationError(task="multiclass", num_classes=None)
