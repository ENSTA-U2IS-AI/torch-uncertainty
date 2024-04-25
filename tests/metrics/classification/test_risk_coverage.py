import matplotlib.pyplot as plt
import torch

from torch_uncertainty.metrics import AURC


class TestAURC:
    """Testing the AURC metric class."""

    def test_compute_zero_binary(self) -> None:
        probs = torch.as_tensor([0.1, 0.2, 0.3, 0.4, 0.2])
        targets = torch.as_tensor([1, 1, 1, 1, 1])
        metric = AURC()
        metric.update(probs, targets)
        assert metric.compute() == 0

    def test_compute_zero_multiclass(self) -> None:
        probs = torch.as_tensor(
            [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.2, 0.8]]
        )
        targets = torch.as_tensor([0, 0, 0, 0, 0]).long()
        metric = AURC()
        metric.update(probs, targets)
        assert metric.compute() == 0

    def test_plot(self) -> None:
        scores = torch.as_tensor([0.2, 0.1, 0.5, 0.3, 0.4])
        values = torch.as_tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        metric = AURC()
        metric.update(scores, values)
        fig, ax = metric.plot()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        assert ax.get_xlabel() == "Coverage (%)"
        assert ax.get_ylabel() == "Risk - Error Rate (%)"
        plt.close(fig)

        metric = AURC()
        metric.update(scores, values)
        fig, ax = metric.plot(plot_oracle=False, plot_value=False)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        assert ax.get_xlabel() == "Coverage (%)"
        assert ax.get_ylabel() == "Risk - Error Rate (%)"
        plt.close(fig)
