import matplotlib.pyplot as plt
import torch

from torch_uncertainty.metrics import AUSE


class TestAUSE:
    """Testing the AUSE metric class."""

    def test_compute_zero(self) -> None:
        values = torch.as_tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        metric = AUSE()
        metric.update(values, values)
        assert metric.compute() == 0

    def test_plot(self) -> None:
        scores = torch.as_tensor([0.2, 0.1, 0.5, 0.3, 0.4])
        values = torch.as_tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        metric = AUSE()
        metric.update(scores, values)
        fig, ax = metric.plot()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        assert ax.get_xlabel() == "Rejection Rate (%)"
        assert ax.get_ylabel() == "Error Rate (%)"
        plt.close(fig)

        metric = AUSE()
        metric.update(scores, values)
        fig, ax = metric.plot(plot_oracle=False, plot_value=False)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        assert ax.get_xlabel() == "Rejection Rate (%)"
        assert ax.get_ylabel() == "Error Rate (%)"
        plt.close(fig)
