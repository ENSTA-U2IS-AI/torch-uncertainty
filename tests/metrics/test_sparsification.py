import matplotlib.pyplot as plt
import pytest
import torch

from torch_uncertainty.metrics import AUSE


@pytest.fixture
def uncertainty_scores() -> torch.Tensor:
    return torch.as_tensor([0.2, 0.1, 0.5, 0.3, 0.4])


@pytest.fixture
def error_values() -> torch.Tensor:
    return torch.as_tensor([0.1, 0.2, 0.3, 0.4, 0.5])


class TestAUSE:
    """Testing the AUSE metric class."""

    def test_compute_zero(self, error_values: torch.Tensor) -> None:
        metric = AUSE()
        metric.update(error_values, error_values)
        res = metric.compute()
        assert res == 0

    def test_plot(
        self, uncertainty_scores: torch.Tensor, error_values: torch.Tensor
    ) -> None:
        metric = AUSE()
        metric.update(uncertainty_scores, error_values)
        fig, ax = metric.plot()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        assert ax.get_xlabel() == "Rejection Rate (%)"
        assert ax.get_ylabel() == "Error Rate (%)"
        plt.close(fig)

        metric = AUSE()
        metric.update(uncertainty_scores, error_values)
        fig, ax = metric.plot(plot_oracle=False, plot_value=False)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        assert ax.get_xlabel() == "Rejection Rate (%)"
        assert ax.get_ylabel() == "Error Rate (%)"
        plt.close(fig)
