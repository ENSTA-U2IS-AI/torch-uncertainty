import matplotlib.pyplot as plt
import pytest
import torch

from torch_uncertainty.metrics.classification import (
    AURC,
    CovAtxRisk,
    RiskAtxCov,
)


class TestAURC:
    """Testing the AURC metric class."""

    def test_compute_binary(self) -> None:
        probs = torch.as_tensor([0.1, 0.2, 0.3, 0.4, 0.2])
        targets = torch.as_tensor([1, 1, 1, 1, 1])
        metric = AURC()
        assert metric(probs, targets).item() == pytest.approx(1)
        targets = torch.as_tensor([0, 0, 0, 0, 0])
        metric = AURC()
        assert metric(probs, targets).item() == pytest.approx(0)
        targets = torch.as_tensor([0, 0, 1, 1, 0])
        metric = AURC()
        value = (0 * 0.4 + 0.25 * 0.2 / 2 + 0.25 * 0.2 + 0.15 * 0.2 / 2) / 0.8
        assert metric(probs, targets).item() == pytest.approx(value)

    def test_compute_multiclass(self) -> None:
        probs = torch.as_tensor(
            [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.2, 0.8]]
        )
        targets = torch.as_tensor([1, 1, 1, 1, 1]).long()
        metric = AURC()
        assert metric(probs, targets).item() == pytest.approx(0)
        targets = torch.as_tensor([0, 0, 0, 0, 0])
        metric = AURC()
        assert metric(probs, targets).item() == pytest.approx(1)
        targets = torch.as_tensor([1, 1, 0, 0, 1])
        metric = AURC()
        value = (0 * 0.4 + 0.25 * 0.2 / 2 + 0.25 * 0.2 + 0.15 * 0.2 / 2) / 0.8
        assert metric(probs, targets).item() == pytest.approx(value)

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
        fig, ax = metric.plot(plot_value=False)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        assert ax.get_xlabel() == "Coverage (%)"
        assert ax.get_ylabel() == "Risk - Error Rate (%)"
        plt.close(fig)


class TestCovAtxRisk:
    """Testing the CovAtxRisk metric class."""

    def test_compute_zero(self) -> None:
        probs = torch.as_tensor(
            [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.8, 0.2]]
        )
        targets = torch.as_tensor([1, 1, 1, 1, 1])
        metric = CovAtxRisk(risk_threshold=0.5)
        # no cov for given risk
        assert torch.isnan(metric(probs, targets))

        probs = torch.as_tensor(
            [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.48, 0.49]
        )
        targets = torch.as_tensor([1, 0, 1, 1, 1, 0, 0, 0, 1])
        metric = CovAtxRisk(risk_threshold=0.55)
        # multiple cov for given risk
        assert metric(probs, targets) == pytest.approx(8 / 9)

        probs = torch.as_tensor([0.1, 0.2, 0.3, 0.4, 0.2])
        targets = torch.as_tensor([0, 0, 1, 1, 1])
        metric = CovAtxRisk(risk_threshold=0.5)
        assert metric(probs, targets) == pytest.approx(4 / 5)

        targets = torch.as_tensor([0, 0, 1, 1, 0])
        metric = CovAtxRisk(risk_threshold=0.5)
        assert metric(probs, targets) == 1

    def test_errors(self):
        with pytest.raises(
            TypeError, match="Expected threshold to be of type float"
        ):
            CovAtxRisk(risk_threshold="0.5")
        with pytest.raises(
            ValueError, match="Threshold should be in the range"
        ):
            CovAtxRisk(risk_threshold=-0.5)


class TestRiskAtxCov:
    """Testing the RiskAtxCov metric class."""

    def test_compute_zero(self) -> None:
        probs = torch.as_tensor(
            [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.8, 0.2]]
        )
        targets = torch.as_tensor([1, 1, 1, 1, 1])
        metric = RiskAtxCov(cov_threshold=0.5)
        assert metric(probs, targets) == 1

        probs = torch.as_tensor([0.1, 0.2, 0.3, 0.4, 0.2])
        targets = torch.as_tensor([0, 0, 1, 1, 1])
        metric = RiskAtxCov(cov_threshold=0.5)
        assert metric(probs, targets) == pytest.approx(1 / 3)

        probs = torch.as_tensor([0.1, 0.19, 0.3, 0.15, 0.4, 0.2])
        targets = torch.as_tensor([0, 0, 1, 0, 1, 1])
        metric = RiskAtxCov(cov_threshold=0.5)
        assert metric(probs, targets) == 0

        probs = torch.as_tensor([0.1, 0.2, 0.3, 0.15, 0.4, 0.2])
        targets = torch.as_tensor([0, 0, 1, 0, 1, 1])
        metric = RiskAtxCov(cov_threshold=0.55)
        assert metric(probs, targets) == 1 / 4
