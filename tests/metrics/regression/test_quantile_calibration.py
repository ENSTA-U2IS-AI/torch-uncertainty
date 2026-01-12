import matplotlib.pyplot as plt
import torch
from torch.distributions import Normal

from torch_uncertainty.metrics import QuantileCalibrationError


class TestQuantileCalibrationError:
    """Testing the QuantileCalibrationError metric class."""

    def test_main(self) -> None:
        torch.manual_seed(42)
        dist = Normal(torch.tensor([[0.0]] * 1000), torch.tensor([[1.0]] * 1000))
        targets = dist.sample()
        qce = QuantileCalibrationError()
        qce.update(dist, targets)
        res = qce.compute()
        assert res.item() < 0.02

        fig, ax = qce.plot()
        assert isinstance(fig, plt.Figure)
        assert ax[0].get_xlabel() == "Top-class Confidence (%)"
        assert ax[0].get_ylabel() == "Success Rate (%)"
        assert ax[1].get_xlabel() == "Top-class Confidence (%)"
        assert ax[1].get_ylabel() == "Density (%)"

        qce2 = QuantileCalibrationError()
        qce2.update(dist, targets, padding_mask=torch.zeros(1000, dtype=torch.bool))
        res2 = qce2.compute()
        assert res2.item() < 0.02
