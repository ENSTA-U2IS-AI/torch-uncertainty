import pytest
import torch
from torch import nn, softmax

from torch_uncertainty.post_processing import TemperatureScaler


def identity_model(x):
    return x


class TestTemperatureScaler:
    """Testing the TemperatureScaler class."""

    def test_main(self):
        scaler = TemperatureScaler(model=nn.Module(), init_val=2)
        scaler.set_temperature(1)

        logits = torch.tensor([[1, 2, 3]], dtype=torch.float32)

        assert scaler.temperature[0].item() == 1.0
        assert torch.all(scaler(logits) == logits)

    def test_fit_biased(self):
        inputs = torch.as_tensor([0.6, 0.4]).repeat(10, 1)
        labels = torch.as_tensor([0.5, 0.5]).repeat(10, 1)

        calibration_set = list(zip(inputs, labels, strict=True))

        scaler = TemperatureScaler(
            model=identity_model, init_val=2, lr=1, max_iter=10
        )
        assert scaler.temperature[0] == 2.0
        scaler.fit(calibration_set)
        assert scaler.temperature[0] > 10  # best is +inf
        assert (
            torch.sum(
                softmax(scaler(torch.as_tensor([[0.6, 0.4]])).detach(), dim=1)
                - torch.as_tensor([[0.5, 0.5]])
            )
            ** 2
            < 0.001
        )
        scaler.fit_predict(calibration_set, progress=False)

    def test_errors(self):
        with pytest.raises(ValueError):
            TemperatureScaler(model=nn.Module(), init_val=-1)

        with pytest.raises(ValueError):
            TemperatureScaler(model=nn.Module(), lr=-1)

        with pytest.raises(ValueError):
            TemperatureScaler(model=nn.Module(), max_iter=-1)

        scaler = TemperatureScaler(
            model=nn.Module(),
        )
        with pytest.raises(ValueError):
            scaler.set_temperature(val=-1)
