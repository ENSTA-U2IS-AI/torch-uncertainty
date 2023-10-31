import pytest
import torch
from torch import softmax

from torch_uncertainty.post_processing import TemperatureScaler


def identity_model(x):
    return x


class TestTemperatureScaler:
    """Testing the TemperatureScaler class."""

    def test_main(self):
        scaler = TemperatureScaler(init_val=2)
        scaler.set_temperature(1)

        logits = torch.tensor([[1, 2, 3]], dtype=torch.float32)

        assert scaler.temperature[0].item() == 1.0
        assert torch.all(scaler(logits) == logits)

    def test_fit_biased(self):
        inputs = torch.as_tensor([0.6, 0.4]).repeat(10, 1)
        labels = torch.as_tensor([0.5, 0.5]).repeat(10, 1)

        calibration_set = list(zip(inputs, labels))

        scaler = TemperatureScaler(init_val=2, lr=1, max_iter=10)
        assert scaler.temperature[0] == 2.0
        scaler.fit(identity_model, calibration_set)
        assert scaler.temperature[0] > 10  # best is +inf
        assert (
            torch.sum(
                (
                    softmax(
                        scaler(torch.as_tensor([[0.6, 0.4]])).detach(), dim=1
                    )
                    - torch.as_tensor([[0.5, 0.5]])
                )
            )
            ** 2
            < 0.001
        )
        scaler.fit_predict(identity_model, calibration_set, progress=False)

    def test_negative_initvalue(self):
        with pytest.raises(ValueError):
            TemperatureScaler(init_val=-1)

    def test_negative_lr(self):
        with pytest.raises(ValueError):
            TemperatureScaler(lr=-1)

    def test_negative_maxiter(self):
        with pytest.raises(ValueError):
            TemperatureScaler(max_iter=-1)

    def test_negative_val(self):
        scaler = TemperatureScaler()
        with pytest.raises(ValueError):
            scaler.set_temperature(val=-1)
