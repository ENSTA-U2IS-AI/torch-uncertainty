# fmt: off
import pytest
import torch

from torch_uncertainty.post_processing import TemperatureScaler


# fmt: on
class TestTemperatureScaler:
    """Testing the TemperatureScaler class."""

    def test_main(self):
        scaler = TemperatureScaler(init_val=2)
        scaler.set_temperature(1)

        logits = torch.tensor([[1, 2, 3]], dtype=torch.float32)

        assert scaler.temperature.item() == 1.0
        assert torch.all(scaler(logits) == logits)

    def test_negative_initvalue(self):
        with pytest.raises(ValueError):
            TemperatureScaler(init_val=-1)
