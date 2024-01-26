import pytest
import torch
from torch import nn

from torch_uncertainty.post_processing import VectorScaler


def identity_model(x):
    return x


class TestVectorScaler:
    """Testing the VectorScaler class."""

    def test_main(self):
        scaler = VectorScaler(model=nn.Module(), num_classes=1, init_w=2)
        scaler.set_temperature(1, 0)

        logits = torch.tensor([[1, 2, 3]], dtype=torch.float32)

        assert scaler.temp_w.item() == 1.0
        assert scaler.temp_b.item() == 0.0
        assert torch.all(scaler(logits) == logits)

        _ = scaler.temperature

    def test_errors(self):
        with pytest.raises(ValueError):
            VectorScaler(model=nn.Module(), num_classes=-1)

        with pytest.raises(TypeError):
            VectorScaler(model=nn.Module(), num_classes=1.8)

        with pytest.raises(ValueError):
            VectorScaler(model=nn.Module(), num_classes=2, lr=-1)

        with pytest.raises(ValueError):
            VectorScaler(model=nn.Module(), num_classes=2, max_iter=-1)
