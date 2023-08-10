# fmt: off
import pytest
import torch

from torch_uncertainty.post_processing import VectorScaler


# fmt: on
def identity_model(x):
    return x


class TestVectorScaler:
    """Testing the VectorScaler class."""

    def test_main(self):
        scaler = VectorScaler(num_classes=1, init_w=2)
        scaler.set_temperature(1, 0)

        logits = torch.tensor([[1, 2, 3]], dtype=torch.float32)

        assert scaler.temp_w.item() == 1.0
        assert scaler.temp_b.item() == 0.0
        assert torch.all(scaler(logits) == logits)

    def test_negative_numclasses(self):
        with pytest.raises(ValueError):
            VectorScaler(num_classes=-1)

    def test_float_numclasses(self):
        with pytest.raises(ValueError):
            VectorScaler(num_classes=1.8)

    def test_negative_lr(self):
        with pytest.raises(ValueError):
            VectorScaler(num_classes=2, lr=-1)

    def test_negative_maxiter(self):
        with pytest.raises(ValueError):
            VectorScaler(num_classes=2, max_iter=-1)
