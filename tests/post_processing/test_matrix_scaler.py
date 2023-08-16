# fmt: off
import pytest
import torch

from torch_uncertainty.post_processing import MatrixScaler


# fmt: on
def identity_model(x):
    return x


class TestMatrixScaler:
    """Testing the MatrixScaler class."""

    def test_main(self):
        scaler = MatrixScaler(num_classes=1, init_w=2)
        scaler.set_temperature(1, 0)

        logits = torch.tensor([[1, 2, 3]], dtype=torch.float32)

        assert scaler.temp_w.item() == 1.0
        assert scaler.temp_b.item() == 0.0
        assert torch.all(scaler(logits) == logits)

        scaler.temperature

    def test_negative_numclasses(self):
        with pytest.raises(ValueError):
            MatrixScaler(num_classes=-1)

    def test_float_numclasses(self):
        with pytest.raises(ValueError):
            MatrixScaler(num_classes=1.8)

    def test_negative_lr(self):
        with pytest.raises(ValueError):
            MatrixScaler(num_classes=2, lr=-1)

    def test_negative_maxiter(self):
        with pytest.raises(ValueError):
            MatrixScaler(num_classes=2, max_iter=-1)
