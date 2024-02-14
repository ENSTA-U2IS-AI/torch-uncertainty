import pytest
import torch
from torch import nn, softmax

from torch_uncertainty.post_processing import (
    MatrixScaler,
    TemperatureScaler,
    VectorScaler,
)


class TestTemperatureScaler:
    """Testing the TemperatureScaler class."""

    def test_main(self):
        scaler = TemperatureScaler(model=nn.Identity(), init_val=2)
        scaler.set_temperature(1)

        logits = torch.tensor([[1, 2, 3]], dtype=torch.float32)

        assert scaler.temperature[0].item() == 1.0
        assert torch.all(scaler(logits) == logits)

    def test_fit_biased(self):
        inputs = torch.as_tensor([0.6, 0.4]).repeat(10, 1)
        labels = torch.as_tensor([0.5, 0.5]).repeat(10, 1)

        calibration_set = list(zip(inputs, labels, strict=True))

        scaler = TemperatureScaler(
            model=nn.Identity(), init_val=2, lr=1, max_iter=10
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
            TemperatureScaler(model=nn.Identity(), init_val=-1)

        with pytest.raises(ValueError):
            TemperatureScaler(model=nn.Identity(), lr=-1)

        with pytest.raises(ValueError):
            TemperatureScaler(model=nn.Identity(), max_iter=-1)

        scaler = TemperatureScaler(
            model=nn.Identity(),
        )
        with pytest.raises(ValueError):
            scaler.set_temperature(val=-1)


class TestVectorScaler:
    """Testing the VectorScaler class."""

    def test_main(self):
        scaler = VectorScaler(model=nn.Identity(), num_classes=1, init_w=2)
        scaler.set_temperature(1, 0)

        logits = torch.tensor([[1, 2, 3]], dtype=torch.float32)

        assert scaler.temp_w.item() == 1.0
        assert scaler.temp_b.item() == 0.0
        assert torch.all(scaler(logits) == logits)

        _ = scaler.temperature

    def test_errors(self):
        with pytest.raises(ValueError):
            VectorScaler(model=nn.Identity(), num_classes=-1)

        with pytest.raises(TypeError):
            VectorScaler(model=nn.Identity(), num_classes=1.8)

        with pytest.raises(ValueError):
            VectorScaler(model=nn.Identity(), num_classes=2, lr=-1)

        with pytest.raises(ValueError):
            VectorScaler(model=nn.Identity(), num_classes=2, max_iter=-1)


class TestMatrixScaler:
    """Testing the MatrixScaler class."""

    def test_main(self):
        scaler = MatrixScaler(model=nn.Identity(), num_classes=1, init_w=2)
        scaler.set_temperature(1, 0)

        logits = torch.tensor([[1, 2, 3]], dtype=torch.float32)

        assert scaler.temp_w.item() == 1.0
        assert scaler.temp_b.item() == 0.0
        assert torch.all(scaler(logits) == logits)

        _ = scaler.temperature

    def test_errors(self):
        with pytest.raises(ValueError):
            MatrixScaler(model=nn.Identity(), num_classes=-1)

        with pytest.raises(TypeError):
            MatrixScaler(model=nn.Identity(), num_classes=1.8)

        with pytest.raises(ValueError):
            MatrixScaler(model=nn.Identity(), num_classes=2, lr=-1)

        with pytest.raises(ValueError):
            MatrixScaler(model=nn.Identity(), num_classes=2, max_iter=-1)
