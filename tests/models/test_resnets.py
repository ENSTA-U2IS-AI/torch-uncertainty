import pytest
import torch

from torch_uncertainty.models.resnet import (
    batched_resnet,
    lpbnn_resnet,
    masked_resnet,
    mimo_resnet,
    packed_resnet,
    resnet,
)


class TestResnet:
    """Testing the ResNet classes."""

    def test_main(self):
        resnet(1, 10, arch=18, conv_bias=True, style="cifar")
        model = resnet(1, 10, arch=50, style="cifar")
        with torch.no_grad():
            model(torch.randn(1, 1, 32, 32))
            model.feats_forward(torch.randn(1, 1, 32, 32))

    def test_mc_dropout(self):
        resnet(1, 10, arch=20, conv_bias=False, style="cifar")
        model = resnet(1, 10, arch=50).eval()
        with torch.no_grad():
            model(torch.randn(1, 1, 32, 32))

    def test_error(self):
        with pytest.raises(ValueError):
            resnet(1, 10, arch=20, style="test")


class TestPackedResnet:
    """Testing the ResNet packed class."""

    def test_main(self):
        model = packed_resnet(1, 10, 20, 2, 2, 1)
        model = packed_resnet(1, 10, 152, 2, 2, 1)
        assert model.check_config(
            {"alpha": 2, "gamma": 1, "groups": 1, "num_estimators": 2}
        )
        assert not model.check_config(
            {"alpha": 1, "gamma": 1, "groups": 1, "num_estimators": 2}
        )

    def test_error(self):
        with pytest.raises(ValueError):
            packed_resnet(1, 10, 20, 2, 2, 1, style="test")


class TestMaskedResnet:
    """Testing the ResNet masked class."""

    def test_main(self):
        model = masked_resnet(1, 10, 20, 2, 2)
        with torch.no_grad():
            model(torch.randn(1, 1, 32, 32))

    def test_error(self):
        with pytest.raises(ValueError):
            masked_resnet(1, 10, 20, 2, 2, style="test")


class TestBatchedResnet:
    """Testing the ResNet batched class."""

    def test_main(self):
        model = batched_resnet(1, 10, 20, 2, conv_bias=True)
        with torch.no_grad():
            model(torch.randn(1, 1, 32, 32))

    def test_error(self):
        with pytest.raises(ValueError):
            batched_resnet(1, 10, 20, 2, style="test")


class TestLPBNNResnet:
    """Testing the ResNet LPBNN class."""

    def test_main(self):
        model = lpbnn_resnet(1, 10, 20, 2, conv_bias=True)
        with torch.no_grad():
            model(torch.randn(1, 1, 32, 32))
        model = lpbnn_resnet(1, 10, 50, 2, conv_bias=False, style="cifar")
        with torch.no_grad():
            model(torch.randn(1, 1, 32, 32))

    def test_error(self):
        with pytest.raises(ValueError):
            lpbnn_resnet(1, 10, 20, 2, style="test")
        with pytest.raises(
            ValueError, match="Unknown ResNet architecture. Got"
        ):
            lpbnn_resnet(1, 10, 42, 2, style="test")


class TestMIMOResnet:
    """Testing the ResNet MIMO class."""

    def test_main(self):
        model = mimo_resnet(1, 10, 34, 2, style="cifar", conv_bias=False)
        model.train()
        model(torch.rand((2, 1, 28, 28)))

    def test_error(self):
        with pytest.raises(ValueError):
            mimo_resnet(1, 10, 101, 2, style="test")
