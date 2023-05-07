# fmt:off
import pytest
import torch

from torch_uncertainty.layers.batchens_layers import BatchConv2d, BatchLinear


# fmt:on
@pytest.fixture
def feat_input() -> torch.Tensor:
    feat = torch.rand((4, 6))
    return feat


@pytest.fixture
def img_input() -> torch.Tensor:
    img = torch.rand((5, 6, 3, 3))
    return img


class TestBatchLinear:
    """Testing the BatchLinear layer class."""

    def test_linear_one_estimator(self, feat_input: torch.Tensor):
        layer = BatchLinear(6, 2, num_estimators=1)
        out = layer(feat_input)
        assert out.shape == torch.Size([4, 2])

    def test_linear_two_estimators(self, feat_input: torch.Tensor):
        layer = BatchLinear(6, 2, num_estimators=2)
        out = layer(feat_input)
        assert out.shape == torch.Size([4, 2])

    def test_linear_one_estimator_no_bias(self, feat_input: torch.Tensor):
        layer = BatchLinear(6, 2, num_estimators=1, bias=False)
        out = layer(feat_input)
        assert out.shape == torch.Size([4, 2])


class TestBatchConv2d:
    """Testing the BatchConv2d layer class."""

    def test_conv_one_estimator(self, img_input: torch.Tensor):
        layer = BatchConv2d(6, 2, num_estimators=1, kernel_size=1)
        out = layer(img_input)
        assert out.shape == torch.Size([5, 2, 3, 3])

    def test_conv_two_estimators(self, img_input: torch.Tensor):
        layer = BatchConv2d(6, 2, num_estimators=2, kernel_size=1)
        out = layer(img_input)
        assert out.shape == torch.Size([5, 2, 3, 3])
