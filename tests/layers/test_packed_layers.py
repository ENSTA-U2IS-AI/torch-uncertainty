# fmt:off
import pytest
import torch

from torch_uncertainty.layers.packed_layers import PackedConv2d, PackedLinear


# fmt:on
@pytest.fixture
def feat_input() -> torch.Tensor:
    feat = torch.rand((6, 1, 1))
    return feat


@pytest.fixture
def feat_input_one_rearrange() -> torch.Tensor:
    feat = torch.rand((1 * 3, 5))
    return feat


@pytest.fixture
def feat_input_two_rearrange() -> torch.Tensor:
    feat = torch.rand((2 * 3, 5))
    return feat


@pytest.fixture
def img_input() -> torch.Tensor:
    img = torch.rand((5, 6, 3, 3))
    return img


class TestPackedLinear:
    """Testing the PackedLinear layer class."""

    def test_linear_one_estimator_no_rearrange(self, feat_input: torch.Tensor):
        layer = PackedLinear(6, 2, alpha=1, num_estimators=1, rearrange=False)
        out = layer(feat_input)
        assert out.shape == torch.Size([2, 1, 1])

    def test_linear_two_estimators_no_rearrange(self, feat_input: torch.Tensor):
        layer = PackedLinear(6, 2, alpha=1, num_estimators=2, rearrange=False)
        out = layer(feat_input)
        assert out.shape == torch.Size([2, 1, 1])

    def test_linear_one_estimator_rearrange(
        self, feat_input_one_rearrange: torch.Tensor
    ):
        layer = PackedLinear(5, 2, alpha=1, num_estimators=1, rearrange=True)
        out = layer(feat_input_one_rearrange)
        assert out.shape == torch.Size([3, 2])

    def test_linear_two_estimator_rearrange(
        self, feat_input_two_rearrange: torch.Tensor
    ):
        layer = PackedLinear(5, 2, alpha=1, num_estimators=1, rearrange=True)
        out = layer(feat_input_two_rearrange)
        assert out.shape == torch.Size([6, 2])


class TestPackedConv2d:
    """Testing the PackedConv2d layer class."""

    def test_conv_one_estimator(self, img_input: torch.Tensor):
        layer = PackedConv2d(6, 2, alpha=1, num_estimators=1, kernel_size=1)
        out = layer(img_input)
        assert out.shape == torch.Size([5, 2, 3, 3])

    def test_conv_two_estimators(self, img_input: torch.Tensor):
        layer = PackedConv2d(6, 2, alpha=1, num_estimators=2, kernel_size=1)
        out = layer(img_input)
        assert out.shape == torch.Size([5, 2, 3, 3])

    def test_conv_one_estimator_gamma2(self, img_input: torch.Tensor):
        layer = PackedConv2d(
            6, 2, alpha=1, num_estimators=1, kernel_size=1, gamma=2
        )
        out = layer(img_input)
        assert out.shape == torch.Size([5, 2, 3, 3])
        assert layer.conv.groups == 1  # and not 2

    def test_conv_two_estimators_gamma2(self, img_input: torch.Tensor):
        layer = PackedConv2d(
            6, 2, alpha=1, num_estimators=2, kernel_size=1, gamma=2
        )
        out = layer(img_input)
        assert out.shape == torch.Size([5, 2, 3, 3])
        assert layer.conv.groups == 2  # and not 4
