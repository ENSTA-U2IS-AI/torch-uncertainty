# fmt:off
import pytest
import torch

from torch_uncertainty.layers.packed import PackedConv2d, PackedLinear


# fmt:on
@pytest.fixture
def feat_input() -> torch.Tensor:
    feat = torch.rand((6, 1))
    return feat


@pytest.fixture
def feat_input_one_rearrange() -> torch.Tensor:
    feat = torch.rand((1 * 3, 5))
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
        assert out.shape == torch.Size([2, 1])

    def test_linear_two_estimators_no_rearrange(self, feat_input: torch.Tensor):
        layer = PackedLinear(6, 2, alpha=1, num_estimators=2, rearrange=False)
        out = layer(feat_input)
        assert out.shape == torch.Size([2, 1])

    def test_linear_one_estimator_rearrange(
        self, feat_input_one_rearrange: torch.Tensor
    ):
        layer = PackedLinear(5, 2, alpha=1, num_estimators=1, rearrange=True)
        out = layer(feat_input_one_rearrange)
        assert out.shape == torch.Size([3, 2])

    def test_linear_two_estimator_rearrange_not_divisible(self):
        feat = torch.rand((2 * 3, 3))
        layer = PackedLinear(5, 1, alpha=1, num_estimators=2, rearrange=True)
        out = layer(feat)
        assert out.shape == torch.Size([6, 1])

    def test_linear_extend(self):
        _ = PackedConv2d(
            5, 3, kernel_size=1, alpha=1, num_estimators=2, gamma=1
        )

    def test_linear_alpha_error(self):
        with pytest.raises(ValueError):
            _ = PackedLinear(5, 2, alpha=None, num_estimators=1, rearrange=True)

        with pytest.raises(ValueError):
            _ = PackedLinear(5, 2, alpha=-1, num_estimators=1, rearrange=True)

    def test_linear_num_estimators_error(self):
        with pytest.raises(ValueError):
            _ = PackedLinear(5, 2, alpha=1, num_estimators=None, rearrange=True)

        with pytest.raises(ValueError):
            _ = PackedLinear(5, 2, alpha=1, num_estimators=1.5, rearrange=True)

        with pytest.raises(ValueError):
            _ = PackedLinear(5, 2, alpha=1, num_estimators=-1, rearrange=True)

    def test_linear_gamma_error(self):
        with pytest.raises(ValueError):
            _ = PackedLinear(
                5, 2, alpha=1, num_estimators=1, gamma=0.5, rearrange=True
            )

        with pytest.raises(ValueError):
            _ = PackedLinear(
                5, 2, alpha=1, num_estimators=1, gamma=-1, rearrange=True
            )


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

    def test_conv_extend(self):
        _ = PackedConv2d(
            5, 3, kernel_size=1, alpha=1, num_estimators=2, gamma=1
        )

    def test_conv_alpha_neg(self):
        with pytest.raises(ValueError):
            _ = PackedConv2d(5, 2, kernel_size=1, alpha=-1, num_estimators=1)

    def test_conv_gamma_float(self):
        with pytest.raises(ValueError):
            _ = PackedConv2d(
                5, 2, kernel_size=1, alpha=1, num_estimators=1, gamma=0.5
            )

    def test_conv_gamma_neg(self):
        with pytest.raises(ValueError):
            _ = PackedConv2d(
                5, 2, kernel_size=1, alpha=1, num_estimators=1, gamma=-0.5
            )
