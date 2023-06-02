# fmt:off
import pytest
import torch

from torch_uncertainty.layers.masksembles_layers import (
    MaskedConv2d,
    MaskedLinear,
)


# fmt:on
@pytest.fixture
def feat_input_odd() -> torch.Tensor:
    feat = torch.rand((5, 10))
    return feat


@pytest.fixture
def feat_input_even() -> torch.Tensor:
    feat = torch.rand((8, 10))
    return feat


@pytest.fixture
def img_input_odd() -> torch.Tensor:
    img = torch.rand((5, 10, 3, 3))
    return img


@pytest.fixture
def img_input_even() -> torch.Tensor:
    img = torch.rand((8, 10, 3, 3))
    return img


class TestMaskedLinear:
    """Testing the MaskedLinear layer class."""

    def test_linear_one_estimator(self, feat_input_odd: torch.Tensor):
        layer = MaskedLinear(10, 2, num_estimators=1, scale=2)
        out = layer(feat_input_odd)
        assert out.shape == torch.Size([5, 2])

    def test_linear_two_estimators_odd(self, feat_input_odd: torch.Tensor):
        layer = MaskedLinear(10, 2, num_estimators=2, scale=2)
        with pytest.raises(Exception):
            _ = layer(feat_input_odd)

    def test_linear_two_estimators_even(self, feat_input_even: torch.Tensor):
        layer = MaskedLinear(10, 2, num_estimators=2, scale=2)
        out = layer(feat_input_even)
        assert out.shape == torch.Size([8, 2])

    def test_linear_c_lt_10(self):
        with pytest.raises(ValueError):
            _ = MaskedLinear(8, 2, num_estimators=1, scale=2)

    def test_linear_s_gt_6(self):
        with pytest.raises(ValueError):
            _ = MaskedLinear(10, 2, num_estimators=1, scale=7)

    def test_linear_s_lt_1(self):
        with pytest.raises(ValueError):
            _ = MaskedLinear(10, 2, num_estimators=1, scale=0)


class TestMaskedConv2d:
    """Testing the MaskedConv2d layer class."""

    def test_conv_one_estimator(self, img_input_odd: torch.Tensor):
        layer = MaskedConv2d(10, 2, num_estimators=1, kernel_size=1, scale=2)
        out = layer(img_input_odd)
        assert out.shape == torch.Size([5, 2, 3, 3])

    def test_conv_two_estimators_odd(self, img_input_odd: torch.Tensor):
        layer = MaskedConv2d(10, 2, num_estimators=2, kernel_size=1, scale=2)
        with pytest.raises(RuntimeError):
            _ = layer(img_input_odd)

    def test_conv_two_estimators_even(self, img_input_even: torch.Tensor):
        layer = MaskedConv2d(10, 2, num_estimators=2, kernel_size=1, scale=2)
        out = layer(img_input_even)
        assert out.shape == torch.Size([8, 2, 3, 3])

    def test_conv_s_lt_1(self):
        with pytest.raises(ValueError):
            _ = MaskedLinear(10, 2, num_estimators=1, scale=0)
