# fmt:off
import pytest
import torch

from torch_uncertainty.layers.bayesian_layers import BayesConv2d, BayesLinear


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


class TestBayesLinear:
    """Testing the BayesLinear layer class."""

    def test_linear(self, feat_input_odd: torch.Tensor) -> None:
        layer = BayesLinear(10, 2)
        out = layer(feat_input_odd)
        assert out.shape == torch.Size([5, 2])

    def test_linear_even(self, feat_input_even: torch.Tensor) -> None:
        layer = BayesLinear(10, 2)
        out = layer(feat_input_even)
        assert out.shape == torch.Size([8, 2])


class TestBayesConv2d:
    """Testing the BayesConv2d layer class."""

    def test_conv2(self, img_input_odd: torch.Tensor) -> None:
        layer = BayesConv2d(10, 2, kernel_size=1)
        out = layer(img_input_odd)
        assert out.shape == torch.Size([5, 2, 3, 3])

    def test_conv2_even(self, img_input_even: torch.Tensor) -> None:
        layer = BayesConv2d(10, 2, kernel_size=1)
        out = layer(img_input_even)
        assert out.shape == torch.Size([8, 2, 3, 3])
