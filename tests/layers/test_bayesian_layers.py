# fmt:off
import pytest
import torch

from torch_uncertainty.layers.bayesian_layers import (
    BayesConv1d,
    BayesConv2d,
    BayesConv3d,
    BayesLinear,
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


@pytest.fixture
def cube_input_odd() -> torch.Tensor:
    img = torch.rand((5, 10, 3, 3, 3))
    return img


@pytest.fixture
def cube_input_even() -> torch.Tensor:
    img = torch.rand((8, 10, 3, 3, 3))
    return img


class TestBayesLinear:
    """Testing the BayesLinear layer class."""

    def test_linear(self, feat_input_odd: torch.Tensor) -> None:
        layer = BayesLinear(10, 2)
        print(layer)
        out = layer(feat_input_odd)
        assert out.shape == torch.Size([5, 2])

    def test_linear_even(self, feat_input_even: torch.Tensor) -> None:
        layer = BayesLinear(10, 2)
        out = layer(feat_input_even)
        assert out.shape == torch.Size([8, 2])


class TestBayesConv1d:
    """Testing the BayesConv1d layer class."""

    def test_conv1(self, feat_input_odd: torch.Tensor) -> None:
        layer = BayesConv1d(5, 2, kernel_size=1)
        print(layer)
        out = layer(feat_input_odd)
        assert out.shape == torch.Size([2, 10])

    def test_conv1_even(self, feat_input_even: torch.Tensor) -> None:
        layer = BayesConv1d(8, 2, kernel_size=1)
        out = layer(feat_input_even)
        assert out.shape == torch.Size([2, 10])


class TestBayesConv2d:
    """Testing the BayesConv2d layer class."""

    def test_conv2(self, img_input_odd: torch.Tensor) -> None:
        layer = BayesConv2d(10, 2, kernel_size=1)
        print(layer)
        out = layer(img_input_odd)
        assert out.shape == torch.Size([5, 2, 3, 3])

    def test_conv2_even(self, img_input_even: torch.Tensor) -> None:
        layer = BayesConv2d(10, 2, kernel_size=1)
        out = layer(img_input_even)
        assert out.shape == torch.Size([8, 2, 3, 3])


class TestBayesConv3d:
    """Testing the BayesConv3d layer class."""

    def test_conv3(self, cube_input_odd: torch.Tensor) -> None:
        layer = BayesConv3d(10, 2, kernel_size=1)
        print(layer)
        out = layer(cube_input_odd)
        assert out.shape == torch.Size([5, 2, 3, 3, 3])

    def test_conv3_even(self, cube_input_even: torch.Tensor) -> None:
        layer = BayesConv3d(10, 2, kernel_size=1)
        out = layer(cube_input_even)
        assert out.shape == torch.Size([8, 2, 3, 3, 3])
