import pytest
import torch

from torch_uncertainty.layers.bayesian import (
    BayesConv1d,
    BayesConv2d,
    BayesConv3d,
    BayesLinear,
    LPBNNConv2d,
    LPBNNLinear,
)
from torch_uncertainty.layers.bayesian.sampler import TrainableDistribution


@pytest.fixture()
def feat_input_odd() -> torch.Tensor:
    return torch.rand((5, 10))


@pytest.fixture()
def feat_input_even() -> torch.Tensor:
    return torch.rand((8, 10))


@pytest.fixture()
def img_input_odd() -> torch.Tensor:
    return torch.rand((5, 10, 3, 3))


@pytest.fixture()
def img_input_even() -> torch.Tensor:
    return torch.rand((8, 10, 3, 3))


@pytest.fixture()
def cube_input_odd() -> torch.Tensor:
    return torch.rand((1, 10, 3, 3, 3))


@pytest.fixture()
def cube_input_even() -> torch.Tensor:
    return torch.rand((2, 10, 3, 3, 3))


class TestBayesLinear:
    """Testing the BayesLinear layer class."""

    def test_linear(self, feat_input_odd: torch.Tensor) -> None:
        layer = BayesLinear(10, 2, sigma_init=0)
        print(layer)
        out = layer(feat_input_odd)
        assert out.shape == torch.Size([5, 2])
        layer.sample()

        layer = BayesLinear(10, 2, sigma_init=0, bias=False)
        out = layer(feat_input_odd)
        assert out.shape == torch.Size([5, 2])
        layer.sample()

    def test_linear_even(self, feat_input_even: torch.Tensor) -> None:
        layer = BayesLinear(10, 2, sigma_init=0)
        out = layer(feat_input_even)
        assert out.shape == torch.Size([8, 2])

        layer.freeze()
        out = layer(feat_input_even)


class TestBayesConv1d:
    """Testing the BayesConv1d layer class."""

    def test_conv1(self, feat_input_odd: torch.Tensor) -> None:
        layer = BayesConv1d(5, 2, kernel_size=1, sigma_init=0)
        print(layer)
        out = layer(feat_input_odd)
        assert out.shape == torch.Size([2, 10])

        layer = BayesConv1d(5, 2, kernel_size=1, sigma_init=0, bias=False)
        out = layer(feat_input_odd)
        assert out.shape == torch.Size([2, 10])

    def test_conv1_even(self, feat_input_even: torch.Tensor) -> None:
        layer = BayesConv1d(
            8, 2, kernel_size=1, sigma_init=0, padding_mode="reflect"
        )
        print(layer)
        out = layer(feat_input_even)
        assert out.shape == torch.Size([2, 10])

        layer.freeze()
        out = layer(feat_input_even)

        layer.__setstate__({"padding_mode": "replicate"})

    def test_error(self):
        with pytest.raises(ValueError):
            BayesConv1d(
                8, 2, kernel_size=1, sigma_init=0, padding_mode="random"
            )


class TestBayesConv2d:
    """Testing the BayesConv2d layer class."""

    def test_conv2(self, img_input_odd: torch.Tensor) -> None:
        layer = BayesConv2d(10, 2, kernel_size=1, sigma_init=0)
        print(layer)
        out = layer(img_input_odd)
        assert out.shape == torch.Size([5, 2, 3, 3])
        layer.sample()

        layer = BayesConv2d(10, 2, kernel_size=1, sigma_init=0, bias=False)
        out = layer(img_input_odd)
        assert out.shape == torch.Size([5, 2, 3, 3])
        layer.sample()

    def test_conv2_even(self, img_input_even: torch.Tensor) -> None:
        layer = BayesConv2d(
            10, 2, kernel_size=1, sigma_init=0, padding_mode="reflect"
        )
        print(layer)
        out = layer(img_input_even)
        assert out.shape == torch.Size([8, 2, 3, 3])

        layer.freeze()
        out = layer(img_input_even)


class TestBayesConv3d:
    """Testing the BayesConv3d layer class."""

    def test_conv3(self, cube_input_odd: torch.Tensor) -> None:
        layer = BayesConv3d(10, 2, kernel_size=1, sigma_init=0)
        print(layer)
        out = layer(cube_input_odd)
        assert out.shape == torch.Size([1, 2, 3, 3, 3])

        layer = BayesConv3d(10, 2, kernel_size=1, sigma_init=0, bias=False)
        print(layer)
        out = layer(cube_input_odd)
        assert out.shape == torch.Size([1, 2, 3, 3, 3])

    def test_conv3_even(self, cube_input_even: torch.Tensor) -> None:
        layer = BayesConv3d(
            10, 2, kernel_size=1, sigma_init=0, padding_mode="reflect"
        )
        print(layer)
        out = layer(cube_input_even)
        assert out.shape == torch.Size([2, 2, 3, 3, 3])

        layer.freeze()
        out = layer(cube_input_even)


class TestTrainableDistribution:
    """Testing the TrainableDistribution class."""

    def test_log_posterior(self) -> None:
        sampler = TrainableDistribution(torch.ones(1), torch.ones(1))
        with pytest.raises(ValueError):
            sampler.log_posterior()


class TestLPBNNLinear:
    """Testing the LPBNNLinear layer class."""

    def test_linear(self, feat_input_odd: torch.Tensor) -> None:
        layer = LPBNNLinear(10, 2, num_estimators=4)
        print(layer)
        out = layer(feat_input_odd.repeat(4, 1))
        assert out.shape == torch.Size([5 * 4, 2])

        layer = LPBNNLinear(10, 2, num_estimators=4, bias=False)
        layer = layer.eval()
        out = layer(feat_input_odd.repeat(4, 1))
        assert out.shape == torch.Size([5 * 4, 2])

    def test_linear_even(self, feat_input_even: torch.Tensor) -> None:
        layer = LPBNNLinear(10, 2, num_estimators=4)
        out = layer(feat_input_even.repeat(4, 1))
        assert out.shape == torch.Size([8 * 4, 2])

        out = layer(feat_input_even)


class TestLPBNNConv2d:
    """Testing the LPBNNConv2d layer class."""

    def test_conv2(self, img_input_odd: torch.Tensor) -> None:
        layer = LPBNNConv2d(10, 2, kernel_size=1, num_estimators=4)
        print(layer)
        out = layer(img_input_odd.repeat(4, 1, 1, 1))
        assert out.shape == torch.Size([5 * 4, 2, 3, 3])

        layer = LPBNNConv2d(
            10, 2, kernel_size=1, num_estimators=4, bias=False, gamma=False
        )
        layer = layer.eval()
        out = layer(img_input_odd.repeat(4, 1, 1, 1))
        assert out.shape == torch.Size([5 * 4, 2, 3, 3])

    def test_conv2_even(self, img_input_even: torch.Tensor) -> None:
        layer = LPBNNConv2d(
            10, 2, kernel_size=1, num_estimators=4, padding_mode="reflect"
        )
        print(layer)
        out = layer(img_input_even.repeat(4, 1, 1, 1))
        assert out.shape == torch.Size([8 * 4, 2, 3, 3])

        out = layer(img_input_even)

    def test_errors(self) -> None:
        with pytest.raises(ValueError, match="std_factor must be"):
            LPBNNConv2d(10, 2, kernel_size=1, num_estimators=1, std_factor=-1)
        with pytest.raises(ValueError, match="num_estimators must be"):
            LPBNNConv2d(10, 2, kernel_size=1, num_estimators=-1)
        with pytest.raises(ValueError, match="hidden_size must be"):
            LPBNNConv2d(10, 2, kernel_size=1, num_estimators=1, hidden_size=-1)
