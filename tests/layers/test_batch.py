import pytest
import torch

from torch_uncertainty.layers.batch_ensemble import BatchConv2d, BatchConvTranspose2d, BatchLinear


@pytest.fixture
def feat_input() -> torch.Tensor:
    return torch.rand((4, 6))


@pytest.fixture
def img_input() -> torch.Tensor:
    return torch.rand((5, 6, 3, 3))


class TestBatchLinear:
    """Testing the BatchLinear layer class."""

    def test_linear_one_estimator(self, feat_input: torch.Tensor) -> None:
        layer = BatchLinear(6, 2, num_estimators=1)
        print(layer)  # noqa: T201
        out = layer(feat_input)
        assert out.shape == torch.Size([4, 2])

    def test_linear_two_estimators(self, feat_input: torch.Tensor) -> None:
        layer = BatchLinear(6, 2, num_estimators=2)
        out = layer(feat_input)
        assert out.shape == torch.Size([4, 2])

    def test_linear_one_estimator_no_bias(self, feat_input: torch.Tensor) -> None:
        layer = BatchLinear(6, 2, num_estimators=1, bias=False)
        out = layer(feat_input)
        assert out.shape == torch.Size([4, 2])

    def test_convert_from_linear(self, feat_input: torch.Tensor) -> None:
        linear = torch.nn.Linear(6, 3)
        layer = BatchLinear.from_linear(linear, num_estimators=2)
        assert layer.linear.weight.shape == torch.Size([3, 6])
        assert layer.linear.bias is None
        assert layer.r_group.shape == torch.Size([2, 6])
        assert layer.s_group.shape == torch.Size([2, 3])
        assert layer.bias.shape == torch.Size([2, 3])
        out = layer(feat_input)
        assert out.shape == torch.Size([4, 3])


class TestBatchConv2d:
    """Testing the BatchConv2d layer class."""

    def test_conv_one_estimator(self, img_input: torch.Tensor) -> None:
        layer = BatchConv2d(6, 2, num_estimators=1, kernel_size=1)
        print(layer)  # noqa: T201
        out = layer(img_input)
        assert out.shape == torch.Size([5, 2, 3, 3])

    def test_conv_two_estimators(self, img_input: torch.Tensor) -> None:
        layer = BatchConv2d(6, 2, num_estimators=2, kernel_size=1)
        out = layer(img_input)
        assert out.shape == torch.Size([5, 2, 3, 3])

    def test_convert_from_conv2d(self, img_input: torch.Tensor) -> None:
        conv = torch.nn.Conv2d(6, 3, 1)
        layer = BatchConv2d.from_conv2d(conv, num_estimators=2)
        assert layer.conv.weight.shape == torch.Size([3, 6, 1, 1])
        assert layer.conv.bias is None
        assert layer.r_group.shape == torch.Size([2, 6])
        assert layer.s_group.shape == torch.Size([2, 3])
        assert layer.bias.shape == torch.Size([2, 3])
        out = layer(img_input)
        assert out.shape == torch.Size([5, 3, 3, 3])


class TestBatchConvTranspose2d:
    """Testing the BatchConvTranspose2d layer class."""

    def test_conv_transpose_one_estimator(self, img_input: torch.Tensor) -> None:
        layer = BatchConvTranspose2d(6, 2, num_estimators=1, kernel_size=1, bias=False)
        print(layer)  # noqa: T201
        out = layer(img_input)
        assert out.shape == torch.Size([5, 2, 3, 3])

    def test_conv_transpose_two_estimators(self, img_input: torch.Tensor) -> None:
        layer = BatchConvTranspose2d(6, 2, num_estimators=2, kernel_size=1)
        out = layer(img_input)
        assert out.shape == torch.Size([5, 2, 3, 3])

    def test_convert_from_conv_transpose2d(self, img_input: torch.Tensor) -> None:
        conv = torch.nn.ConvTranspose2d(6, 3, 1)
        layer = BatchConvTranspose2d.from_conv_transpose2d(conv, num_estimators=2)
        assert layer.conv_transpose.weight.shape == torch.Size([6, 3, 1, 1])
        assert layer.conv_transpose.bias is None
        assert layer.r_group.shape == torch.Size([2, 6])
        assert layer.s_group.shape == torch.Size([2, 3])
        assert layer.bias.shape == torch.Size([2, 3])
        out = layer(img_input)
        assert out.shape == torch.Size([5, 3, 3, 3])
