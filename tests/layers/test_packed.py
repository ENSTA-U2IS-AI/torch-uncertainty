import pytest
import torch

from torch_uncertainty.layers.packed import (
    PackedConv1d,
    PackedConv2d,
    PackedConv3d,
    PackedLinear,
)


@pytest.fixture()
def feat_input() -> torch.Tensor:
    return torch.rand((6, 1))  # (Cin, Lin)


@pytest.fixture()
def feat_input_one_rearrange() -> torch.Tensor:
    return torch.rand((1 * 3, 5))


@pytest.fixture()
def feat_input_16_features() -> torch.Tensor:
    return torch.rand((2, 16))


@pytest.fixture()
def seq_input() -> torch.Tensor:
    return torch.rand((5, 6, 3))


@pytest.fixture()
def img_input() -> torch.Tensor:
    return torch.rand((5, 6, 3, 3))


@pytest.fixture()
def voxels_input() -> torch.Tensor:
    return torch.rand((5, 6, 3, 3, 3))


class TestPackedLinear:
    """Testing the PackedLinear layer class."""

    # Legacy tests
    def test_linear_one_estimator_no_rearrange(self, feat_input: torch.Tensor):
        layer = PackedLinear(
            6, 2, alpha=1, num_estimators=1, rearrange=False, bias=False
        )
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

    def test_linear_full_implementation(
        self, feat_input_16_features: torch.Tensor
    ):
        layer = PackedLinear(
            16, 4, alpha=1, num_estimators=1, implementation="full"
        )
        out = layer(feat_input_16_features)
        assert out.shape == torch.Size([2, 4])
        layer = PackedLinear(
            16, 4, alpha=1, num_estimators=2, implementation="full"
        )
        out = layer(feat_input_16_features)
        assert out.shape == torch.Size([2, 4])

    def test_linear_sparse_implementation(
        self, feat_input_16_features: torch.Tensor
    ):
        layer = PackedLinear(
            16, 4, alpha=1, num_estimators=1, implementation="sparse"
        )
        out = layer(feat_input_16_features)
        assert out.shape == torch.Size([2, 4])
        layer = PackedLinear(
            16, 4, alpha=1, num_estimators=2, implementation="sparse"
        )
        out = layer(feat_input_16_features)
        assert out.shape == torch.Size([2, 4])

    def test_linear_einsum_implementation(
        self, feat_input_16_features: torch.Tensor
    ):
        layer = PackedLinear(
            16, 4, alpha=1, num_estimators=1, implementation="einsum"
        )
        out = layer(feat_input_16_features)
        assert out.shape == torch.Size([2, 4])
        layer = PackedLinear(
            16, 4, alpha=1, num_estimators=2, implementation="einsum"
        )
        out = layer(feat_input_16_features)
        assert out.shape == torch.Size([2, 4])

    def test_linear_extend(self):
        _ = PackedConv2d(
            5, 3, kernel_size=1, alpha=1, num_estimators=2, gamma=1
        )

    def test_linear_failures(self):
        with pytest.raises(ValueError):
            _ = PackedLinear(5, 2, alpha=None, num_estimators=1, rearrange=True)

        with pytest.raises(ValueError):
            _ = PackedLinear(5, 2, alpha=-1, num_estimators=1, rearrange=True)

        with pytest.raises(ValueError):
            _ = PackedLinear(5, 2, alpha=1, num_estimators=None, rearrange=True)

        with pytest.raises(TypeError):
            _ = PackedLinear(5, 2, alpha=1, num_estimators=1.5, rearrange=True)

        with pytest.raises(ValueError):
            _ = PackedLinear(5, 2, alpha=1, num_estimators=-1, rearrange=True)

        with pytest.raises(TypeError):
            _ = PackedLinear(
                5, 2, alpha=1, num_estimators=1, gamma=0.5, rearrange=True
            )

        with pytest.raises(ValueError):
            _ = PackedLinear(
                5, 2, alpha=1, num_estimators=1, gamma=-1, rearrange=True
            )

        with pytest.raises(AssertionError):
            _ = PackedLinear(
                5,
                2,
                alpha=1,
                num_estimators=1,
                gamma=1,
                implementation="invalid",
            )

        with pytest.raises(ValueError):
            layer = PackedLinear(
                16, 4, alpha=1, num_estimators=1, implementation="full"
            )
            layer.implementation = "invalid"
            _ = layer(torch.rand((2, 16)))


class TestPackedConv1d:
    """Testing the PackedConv1d layer class."""

    def test_conv_one_estimator(self, seq_input: torch.Tensor):
        layer = PackedConv1d(6, 2, alpha=1, num_estimators=1, kernel_size=1)
        out = layer(seq_input)
        assert out.shape == torch.Size([5, 2, 3])
        assert layer.weight.shape == torch.Size([2, 6, 1])
        assert layer.bias.shape == torch.Size([2])

    def test_conv_two_estimators(self, seq_input: torch.Tensor):
        layer = PackedConv1d(6, 2, alpha=1, num_estimators=2, kernel_size=1)
        out = layer(seq_input)
        assert out.shape == torch.Size([5, 2, 3])

    def test_conv_one_estimator_gamma2(self, seq_input: torch.Tensor):
        layer = PackedConv1d(
            6, 2, alpha=1, num_estimators=1, kernel_size=1, gamma=2
        )
        out = layer(seq_input)
        assert out.shape == torch.Size([5, 2, 3])
        assert layer.conv.groups == 1  # and not 2

    def test_conv_two_estimators_gamma2(self, seq_input: torch.Tensor):
        layer = PackedConv1d(
            6, 2, alpha=1, num_estimators=2, kernel_size=1, gamma=2
        )
        out = layer(seq_input)
        assert out.shape == torch.Size([5, 2, 3])
        assert layer.conv.groups == 2  # and not 4

    def test_conv_extend(self):
        _ = PackedConv1d(
            5, 3, kernel_size=1, alpha=1, num_estimators=2, gamma=1
        )

    def test_conv1_failures(self):
        with pytest.raises(ValueError):
            _ = PackedConv1d(5, 2, kernel_size=1, alpha=-1, num_estimators=1)

        with pytest.raises(TypeError):
            _ = PackedConv1d(
                5, 2, kernel_size=1, alpha=1, num_estimators=1, gamma=0.5
            )

        with pytest.raises(ValueError):
            _ = PackedConv1d(
                5, 2, kernel_size=1, alpha=1, num_estimators=1, gamma=-1
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

    def test_conv2_failures(self):
        with pytest.raises(ValueError):
            _ = PackedConv2d(5, 2, kernel_size=1, alpha=-1, num_estimators=1)

        with pytest.raises(TypeError):
            _ = PackedConv2d(
                5, 2, kernel_size=1, alpha=1, num_estimators=1, gamma=0.5
            )

        with pytest.raises(ValueError):
            _ = PackedConv2d(
                5, 2, kernel_size=1, alpha=1, num_estimators=1, gamma=-1
            )


class TestPackedConv3d:
    """Testing the PackedConv3d layer class."""

    def test_conv_one_estimator(self, voxels_input: torch.Tensor):
        layer = PackedConv3d(6, 2, alpha=1, num_estimators=1, kernel_size=1)
        out = layer(voxels_input)
        assert out.shape == torch.Size([5, 2, 3, 3, 3])
        assert layer.weight.shape == torch.Size([2, 6, 1, 1, 1])
        assert layer.bias.shape == torch.Size([2])

    def test_conv_two_estimators(self, voxels_input: torch.Tensor):
        layer = PackedConv3d(6, 2, alpha=1, num_estimators=2, kernel_size=1)
        out = layer(voxels_input)
        assert out.shape == torch.Size([5, 2, 3, 3, 3])

    def test_conv_one_estimator_gamma2(self, voxels_input: torch.Tensor):
        layer = PackedConv3d(
            6, 2, alpha=1, num_estimators=1, kernel_size=1, gamma=2
        )
        out = layer(voxels_input)
        assert out.shape == torch.Size([5, 2, 3, 3, 3])
        assert layer.conv.groups == 1  # and not 2

    def test_conv_two_estimators_gamma2(self, voxels_input: torch.Tensor):
        layer = PackedConv3d(
            6, 2, alpha=1, num_estimators=2, kernel_size=1, gamma=2
        )
        out = layer(voxels_input)
        assert out.shape == torch.Size([5, 2, 3, 3, 3])
        assert layer.conv.groups == 2  # and not 4

    def test_conv_extend(self):
        _ = PackedConv3d(
            5, 3, kernel_size=1, alpha=1, num_estimators=2, gamma=1
        )

    def test_conv3_failures(self):
        with pytest.raises(ValueError):
            _ = PackedConv3d(5, 2, kernel_size=1, alpha=-1, num_estimators=1)

        with pytest.raises(TypeError):
            _ = PackedConv3d(
                5, 2, kernel_size=1, alpha=1, num_estimators=1, gamma=0.5
            )

        with pytest.raises(ValueError):
            _ = PackedConv3d(
                5, 2, kernel_size=1, alpha=1, num_estimators=1, gamma=-1
            )
