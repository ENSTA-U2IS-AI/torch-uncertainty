import pytest
import torch
from einops import repeat

from torch_uncertainty.layers.packed import (
    PackedConv1d,
    PackedConv2d,
    PackedConv3d,
    PackedLayerNorm,
    PackedLinear,
    PackedMultiheadAttention,
)


@pytest.fixture()
def feat_input() -> torch.Tensor:
    return torch.rand((6, 1))  # (Cin, Lin)


@pytest.fixture()
def feat_input_one_rearrange() -> torch.Tensor:
    return torch.rand((1 * 3, 5))


@pytest.fixture()
def feat_multi_dim() -> torch.Tensor:
    return torch.rand((1, 2, 3, 4, 6))


@pytest.fixture()
def feat_input_16_features() -> torch.Tensor:
    return torch.rand((3, 16))


@pytest.fixture()
def seq_input() -> torch.Tensor:
    return torch.rand((5, 6, 3))


@pytest.fixture()
def img_input() -> torch.Tensor:
    return torch.rand((5, 6, 3, 3))


@pytest.fixture()
def voxels_input() -> torch.Tensor:
    return torch.rand((5, 6, 3, 3, 3))


@pytest.fixture()
def unbatched_qkv() -> torch.Tensor:
    return torch.rand((3, 6))


@pytest.fixture()
def unbatched_q_kv() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.rand((3, 6)), torch.rand((4, 2))


@pytest.fixture()
def unbatched_q_k_v() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.rand((3, 6)), torch.rand((4, 2)), torch.rand((4, 4))


@pytest.fixture()
def batched_qkv() -> torch.Tensor:
    return torch.rand((2, 3, 6))


@pytest.fixture()
def batched_q_kv() -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.rand((2, 3, 6)),
        torch.rand((2, 4, 2)),
    )


@pytest.fixture()
def batched_q_k_v() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.rand((2, 3, 6)),
        torch.rand((2, 4, 2)),
        torch.rand((2, 4, 4)),
    )


class TestPackedLinear:
    """Testing the PackedLinear layer class."""

    # Legacy tests
    def test_linear_one_estimator_no_rearrange(self, feat_input: torch.Tensor):
        layer = PackedLinear(6, 2, alpha=1, num_estimators=1, rearrange=False, bias=False)
        out = layer(feat_input)
        assert out.shape == torch.Size([2, 1])

    def test_linear_two_estimators_no_rearrange(self, feat_input: torch.Tensor):
        layer = PackedLinear(6, 2, alpha=1, num_estimators=2, rearrange=False)
        out = layer(feat_input)
        assert out.shape == torch.Size([2, 1])

    def test_linear_one_estimator_rearrange(self, feat_input_one_rearrange: torch.Tensor):
        layer = PackedLinear(5, 2, alpha=1, num_estimators=1, rearrange=True)
        out = layer(feat_input_one_rearrange)
        assert out.shape == torch.Size([3, 2])

    def test_linear_two_estimator_rearrange_not_divisible(self):
        feat = torch.rand((2 * 3, 3))
        layer = PackedLinear(5, 1, alpha=1, num_estimators=2, rearrange=True)
        out = layer(feat)
        assert out.shape == torch.Size([6, 1])

    # Full implementation tests
    def test_linear_full_implementation(
        self, feat_input_16_features: torch.Tensor, feat_multi_dim: torch.Tensor
    ):
        layer = PackedLinear(16, 4, alpha=1, num_estimators=1, implementation="full")
        out = layer(feat_input_16_features)
        assert out.shape == torch.Size([3, 4])
        layer = PackedLinear(16, 4, alpha=1, num_estimators=2, implementation="full")
        out = layer(feat_input_16_features)
        assert out.shape == torch.Size([3, 4])
        layer = PackedLinear(6, 2, alpha=1, num_estimators=1, implementation="full")
        out = layer(feat_multi_dim)
        assert out.shape == torch.Size([1, 2, 3, 4, 2])

    # Sparse implementation tests
    def test_linear_sparse_implementation(
        self, feat_input_16_features: torch.Tensor, feat_multi_dim: torch.Tensor
    ):
        layer = PackedLinear(16, 4, alpha=1, num_estimators=1, implementation="sparse")
        out = layer(feat_input_16_features)
        assert out.shape == torch.Size([3, 4])
        layer = PackedLinear(16, 4, alpha=1, num_estimators=2, implementation="sparse")
        out = layer(feat_input_16_features)
        assert out.shape == torch.Size([3, 4])
        layer = PackedLinear(6, 2, alpha=1, num_estimators=1, implementation="sparse")
        out = layer(feat_multi_dim)
        assert out.shape == torch.Size([1, 2, 3, 4, 2])

    # Einsum implementation tests
    def test_linear_einsum_implementation(
        self, feat_input_16_features: torch.Tensor, feat_multi_dim: torch.Tensor
    ):
        layer = PackedLinear(16, 4, alpha=1, num_estimators=1, implementation="einsum")
        out = layer(feat_input_16_features)
        assert out.shape == torch.Size([3, 4])
        layer = PackedLinear(16, 4, alpha=1, num_estimators=2, implementation="einsum")
        out = layer(feat_input_16_features)
        assert out.shape == torch.Size([3, 4])
        layer = PackedLinear(6, 2, alpha=1, num_estimators=1, implementation="einsum")
        out = layer(feat_multi_dim)
        assert out.shape == torch.Size([1, 2, 3, 4, 2])

    def test_linear_extend(self):
        _ = PackedConv2d(5, 3, kernel_size=1, alpha=1, num_estimators=2, gamma=1)

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
            _ = PackedLinear(5, 2, alpha=1, num_estimators=1, gamma=0.5, rearrange=True)

        with pytest.raises(ValueError):
            _ = PackedLinear(5, 2, alpha=1, num_estimators=1, gamma=-1, rearrange=True)

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
            layer = PackedLinear(16, 4, alpha=1, num_estimators=1, implementation="full")
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
        layer = PackedConv1d(6, 2, alpha=1, num_estimators=1, kernel_size=1, gamma=2)
        out = layer(seq_input)
        assert out.shape == torch.Size([5, 2, 3])
        assert layer.conv.groups == 1  # and not 2

    def test_conv_two_estimators_gamma2(self, seq_input: torch.Tensor):
        layer = PackedConv1d(6, 2, alpha=1, num_estimators=2, kernel_size=1, gamma=2)
        out = layer(seq_input)
        assert out.shape == torch.Size([5, 2, 3])
        assert layer.conv.groups == 2  # and not 4

    def test_conv_extend(self):
        _ = PackedConv1d(5, 3, kernel_size=1, alpha=1, num_estimators=2, gamma=1)

    def test_conv1_failures(self):
        with pytest.raises(ValueError):
            _ = PackedConv1d(5, 2, kernel_size=1, alpha=-1, num_estimators=1)

        with pytest.raises(TypeError):
            _ = PackedConv1d(5, 2, kernel_size=1, alpha=1, num_estimators=1, gamma=0.5)

        with pytest.raises(ValueError):
            _ = PackedConv1d(5, 2, kernel_size=1, alpha=1, num_estimators=1, gamma=-1)


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
        layer = PackedConv2d(6, 2, alpha=1, num_estimators=1, kernel_size=1, gamma=2)
        out = layer(img_input)
        assert out.shape == torch.Size([5, 2, 3, 3])
        assert layer.conv.groups == 1  # and not 2

    def test_conv_two_estimators_gamma2(self, img_input: torch.Tensor):
        layer = PackedConv2d(6, 2, alpha=1, num_estimators=2, kernel_size=1, gamma=2)
        out = layer(img_input)
        assert out.shape == torch.Size([5, 2, 3, 3])
        assert layer.conv.groups == 2  # and not 4

    def test_conv_extend(self):
        _ = PackedConv2d(5, 3, kernel_size=1, alpha=1, num_estimators=2, gamma=1)

    def test_conv2_failures(self):
        with pytest.raises(ValueError):
            _ = PackedConv2d(5, 2, kernel_size=1, alpha=-1, num_estimators=1)

        with pytest.raises(TypeError):
            _ = PackedConv2d(5, 2, kernel_size=1, alpha=1, num_estimators=1, gamma=0.5)

        with pytest.raises(ValueError):
            _ = PackedConv2d(5, 2, kernel_size=1, alpha=1, num_estimators=1, gamma=-1)


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
        layer = PackedConv3d(6, 2, alpha=1, num_estimators=1, kernel_size=1, gamma=2)
        out = layer(voxels_input)
        assert out.shape == torch.Size([5, 2, 3, 3, 3])
        assert layer.conv.groups == 1  # and not 2

    def test_conv_two_estimators_gamma2(self, voxels_input: torch.Tensor):
        layer = PackedConv3d(6, 2, alpha=1, num_estimators=2, kernel_size=1, gamma=2)
        out = layer(voxels_input)
        assert out.shape == torch.Size([5, 2, 3, 3, 3])
        assert layer.conv.groups == 2  # and not 4

    def test_conv_extend(self):
        _ = PackedConv3d(5, 3, kernel_size=1, alpha=1, num_estimators=2, gamma=1)

    def test_conv3_failures(self):
        with pytest.raises(ValueError):
            _ = PackedConv3d(5, 2, kernel_size=1, alpha=-1, num_estimators=1)

        with pytest.raises(TypeError):
            _ = PackedConv3d(5, 2, kernel_size=1, alpha=1, num_estimators=1, gamma=0.5)

        with pytest.raises(ValueError):
            _ = PackedConv3d(5, 2, kernel_size=1, alpha=1, num_estimators=1, gamma=-1)


class TestPackedLayerNorm:
    """Testing the PackedGroupNorm layer class."""

    def test_one_estimator_forward(self, batched_qkv: torch.Tensor):
        packed_layer_norm = PackedLayerNorm(
            embed_dim=6,
            num_estimators=1,
            alpha=1,
        )
        out = packed_layer_norm(batched_qkv)
        assert out.shape == torch.Size([2, 3, 6])


class TestPackedMultiheadAttention:
    """Testing the PackedMultiheadAttention layer class."""

    def test_one_estimator_qkv(self, unbatched_qkv: torch.Tensor, batched_qkv: torch.Tensor):
        layer = PackedMultiheadAttention(
            embed_dim=6,
            num_heads=2,
            alpha=1,
            num_estimators=1,
        )
        out, _ = layer(
            query=unbatched_qkv,
            key=unbatched_qkv,
            value=unbatched_qkv,
        )
        assert out.shape == torch.Size([3, 6])

        unbatched_qkv = repeat(unbatched_qkv, "l h -> l b h", b=2)
        out, _ = layer(
            query=unbatched_qkv,
            key=unbatched_qkv,
            value=unbatched_qkv,
        )
        assert out.shape == torch.Size([3, 2, 6])

        layer = PackedMultiheadAttention(
            embed_dim=6,
            num_heads=2,
            alpha=1,
            num_estimators=1,
            batch_first=True,
            bias=False,
        )
        out, _ = layer(
            query=batched_qkv,
            key=batched_qkv,
            value=batched_qkv,
        )
        assert out.shape == torch.Size([2, 3, 6])

    def test_one_estimator_q_kv(
        self,
        unbatched_q_kv: tuple[torch.Tensor, torch.Tensor],
        batched_q_kv: tuple[torch.Tensor, torch.Tensor],
    ):
        layer = PackedMultiheadAttention(
            embed_dim=6,
            num_heads=2,
            alpha=1,
            num_estimators=1,
            kdim=2,
            vdim=2,
            add_zero_attn=True,
        )
        out, _ = layer(
            query=unbatched_q_kv[0],
            key=unbatched_q_kv[1],
            value=unbatched_q_kv[1],
        )
        assert out.shape == torch.Size([3, 6])
        unbatched_q_kv = tuple(repeat(seq, "l h -> l b h", b=2) for seq in unbatched_q_kv)
        out, _ = layer(
            query=unbatched_q_kv[0],
            key=unbatched_q_kv[1],
            value=unbatched_q_kv[1],
        )
        assert out.shape == torch.Size([3, 2, 6])

        layer = PackedMultiheadAttention(
            embed_dim=6,
            num_heads=2,
            alpha=1,
            num_estimators=1,
            kdim=2,
            vdim=2,
            batch_first=True,
        )
        out, _ = layer(
            query=batched_q_kv[0],
            key=batched_q_kv[1],
            value=batched_q_kv[1],
        )
        assert out.shape == torch.Size([2, 3, 6])

    def test_one_estimator_q_k_v(
        self,
        unbatched_q_k_v: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batched_q_k_v: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ):
        layer = PackedMultiheadAttention(
            embed_dim=6,
            num_heads=2,
            alpha=1,
            num_estimators=1,
            kdim=2,
            vdim=4,
            add_bias_kv=True,
        )
        out, _ = layer(
            query=unbatched_q_k_v[0],
            key=unbatched_q_k_v[1],
            value=unbatched_q_k_v[2],
        )
        assert out.shape == torch.Size([3, 6])

        unbatched_q_k_v = tuple(repeat(seq, "l h -> l b h", b=2) for seq in unbatched_q_k_v)

        out, _ = layer(
            query=unbatched_q_k_v[0],
            key=unbatched_q_k_v[1],
            value=unbatched_q_k_v[2],
        )
        assert out.shape == torch.Size([3, 2, 6])

        layer = PackedMultiheadAttention(
            embed_dim=6,
            num_heads=2,
            alpha=1,
            num_estimators=1,
            kdim=2,
            vdim=4,
            batch_first=True,
        )

        layer.eval()

        attn_mask = torch.zeros(3, 4, dtype=torch.bool)
        key_padding_mask = torch.zeros(2, 4, dtype=torch.bool)

        out, _ = layer(
            query=batched_q_k_v[0],
            key=batched_q_k_v[1],
            value=batched_q_k_v[2],
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        assert out.shape == torch.Size([2, 3, 6])
        assert out.isfinite().all()


class TestPackedTransformerEncoderLayer:
    """Testing the PackedTransformerEncoderLayer class."""


class TestPackedTransformerDecoderLayer:
    """Testing the PackedTransformerDecoderLayer class."""
