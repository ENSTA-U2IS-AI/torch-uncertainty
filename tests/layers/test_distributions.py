import pytest
import torch

from torch_uncertainty.layers.distributions import (
    get_dist_conv_layer,
    get_dist_linear_layer,
)


@pytest.fixture()
def feat_input() -> torch.Tensor:
    return torch.rand((3, 8))  # (B, Hin)


def img_input() -> torch.Tensor:
    return torch.rand((3, 2, 32, 32))  # (B, C, H, W)


class TestDistributionLinear:
    """Testing the distribution linear layer classes."""

    def test_normal_linear(self, feat_input: torch.Tensor):
        dist_layer = get_dist_linear_layer("normal")
        layer = dist_layer(
            base_layer=torch.nn.Linear,
            event_dim=2,
            in_features=8,
        )
        out = layer(feat_input)
        assert out.keys() == {"loc", "scale"}
        assert out["loc"].shape == torch.Size([3, 2])
        assert out["scale"].shape == torch.Size([3, 2])

    def test_laplace_linear(self, feat_input: torch.Tensor):
        dist_layer = get_dist_linear_layer("laplace")
        layer = dist_layer(
            base_layer=torch.nn.Linear,
            event_dim=2,
            in_features=8,
        )
        out = layer(feat_input)
        assert out.keys() == {"loc", "scale"}
        assert out["loc"].shape == torch.Size([3, 2])
        assert out["scale"].shape == torch.Size([3, 2])

    def test_cauchy_linear(self, feat_input: torch.Tensor):
        dist_layer = get_dist_linear_layer("cauchy")
        layer = dist_layer(
            base_layer=torch.nn.Linear,
            event_dim=2,
            in_features=8,
        )
        out = layer(feat_input)
        assert out.keys() == {"loc", "scale"}
        assert out["loc"].shape == torch.Size([3, 2])
        assert out["scale"].shape == torch.Size([3, 2])

    def test_student_linear(self, feat_input: torch.Tensor):
        dist_layer = get_dist_linear_layer("student")
        layer = dist_layer(
            base_layer=torch.nn.Linear,
            event_dim=2,
            in_features=8,
        )
        out = layer(feat_input)
        assert out.keys() == {"loc", "scale", "df"}
        assert out["loc"].shape == torch.Size([3, 2])
        assert out["scale"].shape == torch.Size([3, 2])
        assert out["df"].shape == torch.Size([3, 2])

        layer = dist_layer(
            base_layer=torch.nn.Linear,
            event_dim=2,
            in_features=8,
            fixed_df=3.0,
        )
        out = layer(feat_input)
        assert out.keys() == {"loc", "scale", "df"}
        assert out["loc"].shape == torch.Size([3, 2])
        assert out["scale"].shape == torch.Size([3, 2])
        assert out["df"].shape == torch.Size([3, 2])
        assert torch.allclose(out["df"], torch.tensor(3.0))

    def test_nig_linear(self, feat_input: torch.Tensor):
        dist_layer = get_dist_linear_layer("nig")
        layer = dist_layer(
            base_layer=torch.nn.Linear,
            event_dim=2,
            in_features=8,
        )
        out = layer(feat_input)
        assert out.keys() == {"loc", "lmbda", "alpha", "beta"}
        assert out["loc"].shape == torch.Size([3, 2])
        assert out["lmbda"].shape == torch.Size([3, 2])
        assert out["alpha"].shape == torch.Size([3, 2])
        assert out["beta"].shape == torch.Size([3, 2])

    def test_failures(self):
        with pytest.raises(NotImplementedError):
            get_dist_linear_layer("unknown")

        with pytest.raises(ValueError):
            layer_class = get_dist_linear_layer("normal")
            layer_class(
                base_layer=torch.nn.Conv2d,
                event_dim=2,
                in_channels=5,
            )


class TestDistributionConv:
    """Testing the distribution convolutional layer classes."""

    def test_normal_conv(self):
        dist_layer = get_dist_conv_layer("normal")
        layer = dist_layer(
            base_layer=torch.nn.Conv2d,
            event_dim=2,
            in_channels=2,
            kernel_size=3,
        )
        out = layer(img_input())
        assert out.keys() == {"loc", "scale"}
        assert out["loc"].shape == torch.Size([3, 2, 30, 30])
        assert out["scale"].shape == torch.Size([3, 2, 30, 30])

    def test_laplace_conv(self):
        dist_layer = get_dist_conv_layer("laplace")
        layer = dist_layer(
            base_layer=torch.nn.Conv2d,
            event_dim=2,
            in_channels=2,
            kernel_size=3,
        )
        out = layer(img_input())
        assert out.keys() == {"loc", "scale"}
        assert out["loc"].shape == torch.Size([3, 2, 30, 30])
        assert out["scale"].shape == torch.Size([3, 2, 30, 30])

    def test_cauchy_conv(self):
        dist_layer = get_dist_conv_layer("cauchy")
        layer = dist_layer(
            base_layer=torch.nn.Conv2d,
            event_dim=2,
            in_channels=2,
            kernel_size=3,
        )
        out = layer(img_input())
        assert out.keys() == {"loc", "scale"}
        assert out["loc"].shape == torch.Size([3, 2, 30, 30])
        assert out["scale"].shape == torch.Size([3, 2, 30, 30])

    def test_student_conv(self):
        dist_layer = get_dist_conv_layer("student")
        layer = dist_layer(
            base_layer=torch.nn.Conv2d,
            event_dim=2,
            in_channels=2,
            kernel_size=3,
        )
        out = layer(img_input())
        assert out.keys() == {"loc", "scale", "df"}
        assert out["loc"].shape == torch.Size([3, 2, 30, 30])
        assert out["scale"].shape == torch.Size([3, 2, 30, 30])
        assert out["df"].shape == torch.Size([3, 2, 30, 30])

        layer = dist_layer(
            base_layer=torch.nn.Conv2d,
            event_dim=2,
            in_channels=2,
            kernel_size=3,
            fixed_df=3.0,
        )
        out = layer(img_input())
        assert out.keys() == {"loc", "scale", "df"}
        assert out["loc"].shape == torch.Size([3, 2, 30, 30])
        assert out["scale"].shape == torch.Size([3, 2, 30, 30])
        assert out["df"].shape == torch.Size([3, 2, 30, 30])
        assert torch.allclose(out["df"], torch.tensor(3.0))

    def test_nig_conv(self):
        dist_layer = get_dist_conv_layer("nig")
        layer = dist_layer(
            base_layer=torch.nn.Conv2d,
            event_dim=2,
            in_channels=2,
            kernel_size=3,
        )
        out = layer(img_input())
        assert out.keys() == {"loc", "lmbda", "alpha", "beta"}
        assert out["loc"].shape == torch.Size([3, 2, 30, 30])
        assert out["lmbda"].shape == torch.Size([3, 2, 30, 30])
        assert out["alpha"].shape == torch.Size([3, 2, 30, 30])
        assert out["beta"].shape == torch.Size([3, 2, 30, 30])

    def test_failures(self):
        with pytest.raises(NotImplementedError):
            get_dist_conv_layer("unknown")

        with pytest.raises(ValueError):
            layer_class = get_dist_conv_layer("normal")
            layer_class(
                base_layer=torch.nn.Linear,
                event_dim=2,
                in_features=5,
            )
