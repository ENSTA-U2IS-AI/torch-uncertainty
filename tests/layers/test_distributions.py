import pytest
import torch

from torch_uncertainty.layers.distributions import (
    get_dist_linear_layer,
)


@pytest.fixture()
def feat_input() -> torch.Tensor:
    return torch.rand((3, 8))  # (B, Hin)


class TestNormalLinear:
    """Testing the NormalLinear layer class."""

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
