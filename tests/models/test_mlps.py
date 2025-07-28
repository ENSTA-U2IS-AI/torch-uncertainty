import torch

from torch_uncertainty.models.mlp import batched_mlp, bayesian_mlp, mimo_mlp, mlp, packed_mlp


class TestMLPModel:
    """Testing the mlp models."""

    def test_mlps(self) -> None:
        mlp(
            1,
            1,
            hidden_dims=[1, 1, 1],
            dist_family="normal",
        )
        mlp(1, 1, hidden_dims=[])
        packed_mlp(1, 1, hidden_dims=[])
        bayesian_mlp(1, 1, hidden_dims=[1, 1, 1])

    def test_batched_mlp(self) -> None:
        batched_mlp(
            1,
            1,
            hidden_dims=[1, 1, 1],
            dist_family="normal",
        )
        model = batched_mlp(1, 1, hidden_dims=[1, 1], num_estimators=4)
        out = model(torch.randn(10, 1))
        assert out.shape == (40, 1)

    def test_mimo_mlp(self) -> None:
        model = mimo_mlp(1, 1, hidden_dims=[1, 1], num_estimators=4)
        model.train()
        out = model(torch.randn(12, 1))
        assert out.shape == (12, 1)

        model = mimo_mlp(
            1,
            1,
            hidden_dims=[1, 1, 1],
            num_estimators=2,
            dist_family="normal",
        )
        model.eval()
        out = model(torch.randn(10, 1))
        assert isinstance(out, dict)
        assert out["loc"].shape == (20, 1)
        assert out["scale"].shape == (20, 1)
