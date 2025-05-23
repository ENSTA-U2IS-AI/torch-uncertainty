from torch_uncertainty.models.mlp import bayesian_mlp, mlp, packed_mlp


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
