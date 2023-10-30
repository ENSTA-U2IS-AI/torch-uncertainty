from torch_uncertainty.models.mlp import bayesian_mlp, packed_mlp


class TestMLPModel:
    """Testing the mlp models."""

    def test_packed(self):
        packed_mlp(1, 1, hidden_dims=[])

    def test_bayesian(self):
        bayesian_mlp(1, 1, hidden_dims=[1, 1, 1])
