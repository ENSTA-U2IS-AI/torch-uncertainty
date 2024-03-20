from torch_uncertainty.layers.distributions import IndptNormalLayer
from torch_uncertainty.models.mlp import bayesian_mlp, mlp, packed_mlp


class TestMLPModel:
    """Testing the mlp models."""

    def test_mlps(self):
        mlp(
            1,
            1,
            hidden_dims=[1, 1, 1],
            final_layer=IndptNormalLayer,
            final_layer_args={"dim": 1},
        )
        mlp(1, 1, hidden_dims=[])
        packed_mlp(1, 1, hidden_dims=[])
        bayesian_mlp(1, 1, hidden_dims=[1, 1, 1])
