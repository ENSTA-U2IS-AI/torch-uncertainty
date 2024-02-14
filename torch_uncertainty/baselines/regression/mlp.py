from typing import Literal

from torch import nn

from torch_uncertainty.models.mlp import mlp, packed_mlp
from torch_uncertainty.routines.regression import (
    RegressionRoutine,
)


class MLP(RegressionRoutine):
    single = ["std"]
    ensemble = ["packed"]
    versions = {"std": mlp, "packed": packed_mlp}

    def __init__(
        self,
        num_outputs: int,
        in_features: int,
        loss: type[nn.Module],
        version: Literal["std", "packed"],
        hidden_dims: list[int],
        num_estimators: int | None = 1,
        alpha: float | None = None,
        gamma: int = 1,
        **kwargs,
    ) -> None:
        r"""MLP baseline for regression providing support for various versions."""
        params = {
            "in_features": in_features,
            "num_outputs": num_outputs,
            "hidden_dims": hidden_dims,
        }

        if version == "packed":
            params |= {
                "alpha": alpha,
                "num_estimators": num_estimators,
                "gamma": gamma,
            }

        if version not in self.versions:
            raise ValueError(f"Unknown version: {version}")

        model = self.versions[version](**params)

        # version in self.versions:
        super().__init__(
            model=model,
            loss=loss,
            num_estimators=num_estimators,
            dist_estimation=num_outputs,
            mode="mean",
        )
        self.save_hyperparameters()
