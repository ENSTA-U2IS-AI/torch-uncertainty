from typing import Literal

from torch import nn

from torch_uncertainty.models.mlp import mlp, packed_mlp
from torch_uncertainty.routines.regression import (
    RegressionRoutine,
)
from torch_uncertainty.transforms.batch import RepeatTarget


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
        dropout_rate: float = 0.0,
        alpha: float | None = None,
        gamma: int = 1,
    ) -> None:
        r"""MLP baseline for regression providing support for various versions."""
        params = {
            "dropout_rate": dropout_rate,
            "in_features": in_features,
            "num_outputs": num_outputs,
            "hidden_dims": hidden_dims,
        }

        format_batch_fn = nn.Identity()

        if version not in self.versions:
            raise ValueError(f"Unknown version: {version}")

        if version == "packed":
            params |= {
                "alpha": alpha,
                "num_estimators": num_estimators,
                "gamma": gamma,
            }
            format_batch_fn = RepeatTarget(num_repeats=num_estimators)

        model = self.versions[version](**params)

        # version in self.versions:
        super().__init__(
            num_outputs=num_outputs,
            model=model,
            loss=loss,
            num_estimators=num_estimators,
            format_batch_fn=format_batch_fn,
        )
        self.save_hyperparameters()
