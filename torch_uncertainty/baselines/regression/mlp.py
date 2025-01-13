from typing import Literal

from torch import nn

from torch_uncertainty.models.mlp import mlp, packed_mlp
from torch_uncertainty.routines.regression import (
    RegressionRoutine,
)
from torch_uncertainty.transforms.batch import RepeatTarget

ENSEMBLE_METHODS = ["packed"]


class MLPBaseline(RegressionRoutine):
    versions = {"std": mlp, "packed": packed_mlp}

    def __init__(
        self,
        output_dim: int,
        in_features: int,
        loss: nn.Module,
        version: Literal["std", "packed"],
        hidden_dims: list[int],
        num_estimators: int | None = 1,
        dropout_rate: float = 0.0,
        alpha: float | None = None,
        gamma: int = 1,
        dist_family: str | None = None,
        dist_args: dict | None = None,
    ) -> None:
        r"""MLP baseline for regression providing support for various versions."""
        params = {
            "dropout_rate": dropout_rate,
            "in_features": in_features,
            "num_outputs": output_dim,
            "hidden_dims": hidden_dims,
            "dist_family": dist_family,
            "dist_args": dist_args,
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

        super().__init__(
            output_dim=output_dim,
            model=model,
            loss=loss,
            dist_family=dist_family,
            is_ensemble=version in ENSEMBLE_METHODS,
            format_batch_fn=format_batch_fn,
        )
        self.save_hyperparameters(ignore=["loss"])
