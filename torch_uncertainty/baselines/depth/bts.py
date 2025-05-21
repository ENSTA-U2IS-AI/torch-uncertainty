from typing import Literal

from torch import nn

from torch_uncertainty.models.depth.bts import bts_resnet
from torch_uncertainty.routines import PixelRegressionRoutine


class BTSBaseline(PixelRegressionRoutine):
    archs = [50, 101]

    def __init__(
        self,
        loss: nn.Module,
        version: Literal["std"],
        arch: int,
        max_depth: float,
        dist_family: str | None = None,
        pretrained_backbone: bool = True,
    ) -> None:
        params = {
            "arch": arch,
            "dist_family": dist_family,
            "max_depth": max_depth,
            "pretrained_backbone": pretrained_backbone,
        }

        format_batch_fn = nn.Identity()

        if version not in self.versions:
            raise ValueError(f"Unknown version {version}")

        model = bts_resnet(**params)
        super().__init__(
            model=model,
            output_dim=1,
            loss=loss,
            format_batch_fn=format_batch_fn,
            dist_family=dist_family,
        )
        self.save_hyperparameters(ignore=["loss"])
