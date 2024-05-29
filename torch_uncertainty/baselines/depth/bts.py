from typing import Literal

from torch import nn

from torch_uncertainty.models.depth.bts import bts_resnet50, bts_resnet101
from torch_uncertainty.routines import PixelRegressionRoutine


class BTSBaseline(PixelRegressionRoutine):
    single = ["std"]
    versions = {
        "std": [
            bts_resnet50,
            bts_resnet101,
        ]
    }
    archs = [50, 101]

    def __init__(
        self,
        loss: nn.Module,
        version: Literal["std"],
        arch: int,
        max_depth: float,
        num_estimators: int = 1,
        pretrained_backbone: bool = True,
    ) -> None:
        params = {
            "dist_layer": nn.Identity,
            "max_depth": max_depth,
            "pretrained_backbone": pretrained_backbone,
        }

        format_batch_fn = nn.Identity()

        if version not in self.versions:
            raise ValueError(f"Unknown version {version}")

        model = self.versions[version][self.archs.index(arch)](**params)
        super().__init__(
            output_dim=1,
            probabilistic=False,
            model=model,
            loss=loss,
            num_estimators=num_estimators,
            format_batch_fn=format_batch_fn,
        )
        self.save_hyperparameters(ignore=["loss"])
