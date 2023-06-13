# fmt: off
from argparse import ArgumentParser
from typing import Any, Literal

import torch.nn as nn
from pytorch_lightning import LightningModule

from ...models.mlp import mlp
from ...routines.regression import RegressionSingle


# fmt: on
class MLP:
    single = ["vanilla"]
    versions = {"vanilla": mlp}

    def __new__(
        cls,
        num_outputs: int,
        in_features: int,
        loss: nn.Module,
        optimization_procedure: Any,
        version: Literal["vanilla"],
        **kwargs,
    ) -> LightningModule:
        params = {
            "in_features": in_features,
            "num_outputs": num_outputs,
        }

        if version not in cls.versions.keys():
            raise ValueError(f"Unknown version: {version}")

        model = cls.versions[version](**params)
        kwargs.update(params)
        # routine specific parameters
        if version in cls.single:
            return RegressionSingle(
                model=model,
                loss=loss,
                optimization_procedure=optimization_procedure,
                **kwargs,
            )

    @classmethod
    def add_model_specific_args(cls, parser: ArgumentParser) -> ArgumentParser:
        parser = RegressionSingle.add_model_specific_args(parser)
        parser.add_argument(
            "--version",
            type=str,
            choices=cls.versions.keys(),
            default="vanilla",
            help=f"Variation of ResNet. Choose among: {cls.versions.keys()}",
        )
        return parser
