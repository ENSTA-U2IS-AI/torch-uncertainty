# fmt: off
from argparse import ArgumentParser, BooleanOptionalAction
from typing import Any, Literal, Optional

import torch.nn as nn
from pytorch_lightning import LightningModule

from torch_uncertainty.models.wideresnet import (
    batched_wideresnet28x10,
    masked_wideresnet28x10,
    packed_wideresnet28x10,
    wideresnet28x10,
)
from torch_uncertainty.routines.classification import (
    ClassificationEnsemble,
    ClassificationSingle,
)


# fmt: on
class WideResNet:
    single = ["vanilla"]
    ensemble = ["packed", "batched", "masked"]
    versions = {
        "vanilla": [wideresnet28x10],
        "packed": [packed_wideresnet28x10],
        "batched": [batched_wideresnet28x10],
        "masked": [masked_wideresnet28x10],
    }

    def __new__(
        cls,
        num_classes: int,
        in_channels: int,
        loss: nn.Module,
        optimization_procedure: Any,
        version: Literal["vanilla", "packed", "batched", "masked"],
        imagenet_structure: bool = True,
        num_estimators: Optional[int] = None,
        groups: Optional[int] = None,
        scale: Optional[float] = None,
        alpha: Optional[int] = None,
        gamma: Optional[int] = None,
        use_entropy: bool = False,
        use_logits: bool = False,
        use_mi: bool = False,
        use_variation_ratio: bool = False,
        # pretrained: bool = False,
        **kwargs,
    ) -> LightningModule:
        params = {
            "in_channels": in_channels,
            "num_classes": num_classes,
            "imagenet_structure": imagenet_structure,
            "groups": groups,
        }

        if version not in cls.versions.keys():
            raise ValueError(f"Unknown version: {version}")

        # version specific params
        if version == "packed":
            params.update(
                {
                    "num_estimators": num_estimators,
                    "alpha": alpha,
                    "gamma": gamma,
                }
            )
        elif version == "batched":
            params.update(
                {
                    "num_estimators": num_estimators,
                }
            )
        elif version == "masked":
            params.update(
                {
                    "num_estimators": num_estimators,
                    "scale": scale,
                }
            )

        model = cls.versions[version][0](**params)
        kwargs.update(params)
        # routine specific parameters
        if version in cls.single:
            return ClassificationSingle(
                model=model,
                loss=loss,
                optimization_procedure=optimization_procedure,
                use_entropy=use_entropy,
                use_logits=use_logits,
                **kwargs,
            )
        elif version in cls.ensemble:
            return ClassificationEnsemble(
                model=model,
                loss=loss,
                optimization_procedure=optimization_procedure,
                use_entropy=use_entropy,
                use_logits=use_logits,
                use_mi=use_mi,
                use_variation_ratio=use_variation_ratio,
                **kwargs,
            )

    @classmethod
    def add_model_specific_args(cls, parser: ArgumentParser) -> ArgumentParser:
        parser = ClassificationEnsemble.add_model_specific_args(parser)
        parser.add_argument(
            "--version",
            type=str,
            choices=cls.versions.keys(),
            default="vanilla",
            help="Variation of WideResNet. "
            + f"Choose among: {cls.versions.keys()}",
        )
        parser.add_argument(
            "--groups",
            type=int,
            default=1,
            help="Number of groups for vanilla or masked wideresnet",
        )
        parser.add_argument(
            "--scale",
            type=float,
            default=None,
            help="Scale for masked wideresnet",
        )
        parser.add_argument(
            "--alpha",
            type=int,
            default=None,
            help="Alpha for packed wideresnet",
        )
        parser.add_argument(
            "--gamma",
            type=int,
            default=None,
            help="Gamma for packed wideresnet",
        )
        parser.add_argument(
            "--pretrained",
            dest="pretrained",
            action=BooleanOptionalAction,
            default=False,
        )

        return parser
