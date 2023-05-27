# fmt: off
from argparse import ArgumentParser, BooleanOptionalAction
from typing import Any, Literal, Optional

import torch.nn as nn
from pytorch_lightning import LightningModule

from torch_uncertainty.models.resnet import (
    batched_resnet18,
    batched_resnet34,
    batched_resnet50,
    batched_resnet101,
    batched_resnet152,
    masked_resnet18,
    masked_resnet34,
    masked_resnet50,
    masked_resnet101,
    masked_resnet152,
    packed_resnet18,
    packed_resnet34,
    packed_resnet50,
    packed_resnet101,
    packed_resnet152,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)
from torch_uncertainty.routines.classification import (
    ClassificationEnsemble,
    ClassificationSingle,
)


# fmt: on
class ResNet:
    single = ["vanilla"]
    ensemble = ["packed", "batched", "masked"]
    versions = {
        "vanilla": [resnet18, resnet34, resnet50, resnet101, resnet152],
        "packed": [
            packed_resnet18,
            packed_resnet34,
            packed_resnet50,
            packed_resnet101,
            packed_resnet152,
        ],
        "batched": [
            batched_resnet18,
            batched_resnet34,
            batched_resnet50,
            batched_resnet101,
            batched_resnet152,
        ],
        "masked": [
            masked_resnet18,
            masked_resnet34,
            masked_resnet50,
            masked_resnet101,
            masked_resnet152,
        ],
    }
    archs = [18, 34, 50, 101, 152]

    def __new__(
        cls,
        num_classes: int,
        in_channels: int,
        loss: nn.Module,
        optimization_procedure: Any,
        version: Literal["vanilla", "packed", "batched", "masked"],
        arch: int,
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
        **kwargs,
    ) -> LightningModule:
        params = {
            "in_channels": in_channels,
            "num_classes": num_classes,
            "imagenet_structure": imagenet_structure,
        }
        # version specific parameters
        if version == "vanilla":
            # TODO: check parameters
            params.update({"groups": groups})
        elif version == "packed":
            # TODO: check parameters
            params.update(
                {
                    "num_estimators": num_estimators,
                    "alpha": alpha,
                    "gamma": gamma,
                }
            )
        elif version == "batched":
            # TODO: check parameters
            params.update({"num_estimators": num_estimators})
        elif version == "masked":
            # TODO: check parameters
            params.update(
                {
                    "num_estimators": num_estimators,
                    "scale": scale,
                    "groups": groups,
                }
            )
        else:
            raise ValueError(f"Unknown version: {version}")

        model = cls.versions[version][cls.archs.index(arch)](**params)
        kwargs.update(params)
        print(kwargs)
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
        parser.add_argument(
            "--version",
            type=str,
            choices=cls.versions.keys(),
            required=True,
            help=f"Variation of ResNet. Choose among: {cls.versions.keys()}",
        )
        parser.add_argument(
            "--arch",
            type=int,
            choices=cls.archs,
            required=True,
            help=f"Architecture of ResNet. Choose among: {cls.archs}",
        )
        # parser.add_argument(
        #     "--imagenet_structure",
        #     action=BooleanOptionalAction,
        #     default=True,
        #     help="Use imagenet structure",
        # )
        parser.add_argument(
            "--num_estimators",
            type=int,
            default=None,
            help="Number of estimators for ensemble",
        )
        parser.add_argument(
            "--groups",
            type=int,
            default=1,
            help="Number of groups for vanilla or masked resnet",
        )
        parser.add_argument(
            "--scale",
            type=float,
            default=None,
            help="Scale for masked resnet",
        )
        parser.add_argument(
            "--alpha",
            type=int,
            default=None,
            help="Alpha for packed resnet",
        )
        parser.add_argument(
            "--gamma",
            type=int,
            default=None,
            help="Gamma for packed resnet",
        )
        # FIXME: should be a str to choose among the available OOD criteria
        # rather than a boolean, but it is not possible since
        # ClassificationSingle and ClassificationEnsemble have different OOD
        # criteria.
        parser.add_argument(
            "--entropy",
            dest="use_entropy",
            action=BooleanOptionalAction,
            default=False,
        )
        parser.add_argument(
            "--logits",
            dest="use_logits",
            action=BooleanOptionalAction,
            default=False,
        )
        parser.add_argument(
            "--mutual_information",
            dest="use_mi",
            action=BooleanOptionalAction,
            default=False,
        )
        parser.add_argument(
            "--variation_ratio",
            dest="use_variation_ratio",
            action=BooleanOptionalAction,
            default=False,
        )

        return parser
