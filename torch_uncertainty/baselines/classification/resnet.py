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
    r"""ResNet backbone baseline for classification providing support for
    various versions and architectures.

    Args:
        num_classes (int): Number of classes to predict.
        in_channels (int): Number of input channels.
        loss (nn.Module): Training loss.
        optimization_procedure (Any): Optimization procedure, corresponds to
            what expect the `LightningModule.configure_optimizers()
            <https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#configure-optimizers>`_
            method.
        version (str):
            Determines which ResNet version to use:

            - ``"vanilla"``: original ResNet
            - ``"packed"``: Packed-Ensembles ResNet
            - ``"batched"``: BatchEnsemble ResNet
            - ``"masked"``: Masksemble ResNet

        arch (int):
            Determines which ResNet architecture to use:

            - ``18``: ResNet-18
            - ``32``: ResNet-32
            - ``50``: ResNet-50
            - ``101``: ResNet-101
            - ``152``: ResNet-152

        imagenet_structure (bool, optional): Whether to use the ImageNet
            structure. Defaults to ``True``.
        num_estimators (int, optional): Number of estimators in the ensemble.
            Only used if :attr:`version` is either ``"packed"``, ``"batched"``
            or ``"masked"`` Defaults to ``None``.
        groups (int, optional): Number of groups in convolutions. Defaults to
            ``1``.
        scale (float, optional): Expansion factor affecting the width of the
            estimators. Only used if :attr:`version` is ``"masked"``. Defaults
            to ``None``.
        alpha (float, optional): Expansion factor affecting the width of the
            estimators. Only used if :attr:`version` is ``"packed"``. Defaults
            to ``None``.
        gamma (int, optional): Number of groups within each estimator. Only
            used if :attr:`version` is ``"packed"`` and scales with
            :attr:`groups`. Defaults to ``1s``.
        use_entropy (bool, optional): Indicates whether to use the entropy
            values as the OOD criterion or not. Defaults to ``False``.
        use_logits (bool, optional): Indicates whether to use the logits as the
            OOD criterion or not. Defaults to ``False``.
        use_mi (bool, optional): Indicates whether to use the mutual
            information as the OOD criterion or not. Defaults to ``False``.
        use_variation_ratio (bool, optional): Indicates whether to use the
            variation ratio as the OOD criterion or not. Defaults to ``False``.
        pretrained (bool, optional): Indicates whether to use the pretrained
            weights or not. Only used if :attr:`version` is ``"packed"``.
            Defaults to ``False``.

    Raises:
        ValueError: If :attr:`version` is not either ``"vanilla"``,
            ``"packed"``, ``"batched"`` or ``"masked"``.

    Returns:
        LightningModule: ResNet baseline ready for training and evaluation.
    """

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
        groups: Optional[int] = 1,
        scale: Optional[float] = None,
        alpha: Optional[float] = None,
        gamma: Optional[int] = 1,
        use_entropy: bool = False,
        use_logits: bool = False,
        use_mi: bool = False,
        use_variation_ratio: bool = False,
        pretrained: bool = False,
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

        if version == "packed":
            params.update(
                {
                    "num_estimators": num_estimators,
                    "alpha": alpha,
                    "gamma": gamma,
                    "pretrained": pretrained,
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

        model = cls.versions[version][cls.archs.index(arch)](**params)
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
            help=f"Variation of ResNet. Choose among: {cls.versions.keys()}",
        )
        parser.add_argument(
            "--arch",
            type=int,
            choices=cls.archs,
            default=18,
            help=f"Architecture of ResNet. Choose among: {cls.archs}",
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
            type=float,
            default=None,
            help="Alpha for packed resnet",
        )
        parser.add_argument(
            "--gamma",
            type=int,
            default=1,
            help="Gamma for packed resnet",
        )
        parser.add_argument(
            "--pretrained",
            dest="pretrained",
            action=BooleanOptionalAction,
            default=False,
        )
        return parser
