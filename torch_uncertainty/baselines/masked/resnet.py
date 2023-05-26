# fmt: off
from argparse import ArgumentParser, BooleanOptionalAction
from typing import Any, Dict, Literal

import torch
import torch.nn as nn

from torch_uncertainty.models.resnet import (
    masked_resnet18,
    masked_resnet34,
    masked_resnet50,
    masked_resnet101,
    masked_resnet152,
)
from torch_uncertainty.routines.classification import ClassificationEnsemble

# fmt: on
archs = [
    masked_resnet18,
    masked_resnet34,
    masked_resnet50,
    masked_resnet101,
    masked_resnet152,
]
choices = [18, 34, 50, 101, 152]


class MaskedResNet(ClassificationEnsemble):
    r"""LightningModule for Masksembles ResNet.

    Args:
        num_classes (int): Number of classes to predict.
        num_estimators (int): Number of estimators in the ensemble.
        in_channels (int): Number of input channels.
        scale (int): Expansion factor affecting the width of the estimators.
        groups (int): Number of groups within each estimator.
        arch (int):
            Determines which ResNet architecture to use:

            - ``18``: ResNet-18
            - ``32``: ResNet-32
            - ``50``: ResNet-50
            - ``101``: ResNet-101
            - ``152``: ResNet-152

        loss (torch.nn.Module): Training loss.
        optimization_procedure (Any): Optimization procedure, corresponds to
            what expect the `LightningModule.configure_optimizers()
            <https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#configure-optimizers>`_
            method.

    Raises:
            ValueError: If :attr:`scale`:math:`<1`.
            ValueError: If :attr:`groups`:math:`<1`.
    """

    def __init__(
        self,
        num_classes: int,
        num_estimators: int,
        in_channels: int,
        scale: int,
        groups: int,
        arch: Literal[18, 34, 50, 101, 152],
        loss: nn.Module,
        optimization_procedure: Any,
        imagenet_structure: bool = True,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            num_estimators=num_estimators,
            **kwargs,
        )

        if scale < 1:
            raise ValueError(f"Attribute `scale` should be >= 1, not {scale}.")
        if groups < 1:
            raise ValueError(
                f"Attribute `groups` should be >= 1, not {groups}."
            )

        # construct config
        self.save_hyperparameters(ignore=["loss", "optimization_procedure"])

        self.loss = loss
        self.optimization_procedure = optimization_procedure

        self.model = archs[choices.index(arch)](
            in_channels=in_channels,
            num_estimators=num_estimators,
            scale=scale,
            groups=groups,
            num_classes=num_classes,
            imagenet_structure=imagenet_structure,
        )

        # to log the graph
        self.example_input_array = torch.randn(1, in_channels, 32, 32)

    def configure_optimizers(self) -> dict:
        return self.optimization_procedure(self)

    @property
    def criterion(self) -> nn.Module:
        return self.loss()

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        input = input.repeat(self.num_estimators, 1, 1, 1)
        return self.model.forward(input)

    @staticmethod
    def add_model_specific_args(
        parent_parser: ArgumentParser,
    ) -> ArgumentParser:
        """Defines the model's attributes via command-line options:

        - ``--arch [int]``: defines :attr:`arch`. Defaults to ``18``.
        - ``--num_estimators [int]``: defines :attr:`num_estimators`. Defaults
          to ``1``.
        - ``--scale [int]``: defines :attr:`scale`. Defaults to ``1``.
        - ``--imagenet_structure``: sets :attr:`imagenet_structure`. Defaults
          to ``True``.
        - ``--groups [int]``: defines :attr:`groups`. Defaults to ``1``.

        Example:

            .. parsed-literal::

                python script.py --arch 18 --num_estimators 4 --scale 2.0
        """
        parent_parser = ClassificationEnsemble.add_model_specific_args(
            parent_parser
        )
        parent_parser.add_argument(
            "--arch",
            type=int,
            default=18,
            choices=choices,
            help="Type of ResNet",
        )
        parent_parser.add_argument(
            "--imagenet_structure",
            action=BooleanOptionalAction,
            default=True,
            help="Use imagenet structure",
        )
        parent_parser.add_argument("--scale", type=float, default=2.0)
        return parent_parser
