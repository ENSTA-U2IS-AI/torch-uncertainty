# fmt: off
from argparse import ArgumentParser, BooleanOptionalAction
from typing import Any

import torch
import torch.nn as nn

from torch_uncertainty.models.resnet import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)
from torch_uncertainty.routines.classification import ClassificationSingle

# fmt: on
archs = [resnet18, resnet34, resnet50, resnet101, resnet152]
choices = [18, 34, 50, 101, 152]


class ResNet(ClassificationSingle):
    r"""LightningModule for Vanilla ResNet.

    Args:
        num_classes (int): Number of classes to predict.
        in_channels (int): Number of input channels.
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
        groups (int, optional): Number of groups in convolutions. Defaults to
            ``1``.
        imagenet_structure (bool, optional): Whether to use the ImageNet
            structure. Defaults to ``True``.

    Raises:
        ValueError: If :attr:`groups` :math:`<1`.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        arch: int,
        loss: nn.Module,
        optimization_procedure: Any,
        groups: int = 1,
        imagenet_structure: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            **kwargs,
        )

        self.save_hyperparameters(ignore=["loss", "optimization_procedure"])
        if groups < 1:
            raise ValueError(
                f"Number of groups must be at least 1, not {groups}"
            )

        self.loss = loss
        self.optimization_procedure = optimization_procedure

        self.model = archs[choices.index(arch)](
            in_channels=in_channels,
            num_classes=num_classes,
            groups=groups,
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
        return self.model.forward(input)

    @staticmethod
    def add_model_specific_args(
        parent_parser: ArgumentParser,
    ) -> ArgumentParser:
        """Defines the model's attributes via command-line options:

        - ``--arch [int]``: defines :attr:`arch`. Defaults to ``18``.
        - ``--groups [int]``: defines :attr:`groups`. Defaults to ``1``.
        - ``--imagenet_structure``: sets :attr:`imagenet_structure`. Defaults
          to ``True``.

        Example:

            .. parsed-literal::

                python script.py --groups 2 --no-imagenet_structure
        """
        parent_parser = ClassificationSingle.add_model_specific_args(
            parent_parser
        )
        parent_parser.add_argument(
            "--arch",
            type=int,
            default=18,
            choices=choices,
            help="Type of ResNet",
        )
        parent_parser.add_argument("--groups", type=int, default=1)
        parent_parser.add_argument(
            "--imagenet_structure",
            action=BooleanOptionalAction,
            default=True,
            help="Use imagenet structure",
        )
        return parent_parser
