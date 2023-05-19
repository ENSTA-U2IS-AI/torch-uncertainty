# fmt: off
from argparse import ArgumentParser
from typing import Any

import torch
import torch.nn as nn

from torch_uncertainty.routines.classification import ClassificationSingle

from .model import dummy_model


# fmt: on
class Dummy(ClassificationSingle):
    r"""LightningModule for Vanilla ResNet.

    Args:
        num_classes (int): Number of classes to predict.
        in_channels (int): Number of input channels.
        loss (torch.nn.Module): Training loss.
        optimization_procedure (Any): Optimization procedure, corresponds to
            what expect the `LightningModule.configure_optimizers()
            <https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#configure-optimizers>`_
            method.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        loss: nn.Module,
        optimization_procedure: Any,
        **kwargs,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
        )

        self.save_hyperparameters(ignore=["loss", "optimization_procedure"])

        self.loss = loss
        self.optimization_procedure = optimization_procedure

        self.model = dummy_model(
            in_channels=in_channels,
            num_classes=num_classes,
        )

        # to log the graph
        self.example_input_array = torch.randn(1, in_channels, 8, 8)

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
        return parent_parser
