# fmt: off
from argparse import ArgumentParser
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
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        arch: int,
        loss: nn.Module,
        optimization_procedure: Any,
        groups: int = 1,
        use_entropy: bool = False,
        use_logits: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            use_entropy=use_entropy,
            use_logits=use_logits,
        )

        self.save_hyperparameters(ignore=["loss", "optimization_procedure"])
        assert groups >= 1

        self.loss = loss
        self.optimization_procedure = optimization_procedure

        self.model = archs[choices.index(arch)](
            in_channels=in_channels,
            num_classes=num_classes,
            groups=groups,
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
        parent_parser.add_argument(
            "--arch",
            type=int,
            default=18,
            choices=choices,
            help="Type of ResNet",
        )
        parent_parser.add_argument(
            "--entropy", dest="use_entropy", action="store_true"
        )
        parent_parser.add_argument("--groups", type=int, default=1)
        parent_parser.add_argument(
            "--logits", dest="use_logits", action="store_true"
        )
        return parent_parser
