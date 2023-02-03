# fmt: off
from argparse import ArgumentParser, Namespace
from typing import Dict, Union

import torch
import torch.nn as nn

from torch_uncertainty.models.resnet.std import (
    ResNet18,
    ResNet34,
    ResNet50,
    ResNet101,
    ResNet152,
)
from torch_uncertainty.routines.classification import ClassificationSingle

# fmt: on
archs = [ResNet18, ResNet34, ResNet50, ResNet101, ResNet152]
choices = [18, 34, 50, 101, 152]


class ResNet(ClassificationSingle):
    def __init__(
        self,
        loss,
        optimization_procedure,
        num_classes: int,
        in_channels: int,
        config: Union[Dict, Namespace],
    ) -> None:
        if isinstance(config, Namespace):
            config = vars(config)

        super().__init__(num_classes, config)
        self.save_hyperparameters(config)
        assert config["groups"] >= 1

        self.loss = loss
        self.optimization_procedure = optimization_procedure

        self.model = archs[choices.index(config["arch"])](
            in_channels=in_channels,
            num_classes=num_classes,
            groups=config["groups"],
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
        return parent_parser
