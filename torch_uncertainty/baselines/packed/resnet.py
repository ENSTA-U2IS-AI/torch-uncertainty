# fmt: off
from argparse import ArgumentParser, Namespace
from typing import Dict, Union

import torch
import torch.nn as nn

from torch_uncertainty.models.resnet.packed import (
    PackedResNet18,
    PackedResNet34,
    PackedResNet50,
    PackedResNet101,
    PackedResNet152,
)
from torch_uncertainty.routines.classification import ClassificationEnsemble

# fmt: on
archs = [
    PackedResNet18,
    PackedResNet34,
    PackedResNet50,
    PackedResNet101,
    PackedResNet152,
]
choices = [18, 34, 50, 101, 152]


class PackedResNet(ClassificationEnsemble):
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

        num_estimators: int = config.get("num_estimators", 1)
        super().__init__(num_classes, num_estimators, config)

        self.alpha: int = config.get("alpha", 1)
        self.gamma: int = config.get("gamma", 1)
        assert self.alpha >= 1
        assert self.gamma >= 1

        self.save_hyperparameters(config)

        self.loss = loss
        self.optimization_procedure = optimization_procedure

        self.model = archs[choices.index(config["arch"])](
            in_channels=in_channels,
            num_estimators=self.num_estimators,
            alpha=self.alpha,
            gamma=self.gamma,
            num_classes=num_classes,
        )

        # to log the graph
        self.example_input_array = torch.randn(1, in_channels, 32, 32)

    def configure_optimizers(self) -> dict:
        return self.optimization_procedure(self)

    @property
    def criterion(self) -> nn.Module:
        return self.loss()

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        input = input.repeat(1, self.num_estimators, 1, 1)
        return self.model.forward(input)

    @staticmethod
    def add_model_specific_args(
        parent_parser: ArgumentParser,
    ) -> ArgumentParser:
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
        parent_parser.add_argument("--alpha", type=int, default=1)
        parent_parser.add_argument("--gamma", type=int, default=1)
        parent_parser.add_argument("--num_estimators", type=int, default=1)
        return parent_parser
