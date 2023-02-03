# fmt: off
from argparse import Namespace
from pathlib import Path
from typing import Dict, Union

import torch
import torch.nn as nn

from torch_uncertainty import cli_main
from torch_uncertainty.datamodules import CIFAR10DataModule
from torch_uncertainty.models.resnet.std import ResNet18_STD
from torch_uncertainty.optimization_procedures import optim_cifar10_resnet18
from torch_uncertainty.routines.classification import ClassificationSingle


# fmt: on
class STD(ClassificationSingle):
    def __init__(self, config: Union[Dict, Namespace]) -> None:
        if isinstance(config, Namespace):
            config = vars(config)

        super().__init__()
        self.save_hyperparameters(config)

        self.num_classes = 10
        self.model = ResNet18_STD(self.num_classes)

        # to log the graph
        self.example_input_array = torch.randn(1, 3, 32, 32)

    def configure_optimizers(self) -> dict:
        return optim_cifar10_resnet18(self)

    @property
    def criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.model.forward(input)


if __name__ == "__main__":
    root = Path(__file__).parent.absolute()
    cli_main(STD, CIFAR10DataModule, root, "std")
