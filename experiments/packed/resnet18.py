# fmt: off
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
from pysemble import cli_main
from pysemble.datamodules import CIFAR10DataModule
from pysemble.networks.resnet_groupens import ResNet18_GrE
from pysemble.optimization_procedures import optim_cifar10_resnet18
from pysemble.routines import Ensemble
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor


# fmt: on
class GroupEns(Ensemble):
    def __init__(self, config: Union[Dict[str, Any], Namespace]) -> None:
        if isinstance(config, Namespace):
            config = vars(config)

        super().__init__(**config)
        self.save_hyperparameters(config)

        # To log the model graph in tensorboard.
        self.example_input_array = torch.randn(1, 3, 32, 32)

        self.n_estimators: int = config.get("n_estimators", 4)
        self.augmentation: int = config.get("augmentation", 2)
        self.n_subgroups: int = config.get("n_subgroups", 1)

        self.model = ResNet18_GrE(
            n_estimators=self.n_estimators,
            augmentation=self.augmentation,
            n_subgroups=self.n_subgroups,
            num_classes=10,
        )
        # self.model = torch.compile(self.model)

    def configure_optimizers(self) -> dict:
        return optim_cifar10_resnet18(self)

    @property
    def criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = Ensemble.add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("ResNet")
        parser.add_argument("--n_estimators", type=int, default=4)
        parser.add_argument("--augmentation", type=float, default=2)
        parser.add_argument("--n_subgroups", type=int, default=1)
        return parent_parser

    def training_step(  # type: ignore
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        inputs, targets = batch
        targets = targets.repeat(self.n_estimators)
        return super().training_step((inputs, targets), batch_idx)

    def forward(self, input: Tensor) -> Tensor:  # type: ignore
        input = input.repeat(1, self.n_estimators, 1, 1)
        return self.model.forward(input)


if __name__ == "__main__":
    root = Path(__file__).parent.absolute()
    cli_main(GroupEns, CIFAR10DataModule, root, "groupens")
