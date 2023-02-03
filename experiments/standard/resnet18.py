# fmt: off
from pathlib import Path

import torch.nn as nn

from torch_uncertainty import cli_main
from torch_uncertainty.baselines.standard import ResNet
from torch_uncertainty.datamodules import CIFAR10DataModule
from torch_uncertainty.optimization_procedures import optim_cifar10_resnet18

# fmt: on

if __name__ == "__main__":
    root = Path(__file__).parent.absolute().parents[1]
    # print(root)
    cli_main(
        ResNet,
        CIFAR10DataModule,
        nn.CrossEntropyLoss,
        optim_cifar10_resnet18,
        root,
        "std",
    )
