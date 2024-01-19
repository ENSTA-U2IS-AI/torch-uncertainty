import torch
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from lightning.pytorch.loggers import TensorBoardLogger  # noqa: F401

from torch_uncertainty.baselines import ResNet
from torch_uncertainty.datamodules import CIFAR10DataModule
from torch_uncertainty.lightning_cli import MySaveConfigCallback


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_optimizer_args(torch.optim.SGD)
        parser.add_lr_scheduler_args(torch.optim.lr_scheduler.MultiStepLR)


def cli_main():
    _ = MyLightningCLI(
        ResNet, CIFAR10DataModule, save_config_callback=MySaveConfigCallback
    )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    cli_main()
