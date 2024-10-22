import torch
from lightning.pytorch.cli import LightningArgumentParser

from torch_uncertainty import TULightningCLI
from torch_uncertainty.baselines.classification import VGGBaseline
from torch_uncertainty.datamodules import CIFAR10DataModule


class ResNetCLI(TULightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_arguments_to_parser(parser)
        parser.add_optimizer_args(torch.optim.Adam)
        parser.add_lr_scheduler_args(torch.optim.lr_scheduler.MultiStepLR)


def cli_main() -> ResNetCLI:
    return ResNetCLI(VGGBaseline, CIFAR10DataModule)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    cli = cli_main()
    if (
        (not cli.trainer.fast_dev_run)
        and cli.subcommand == "fit"
        and cli._get(cli.config, "eval_after_fit")
    ):
        cli.trainer.test(datamodule=cli.datamodule, ckpt_path="best")
