import torch
from lightning.pytorch.cli import LightningArgumentParser
from lightning.pytorch.loggers import TensorBoardLogger  # noqa: F401

from torch_uncertainty.baselines.segmentation import SegFormer
from torch_uncertainty.datamodules import CamVidDataModule
from torch_uncertainty.utils import TULightningCLI


class SegFormerCLI(TULightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_optimizer_args(torch.optim.SGD)
        parser.add_lr_scheduler_args(torch.optim.lr_scheduler.MultiStepLR)


def cli_main() -> SegFormerCLI:
    return SegFormerCLI(SegFormer, CamVidDataModule)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    cli = cli_main()

    if cli.subcommand == "fit" and cli._get(cli.config, "eval_after_fit"):
        cli.trainer.test(datamodule=cli.datamodule, ckpt_path="best")