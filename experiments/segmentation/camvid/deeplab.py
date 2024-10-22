import torch
from lightning.pytorch.cli import LightningArgumentParser

from torch_uncertainty import TULightningCLI
from torch_uncertainty.baselines.segmentation import DeepLabBaseline
from torch_uncertainty.datamodules.segmentation import CamVidDataModule
from torch_uncertainty.utils.learning_rate import PolyLR


class DeepLabV3CLI(TULightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_optimizer_args(torch.optim.SGD)
        parser.add_lr_scheduler_args(PolyLR)


def cli_main() -> DeepLabV3CLI:
    return DeepLabV3CLI(DeepLabBaseline, CamVidDataModule)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    cli = cli_main()
    if (
        (not cli.trainer.fast_dev_run)
        and cli.subcommand == "fit"
        and cli._get(cli.config, "eval_after_fit")
    ):
        cli.trainer.test(datamodule=cli.datamodule, ckpt_path="best")
