import torch
from lightning.pytorch.cli import LightningArgumentParser

from torch_uncertainty import TULightningCLI
from torch_uncertainty.baselines.segmentation import SegFormerBaseline
from torch_uncertainty.datamodules.segmentation import CityscapesDataModule


class SegFormerCLI(TULightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_optimizer_args(torch.optim.AdamW)
        parser.add_lr_scheduler_args(torch.optim.lr_scheduler.StepLR)


def cli_main() -> SegFormerCLI:
    return SegFormerCLI(SegFormerBaseline, CityscapesDataModule)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    cli = cli_main()
    if (
        (not cli.trainer.fast_dev_run)
        and cli.subcommand == "fit"
        and cli._get(cli.config, "eval_after_fit")
    ):
        cli.trainer.test(datamodule=cli.datamodule, ckpt_path="best")
