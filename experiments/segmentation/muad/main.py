import torch

from torch_uncertainty import TULightningCLI
from torch_uncertainty.datamodules.segmentation import MUADDataModule
from torch_uncertainty.routines import SegmentationRoutine


def cli_main() -> TULightningCLI:
    return TULightningCLI(SegmentationRoutine, MUADDataModule)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    cli = cli_main()
    if (
        (not cli.trainer.fast_dev_run)
        and cli.subcommand == "fit"
        and cli._get(cli.config, "eval_after_fit")
    ):
        cli.trainer.test(datamodule=cli.datamodule, ckpt_path="best")
