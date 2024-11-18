import torch
from lightning.pytorch.cli import LightningArgumentParser

from torch_uncertainty import TULightningCLI
from torch_uncertainty.baselines.regression import MLPBaseline
from torch_uncertainty.datamodules import UCIRegressionDataModule


class MLPCLI(TULightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_optimizer_args(torch.optim.Adam)


def cli_main() -> MLPCLI:
    return MLPCLI(MLPBaseline, UCIRegressionDataModule)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    cli = cli_main()
    if (
        (not cli.trainer.fast_dev_run)
        and cli.subcommand == "fit"
        and cli._get(cli.config, "eval_after_fit")
    ):
        cli.trainer.test(datamodule=cli.datamodule, ckpt_path="best")
