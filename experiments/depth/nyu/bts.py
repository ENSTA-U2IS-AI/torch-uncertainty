import torch
from lightning.pytorch.cli import LightningArgumentParser

from torch_uncertainty import TULightningCLI
from torch_uncertainty.baselines.depth import BTSBaseline
from torch_uncertainty.datamodules.depth import NYUv2DataModule
from torch_uncertainty.utils.learning_rate import PolyLR


class BTSCLI(TULightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_optimizer_args(torch.optim.AdamW)
        parser.add_lr_scheduler_args(PolyLR)


def cli_main() -> BTSCLI:
    return BTSCLI(BTSBaseline, NYUv2DataModule)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    cli = cli_main()
    if (
        (not cli.trainer.fast_dev_run)
        and cli.subcommand == "fit"
        and cli._get(cli.config, "eval_after_fit")
    ):
        cli.trainer.test(datamodule=cli.datamodule, ckpt_path="best")
