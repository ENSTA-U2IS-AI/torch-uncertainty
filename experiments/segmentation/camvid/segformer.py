import torch
from lightning.pytorch.cli import LightningArgumentParser
from torch import nn

from torch_uncertainty import TULightningCLI
from torch_uncertainty.datamodules.segmentation import CamVidDataModule
from torch_uncertainty.models.segmentation.segformer import seg_former
from torch_uncertainty.routines.segmentation import SegmentationRoutine


class SegFormerModel(SegmentationRoutine):
    def __init__(
        self,
        num_classes: int,
        loss: nn.Module,
        arch: int = 0,
        optim_recipe: dict | None = None,
        **kwargs,
    ) -> None:
        """SegFormer model for segmentation."""
        model = seg_former(num_classes=num_classes, arch=arch)
        super().__init__(
            num_classes=num_classes,
            model=model,
            loss=loss,
            optim_recipe=optim_recipe,
            format_batch_fn=nn.Identity(),
            **kwargs,
        )
        self.save_hyperparameters(ignore=["loss"])


class SegFormerCLI(TULightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_optimizer_args(torch.optim.SGD)
        parser.add_lr_scheduler_args(torch.optim.lr_scheduler.MultiStepLR)


def cli_main() -> SegFormerCLI:
    return SegFormerCLI(SegFormerModel, CamVidDataModule)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    cli = cli_main()
    if (
        (not cli.trainer.fast_dev_run)
        and cli.subcommand == "fit"
        and cli._get(cli.config, "eval_after_fit")
    ):
        cli.trainer.test(datamodule=cli.datamodule, ckpt_path="best")
