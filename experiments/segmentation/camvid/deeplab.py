import torch
from lightning.pytorch.cli import LightningArgumentParser
from torch import nn
from torch.optim.lr_scheduler import PolynomialLR

from torch_uncertainty import TULightningCLI
from torch_uncertainty.datamodules.segmentation import CamVidDataModule
from torch_uncertainty.models.segmentation import deep_lab_v3_resnet
from torch_uncertainty.routines.segmentation import SegmentationRoutine


class DeepLabModel(SegmentationRoutine):
    def __init__(
        self,
        num_classes: int,
        loss: nn.Module,
        arch: int = 50,
        style: str = "v3",
        output_stride: int = 16,
        separable: bool = False,
        pretrained_backbone: bool = True,
        optim_recipe: dict | None = None,
        **kwargs,
    ) -> None:
        """DeepLab model for segmentation."""
        model = deep_lab_v3_resnet(
            num_classes=num_classes,
            arch=arch,
            style=style,
            output_stride=output_stride,
            separable=separable,
            pretrained_backbone=pretrained_backbone,
        )
        super().__init__(
            num_classes=num_classes,
            model=model,
            loss=loss,
            optim_recipe=optim_recipe,
            format_batch_fn=nn.Identity(),
            **kwargs,
        )
        self.save_hyperparameters(ignore=["loss"])


class DeepLabV3CLI(TULightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_optimizer_args(torch.optim.SGD)
        parser.add_lr_scheduler_args(PolynomialLR)


def cli_main() -> DeepLabV3CLI:
    return DeepLabV3CLI(DeepLabModel, CamVidDataModule)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    cli = cli_main()
    if (
        (not cli.trainer.fast_dev_run)
        and cli.subcommand == "fit"
        and cli._get(cli.config, "eval_after_fit")
    ):
        cli.trainer.test(datamodule=cli.datamodule, ckpt_path="best")
