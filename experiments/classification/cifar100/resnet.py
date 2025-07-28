import torch
from lightning.pytorch.cli import LightningArgumentParser
from torch import nn

from torch_uncertainty import TULightningCLI
from torch_uncertainty.datamodules import CIFAR100DataModule
from torch_uncertainty.models.classification import resnet
from torch_uncertainty.routines.classification import ClassificationRoutine


class ResNetModel(ClassificationRoutine):
    def __init__(
        self,
        num_classes: int,
        loss: nn.Module,
        in_channels: int = 3,
        arch: int = 18,
        style: str = "cifar",
        normalization_layer: type[nn.Module] = nn.BatchNorm2d,
        optim_recipe: dict | None = None,
        **kwargs,
    ) -> None:
        """ResNet model for classification."""
        model = resnet(
            arch=arch,
            num_classes=num_classes,
            in_channels=in_channels,
            style=style,
            normalization_layer=normalization_layer,
        )
        super().__init__(
            num_classes=num_classes,
            model=model,
            loss=loss,
            optim_recipe=optim_recipe,
            **kwargs,
        )
        self.save_hyperparameters(ignore=["loss"])


class ResNetCLI(TULightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_arguments_to_parser(parser)
        parser.add_optimizer_args(torch.optim.SGD)
        parser.add_lr_scheduler_args(torch.optim.lr_scheduler.MultiStepLR)


def cli_main() -> ResNetCLI:
    return ResNetCLI(ResNetModel, CIFAR100DataModule)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    cli = cli_main()
    if (
        (not cli.trainer.fast_dev_run)
        and cli.subcommand == "fit"
        and cli._get(cli.config, "eval_after_fit")
    ):
        cli.trainer.test(datamodule=cli.datamodule, ckpt_path="best")
