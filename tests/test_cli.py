import sys

from torch import nn

from torch_uncertainty.datamodules import CIFAR10DataModule
from torch_uncertainty.models.classification import resnet
from torch_uncertainty.routines.classification import ClassificationRoutine
from torch_uncertainty.utils.cli import TULightningCLI, TUSaveConfigCallback


class SimpleResNetModel(ClassificationRoutine):
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        loss: nn.Module,
        arch: int = 18,
        **kwargs,
    ) -> None:
        """Simple ResNet model for testing."""
        model = resnet(
            arch=arch,
            num_classes=num_classes,
            in_channels=in_channels,
            style="cifar",
        )
        super().__init__(
            num_classes=num_classes,
            model=model,
            loss=loss,
            **kwargs,
        )
        self.save_hyperparameters(ignore=["loss"])


class TestCLI:
    """Testing torch-uncertainty CLI."""

    def test_cli_init(self) -> None:
        """Test CLI initialization."""
        sys.argv = [
            "file.py",
            "--model.in_channels",
            "3",
            "--model.num_classes",
            "10",
            "--model.arch",
            "18",
            "--model.loss.class_path",
            "torch.nn.CrossEntropyLoss",
            "--data.root",
            "./data",
            "--data.batch_size",
            "4",
            "--trainer.callbacks+=ModelCheckpoint",
            "--trainer.callbacks.monitor=val/cls/Acc",
            "--trainer.callbacks.mode=max",
        ]
        cli = TULightningCLI(SimpleResNetModel, CIFAR10DataModule, run=False)
        assert cli.eval_after_fit_default is False
        assert cli.save_config_callback == TUSaveConfigCallback
        assert isinstance(cli.trainer.callbacks[0], TUSaveConfigCallback)
        cli.trainer.callbacks[0].setup(cli.trainer, cli.model, stage="fit")
        cli.trainer.callbacks[0].already_saved = True
        cli.trainer.callbacks[0].setup(cli.trainer, cli.model, stage="fit")
