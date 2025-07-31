import sys

from torch_uncertainty.models import resnet
from torch_uncertainty.routines import ClassificationRoutine
from torch_uncertainty.datamodules import CIFAR10DataModule
from torch_uncertainty.utils.cli import TULightningCLI, TUSaveConfigCallback


class TestCLI:
    """Testing torch-uncertainty CLI."""

    def test_cli_init(self) -> None:
        """Test CLI initialization."""
        sys.argv = [
            "file.py",
            "--model.model.class_path",
            "torch_uncertainty.models.resnet",
            "--model.model.init_args.in_channels",
            "3",
            "--model.model.init_args.num_classes",
            "10",
            "--model.model.init_args.arch",
            "18",
            "--model.num_classes",
            "10",
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
        cli = TULightningCLI(ClassificationRoutine, CIFAR10DataModule, run=False)
        assert cli.eval_after_fit_default is False
        assert cli.save_config_callback == TUSaveConfigCallback
        assert isinstance(cli.trainer.callbacks[0], TUSaveConfigCallback)
        cli.trainer.callbacks[0].setup(cli.trainer, cli.model, stage="fit")
        cli.trainer.callbacks[0].already_saved = True
        cli.trainer.callbacks[0].setup(cli.trainer, cli.model, stage="fit")
