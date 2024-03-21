from cli_test_helpers import ArgvContext

from torch_uncertainty.baselines.classification import ResNetBaseline
from torch_uncertainty.datamodules import CIFAR10DataModule
from torch_uncertainty.utils.cli import TULightningCLI, TUSaveConfigCallback


class TestCLI:
    """Testing torch-uncertainty CLI."""

    def test_cli_init(self):
        """Test CLI initialization."""
        with ArgvContext(
            "file.py",
            "--model.in_channels",
            "3",
            "--model.num_classes",
            "10",
            "--model.version",
            "std",
            "--model.arch",
            "18",
            "--model.loss",
            "torch.nn.CrossEntropyLoss",
            "--data.root",
            "./data",
            "--data.batch_size",
            "32",
        ):
            cli = TULightningCLI(ResNetBaseline, CIFAR10DataModule, run=False)
            assert cli.eval_after_fit_default is False
            assert cli.save_config_callback == TUSaveConfigCallback
