# fmt:off
from pathlib import Path

import torch.nn as nn
from cli_test_helpers import ArgvContext

from torch_uncertainty import cls_main, init_args
from torch_uncertainty.optimization_procedures import optim_cifar10_resnet18

from .._dummies import DummyBaseline, DummyDataModule


# fmt:on
class TestClassificationSingle:
    """Testing the classification routine with a single model."""

    def test_cls_main_resnet(self):
        root = Path(__file__).parent.absolute().parents[0]
        with ArgvContext(""):
            args = init_args(DummyBaseline, DummyDataModule)

            # datamodule
            args.root = str(root / "data")
            dm = DummyDataModule(**vars(args))

            model = DummyBaseline(
                num_classes=dm.num_classes,
                in_channels=dm.num_channels,
                loss=nn.CrossEntropyLoss,
                optimization_procedure=optim_cifar10_resnet18,
                baseline_type="single",
                **vars(args),
            )

            cls_main(model, dm, root, "dummy", "classification", args)


class TestClassificationEnsemble:
    """Testing the classification routine with an ensemble model."""

    def test_cls_main_resnet(self):
        root = Path(__file__).parent.absolute().parents[0]
        with ArgvContext(""):
            args = init_args(DummyBaseline, DummyDataModule)

            # datamodule
            args.root = str(root / "data")
            dm = DummyDataModule(**vars(args))

            model = DummyBaseline(
                num_classes=dm.num_classes,
                in_channels=dm.num_channels,
                loss=nn.CrossEntropyLoss,
                optimization_procedure=optim_cifar10_resnet18,
                baseline_type="ensemble",
                **vars(args),
            )

            cls_main(model, dm, root, "dummy", "classification", args)
