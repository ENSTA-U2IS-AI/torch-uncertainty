import pytest
from torch import nn
from torchvision.datasets import MNIST

from tests._dummies.dataset import DummyClassificationDataset
from torch_uncertainty.datamodules import MNISTDataModule
from torch_uncertainty.transforms import Cutout


class TestMNISTDataModule:
    """Testing the MNISTDataModule datamodule class."""

    def test_mnist_cutout(self):
        dm = MNISTDataModule(
            root="./data/", batch_size=128, cutout=16, val_split=0.1
        )

        assert dm.dataset == MNIST
        assert isinstance(dm.train_transform.transforms[0], Cutout)

        dm = MNISTDataModule(
            root="./data/", batch_size=128, ood_ds="not", cutout=0, val_split=0
        )
        assert isinstance(dm.train_transform.transforms[0], nn.Identity)

        with pytest.raises(ValueError):
            MNISTDataModule(root="./data/", batch_size=128, ood_ds="other")

        MNISTDataModule(root="./data/", batch_size=128, test_alt="c")

        dm.dataset = DummyClassificationDataset
        dm.ood_dataset = DummyClassificationDataset

        dm.prepare_data()
        dm.setup()

        dm.train_dataloader()
        dm.val_dataloader()
        dm.test_dataloader()

        with pytest.raises(ValueError):
            dm.setup("other")

        dm.eval_ood = True
        dm.val_split = 0.1
        dm.prepare_data()
        dm.setup()
        dm.test_dataloader()
