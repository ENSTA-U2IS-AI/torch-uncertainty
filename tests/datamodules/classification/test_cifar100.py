import pytest
from torch import nn
from torchvision.datasets import CIFAR100

from tests._dummies.dataset import DummyClassificationDataset
from torch_uncertainty.datamodules import CIFAR100DataModule
from torch_uncertainty.transforms import Cutout


class TestCIFAR100DataModule:
    """Testing the CIFAR100DataModule datamodule class."""

    def test_cifar100(self) -> None:
        dm = CIFAR100DataModule(
            root="./data/",
            batch_size=128,
            train_transform=nn.Identity(),
            test_transform=nn.Identity(),
            num_tta=2,
        )
        dm = CIFAR100DataModule(
            root="./data/",
            batch_size=128,
            train_transform=nn.Identity(),
            test_transform=nn.Identity(),
        )
        assert isinstance(dm.train_transform, nn.Identity)
        assert isinstance(dm.test_transform, nn.Identity)

        dm = CIFAR100DataModule(root="./data/", batch_size=128, cutout=16)

        assert dm.dataset == CIFAR100
        assert isinstance(dm.train_transform.transforms[2], Cutout)

        dm.dataset = DummyClassificationDataset
        dm.shift_dataset = DummyClassificationDataset

        dm.prepare_data()
        dm.setup()

        dm.train_dataloader()
        dm.val_dataloader()
        dm.test_dataloader()

        dm.eval_shift = True
        dm.prepare_data()
        dm.setup("test")
        dm.test_dataloader()

        dm = CIFAR100DataModule(
            root="./data/",
            batch_size=128,
            cutout=0,
            val_split=0.1,
            num_dataloaders=2,
            basic_augment=False,
        )
        dm.dataset = DummyClassificationDataset
        dm.shift_dataset = DummyClassificationDataset
        dm.setup()
        dm.setup("test")
        dm.train_dataloader()
        with pytest.raises(ValueError):
            dm.setup("other")

        with pytest.raises(ValueError):
            dm = CIFAR100DataModule(
                root="./data/",
                batch_size=128,
                num_dataloaders=1,
                cutout=8,
                randaugment=True,
            )

        dm = CIFAR100DataModule(root="./data/", batch_size=128, randaugment=True)

        dm = CIFAR100DataModule(root="./data/", batch_size=128, auto_augment="rand-m9-n2-mstd0.5")

    def test_cifar100_cv(self) -> None:
        dm = CIFAR100DataModule(root="./data/", batch_size=128)
        dm.dataset = lambda root, train, download, transform: DummyClassificationDataset(
            root,
            train=train,
            download=download,
            transform=transform,
            num_images=20,
        )
        dm.make_cross_val_splits(2, 1)

        dm = CIFAR100DataModule(root="./data/", batch_size=128, val_split=0.1)
        dm.dataset = lambda root, train, download, transform: DummyClassificationDataset(
            root,
            train=train,
            download=download,
            transform=transform,
            num_images=20,
        )
        dm.make_cross_val_splits(2, 1)

    def test_ood_defaults_and_get_indices(self, monkeypatch) -> None:
        dm = CIFAR100DataModule(
            root="./data/",
            batch_size=16,
            train_transform=nn.Identity(),
            test_transform=nn.Identity(),
            eval_ood=True,
            eval_shift=True,
        )

        dm.dataset = DummyClassificationDataset
        dm.shift_dataset = DummyClassificationDataset

        def _mock_get_ood(**_):
            test_ood = DummyClassificationDataset(
                root="./data/", train=False, download=False, transform=nn.Identity(), num_images=5
            )
            val_ood = DummyClassificationDataset(
                root="./data/", train=False, download=False, transform=nn.Identity(), num_images=5
            )
            near_default = {
                "example1": DummyClassificationDataset(
                    root="./data/",
                    train=False,
                    download=False,
                    transform=nn.Identity(),
                    num_images=5,
                ),
                "example2": DummyClassificationDataset(
                    root="./data/",
                    train=False,
                    download=False,
                    transform=nn.Identity(),
                    num_images=5,
                ),
            }
            far_default = {
                "example3": DummyClassificationDataset(
                    root="./data/",
                    train=False,
                    download=False,
                    transform=nn.Identity(),
                    num_images=5,
                ),
                "example4": DummyClassificationDataset(
                    root="./data/",
                    train=False,
                    download=False,
                    transform=nn.Identity(),
                    num_images=5,
                ),
                "example5": DummyClassificationDataset(
                    root="./data/",
                    train=False,
                    download=False,
                    transform=nn.Identity(),
                    num_images=5,
                ),
            }
            return test_ood, val_ood, near_default, far_default

        monkeypatch.setattr(
            "torch_uncertainty.datamodules.classification.cifar100.get_ood_datasets",
            _mock_get_ood,
        )
        dm.setup("test")

        assert hasattr(dm, "near_oods")
        assert len(dm.near_oods) == 2

        assert hasattr(dm, "far_oods")
        assert len(dm.far_oods) == 3

        # dataset_name must exist for all OOD datasets
        for ds in [dm.val_ood, *dm.near_oods, *dm.far_oods]:
            assert hasattr(ds, "dataset_name")
            assert ds.dataset_name in {"dummy", ds.__class__.__name__.lower()}

        assert dm.near_ood_names == [ds.dataset_name for ds in dm.near_oods]
        assert dm.far_ood_names == [ds.dataset_name for ds in dm.far_oods]

        # test_dataloader order & count:
        loaders = dm.test_dataloader()
        expected = 1 + 1 + 1 + len(dm.near_oods) + len(dm.far_oods) + 1
        assert len(loaders) == expected

        idx = dm.get_indices()
        assert idx["test"] == [0]
        assert idx["test_ood"] == [1]
        assert idx["val_ood"] == [2]
        assert idx["near_oods"] == list(range(3, 3 + len(dm.near_oods)))
        assert idx["far_oods"] == list(
            range(3 + len(dm.near_oods), 3 + len(dm.near_oods) + len(dm.far_oods))
        )
        assert idx["shift"] == [3 + len(dm.near_oods) + len(dm.far_oods)]
