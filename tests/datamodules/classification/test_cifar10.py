import pytest
from torch import nn
from torchvision.datasets import CIFAR10

from tests._dummies.dataset import DummyClassificationDataset
from torch_uncertainty.datamodules import CIFAR10DataModule
from torch_uncertainty.transforms import Cutout


class TestCIFAR10DataModule:
    """Testing the CIFAR10DataModule datamodule class."""

    def test_cifar10_main(self) -> None:
        dm = CIFAR10DataModule(
            root="./data/",
            batch_size=128,
            train_transform=nn.Identity(),
            test_transform=nn.Identity(),
            num_tta=2,
        )
        dm = CIFAR10DataModule(
            root="./data/",
            batch_size=128,
            train_transform=nn.Identity(),
            test_transform=nn.Identity(),
        )
        assert isinstance(dm.train_transform, nn.Identity)
        assert isinstance(dm.test_transform, nn.Identity)
        dm = CIFAR10DataModule(root="./data/", batch_size=128, cutout=16, postprocess_set="test")

        assert dm.dataset == CIFAR10
        assert isinstance(dm.train_transform.transforms[2], Cutout)

        dm.dataset = DummyClassificationDataset
        dm.shift_dataset = DummyClassificationDataset

        dm.prepare_data()
        dm.setup()

        with pytest.raises(ValueError):
            dm.setup("xxx")

        # test abstract methods
        dm.get_train_set()
        dm.get_val_set()
        dm.get_test_set()

        dm.train_dataloader()
        dm.val_dataloader()
        dm.test_dataloader()

        dm.eval_shift = True
        dm.prepare_data()
        dm.setup("test")
        dm.test_dataloader()

        dm = CIFAR10DataModule(
            root="./data/",
            batch_size=128,
            cutout=16,
            test_alt="h",
            basic_augment=False,
        )
        dm.dataset = DummyClassificationDataset
        dm.setup("test")

        dm = CIFAR10DataModule(
            root="./data/",
            batch_size=128,
            num_dataloaders=2,
            val_split=0.1,
            randaugment=True,
        )
        dm.dataset = DummyClassificationDataset
        dm.setup()
        dm.train_dataloader()

        dm = CIFAR10DataModule(
            root="./data/",
            batch_size=128,
            num_dataloaders=1,
            val_split=0.1,
            num_tta=64,
            eval_shift=True,
        )
        dm.dataset = DummyClassificationDataset
        dm.shift_dataset = DummyClassificationDataset
        dm.setup()
        dm.get_val_set()
        dm.get_test_set()
        dm.get_shift_set()
        idx = dm.get_indices()
        assert idx["shift"] == [1]

        with pytest.raises(ValueError):
            CIFAR10DataModule(
                root="./data/",
                batch_size=128,
                cutout=8,
                num_dataloaders=2,
                val_split=0.1,
                auto_augment="rand-m9-n2-mstd0.5",
            )

        dm = CIFAR10DataModule(
            root="./data/",
            batch_size=128,
            test_alt="h",
        )
        with pytest.raises(ValueError, match="CIFAR-H can only be used in testing."):
            dm.setup("fit")

        with pytest.raises(ValueError, match="Test set "):
            CIFAR10DataModule(
                root="./data/",
                batch_size=128,
                test_alt="x",
            )

        CIFAR10DataModule(
            root="./data/",
            batch_size=128,
            num_dataloaders=2,
            val_split=0.1,
            auto_augment="rand-m9-n2-mstd0.5",
            num_tta=4,
        )

    def test_cifar10_cv(self) -> None:
        dm = CIFAR10DataModule(root="./data/", batch_size=128)
        dm.dataset = lambda root, train, download, transform: DummyClassificationDataset(
            root,
            train=train,
            download=download,
            transform=transform,
            num_images=20,
        )
        dm.make_cross_val_splits(2, 1)

        dm = CIFAR10DataModule(root="./data/", batch_size=128, val_split=0.1)
        dm.dataset = lambda root, train, download, transform: DummyClassificationDataset(
            root,
            train=train,
            download=download,
            transform=transform,
            num_images=20,
        )
        dm.make_cross_val_splits(2, 1)

    def test_ood_defaults_and_get_indices(self, monkeypatch) -> None:
        dm = CIFAR10DataModule(
            root="./data/",
            batch_size=16,
            train_transform=nn.Identity(),
            test_transform=nn.Identity(),
            eval_ood=True,
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
            "torch_uncertainty.datamodules.classification.cifar10.get_ood_datasets",
            _mock_get_ood,
        )
        dm.setup("test")

        assert hasattr(dm, "near_oods")
        assert len(dm.near_oods) == 2
        assert hasattr(dm, "far_oods")
        assert len(dm.far_oods) == 3

        for ds in [dm.val_ood, *dm.near_oods, *dm.far_oods]:
            assert hasattr(ds, "dataset_name")
            assert ds.dataset_name in {"dummy", ds.__class__.__name__.lower()}

        assert dm.near_ood_names == [ds.dataset_name for ds in dm.near_oods]
        assert dm.far_ood_names == [ds.dataset_name for ds in dm.far_oods]

        loaders = dm.test_dataloader()
        expected = 1 + 1 + 1 + len(dm.near_oods) + len(dm.far_oods)
        assert len(loaders) == expected

        idx = dm.get_indices()
        assert idx["test"] == [0]
        assert idx["test_ood"] == [1]
        assert idx["val_ood"] == [2]
        assert idx["near_oods"] == list(range(3, 3 + len(dm.near_oods)))
        assert idx["far_oods"] == list(
            range(3 + len(dm.near_oods), 3 + len(dm.near_oods) + len(dm.far_oods))
        )
        assert idx["shift"] == []

    def test_user_supplied_near_far_ood_typecheck_and_override(self, monkeypatch, tmp_path):
        """Covers user-provided OOD lists (type checks and overrides)."""
        # invalid near
        dm_bad_near = CIFAR10DataModule(
            root=tmp_path,
            batch_size=16,
            eval_ood=True,
            near_ood_datasets=[123],
        )
        dm_bad_near.dataset = DummyClassificationDataset
        dm_bad_near.shift_dataset = DummyClassificationDataset
        with pytest.raises(TypeError, match="near_ood_datasets must be Dataset objects"):
            dm_bad_near.setup("test")

        # invalid far
        dm_bad_far = CIFAR10DataModule(
            root=tmp_path,
            batch_size=16,
            eval_ood=True,
            far_ood_datasets=[object()],
        )
        dm_bad_far.dataset = DummyClassificationDataset
        dm_bad_far.shift_dataset = DummyClassificationDataset
        with pytest.raises(TypeError, match="far_ood_datasets must be Dataset objects"):
            dm_bad_far.setup("test")

        # valid override
        near_custom = [DummyClassificationDataset(root="./data/", num_images=2)]
        far_custom = [DummyClassificationDataset(root="./data/", num_images=1)]

        dm = CIFAR10DataModule(
            root=tmp_path,
            batch_size=16,
            eval_ood=True,
            near_ood_datasets=near_custom,
            far_ood_datasets=far_custom,
        )
        dm.dataset = DummyClassificationDataset
        dm.shift_dataset = DummyClassificationDataset

        def _fake_get_ood(**_):
            return (
                DummyClassificationDataset(root="./data/", num_images=1),
                DummyClassificationDataset(root="./data/", num_images=1),
                {"near_default": DummyClassificationDataset(root="./data/", num_images=1)},
                {"far_default": DummyClassificationDataset(root="./data/", num_images=1)},
            )

        monkeypatch.setattr(
            "torch_uncertainty.datamodules.classification.cifar10.get_ood_datasets",
            _fake_get_ood,
        )
        dm.setup("test")

        assert dm.near_oods is near_custom
        assert dm.far_oods is far_custom
        for ds in [dm.val_ood, *dm.near_oods, *dm.far_oods]:
            assert hasattr(ds, "dataset_name")

        idx = dm.get_indices()
        assert idx["test"] == [0]
        assert idx["test_ood"] == [1]
        assert idx["val_ood"] == [2]
        assert idx["near_oods"] == [3]
        assert idx["far_oods"] == [4]
        assert idx["shift"] == []
