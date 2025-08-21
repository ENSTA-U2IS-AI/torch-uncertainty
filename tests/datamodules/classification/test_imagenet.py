from pathlib import Path

import pytest
from torch import nn

from tests._dummies.dataset import DummyClassificationDataset
from torch_uncertainty.datamodules import ImageNetDataModule


class TestImageNetDataModule:
    """Testing the ImageNetDataModule datamodule class."""

    def test_imagenet(self, monkeypatch) -> None:
        dm = ImageNetDataModule(
            root="./data/",
            batch_size=128,
            val_split=0.1,
            train_transform=nn.Identity(),
            test_transform=nn.Identity(),
            num_tta=2,
        )
        dm = ImageNetDataModule(
            root="./data/",
            batch_size=128,
            val_split=0.1,
            train_transform=nn.Identity(),
            test_transform=nn.Identity(),
        )
        assert isinstance(dm.train_transform, nn.Identity)
        assert isinstance(dm.test_transform, nn.Identity)
        dm.dataset = DummyClassificationDataset

        fake_root = "./data/"
        mod_name = ImageNetDataModule.__module__

        def _fake_download_and_extract(name, dest_root):  # noqa: ARG001
            return str(fake_root)

        monkeypatch.setattr(
            f"{mod_name}.download_and_extract_hf_dataset",
            _fake_download_and_extract,
            raising=True,
        )

        dm.prepare_data()
        dm.setup()

        path = Path(__file__).parent.resolve() / "../../assets/dummy_indices.yaml"
        dm = ImageNetDataModule(root="./data/", batch_size=128, val_split=path)
        dm.shift_dataset = DummyClassificationDataset
        dm.setup("fit")
        dm.setup("test")
        dm.val_dataloader()
        dm.test_dataloader()

        dm.val_split = None
        dm.setup("fit")
        dm.val_dataloader()
        dm.test_dataloader()

        dm.eval_shift = True
        dm.prepare_data()
        dm.setup("test")
        dm.test_dataloader()

        ImageNetDataModule(
            root="./data/",
            batch_size=128,
            val_split=path,
            rand_augment_opt="rand-m9-n1",
            basic_augment=False,
        )

        with pytest.raises(ValueError):
            dm.setup("other")

        for test_alt in ["r", "o", "a"]:
            dm = ImageNetDataModule(root="./data/", batch_size=128, test_alt=test_alt)

        with pytest.raises(ValueError):
            dm = ImageNetDataModule(root="./data/", batch_size=128, test_alt="x")

        for procedure in ["ViT", "A3"]:
            dm = ImageNetDataModule(
                root="./data/",
                batch_size=128,
                procedure=procedure,
            )

        with pytest.raises(ValueError):
            dm = ImageNetDataModule(root="./data/", batch_size=128, procedure="A2")

        with pytest.raises(FileNotFoundError):
            dm._verify_splits(split="test")

        dm.root = Path("./tests/testlog")
        with pytest.raises(FileNotFoundError):
            dm._verify_splits(split="test")

    def test_ood_defaults_and_get_indices(self, monkeypatch) -> None:
        dm = ImageNetDataModule(
            root="./data/",
            batch_size=16,
            train_transform=nn.Identity(),
            test_transform=nn.Identity(),
            eval_ood=True,
            eval_shift=True,
        )

        fake_root = "./data/"
        mod_name = ImageNetDataModule.__module__

        def _fake_download_and_extract(name, dest_root):  # noqa: ARG001
            return str(fake_root)

        monkeypatch.setattr(
            f"{mod_name}.download_and_extract_hf_dataset",
            _fake_download_and_extract,
            raising=True,
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
            "torch_uncertainty.datamodules.classification.imagenet.get_ood_datasets",
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
