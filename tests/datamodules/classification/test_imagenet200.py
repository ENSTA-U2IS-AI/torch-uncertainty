# tests/datamodules/test_imagenet200.py
from pathlib import Path

import pytest
from torch import nn

from tests._dummies.dataset import DummyClassificationDataset
from torch_uncertainty.datamodules import ImageNet200DataModule


class TestImageNet200DataModule:
    """Testing the ImageNet200DataModule datamodule class."""

    def test_imagenet200(self, monkeypatch) -> None:
        dm = ImageNet200DataModule(
            root="./data/",
            batch_size=128,
            val_split=0.1,
            basic_augment=False,
        )

        mod_name = ImageNet200DataModule.__module__
        fake_root = "./data/"

        def _fake_download_and_extract(name, dest_root):  # noqa: ARG001
            return str(fake_root)

        def _fake_download_and_extract_splits_from_hf(root):  # noqa: ARG001
            return Path(fake_root)

        def _fake_filelistdataset(root, list_file, transform):  # noqa: ARG001
            return DummyClassificationDataset(
                root=str(root),
                train=False,
                download=False,
                transform=transform,
                num_images=10,
            )

        monkeypatch.setattr(
            f"{mod_name}.download_and_extract_hf_dataset",
            _fake_download_and_extract,
            raising=True,
        )
        monkeypatch.setattr(
            f"{mod_name}.download_and_extract_splits_from_hf",
            _fake_download_and_extract_splits_from_hf,
            raising=True,
        )
        monkeypatch.setattr(
            f"{mod_name}.FileListDataset",
            _fake_filelistdataset,
            raising=True,
        )

        dm.dataset = DummyClassificationDataset

        dm.prepare_data()
        dm.setup("fit")
        dm.setup("test")

        _ = dm.val_dataloader()
        loaders = dm.test_dataloader()
        assert len(loaders) == 1

        path = Path(__file__).parent.resolve() / "../../assets/dummy_indices.yaml"
        dm = ImageNet200DataModule(
            root="./data/", batch_size=128, val_split=path, basic_augment=False
        )
        dm.setup("fit")
        dm.setup("test")
        _ = dm.val_dataloader()
        _ = dm.test_dataloader()

        dm.eval_shift = True
        dm.shift_dataset = DummyClassificationDataset
        dm.prepare_data()
        dm.setup("test")
        loaders = dm.test_dataloader()
        assert len(loaders) == 2

        ImageNet200DataModule(
            root="./data/",
            batch_size=128,
            val_split=path,
            rand_augment_opt="rand-m9-n1",
            basic_augment=False,
        )

        with pytest.raises(ValueError):
            dm.setup("other")

        for test_alt in ["r", "o", "a"]:
            _ = ImageNet200DataModule(
                root="./data/", batch_size=128, test_alt=test_alt, basic_augment=False
            )

        with pytest.raises(ValueError):
            _ = ImageNet200DataModule(root="./data/", batch_size=128, test_alt="x")

        for procedure in ["ViT", "A3"]:
            _ = ImageNet200DataModule(
                root="./data/", batch_size=128, procedure=procedure, basic_augment=False
            )

        with pytest.raises(ValueError):
            _ = ImageNet200DataModule(root="./data/", batch_size=128, procedure="A2")

        with pytest.raises(FileNotFoundError):
            dm._verify_splits(split="test")

        dm.root = Path("./tests/testlog")
        with pytest.raises(FileNotFoundError):
            dm._verify_splits(split="test")

    def test_ood_defaults_and_get_indices(self, monkeypatch) -> None:
        dm = ImageNet200DataModule(
            root="./data/",
            batch_size=16,
            eval_ood=True,
            eval_shift=True,
            basic_augment=False,
        )

        mod_name = ImageNet200DataModule.__module__
        fake_root = "./data/"

        def _fake_download_and_extract(name, dest_root):  # noqa: ARG001
            return str(fake_root)

        def _fake_download_and_extract_splits_from_hf(root):  # noqa: ARG001
            return Path(fake_root)

        def _fake_filelistdataset(root, list_file, transform):  # noqa: ARG001
            return DummyClassificationDataset(
                root=str(root),
                train=False,
                download=False,
                transform=transform,
                num_images=10,
            )

        # Return OOD sets + defaults
        def _mock_get_ood_datasets(**_):
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
            f"{mod_name}.download_and_extract_hf_dataset",
            _fake_download_and_extract,
            raising=True,
        )
        monkeypatch.setattr(
            f"{mod_name}.download_and_extract_splits_from_hf",
            _fake_download_and_extract_splits_from_hf,
            raising=True,
        )
        monkeypatch.setattr(
            f"{mod_name}.FileListDataset",
            _fake_filelistdataset,
            raising=True,
        )
        monkeypatch.setattr(f"{mod_name}.get_ood_datasets", _mock_get_ood_datasets, raising=True)

        dm.dataset = DummyClassificationDataset
        dm.shift_dataset = DummyClassificationDataset

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
