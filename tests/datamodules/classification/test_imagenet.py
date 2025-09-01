from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from tests._dummies.dataset import DummyClassificationDataset
from torch_uncertainty.datamodules import ImageNetDataModule


class TestImageNetDataModule:
    """Testing the ImageNetDataModule datamodule class."""

    class _TinyImgDataset(Dataset):
        def __init__(self, n=3):
            self.n = n

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            return torch.zeros(3, 224, 224), 0

    class _DummyFileListDataset(Dataset):
        def __init__(self, root, list_file, transform):
            self.ds = TestImageNetDataModule._TinyImgDataset()

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, i):
            return self.ds[i]

    class _DummyImageFolder(Dataset):
        def __init__(self, root, transform=None):
            self.ds = TestImageNetDataModule._TinyImgDataset()
            self.transform = transform

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, i):
            x, y = self.ds[i]
            return (self.transform(x) if self.transform is not None else x), y

    @staticmethod
    def _fake_download_and_extract(_name, dest_root):
        return str(dest_root)

    @staticmethod
    def _fake_download_and_extract_splits_from_hf(root):
        return Path(root)

    @staticmethod
    def _fake_get_ood_datasets(**_):
        test_ood = DummyClassificationDataset(root="./data/", num_images=2)
        val_ood = DummyClassificationDataset(root="./data/", num_images=2)
        near_default = {
            "near1": DummyClassificationDataset(root="./data/", num_images=1),
            "near2": DummyClassificationDataset(root="./data/", num_images=1),
        }
        far_default = {
            "far1": DummyClassificationDataset(root="./data/", num_images=1),
            "far2": DummyClassificationDataset(root="./data/", num_images=1),
            "far3": DummyClassificationDataset(root="./data/", num_images=1),
        }
        return test_ood, val_ood, near_default, far_default

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

        monkeypatch.setattr(
            "torch_uncertainty.datamodules.classification.imagenet.get_ood_datasets",
            self._fake_get_ood_datasets,
        )
        dm.setup("test")

        assert hasattr(dm, "near_oods")
        assert len(dm.near_oods) == 2

        assert hasattr(dm, "far_oods")
        assert len(dm.far_oods) == 3

        for ds in [dm.val_ood, *dm.near_oods, *dm.far_oods]:
            assert hasattr(ds, "dataset_name")

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

    def test_setup_fit_rejects_test_alt(self, monkeypatch, tmp_path):
        """setup('fit') must raise when test_alt is provided."""
        mod_name = ImageNetDataModule.__module__
        monkeypatch.setattr(
            f"{mod_name}.download_and_extract_hf_dataset",
            self._fake_download_and_extract,
            raising=True,
        )
        monkeypatch.setattr(
            f"{mod_name}.download_and_extract_splits_from_hf",
            self._fake_download_and_extract_splits_from_hf,
            raising=True,
        )
        monkeypatch.setattr(f"{mod_name}.ImageNetR", DummyClassificationDataset, raising=True)

        dm = ImageNetDataModule(
            root=tmp_path,
            batch_size=8,
            test_alt="r",
            basic_augment=False,
            num_workers=0,
            persistent_workers=False,
            pin_memory=False,
        )
        with pytest.raises(ValueError, match="test_alt.*not supported for training"):
            dm.setup("fit")

    def test_setup_test_with_test_alt(self, monkeypatch, tmp_path):
        """setup('test') with test_alt uses alt dataset constructor."""
        mod_name = ImageNetDataModule.__module__
        monkeypatch.setattr(f"{mod_name}.ImageNetR", DummyClassificationDataset, raising=True)
        monkeypatch.setattr(
            f"{mod_name}.download_and_extract_hf_dataset",
            self._fake_download_and_extract,
            raising=True,
        )
        monkeypatch.setattr(
            f"{mod_name}.download_and_extract_splits_from_hf",
            self._fake_download_and_extract_splits_from_hf,
            raising=True,
        )

        dm = ImageNetDataModule(
            root=tmp_path,
            batch_size=8,
            test_alt="r",
            basic_augment=False,
            num_workers=0,
            persistent_workers=False,
            pin_memory=False,
        )
        dm.setup("test")
        assert isinstance(dm.test, DummyClassificationDataset)
