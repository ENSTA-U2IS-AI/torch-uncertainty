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
        test_ood = TestImageNetDataModule._TinyImgDataset(2)
        val_ood = TestImageNetDataModule._TinyImgDataset(2)
        near_default = {
            "near1": TestImageNetDataModule._TinyImgDataset(1),
            "near2": TestImageNetDataModule._TinyImgDataset(1),
        }
        far_default = {
            "far1": TestImageNetDataModule._TinyImgDataset(1),
            "far2": TestImageNetDataModule._TinyImgDataset(1),
            "far3": TestImageNetDataModule._TinyImgDataset(1),
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

    def test_get_indices_no_ood_and_train_dataloader(self, monkeypatch, tmp_path):
        """Covers get_indices() when eval_ood=False and train_dataloader success/failure."""
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
        monkeypatch.setattr(f"{mod_name}.FileListDataset", self._DummyFileListDataset, raising=True)
        monkeypatch.setattr(f"{mod_name}.ImageFolder", self._DummyImageFolder, raising=True)

        dm = ImageNetDataModule(
            root=tmp_path,
            batch_size=8,
            eval_ood=False,
            eval_shift=False,
            basic_augment=False,
            num_workers=0,
            persistent_workers=False,
            pin_memory=False,
        )

        dm.prepare_data()
        dm.setup("fit")
        dm.setup("test")

        loaders = dm.test_dataloader()
        assert isinstance(loaders, list)
        assert len(loaders) == 1

        idx = dm.get_indices()
        assert idx["test"] == [0]
        assert idx["test_ood"] == []
        assert idx["val_ood"] == []
        assert idx["near_oods"] == []
        assert idx["far_oods"] == []
        assert idx["shift"] == []

        data_dir = tmp_path / "imagenet_fake"
        (data_dir / "train").mkdir(parents=True, exist_ok=True)
        dm.data_dir = str(data_dir)

        batch = next(iter(dm.train_dataloader()))
        assert isinstance(batch, list | tuple)
        assert len(batch) == 2

        x, y = batch
        assert torch.is_tensor(x)
        assert x.ndim == 4
        assert torch.is_tensor(y)
        assert y.ndim in (0, 1)

        dm.data_dir = str(tmp_path / "no_train_here")
        with pytest.raises(RuntimeError, match="ImageNet training data not found"):
            dm.train_dataloader()

    def test_user_supplied_near_far_ood_typecheck_and_override(self, monkeypatch, tmp_path):
        """Covers user-provided near/far OOD lists: type checks & override."""
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
        monkeypatch.setattr(f"{mod_name}.FileListDataset", self._DummyFileListDataset, raising=True)
        monkeypatch.setattr(
            f"{mod_name}.get_ood_datasets", self._fake_get_ood_datasets, raising=True
        )

        dm_bad_near = ImageNetDataModule(
            root=tmp_path,
            batch_size=8,
            eval_ood=True,
            basic_augment=False,
            near_ood_datasets=[123, "nope"],
            num_workers=0,
            persistent_workers=False,
            pin_memory=False,
        )
        with pytest.raises(TypeError, match="near_ood_datasets must be Dataset objects"):
            dm_bad_near.setup("test")

        dm_bad_far = ImageNetDataModule(
            root=tmp_path,
            batch_size=8,
            eval_ood=True,
            basic_augment=False,
            far_ood_datasets=[object()],
            num_workers=0,
            persistent_workers=False,
            pin_memory=False,
        )
        with pytest.raises(TypeError, match="far_ood_datasets must be Dataset objects"):
            dm_bad_far.setup("test")

        near_custom = [self._TinyImgDataset(2), self._TinyImgDataset(3)]
        far_custom = [self._TinyImgDataset(1)]

        dm = ImageNetDataModule(
            root=tmp_path,
            batch_size=8,
            eval_ood=True,
            basic_augment=False,
            near_ood_datasets=near_custom,
            far_ood_datasets=far_custom,
            num_workers=0,
            persistent_workers=False,
            pin_memory=False,
        )
        dm.setup("test")

        assert hasattr(dm, "near_oods")
        assert dm.near_oods is near_custom
        assert hasattr(dm, "far_oods")
        assert dm.far_oods is far_custom

        for ds in [dm.val_ood, *dm.near_oods, *dm.far_oods]:
            assert hasattr(ds, "dataset_name")
            assert isinstance(ds.dataset_name, str)
            assert ds.dataset_name

        loaders = dm.test_dataloader()
        expected = 1 + 1 + 1 + len(near_custom) + len(far_custom)
        assert isinstance(loaders, list)
        assert len(loaders) == expected

        idx = dm.get_indices()
        assert idx["test"] == [0]
        assert idx["test_ood"] == [1]
        assert idx["val_ood"] == [2]
        assert idx["near_oods"] == list(range(3, 3 + len(near_custom)))
        assert idx["far_oods"] == list(
            range(3 + len(near_custom), 3 + len(near_custom) + len(far_custom))
        )
        assert idx["shift"] == []
