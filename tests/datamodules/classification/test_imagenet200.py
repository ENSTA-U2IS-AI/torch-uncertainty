from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from torch_uncertainty.datamodules import ImageNet200DataModule


class TinyImgDataset(Dataset):
    """A tiny image dataset returning zeros."""

    def __init__(self, n=3):
        """Initialize dataset with n samples."""
        self.n = n

    def __len__(self):
        """Return the number of samples."""
        return self.n

    def __getitem__(self, idx):
        """Return a (C,H,W) tensor and a label."""
        return torch.zeros(3, 224, 224), 0


class DummyFileListDataset(Dataset):
    """Stand-in for FileListDataset."""

    def __init__(self, root, list_file, transform):
        """Initialize with a fixed TinyImgDataset backend."""
        self.ds = TinyImgDataset()

    def __len__(self):
        """Return the number of samples."""
        return len(self.ds)

    def __getitem__(self, i):
        """Return a sample from the inner dataset."""
        return self.ds[i]


class DummyImageFolder(Dataset):
    """Stand-in for torchvision.datasets.ImageFolder."""

    def __init__(self, root, transform=None):
        """Initialize with a fixed TinyImgDataset backend and optional transform."""
        self.ds = TinyImgDataset()
        self.transform = transform

    def __len__(self):
        """Return the number of samples."""
        return len(self.ds)

    def __getitem__(self, i):
        """Return (image, label), applying transform if provided."""
        x, y = self.ds[i]
        return (self.transform(x) if self.transform is not None else x), y


def _fake_download_and_extract(name, dest_root):  # noqa: ARG001
    """Return the destination root unchanged (avoid I/O)."""
    return str(dest_root)


def _fake_download_and_extract_splits_from_hf(root):
    """Return a Path to the provided root (avoid I/O)."""
    return Path(root)


def _fake_get_ood_datasets(**_):
    """Return tiny OOD datasets and minimal defaults (no I/O)."""
    test_ood = TinyImgDataset(2)
    val_ood = TinyImgDataset(2)
    near_default = {
        "near1": TinyImgDataset(1),
        "near2": TinyImgDataset(1),
    }
    far_default = {
        "far1": TinyImgDataset(1),
        "far2": TinyImgDataset(1),
        "far3": TinyImgDataset(1),
    }
    return test_ood, val_ood, near_default, far_default


def test_get_indices_no_ood_and_test_dataloader(monkeypatch, tmp_path):
    """Covers get_indices() mapping when eval_ood=False and basic test loader count."""
    mod = ImageNet200DataModule.__module__

    monkeypatch.setattr(
        f"{mod}.download_and_extract_hf_dataset", _fake_download_and_extract, raising=True
    )
    monkeypatch.setattr(
        f"{mod}.download_and_extract_splits_from_hf",
        _fake_download_and_extract_splits_from_hf,
        raising=True,
    )
    monkeypatch.setattr(f"{mod}.FileListDataset", DummyFileListDataset, raising=True)

    dm = ImageNet200DataModule(
        root=tmp_path,
        batch_size=32,
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
    assert "test" in idx
    assert idx["test"] == [0]
    assert idx["test_ood"] == []
    assert idx["val_ood"] == []
    assert idx["near_oods"] == []
    assert idx["far_oods"] == []
    assert idx["shift"] == []


def test_train_dataloader_success_and_failure(monkeypatch, tmp_path):
    """Cover train_dataloader() success with a fake train/ dir and failure when missing."""
    mod = ImageNet200DataModule.__module__

    monkeypatch.setattr(
        f"{mod}.download_and_extract_hf_dataset", _fake_download_and_extract, raising=True
    )
    monkeypatch.setattr(
        f"{mod}.download_and_extract_splits_from_hf",
        _fake_download_and_extract_splits_from_hf,
        raising=True,
    )
    monkeypatch.setattr(f"{mod}.FileListDataset", DummyFileListDataset, raising=True)
    monkeypatch.setattr(f"{mod}.ImageFolder", DummyImageFolder, raising=True)

    dm = ImageNet200DataModule(
        root=tmp_path,
        batch_size=8,
        basic_augment=False,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )

    dm.prepare_data()
    dm.setup("fit")
    dm.setup("test")

    data_dir = tmp_path / "imagenet_fake"
    train_dir = data_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    dm.data_dir = str(data_dir)

    loader = dm.train_dataloader()
    batch = next(iter(loader))

    assert isinstance(batch, list | tuple)
    assert len(batch) == 2

    x, y = batch[0], batch[1]

    assert torch.is_tensor(x)
    assert x.ndim == 4
    assert torch.is_tensor(y)
    assert y.ndim in (0, 1)

    dm.data_dir = str(tmp_path / "no_train_here")
    with pytest.raises(RuntimeError, match="ImageNet training data not found"):
        dm.train_dataloader()


def test_user_supplied_near_far_ood_instances_and_typecheck(monkeypatch, tmp_path):
    """Exercise user-provided OOD lists: type checks and successful overrides."""
    mod = ImageNet200DataModule.__module__

    monkeypatch.setattr(
        f"{mod}.download_and_extract_hf_dataset", _fake_download_and_extract, raising=True
    )
    monkeypatch.setattr(
        f"{mod}.download_and_extract_splits_from_hf",
        _fake_download_and_extract_splits_from_hf,
        raising=True,
    )
    monkeypatch.setattr(f"{mod}.FileListDataset", DummyFileListDataset, raising=True)
    monkeypatch.setattr(f"{mod}.get_ood_datasets", _fake_get_ood_datasets, raising=True)

    dm_bad = ImageNet200DataModule(
        root=tmp_path,
        batch_size=16,
        eval_ood=True,
        basic_augment=False,
        near_ood_datasets=[123, "not a dataset"],  # invalid
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )
    with pytest.raises(TypeError, match="near_ood_datasets must be Dataset objects"):
        dm_bad.setup("test")

    dm_bad2 = ImageNet200DataModule(
        root=tmp_path,
        batch_size=16,
        eval_ood=True,
        basic_augment=False,
        far_ood_datasets=[object()],
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )
    with pytest.raises(TypeError, match="far_ood_datasets must be Dataset objects"):
        dm_bad2.setup("test")

    near_custom = [TinyImgDataset(2), TinyImgDataset(3)]
    far_custom = [TinyImgDataset(1)]

    dm = ImageNet200DataModule(
        root=tmp_path,
        batch_size=16,
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

    loaders = dm.test_dataloader()
    expected = (
        1 + 1 + 1 + len(near_custom) + len(far_custom)
    )  # ID + test_ood + val_ood + near + far
    assert isinstance(loaders, list)
    assert len(loaders) == expected

    idx = dm.get_indices()
    assert idx["test"] == [0]
    assert idx["test_ood"] == [1]
    assert idx["val_ood"] == [2]
    start_near = 3
    assert idx["near_oods"] == list(range(start_near, start_near + len(near_custom)))
    start_far = start_near + len(near_custom)
    assert idx["far_oods"] == list(range(start_far, start_far + len(far_custom)))
    assert idx["shift"] == []
