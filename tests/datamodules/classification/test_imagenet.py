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

    def test_guard_test_alt_with_eval_ood(self):
        with pytest.raises(ValueError, match="test_alt.*not supported.*ood_eval"):
            ImageNetDataModule(
                root="./data/",
                batch_size=8,
                test_alt="r",
                eval_ood=True,
            )

    def test_near_far_instances_used_and_named(self, monkeypatch, tmp_path):
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
        monkeypatch.setattr(
            f"{mod_name}.FileListDataset",
            self._DummyFileListDataset,
            raising=True,
        )
        monkeypatch.setattr(
            "torch_uncertainty.datamodules.classification.imagenet.get_ood_datasets",
            self._fake_get_ood_datasets,
        )

        class NearDS(Dataset):
            def __len__(self):
                return 1

            def __getitem__(self, i):
                return torch.zeros(3, 224, 224), 0

        class FarDS(Dataset):
            def __len__(self):
                return 1

            def __getitem__(self, i):
                return torch.zeros(3, 224, 224), 0

        near_list = [NearDS(), NearDS()]
        far_list = [FarDS()]

        dm = ImageNetDataModule(
            root=tmp_path,
            batch_size=8,
            eval_ood=True,
            train_transform=nn.Identity(),
            test_transform=nn.Identity(),
            near_ood_datasets=near_list,
            far_ood_datasets=far_list,
        )
        dm.setup("test")

        assert dm.near_oods is near_list
        assert dm.far_oods is far_list
        assert all(hasattr(ds, "dataset_name") for ds in dm.near_oods)
        assert all(hasattr(ds, "dataset_name") for ds in dm.far_oods)
        assert {ds.dataset_name for ds in dm.near_oods} == {"neards"}
        assert {ds.dataset_name for ds in dm.far_oods} == {"fards"}

    def test_near_far_type_errors(self, monkeypatch, tmp_path):
        # Avoid split file access so we can reach the TypeError branches
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
        monkeypatch.setattr(
            f"{mod_name}.FileListDataset",
            self._DummyFileListDataset,
            raising=True,
        )
        monkeypatch.setattr(
            "torch_uncertainty.datamodules.classification.imagenet.get_ood_datasets",
            self._fake_get_ood_datasets,
        )

        dm_bad_near = ImageNetDataModule(
            root=tmp_path,
            batch_size=8,
            eval_ood=True,
            near_ood_datasets=[123],
            test_transform=nn.Identity(),
        )
        with pytest.raises(TypeError, match="near_ood_datasets.*Dataset"):
            dm_bad_near.setup("test")

        dm_bad_far = ImageNetDataModule(
            root=tmp_path,
            batch_size=8,
            eval_ood=True,
            far_ood_datasets=["nope"],
            test_transform=nn.Identity(),
        )
        with pytest.raises(TypeError, match="far_ood_datasets.*Dataset"):
            dm_bad_far.setup("test")

    def test_train_dataloader_success_and_missing(self, monkeypatch, tmp_path):
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
        monkeypatch.setattr(f"{mod_name}.ImageFolder", self._DummyImageFolder, raising=True)

        dm = ImageNetDataModule(
            root=tmp_path,
            batch_size=4,
            train_transform=nn.Identity(),
            test_transform=nn.Identity(),
        )
        dm.data_dir = str(tmp_path)
        (tmp_path / "train").mkdir(parents=True, exist_ok=True)
        loader = dm.train_dataloader()
        batch = next(iter(loader))
        assert isinstance(batch, list | tuple)
        assert len(batch) == 2

        dm2 = ImageNetDataModule(
            root=tmp_path / "other",
            batch_size=4,
            train_transform=nn.Identity(),
            test_transform=nn.Identity(),
        )
        dm2.data_dir = str(tmp_path / "no_train_here")
        with pytest.raises(RuntimeError, match="ImageNet training data not found"):
            dm2.train_dataloader()

    def test_get_indices_without_ood_or_shift(self, tmp_path):
        dm = ImageNetDataModule(
            root=tmp_path,
            batch_size=8,
            eval_ood=False,
            eval_shift=False,
        )
        idx = dm.get_indices()
        assert idx["test"] == [0]
        assert idx["test_ood"] == []
        assert idx["val_ood"] == []
        assert idx["near_oods"] == []
        assert idx["far_oods"] == []
        assert idx["shift"] == []

    def test_tta_wraps_ood_sets(self, monkeypatch, tmp_path):
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
        monkeypatch.setattr(
            f"{mod_name}.FileListDataset",
            self._DummyFileListDataset,
            raising=True,
        )
        monkeypatch.setattr(
            "torch_uncertainty.datamodules.classification.imagenet.get_ood_datasets",
            self._fake_get_ood_datasets,
        )

        dm = ImageNetDataModule(
            root=tmp_path,
            batch_size=8,
            eval_ood=True,
            num_tta=2,
            train_transform=nn.Identity(),
            test_transform=nn.Identity(),
            num_workers=0,
            persistent_workers=False,
            pin_memory=False,
        )

        dm.setup("test")

        val_ood_wrapped = dm.get_val_ood_set()
        test_ood_wrapped = dm.get_test_ood_set()
        near_wrapped = dm.get_near_ood_set()
        far_wrapped = dm.get_far_ood_set()

        assert len(val_ood_wrapped) == len(dm.val_ood) * dm.num_tta
        assert len(test_ood_wrapped) == len(dm.test_ood) * dm.num_tta
        assert all(
            len(w) == len(b) * dm.num_tta for w, b in zip(near_wrapped, dm.near_oods, strict=False)
        )
        assert all(
            len(w) == len(b) * dm.num_tta for w, b in zip(far_wrapped, dm.far_oods, strict=False)
        )

        def _assert_first_block_repeat(wrapped_ds, num_tta: int):
            if len(wrapped_ds) == 0 or num_tta < 2:
                return
            x0, y0 = wrapped_ds[0]
            x1, y1 = wrapped_ds[1]
            assert y0 == y1
            if torch.is_tensor(x0) and torch.is_tensor(x1):
                assert x0.shape == x1.shape
            else:
                assert type(x0) is type(x1)
                if hasattr(x0, "size") and hasattr(x1, "size"):
                    assert x0.size == x1.size

        _assert_first_block_repeat(val_ood_wrapped, dm.num_tta)
        _assert_first_block_repeat(test_ood_wrapped, dm.num_tta)
        for w in near_wrapped:
            _assert_first_block_repeat(w, dm.num_tta)
        for w in far_wrapped:
            _assert_first_block_repeat(w, dm.num_tta)
