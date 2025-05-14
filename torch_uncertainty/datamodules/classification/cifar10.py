import logging
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from numpy.typing import ArrayLike
from timm.data.auto_augment import rand_augment_transform
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2

from torch_uncertainty.datamodules.abstract import TUDataModule
from torch_uncertainty.datasets import AggregatedDataset
from torch_uncertainty.datasets.classification import CIFAR10C, CIFAR10H
from torch_uncertainty.datasets.ood.utils import get_ood_datasets
from torch_uncertainty.transforms import Cutout
from torch_uncertainty.utils import create_train_val_split

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logging.getLogger("faiss").setLevel(logging.WARNING)


class CIFAR10DataModule(TUDataModule):
    num_classes = 10
    num_channels = 3
    input_shape = (3, 32, 32)
    training_task = "classification"
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    def __init__(
        self,
        root: str | Path,
        batch_size: int,
        eval_batch_size: int | None = None,
        eval_ood: bool = False,
        eval_shift: bool = False,
        shift_severity: int = 1,
        val_split: float | None = None,
        postprocess_set: Literal["val", "test"] = "val",
        num_workers: int = 1,
        basic_augment: bool = True,
        cutout: int | None = None,
        auto_augment: str | None = None,
        test_alt: Literal["h"] | None = None,
        num_dataloaders: int = 1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        near_ood_datasets: list | None = None,
        far_ood_datasets: list | None = None,
    ) -> None:
        """DataModule for CIFAR10.

        Args:
            root (str | Path): Root directory of the datasets.
            batch_size (int): Number of samples per batch during training.
            eval_batch_size (int | None) : Number of samples per batch during evaluation (val
                and test). Set to batch_size if None. Defaults to None.
            eval_ood (bool): Whether to evaluate on out-of-distribution data. Defaults to ``False``.
            near_ood_datasets (list, optional): list of near OOD dataset classes must be subclass of torch.utils.data.Dataset. Defaults to CIFAR-100, Tiny ImageNet (OpenOOD splits)
            far_ood_datasets (list, optional): list of far OOD dataset classes must be subclass of torch.utils.data.Dataset. Defaults to MNIST, SVHN, Textures, Places365 (OpenOOD splits)
            eval_shift (bool): Whether to evaluate on shifted data. Defaults to ``False``.
            val_split (float): Share of samples to use for validation. Defaults
                to ``0.0``.
            postprocess_set (str, optional): The post-hoc calibration dataset to
                use for the post-processing method. Defaults to ``val``.
            num_workers (int): Number of workers to use for data loading. Defaults
                to ``1``.
            basic_augment (bool): Whether to apply base augmentations. Defaults to
                ``True``.
            cutout (int): Size of cutout to apply to images. Defaults to ``None``.
            randaugment (bool): Whether to apply RandAugment. Defaults to
                ``False``.
            auto_augment (str): Which auto-augment to apply. Defaults to ``None``.
            test_alt (str): Which test set to use. Defaults to ``None``.
            shift_severity (int): Severity of corruption to apply for
                CIFAR10-C. Defaults to ``1``.
            num_dataloaders (int): Number of dataloaders to use. Defaults to ``1``.
            pin_memory (bool): Whether to pin memory. Defaults to ``True``.
            persistent_workers (bool): Whether to use persistent workers. Defaults
                to ``True``.
        """
        super().__init__(
            root=root,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            val_split=val_split,
            postprocess_set=postprocess_set,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        self.num_dataloaders = num_dataloaders
        self.eval_ood = eval_ood
        self.eval_shift = eval_shift

        if test_alt == "h":
            self.dataset = CIFAR10H
        elif test_alt is None:
            self.dataset = CIFAR10
        else:
            raise ValueError(f"Test set {test_alt} is not supported.")

        self.test_alt = test_alt
        self.shift_severity = shift_severity
        self.shift_dataset = CIFAR10C

        self.near_ood_datasets = near_ood_datasets or []  # List of near OOD dataset classes
        self.far_ood_datasets = far_ood_datasets or []  # List of far OOD dataset classes

        if (cutout is not None) + int(auto_augment is not None) > 1:
            raise ValueError(
                "Only one data augmentation can be chosen at a time. Raise a "
                "GitHub issue if needed."
            )

        if basic_augment:
            basic_transform = v2.Compose(
                [
                    v2.RandomCrop(32, padding=4),
                    v2.RandomHorizontalFlip(),
                ]
            )
        else:
            basic_transform = nn.Identity()

        if cutout:
            main_transform = Cutout(cutout)
        elif auto_augment:
            main_transform = rand_augment_transform(auto_augment, {})
        else:
            main_transform = nn.Identity()

        self.train_transform = v2.Compose(
            [
                v2.ToImage(),
                basic_transform,
                main_transform,
                v2.ToDtype(dtype=torch.float32, scale=True),
                v2.Normalize(mean=self.mean, std=self.std),
            ]
        )

        self.test_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(32),
                v2.CenterCrop(32),
                v2.ToDtype(dtype=torch.float32, scale=True),
                v2.Normalize(mean=self.mean, std=self.std),
            ]
        )

    def prepare_data(self) -> None:  # coverage: ignore
        if self.test_alt is None:
            self.dataset(self.root, train=True, download=True)
            self.dataset(self.root, train=False, download=True)
        else:
            self.dataset(
                self.root,
                download=True,
            )

        if self.eval_shift:
            self.shift_dataset(
                self.root,
                shift_severity=self.shift_severity,
                download=True,
            )

    def setup(self, stage: Literal["fit", "test"] | None = None) -> None:
        if stage == "fit" or stage is None:
            if self.test_alt == "h":
                raise ValueError("CIFAR-H can only be used in testing.")
            full = self.dataset(
                self.root,
                train=True,
                download=False,
                transform=self.train_transform,
            )
            if self.val_split:
                self.train, self.val = create_train_val_split(
                    full,
                    self.val_split,
                    self.test_transform,
                )
            else:
                self.train = full
                self.val = self.dataset(
                    self.root,
                    train=False,
                    download=False,
                    transform=self.test_transform,
                )
        if stage == "test" or stage is None:
            if self.test_alt is None:
                self.test = self.dataset(
                    self.root,
                    train=False,
                    download=False,
                    transform=self.test_transform,
                )
            else:
                self.test = self.dataset(
                    self.root,
                    transform=self.test_transform,
                    shift_severity=self.shift_severity,
                )

            if self.eval_ood:
                self.val_ood, near_default, far_default = get_ood_datasets(
                    root=self.root,
                    dataset_id="CIFAR10",
                    transform=self.test_transform,
                )

                if self.near_ood_datasets:
                    if not all(isinstance(ds, Dataset) for ds in self.near_ood_datasets):
                        raise TypeError("All entries in near_ood_datasets must be Dataset objects")
                    self.near_oods = self.near_ood_datasets
                else:
                    self.near_oods = list(near_default.values())

                if self.far_ood_datasets:
                    if not all(isinstance(ds, Dataset) for ds in self.far_ood_datasets):
                        raise TypeError("All entries in far_ood_datasets must be Dataset objects")
                    self.far_oods = self.far_ood_datasets
                else:
                    self.far_oods = list(far_default.values())

                for ds in [self.val_ood, *self.near_oods, *self.far_oods]:
                    if not hasattr(ds, "dataset_name"):
                        ds.dataset_name = ds.__class__.__name__.lower()

                self.near_ood_names = [ds.dataset_name for ds in self.near_oods]
                self.far_ood_names = [ds.dataset_name for ds in self.far_oods]

        if stage not in ["fit", "test", None]:
            raise ValueError(f"Stage {stage} is not supported.")

    def train_dataloader(self) -> DataLoader:
        r"""Get the training dataloader for CIFAR10.

        Return:
            DataLoader: CIFAR10 training dataloader.
        """
        if self.num_dataloaders > 1:
            return self._data_loader(
                AggregatedDataset(self.train, self.num_dataloaders),
                shuffle=True,
                training=True,
            )
        return self._data_loader(self.train, training=True, shuffle=True)

    def test_dataloader(self):
        loaders = [self._data_loader(self.test, training=False)]
        if self.eval_ood:
            loaders.append(self._data_loader(self.val_ood, training=False))

            loaders.extend(self._data_loader(ds, training=False) for ds in self.near_oods)

            loaders.extend(self._data_loader(ds, training=False) for ds in self.far_oods)

        if self.eval_shift:
            loaders.append(self._data_loader(self.shift_dataset, training=False))
        return loaders

    def _get_train_data(self) -> ArrayLike:
        if self.val_split:
            return self.train.dataset.data[self.train.indices]
        return self.train.data

    def _get_train_targets(self) -> ArrayLike:
        if self.val_split:
            return np.array(self.train.dataset.targets)[self.train.indices]
        return np.array(self.train.targets)

    def get_indices(self):
        idx = 0
        indices = {}
        indices["test"] = [idx]
        idx += 1
        if self.eval_ood:
            indices["val_ood"] = [idx]
            idx += 1
            n_near = len(self.near_oods)
            indices["near_oods"] = list(range(idx, idx + n_near))
            idx += n_near
            n_far = len(self.far_oods)
            indices["far_oods"] = list(range(idx, idx + n_far))
            idx += n_far
        else:
            indices["val_ood"] = []
            indices["near_oods"] = []
            indices["far_oods"] = []
        if self.eval_shift:
            indices["shift"] = [idx]
        else:
            indices["shift"] = []
        return indices
