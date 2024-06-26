from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torchvision.transforms as T
from numpy.typing import ArrayLike
from timm.data.auto_augment import rand_augment_transform
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100, SVHN

from torch_uncertainty.datamodules.abstract import TUDataModule
from torch_uncertainty.datasets import AggregatedDataset
from torch_uncertainty.datasets.classification import CIFAR100C
from torch_uncertainty.transforms import Cutout
from torch_uncertainty.utils import create_train_val_split


class CIFAR100DataModule(TUDataModule):
    num_classes = 100
    num_channels = 3
    input_shape = (3, 32, 32)
    training_task = "classification"

    def __init__(
        self,
        root: str | Path,
        batch_size: int,
        eval_ood: bool = False,
        val_split: float | None = None,
        cutout: int | None = None,
        randaugment: bool = False,
        auto_augment: str | None = None,
        test_alt: Literal["c"] | None = None,
        corruption_severity: int = 1,
        num_dataloaders: int = 1,
        num_workers: int = 1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ) -> None:
        """DataModule for CIFAR100.

        Args:
            root (str): Root directory of the datasets.
            eval_ood (bool): Whether to evaluate out-of-distribution
                performance.
            batch_size (int): Number of samples per batch.
            val_split (float): Share of samples to use for validation. Defaults
                to ``0.0``.
            cutout (int): Size of cutout to apply to images. Defaults to ``None``.
            randaugment (bool): Whether to apply RandAugment. Defaults to
                ``False``.
            auto_augment (str): Which auto-augment to apply. Defaults to ``None``.
            test_alt (str): Which test set to use. Defaults to ``None``.
            corruption_severity (int): Severity of corruption to apply to
                CIFAR100-C. Defaults to ``1``.
            num_dataloaders (int): Number of dataloaders to use. Defaults to ``1``.
            num_workers (int): Number of workers to use for data loading. Defaults
                to ``1``.
            pin_memory (bool): Whether to pin memory. Defaults to ``True``.
            persistent_workers (bool): Whether to use persistent workers. Defaults
                to ``True``.
        """
        super().__init__(
            root=root,
            batch_size=batch_size,
            val_split=val_split,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        self.eval_ood = eval_ood
        self.num_dataloaders = num_dataloaders

        if test_alt == "c":
            self.dataset = CIFAR100C
        else:
            self.dataset = CIFAR100

        self.test_alt = test_alt

        self.ood_dataset = SVHN

        self.corruption_severity = corruption_severity

        if (cutout is not None) + randaugment + int(
            auto_augment is not None
        ) > 1:
            raise ValueError(
                "Only one data augmentation can be chosen at a time. Raise a "
                "GitHub issue if needed."
            )

        if cutout:
            main_transform = Cutout(cutout)
        elif randaugment:
            main_transform = T.RandAugment(num_ops=2, magnitude=20)
        elif auto_augment:
            main_transform = rand_augment_transform(auto_augment, {})
        else:
            main_transform = nn.Identity()

        self.train_transform = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                main_transform,
                T.ToTensor(),
                T.ConvertImageDtype(torch.float32),
                T.Normalize(
                    (0.5071, 0.4867, 0.4408),
                    (0.2675, 0.2565, 0.2761),
                ),
            ]
        )
        self.test_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    (0.5071, 0.4867, 0.4408),
                    (0.2675, 0.2565, 0.2761),
                ),
            ]
        )

    def prepare_data(self) -> None:  # coverage: ignore
        if self.test_alt is None:
            self.dataset(self.root, train=True, download=True)
            self.dataset(self.root, train=False, download=True)

        if self.eval_ood:
            self.ood_dataset(
                self.root,
                split="test",
                download=True,
                transform=self.test_transform,
            )

    def setup(self, stage: Literal["fit", "test"] | None = None) -> None:
        if stage == "fit" or stage is None:
            if self.test_alt == "c":
                raise ValueError("CIFAR-C can only be used in testing.")
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
                    severity=self.corruption_severity,
                )
            if self.eval_ood:
                self.ood = self.ood_dataset(
                    self.root,
                    split="test",
                    download=False,
                    transform=self.test_transform,
                )
        if stage not in ["fit", "test", None]:
            raise ValueError(f"Stage {stage} is not supported.")

    def train_dataloader(self) -> DataLoader:
        """Get the training dataloader for CIFAR100.

        Return:
            DataLoader: CIFAR100 training dataloader.
        """
        if self.num_dataloaders > 1:
            return self._data_loader(
                AggregatedDataset(self.train, self.num_dataloaders),
                shuffle=True,
            )
        return self._data_loader(self.train, shuffle=True)

    def test_dataloader(self) -> list[DataLoader]:
        r"""Get test dataloaders.

        Return:
            List[DataLoader]: test set for in distribution data
            and out-of-distribution data.
        """
        dataloader = [self._data_loader(self.test)]
        if self.eval_ood:
            dataloader.append(self._data_loader(self.ood))
        return dataloader

    def _get_train_data(self) -> ArrayLike:
        if self.val_split:
            return self.train.dataset.data[self.train.indices]
        return self.train.data

    def _get_train_targets(self) -> ArrayLike:
        if self.val_split:
            return np.array(self.train.dataset.targets)[self.train.indices]
        return np.array(self.train.targets)
