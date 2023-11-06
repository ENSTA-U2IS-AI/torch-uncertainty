from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torchvision.transforms as T
from numpy.typing import ArrayLike
from timm.data.auto_augment import rand_augment_transform
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10, SVHN

from torch_uncertainty.datasets import AggregatedDataset
from torch_uncertainty.datasets.classification import CIFAR10C, CIFAR10H
from torch_uncertainty.transforms import Cutout

from .abstract import AbstractDataModule


class CIFAR10DataModule(AbstractDataModule):
    num_classes = 10
    num_channels = 3
    input_shape = (3, 32, 32)
    training_task = "classification"

    def __init__(
        self,
        root: str | Path,
        evaluate_ood: bool,
        batch_size: int,
        val_split: float = 0.0,
        num_workers: int = 1,
        cutout: int | None = None,
        auto_augment: str | None = None,
        test_alt: Literal["c", "h"] | None = None,
        corruption_severity: int = 1,
        num_dataloaders: int = 1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        **kwargs,
    ) -> None:
        """DataModule for CIFAR10.

        Args:
            root (str): Root directory of the datasets.
            evaluate_ood (bool): Whether to evaluate on out-of-distribution data.
            batch_size (int): Number of samples per batch.
            val_split (float): Share of samples to use for validation. Defaults
                to ``0.0``.
            num_workers (int): Number of workers to use for data loading. Defaults
                to ``1``.
            cutout (int): Size of cutout to apply to images. Defaults to ``None``.
            randaugment (bool): Whether to apply RandAugment. Defaults to
                ``False``.
            auto_augment (str): Which auto-augment to apply. Defaults to ``None``.
            test_alt (str): Which test set to use. Defaults to ``None``.
            corruption_severity (int): Severity of corruption to apply for
                CIFAR10-C. Defaults to ``1``.
            num_dataloaders (int): Number of dataloaders to use. Defaults to ``1``.
            pin_memory (bool): Whether to pin memory. Defaults to ``True``.
            persistent_workers (bool): Whether to use persistent workers. Defaults
                to ``True``.
            kwargs: Additional arguments.
        """
        super().__init__(
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        self.val_split = val_split
        self.num_dataloaders = num_dataloaders
        self.evaluate_ood = evaluate_ood

        if test_alt == "c":
            self.dataset = CIFAR10C
        elif test_alt == "h":
            self.dataset = CIFAR10H
        else:
            self.dataset = CIFAR10

        self.test_alt = test_alt
        self.corruption_severity = corruption_severity
        self.ood_dataset = SVHN

        if (cutout is not None) + int(auto_augment is not None) > 1:
            raise ValueError(
                "Only one data augmentation can be chosen at a time. Raise a "
                "GitHub issue if needed."
            )

        if cutout:
            main_transform = Cutout(cutout)
        elif auto_augment:
            main_transform = rand_augment_transform(auto_augment, {})
        else:
            main_transform = nn.Identity()

        self.transform_train = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                main_transform,
                T.ToTensor(),
                T.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010),
                ),
            ]
        )

        self.transform_test = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010),
                ),
            ]
        )

    def prepare_data(self) -> None:  # coverage: ignore
        if self.test_alt is None:
            self.dataset(self.root, train=True, download=True)
            self.dataset(self.root, train=False, download=True)
        elif self.test_alt == "c":
            self.dataset(
                self.root,
                severity=self.corruption_severity,
                download=True,
            )
        else:
            self.dataset(
                self.root,
                download=True,
            )

        if self.evaluate_ood:
            self.ood_dataset(self.root, split="test", download=True)

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            if self.test_alt in ("c", "h"):
                raise ValueError("CIFAR-C and H can only be used in testing.")
            full = self.dataset(
                self.root,
                train=True,
                download=False,
                transform=self.transform_train,
            )
            if self.val_split:
                self.train, self.val = random_split(
                    full,
                    [
                        1 - self.val_split,
                        self.val_split,
                    ],
                )
            else:
                self.train = full
                self.val = self.dataset(
                    self.root,
                    train=False,
                    download=False,
                    transform=self.transform_test,
                )
        elif stage == "test":
            if self.test_alt is None:
                self.test = self.dataset(
                    self.root,
                    train=False,
                    download=False,
                    transform=self.transform_test,
                )
            else:
                self.test = self.dataset(
                    self.root,
                    transform=self.transform_test,
                    severity=self.corruption_severity,
                )
            if self.evaluate_ood:
                self.ood = self.ood_dataset(
                    self.root,
                    split="test",
                    download=False,
                    transform=self.transform_test,
                )
        else:
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
            )
        return self._data_loader(self.train, shuffle=True)

    def test_dataloader(self) -> list[DataLoader]:
        r"""Get test dataloaders.

        Return:
            List[DataLoader]: test set for in distribution data
            and out-of-distribution data.
        """
        dataloader = [self._data_loader(self.test)]
        if self.evaluate_ood:
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

    @classmethod
    def add_argparse_args(
        cls,
        parent_parser: ArgumentParser,
        **kwargs: Any,
    ) -> ArgumentParser:
        p = super().add_argparse_args(parent_parser)

        # Arguments for CIFAR10
        p.add_argument("--cutout", type=int, default=0)
        p.add_argument("--auto_augment", type=str)
        p.add_argument("--test_alt", choices=["c", "h"], default=None)
        p.add_argument(
            "--severity", dest="corruption_severity", type=int, default=None
        )
        p.add_argument("--evaluate_ood", action="store_true")
        return parent_parser
