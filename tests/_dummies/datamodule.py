from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import numpy as np
import torchvision.transforms as T
from numpy.typing import ArrayLike
from torch.utils.data import DataLoader

from torch_uncertainty.datamodules.abstract import AbstractDataModule

from .dataset import DummyClassificationDataset, DummyRegressionDataset


class DummyClassificationDataModule(AbstractDataModule):
    num_channels = 1
    image_size: int = 4
    training_task = "classification"

    def __init__(
        self,
        root: str | Path,
        evaluate_ood: bool,
        batch_size: int,
        num_classes: int = 2,
        num_workers: int = 1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        self.evaluate_ood = evaluate_ood
        self.num_classes = num_classes

        self.dataset = DummyClassificationDataset
        self.ood_dataset = DummyClassificationDataset

        self.transform_train = T.ToTensor()
        self.transform_test = T.ToTensor()

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self.train = self.dataset(
                self.root,
                num_channels=self.num_channels,
                num_classes=self.num_classes,
                image_size=self.image_size,
                transform=self.transform_train,
            )
            self.val = self.dataset(
                self.root,
                num_channels=self.num_channels,
                num_classes=self.num_classes,
                image_size=self.image_size,
                transform=self.transform_test,
            )
        elif stage == "test":
            self.test = self.dataset(
                self.root,
                num_channels=self.num_channels,
                num_classes=self.num_classes,
                image_size=self.image_size,
                transform=self.transform_test,
            )
            self.ood = self.ood_dataset(
                self.root,
                num_channels=self.num_channels,
                num_classes=self.num_classes,
                image_size=self.image_size,
                transform=self.transform_test,
            )

    def test_dataloader(self) -> DataLoader | list[DataLoader]:
        dataloader = [self._data_loader(self.test)]
        if self.evaluate_ood:
            dataloader.append(self._data_loader(self.ood))
        return dataloader

    def _get_train_data(self) -> ArrayLike:
        return self.train.data

    def _get_train_targets(self) -> ArrayLike:
        return np.array(self.train.targets)

    @classmethod
    def add_argparse_args(
        cls,
        parent_parser: ArgumentParser,
        **kwargs: Any,
    ) -> ArgumentParser:
        p = super().add_argparse_args(parent_parser)
        p.add_argument("--evaluate_ood", action="store_true")
        return parent_parser


class DummyRegressionDataModule(AbstractDataModule):
    in_features = 4
    training_task = "regression"

    def __init__(
        self,
        root: str | Path,
        evaluate_ood: bool,
        batch_size: int,
        out_features: int = 2,
        num_workers: int = 1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        self.evaluate_ood = evaluate_ood
        self.out_features = out_features

        self.dataset = DummyRegressionDataset
        self.ood_dataset = DummyRegressionDataset

        self.transform_train = None
        self.transform_test = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self.train = self.dataset(
                self.root,
                out_features=self.out_features,
                transform=self.transform_train,
            )
            self.val = self.dataset(
                self.root,
                out_features=self.out_features,
                transform=self.transform_test,
            )
        elif stage == "test":
            self.test = self.dataset(
                self.root,
                out_features=self.out_features,
                transform=self.transform_test,
            )
        if self.evaluate_ood:
            self.ood = self.ood_dataset(
                self.root,
                out_features=self.out_features,
                transform=self.transform_test,
            )

    def test_dataloader(self) -> DataLoader | list[DataLoader]:
        dataloader = [self._data_loader(self.test)]
        if self.evaluate_ood:
            dataloader.append(self._data_loader(self.ood))
        return dataloader

    @classmethod
    def add_argparse_args(
        cls,
        parent_parser: ArgumentParser,
        **kwargs: Any,
    ) -> ArgumentParser:
        p = super().add_argparse_args(parent_parser)
        p.add_argument("--evaluate_ood", action="store_true")
        return parent_parser
