from argparse import ArgumentParser
from pathlib import Path
from typing import Any, List, Optional, Union

import torchvision.transforms as T
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from .dataset import DummyClassificationDataset, DummyRegressionDataset


class DummyClassificationDataModule(LightningDataModule):
    num_channels = 1
    image_size: int = 4
    training_task = "classification"

    def __init__(
        self,
        root: Union[str, Path],
        evaluate_ood: bool,
        batch_size: int,
        num_classes: int = 2,
        num_workers: int = 1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        root = Path(root)

        self.root: Path = root
        self.evaluate_ood = evaluate_ood
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.dataset = DummyClassificationDataset
        self.ood_dataset = DummyClassificationDataset

        self.transform_train = T.ToTensor()
        self.transform_test = T.ToTensor()

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
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

    def train_dataloader(self) -> DataLoader:
        return self._data_loader(self.train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._data_loader(self.val)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        dataloader = [self._data_loader(self.test)]
        if self.evaluate_ood:
            dataloader.append(self._data_loader(self.ood))
        return dataloader

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    @classmethod
    def add_argparse_args(
        cls,
        parent_parser: ArgumentParser,
        **kwargs: Any,
    ) -> ArgumentParser:
        p = parent_parser.add_argument_group("datamodule")
        p.add_argument("--root", type=str, default="./data/")
        p.add_argument("--batch_size", type=int, default=2)
        p.add_argument("--num_workers", type=int, default=1)
        p.add_argument("--evaluate_ood", action="store_true")
        return parent_parser


class DummyRegressionDataModule(LightningDataModule):
    in_features = 4
    training_task = "regression"

    def __init__(
        self,
        root: Union[str, Path],
        evaluate_ood: bool,
        batch_size: int,
        out_features: int = 2,
        num_workers: int = 1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        if isinstance(root, str):
            root = Path(root)
        self.root: Path = root
        self.evaluate_ood = evaluate_ood
        self.batch_size = batch_size
        self.out_features = out_features
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.dataset = DummyRegressionDataset
        self.ood_dataset = DummyRegressionDataset

        self.transform_train = None
        self.transform_test = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
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

    def train_dataloader(self) -> DataLoader:
        return self._data_loader(self.train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._data_loader(self.val)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        dataloader = [self._data_loader(self.test)]
        if self.evaluate_ood:
            dataloader.append(self._data_loader(self.ood))
        return dataloader

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    @classmethod
    def add_argparse_args(
        cls,
        parent_parser: ArgumentParser,
        **kwargs: Any,
    ) -> ArgumentParser:
        p = parent_parser.add_argument_group("datamodule")
        p.add_argument("--root", type=str, default="./data/")
        p.add_argument("--batch_size", type=int, default=2)
        p.add_argument("--num_workers", type=int, default=1)
        p.add_argument("--evaluate_ood", action="store_true")
        return parent_parser
