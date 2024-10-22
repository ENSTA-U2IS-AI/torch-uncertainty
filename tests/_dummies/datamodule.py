from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.v2 as T
from numpy.typing import ArrayLike
from torch.utils.data import DataLoader
from torchvision import tv_tensors

from torch_uncertainty.datamodules import TUDataModule

from .dataset import (
    DummPixelRegressionDataset,
    DummyClassificationDataset,
    DummyRegressionDataset,
    DummySegmentationDataset,
)


class DummyClassificationDataModule(TUDataModule):
    num_channels = 1
    image_size: int = 4
    training_task = "classification"

    def __init__(
        self,
        root: str | Path,
        batch_size: int,
        num_classes: int = 2,
        num_workers: int = 1,
        eval_ood: bool = False,
        eval_shift: bool = False,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        num_images: int = 2,
    ) -> None:
        super().__init__(
            root=root,
            val_split=None,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        self.eval_ood = eval_ood
        self.eval_shift = eval_shift
        self.num_classes = num_classes
        self.num_images = num_images

        self.dataset = DummyClassificationDataset
        self.ood_dataset = DummyClassificationDataset
        self.shift_dataset = DummyClassificationDataset

        self.train_transform = T.ToTensor()
        self.test_transform = T.ToTensor()

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self.train = self.dataset(
                self.root,
                num_channels=self.num_channels,
                num_classes=self.num_classes,
                image_size=self.image_size,
                transform=self.train_transform,
                num_images=self.num_images,
            )
            self.val = self.dataset(
                self.root,
                num_channels=self.num_channels,
                num_classes=self.num_classes,
                image_size=self.image_size,
                transform=self.test_transform,
                num_images=self.num_images,
            )
        elif stage == "test":
            self.test = self.dataset(
                self.root,
                num_channels=self.num_channels,
                num_classes=self.num_classes,
                image_size=self.image_size,
                transform=self.test_transform,
                num_images=self.num_images,
            )
            self.ood = self.ood_dataset(
                self.root,
                num_channels=self.num_channels,
                num_classes=self.num_classes,
                image_size=self.image_size,
                transform=self.test_transform,
                num_images=self.num_images,
            )
            self.shift = self.shift_dataset(
                self.root,
                num_channels=self.num_channels,
                num_classes=self.num_classes,
                image_size=self.image_size,
                transform=self.test_transform,
                num_images=self.num_images,
            )
            self.shift.shift_severity = 1

    def test_dataloader(self) -> DataLoader | list[DataLoader]:
        dataloader = [self._data_loader(self.test)]
        if self.eval_ood:
            dataloader.append(self._data_loader(self.ood))
        if self.eval_shift:
            dataloader.append(self._data_loader(self.shift))
        return dataloader

    def _get_train_data(self) -> ArrayLike:
        return self.train.data

    def _get_train_targets(self) -> ArrayLike:
        return np.array(self.train.targets)


class DummyRegressionDataModule(TUDataModule):
    in_features = 4
    training_task = "regression"

    def __init__(
        self,
        root: str | Path,
        batch_size: int,
        out_features: int = 2,
        num_workers: int = 1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ) -> None:
        super().__init__(
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            val_split=0,
        )

        self.out_features = out_features

        self.dataset = DummyRegressionDataset
        self.ood_dataset = DummyRegressionDataset

        self.train_transform = None
        self.test_transform = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self.train = self.dataset(
                self.root,
                out_features=self.out_features,
                transform=self.train_transform,
            )
            self.val = self.dataset(
                self.root,
                out_features=self.out_features,
                transform=self.test_transform,
            )
        elif stage == "test":
            self.test = self.dataset(
                self.root,
                out_features=self.out_features,
                transform=self.test_transform,
            )

    def test_dataloader(self) -> DataLoader | list[DataLoader]:
        return [self._data_loader(self.test)]


class DummySegmentationDataModule(TUDataModule):
    num_channels = 3
    training_task = "segmentation"
    mean = [0.0, 0.0, 0.0]
    std = [1.0, 1.0, 1.0]

    def __init__(
        self,
        root: str | Path,
        batch_size: int,
        num_classes: int = 2,
        num_workers: int = 1,
        image_size: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        num_images: int = 2,
    ) -> None:
        super().__init__(
            root=root,
            val_split=None,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        self.num_classes = num_classes
        self.num_channels = 3
        self.num_images = num_images
        self.image_size = image_size

        self.dataset = DummySegmentationDataset

        self.train_transform = T.ToDtype(
            dtype={
                tv_tensors.Image: torch.float32,
                tv_tensors.Mask: torch.int64,
                "others": None,
            },
            scale=True,
        )
        self.test_transform = T.ToDtype(
            dtype={
                tv_tensors.Image: torch.float32,
                tv_tensors.Mask: torch.int64,
                "others": None,
            },
            scale=True,
        )

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self.train = self.dataset(
                self.root,
                num_channels=self.num_channels,
                num_classes=self.num_classes,
                image_size=self.image_size,
                transforms=self.train_transform,
                num_images=self.num_images,
            )
            self.val = self.dataset(
                self.root,
                num_channels=self.num_channels,
                num_classes=self.num_classes,
                image_size=self.image_size,
                transforms=self.test_transform,
                num_images=self.num_images,
            )
        elif stage == "test":
            self.test = self.dataset(
                self.root,
                num_channels=self.num_channels,
                num_classes=self.num_classes,
                image_size=self.image_size,
                transforms=self.test_transform,
                num_images=self.num_images,
            )

    def test_dataloader(self) -> DataLoader | list[DataLoader]:
        return [self._data_loader(self.test)]

    def _get_train_data(self) -> ArrayLike:
        return self.train.data

    def _get_train_targets(self) -> ArrayLike:
        return np.array(self.train.targets)


class DummyPixelRegressionDataModule(TUDataModule):
    num_channels = 3
    training_task = "pixel_regression"

    def __init__(
        self,
        root: str | Path,
        batch_size: int,
        output_dim: int = 2,
        num_workers: int = 1,
        image_size: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        num_images: int = 2,
    ) -> None:
        super().__init__(
            root=root,
            val_split=None,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        self.output_dim = output_dim
        self.num_channels = 3
        self.num_images = num_images
        self.image_size = image_size

        self.dataset = DummPixelRegressionDataset

        self.train_transform = T.ToDtype(
            dtype={
                tv_tensors.Image: torch.float32,
                tv_tensors.Mask: torch.float32,
                "others": None,
            },
            scale=True,
        )
        self.test_transform = T.ToDtype(
            dtype={
                tv_tensors.Image: torch.float32,
                tv_tensors.Mask: torch.float32,
                "others": None,
            },
            scale=True,
        )

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self.train = self.dataset(
                self.root,
                num_channels=self.num_channels,
                output_dim=self.output_dim,
                image_size=self.image_size,
                transforms=self.train_transform,
                num_images=self.num_images,
            )
            self.val = self.dataset(
                self.root,
                num_channels=self.num_channels,
                output_dim=self.output_dim,
                image_size=self.image_size,
                transforms=self.test_transform,
                num_images=self.num_images,
            )
        elif stage == "test":
            self.test = self.dataset(
                self.root,
                num_channels=self.num_channels,
                output_dim=self.output_dim,
                image_size=self.image_size,
                transforms=self.test_transform,
                num_images=self.num_images,
            )

    def test_dataloader(self) -> DataLoader | list[DataLoader]:
        return [self._data_loader(self.test)]

    def _get_train_data(self) -> ArrayLike:
        return self.train.data

    def _get_train_targets(self) -> ArrayLike:
        return np.array(self.train.targets)
