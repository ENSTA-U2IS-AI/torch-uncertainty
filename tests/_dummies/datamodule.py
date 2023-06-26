# fmt: off
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, List, Optional, Union

import torchvision.transforms as T
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from .dataset import DummyDataset


# fmt: on
class DummyDataModule(LightningDataModule):
    num_channels = 1
    image_size: int = 8
    training_task = "classification"

    def __init__(
        self,
        root: Union[str, Path],
        batch_size: int,
        num_classes: int = 10,
        num_workers: int = 1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        root = Path(root)

        self.root: Path = root
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.dataset = DummyDataset
        self.ood_dataset = DummyDataset

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
        r"""Gets the training dataloader for DummyDataset.
        Returns:
            DataLoader: DummyDataset training dataloader.
        """
        return self._data_loader(self.train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        r"""Gets the validation dataloader for DummyDataset.
        Returns:
            DataLoader: DummyDataset validation dataloader.
        """
        return self._data_loader(self.val)

    def test_dataloader(self) -> List[DataLoader]:
        r"""Gets the test dataloaders for DummyDataset.
        Returns:
            List[DataLoader]: Dataloaders of the DummyDataset test set (in
                distribution data) and SVHN test split (out-of-distribution
                data).
        """
        return [self._data_loader(self.test), self._data_loader(self.ood)]

    def _data_loader(
        self, dataset: Dataset, shuffle: bool = False
    ) -> DataLoader:
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
        return parent_parser
