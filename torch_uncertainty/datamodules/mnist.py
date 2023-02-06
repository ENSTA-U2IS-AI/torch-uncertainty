# fmt: off
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, List, Optional, Union

import torchvision.transforms as T
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import MNIST, FashionMNIST


# fmt: on
class MNISTDataModule(LightningDataModule):
    def __init__(
        self,
        root: Union[str, Path],
        batch_size: int,
        val_split: int = 0,
        num_workers: int = 1,
        enable_cutout: bool = False,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        if isinstance(root, str):
            root = Path(root)

        self.root: Path = root
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.enable_cutout = enable_cutout
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.dataset = MNIST
        self.ood_dataset = FashionMNIST
        self.num_classes = 10

        self.transform_train = T.Compose(
            [
                T.ToTensor(),
                T.RandomCrop(28, padding=4),
                T.Normalize((0.1307,), (0.3081,)),
            ]
        )
        self.transform_test = T.Compose(
            [
                T.ToTensor(),
                T.CenterCrop(28),
                T.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def prepare_data(self) -> None:
        self.dataset(self.root, train=True, download=True)
        self.dataset(self.root, train=False, download=True)
        self.ood_dataset(self.root, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            full = self.dataset(
                self.root,
                train=True,
                download=False,
                transform=self.transform_train,
            )
            self.train, self.val = random_split(
                full, [len(full) - self.val_split, self.val_split]
            )
            if self.val_split == 0:
                self.val = self.dataset(
                    self.root,
                    train=False,
                    download=False,
                    transform=self.transform_test,
                )
        if stage == "test" or stage is None:
            self.test = self.dataset(
                self.root,
                train=False,
                download=False,
                transform=self.transform_test,
            )
            self.ood = self.ood_dataset(
                self.root,
                download=False,
                transform=self.transform_test,
            )

    def train_dataloader(self) -> DataLoader:
        r"""Gets the training dataloader for MNIST.
        Returns:
            DataLoader: MNIST training dataloader.
        """
        return self._data_loader(self.train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        r"""Gets the validation dataloader for MNIST.
        Returns:
            DataLoader: MNIST validation dataloader.
        """
        return self._data_loader(self.val)

    def predict_dataloader(self) -> DataLoader:
        r"""Gets the validation dataloader for MNIST.
        Returns:
            DataLoader: MNIST validation dataloader.
        """
        return self._data_loader(self.val)

    def test_dataloader(self) -> List[DataLoader]:
        r"""Gets the test dataloaders for MNIST.
        Returns:
            List[DataLoader]: Dataloaders of the MNIST test set (in
                distribution data) and FashionMNIST test split
                (out-of-distribution data).
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
        p.add_argument("--batch_size", type=int, default=128)
        p.add_argument("--val_split", type=int, default=0)
        p.add_argument("--num_workers", type=int, default=4)
        return parent_parser
