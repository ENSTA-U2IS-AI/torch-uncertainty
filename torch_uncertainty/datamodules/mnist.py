# fmt: off
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, List, Optional, Union

import torchvision.transforms as T
from pytorch_lightning import LightningDataModule
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import MNIST, FashionMNIST

from torch_uncertainty.transforms import Cutout


# fmt: on
class MNISTDataModule(LightningDataModule):
    """DataModule for MNIST.

    Args:
        root (str): Root directory of the datasets.
        batch_size (int): Number of samples per batch.
        val_split (float): Share of samples to use for validation. Defaults
            to ``0.0``.
        num_workers (int): Number of workers to use for data loading. Defaults
            to ``1``.
        cutout (int): Size of cutout to apply to images. Defaults to ``None``.
        pin_memory (bool): Whether to pin memory. Defaults to ``True``.
        persistent_workers (bool): Whether to use persistent workers. Defaults
            to ``True``.
    """

    num_classes = 10
    num_channels = 1
    input_shape = (1, 28, 28)
    training_task = "classification"

    def __init__(
        self,
        root: Union[str, Path],
        ood_detection: bool,
        batch_size: int,
        val_split: float = 0.0,
        num_workers: int = 1,
        cutout: Optional[int] = None,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        if isinstance(root, str):
            root = Path(root)

        self.root: Path = root
        self.ood_detection = ood_detection
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.dataset = MNIST
        self.ood_dataset = FashionMNIST
        self.num_classes = 10

        if cutout:
            main_transform = Cutout(cutout)
        else:
            main_transform = nn.Identity()

        self.transform_train = T.Compose(
            [
                main_transform,
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
        """Download the datasets."""
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
                full,
                [
                    int(len(full) * (1 - self.val_split)),
                    len(full) - int(len(full) * (1 - self.val_split)),
                ],
            )
            if self.val_split == 0:
                self.val = self.dataset(
                    self.root,
                    train=False,
                    download=False,
                    transform=self.transform_test,
                )
        if stage == "test":
            self.test = self.dataset(
                self.root,
                train=False,
                download=False,
                transform=self.transform_test,
            )

        if self.ood_detection:
            self.ood = self.ood_dataset(
                self.root,
                download=False,
                transform=self.transform_test,
            )

    def train_dataloader(self) -> DataLoader:
        r"""Get the training dataloader for MNIST.

        Return:
            DataLoader: MNIST training dataloader.
        """
        return self._data_loader(self.train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        r"""Get the validation dataloader for MNIST.

        Return:
            DataLoader: MNIST validation dataloader.
        """
        return self._data_loader(self.val)

    def test_dataloader(self) -> List[DataLoader]:
        r"""Get the test dataloaders for MNIST.

        Return:
            List[DataLoader]: Dataloaders of the MNIST test set (in
                distribution data) and FashionMNIST test split
                (out-of-distribution data).
        """
        dataloader = [self._data_loader(self.test)]
        if self.ood_detection:
            dataloader.append(self._data_loader(self.ood))
        return dataloader

    def _data_loader(
        self, dataset: Dataset, shuffle: bool = False
    ) -> DataLoader:
        """Create a dataloader for a given dataset.

        Args:
            dataset (Dataset): Dataset to create a dataloader for.
            shuffle (bool, optional): Whether to shuffle the dataset. Defaults
                to False.

        Return:
            DataLoader: Dataloader for the given dataset.
        """
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
        p.add_argument(
            "--evaluate_ood", dest="ood_detection", action="store_true"
        )
        return parent_parser
