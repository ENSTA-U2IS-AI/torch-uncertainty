# fmt: off
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Any, Optional, Tuple, Union

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

from ..datasets.uci_regression import UCIRegression


# fmt: on
class UCIDataModule(LightningDataModule):
    """The UCI regression datasets.

    Args:
        root (string): Root directory of the datasets.
        batch_size (int): The batch size for training and testing.
        dataset_name (string, optional): The name of the dataset. One of
            "boston-housing", "concrete", "energy", "kin8nm",
            "naval-propulsion-plant", "power-plant", "protein",
            "wine-quality-red", and "yacht".
        val_split (float, optional): Share of validation samples. Defaults
            to ``0``.
        num_workers (int, optional): How many subprocesses to use for data
            loading. Defaults to ``1``.
        pin_memory (bool, optional): Whether to pin memory in the GPU. Defaults
            to ``True``.
        persistent_workers (bool, optional): Whether to use persistent workers.
            Defaults to ``True``.
    """

    training_task = "regression"

    def __init__(
        self,
        root: Union[str, Path],
        batch_size: int,
        dataset_name: str,
        val_split: float = 0.0,
        num_workers: int = 1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        input_shape: Optional[Tuple[int, ...]] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        if isinstance(root, str):
            root = Path(root)
        self.root: Path = root
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers

        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.dataset = partial(UCIRegression, dataset_name=dataset_name)
        self.input_shape = input_shape

    def prepare_data(self) -> None:
        """Download the dataset."""
        self.dataset(root=self.root, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        """Split the datasets into train, val, and test."""
        full = self.dataset(
            self.root,
            download=False,
        )
        self.train, self.test, self.val = random_split(
            full,
            [
                int(len(full) * (0.8 - self.val_split)),
                int(len(full) * 0.2),
                len(full)
                - int(len(full) * 0.2)
                - int(len(full) * (0.8 - self.val_split)),
            ],
        )
        if self.val_split == 0:
            self.val = self.test

    def train_dataloader(self) -> DataLoader:
        """Get the training dataloader for UCI Regression.

        Return:
            DataLoader: UCI Regression training dataloader.
        """
        return self._data_loader(self.train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Get the validation dataloader for UCI Regression.

        Return:
            DataLoader: UCI Regression validation dataloader.
        """
        return self._data_loader(self.val)

    def test_dataloader(self) -> DataLoader:
        """Get the test dataloader for UCI Regression.

        Return:
            DataLoader: UCI Regression test dataloader.
        """
        return self._data_loader(self.test)

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
        p.add_argument("--val_split", type=float, default=0)
        p.add_argument("--num_workers", type=int, default=4)
        return parent_parser
