from argparse import ArgumentParser
from typing import Any, Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split


class CubicDataModule(LightningDataModule):
    training_task = "regression"

    def __init__(
        self,
        batch_size: int,
        val_split: float = 0.0,
        num_workers: int = 1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        lower_bound: int = -4.0,
        upper_bound: int = 4.0,
        num_samples: int = 5000,
        noise_mean: float = 0.0,
        noise_std: float = 3.0,
        **kwargs,
    ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers

        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.num_samples = num_samples
        self.noise = (noise_mean, noise_std)

    def generate_cubic_dataset(self) -> TensorDataset:
        x = torch.linspace(
            self.lower_bound, self.upper_bound, self.num_samples
        ).unsqueeze(1)
        y = x**3 + torch.normal(*self.noise, size=x.size())
        return TensorDataset(x, y)

    def setup(self, stage: Optional[str] = None) -> None:
        """Split the datasets into train, val, and test."""
        full = self.generate_cubic_dataset()
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
        """Get the training dataloader for the Cubic Regression.

        Return:
            DataLoader: Cubic Regression training dataloader.
        """
        return self._data_loader(self.train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Get the validation dataloader for the Cubic Regression.

        Return:
            DataLoader: Cubic Regression validation dataloader.
        """
        return self._data_loader(self.val)

    def test_dataloader(self) -> DataLoader:
        """Get the test dataloader for the Cubic Regression.

        Return:
            DataLoader: Cubic Regression test dataloader.
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
        p.add_argument("--batch_size", type=int, default=128)
        p.add_argument("--val_split", type=float, default=0)
        p.add_argument("--num_workers", type=int, default=4)
        return parent_parser
