from functools import partial
from pathlib import Path

from torch import Generator
from torch.utils.data import random_split

from torch_uncertainty.datasets.regression import UCIRegression

from .abstract import TUDataModule


class UCIRegressionDataModule(TUDataModule):
    training_task = "regression"

    def __init__(
        self,
        root: str | Path,
        batch_size: int,
        dataset_name: str,
        val_split: float = 0.0,
        num_workers: int = 1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        input_shape: tuple[int, ...] | None = None,
        split_seed: int = 42,
    ) -> None:
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
            input_shape (tuple, optional): The shape of the input data. Defaults to
                ``None``.
            split_seed (int, optional): The seed to use for splitting the dataset.
                Defaults to ``42``.
        """
        super().__init__(
            root=root,
            batch_size=batch_size,
            val_split=val_split,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        self.dataset = partial(
            UCIRegression, dataset_name=dataset_name, seed=split_seed
        )
        self.input_shape = input_shape
        self.gen = Generator().manual_seed(split_seed)

    def prepare_data(self) -> None:
        """Download the dataset."""
        self.dataset(root=self.root, download=True)

    # ruff: noqa: ARG002
    def setup(self, stage: str | None = None) -> None:
        """Split the datasets into train, val, and test."""
        full = self.dataset(
            self.root,
            download=False,
        )
        self.train, self.test, self.val = random_split(
            full,
            [
                0.8 - self.val_split,
                0.2,
                self.val_split,
            ],
            generator=self.gen,
        )
        if self.val_split == 0:
            self.val = self.test
