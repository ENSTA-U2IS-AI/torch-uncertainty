from pathlib import Path

from torch.utils.data import Dataset

from torch_uncertainty.datamodules.abstract import TUDataModule
from torch_uncertainty.utils import create_train_val_split


class UCIClassificationDataModule(TUDataModule):
    training_task = "classification"

    def __init__(
        self,
        root: str | Path,
        dataset: type[Dataset],
        batch_size: int,
        val_split: float = 0.0,
        test_split: float = 0.2,
        num_workers: int = 1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        binary: bool = True,
    ) -> None:
        """The UCI classification datamodule base class.

        Args:
            root (string): Root directory of the datasets.
            dataset (type[Dataset]): The UCI classification dataset class.
            batch_size (int): The batch size for training and testing.
            val_split (float, optional): Share of validation samples among the
                non-test samples. Defaults to ``0``.
            test_split (float, optional): Share of test samples. Defaults to ``0.2``.
            num_workers (int, optional): How many subprocesses to use for data
                loading. Defaults to ``1``.
            pin_memory (bool, optional): Whether to pin memory in the GPU. Defaults
                to ``True``.
            persistent_workers (bool, optional): Whether to use persistent workers.
                Defaults to ``True``.
            binary (bool, optional): Whether to use binary classification. Defaults
                to ``True``.

        """
        super().__init__(
            root=root,
            batch_size=batch_size,
            val_split=val_split,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        self.dataset = dataset
        self.test_split = test_split
        self.binary = binary

    def prepare_data(self) -> None:
        """Download the dataset."""
        self.dataset(root=self.root, download=True)

    # ruff: noqa: ARG002
    def setup(self, stage: str | None = None) -> None:
        """Split the datasets into train, val, and test."""
        if stage == "fit" or stage is None:
            full = self.dataset(
                self.root,
                train=True,
                download=False,
                binary=self.binary,
                test_split=self.test_split,
            )

            if self.val_split:
                self.train, self.val = create_train_val_split(
                    full,
                    self.val_split,
                )
            else:
                self.train = full
                self.val = self.dataset(
                    self.root,
                    train=False,
                    download=False,
                    binary=self.binary,
                    test_split=self.test_split,
                )
        if stage == "test" or stage is None:
            self.test = self.dataset(
                self.root,
                train=False,
                download=False,
                binary=self.binary,
                test_split=self.test_split,
            )
        if stage not in ["fit", "test", None]:
            raise ValueError(f"Stage {stage} is not supported.")
