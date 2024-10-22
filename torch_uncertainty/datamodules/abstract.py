from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

from lightning.pytorch.core import LightningDataModule
from numpy.typing import ArrayLike
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler


class TUDataModule(ABC, LightningDataModule):
    training_task: str
    train: Dataset
    val: Dataset
    test: Dataset

    def __init__(
        self,
        root: str | Path,
        batch_size: int,
        val_split: float | None,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
    ) -> None:
        """Abstract DataModule class.

        This class implements the basic functionality of a DataModule. It includes
        setters and getters for the datasets, dataloaders, as well as the crossval
        logic. It also provide the basic argparse arguments for the datamodules.

        Args:
            root (str): Root directory of the datasets.
            batch_size (int): Number of samples per batch.
            val_split (float): Share of samples to use for validation.
            num_workers (int): Number of workers to use for data loading.
            pin_memory (bool): Whether to pin memory.
            persistent_workers (bool): Whether to use persistent workers.
        """
        super().__init__()

        self.root = Path(root)
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers

        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

    @abstractmethod
    def setup(self, stage: Literal["fit", "test"] | None = None) -> None:
        pass

    def get_train_set(self) -> Dataset:
        """Get the training set."""
        return self.train

    def get_val_set(self) -> Dataset:
        """Get the validation set."""
        return self.val

    def get_test_set(self) -> Dataset:
        """Get the test set."""
        return self.test

    def train_dataloader(self) -> DataLoader:
        r"""Get the training dataloader.

        Return:
            DataLoader: training dataloader.
        """
        return self._data_loader(self.train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        r"""Get the validation dataloader.

        Return:
            DataLoader: validation dataloader.
        """
        return self._data_loader(self.val)

    def test_dataloader(self) -> list[DataLoader]:
        r"""Get test dataloaders.

        Return:
            list[DataLoader]: test set for in distribution data
            and out-of-distribution data.
        """
        return [self._data_loader(self.test)]

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

    # These two functions have to be defined in each datamodule
    # by setting the correct path to the matrix of data for each dataset.
    # It is generally "Dataset.samples" or "Dataset.data"
    # They are used for constructing cross validation splits
    def _get_train_data(self) -> ArrayLike:
        raise NotImplementedError

    def _get_train_targets(self) -> ArrayLike:
        raise NotImplementedError

    def make_cross_val_splits(
        self, n_splits: int = 10, train_over: int = 4
    ) -> list:
        self.setup("fit")
        skf = StratifiedKFold(n_splits)
        cv_dm = []

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(self._get_train_data(), self._get_train_targets())
        ):
            if fold >= train_over:
                break

            fold_dm = CrossValDataModule(
                root=self.root,
                train_idx=train_idx,
                val_idx=val_idx,
                datamodule=self,
                batch_size=self.batch_size,
                val_split=self.val_split,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
            )
            cv_dm.append(fold_dm)

        return cv_dm


class CrossValDataModule(TUDataModule):
    def __init__(
        self,
        root: str | Path,
        train_idx: ArrayLike,
        val_idx: ArrayLike,
        datamodule: TUDataModule,
        batch_size: int,
        val_split: float,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
    ) -> None:
        super().__init__(
            root=root,
            batch_size=batch_size,
            val_split=val_split,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        self.train_idx = train_idx
        self.val_idx = val_idx
        self.dm = datamodule

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self.train = self.dm.train
            self.val = self.dm.val
        elif stage == "test":
            self.test = self.val
        else:
            raise ValueError(f"Stage {stage} not supported.")

    def _data_loader(self, dataset: Dataset, idx: ArrayLike) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            sampler=SubsetRandomSampler(idx),
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def get_train_set(self) -> Dataset:
        """Get the training set for the current fold."""
        return self.dm.train

    def get_val_set(self) -> Dataset:
        """Get the validation set for the current fold."""
        return self.dm.val

    def get_test_set(self) -> Dataset:
        """Get the test set for the current fold."""
        return self.dm.val

    def train_dataloader(self) -> DataLoader:
        """Get the training dataloader for the current fold."""
        return self._data_loader(self.dm.get_train_set(), self.train_idx)

    def val_dataloader(self) -> DataLoader:
        """Get the validation dataloader for the current fold."""
        return self._data_loader(self.dm.get_train_set(), self.val_idx)

    def test_dataloader(self) -> DataLoader:
        """Get the test dataloader for the current fold."""
        return self._data_loader(self.dm.get_train_set(), self.val_idx)
