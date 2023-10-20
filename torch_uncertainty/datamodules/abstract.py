from argparse import ArgumentParser
from pathlib import Path
from typing import Any, List, Optional, Union

from pytorch_lightning import LightningDataModule
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from numpy.typing import ArrayLike


class AbstractDataModule(LightningDataModule):
    training_task = ""

    def __init__(
        self,
        root: Union[str, Path],
        batch_size: int,
        num_workers: int = 1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        if isinstance(root, str):
            root = Path(root)
        self.root: Path = root
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

    def setup(self, stage: Optional[str] = None) -> None:
        self.train = Dataset()
        self.val = Dataset()
        self.test = Dataset()

    def get_train_set(self) -> Dataset:
        return self.train

    def get_test_set(self) -> Dataset:
        return self.test

    def get_val_set(self) -> Dataset:
        return self.val

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

    def test_dataloader(self) -> List[DataLoader]:
        r"""Get test dataloaders.

        Return:
            List[DataLoader]: test set for in distribution data
            and out-of-distribution data.
        """
        dataloader = [self._data_loader(self.test)]
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

    # These two functions have to be defined in each datamodule
    # by setting the correct path to the matrix of data for each dataset.
    # It is generally "Dataset.samples" or "Dataset.data"
    # They are used for constructing cross validation splits
    def _get_train_data(self) -> ArrayLike:
        pass

    def _get_train_targets(self) -> ArrayLike:
        pass

    def make_cross_val_splits(self, n_splits=10, train_over=4) -> list:
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
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
            )

            cv_dm.append(fold_dm)

        return cv_dm

    @classmethod
    def add_argparse_args(
        cls,
        parent_parser: ArgumentParser,
        **kwargs: Any,
    ) -> ArgumentParser:
        p = parent_parser.add_argument_group("datamodule")
        p.add_argument("--root", type=str, default="./data/")
        p.add_argument("--batch_size", type=int, default=128)
        p.add_argument("--val_split", type=float, default=0.0)
        p.add_argument("--num_workers", type=int, default=4)
        p.add_argument("--use_cv", action="store_true")
        p.add_argument("--n_splits", type=int, default=10)
        p.add_argument("--train_over", type=int, default=4)
        return parent_parser


class CrossValDataModule(AbstractDataModule):
    def __init__(
        self,
        root: str | Path,
        train_idx: ArrayLike,
        val_idx: ArrayLike,
        datamodule: AbstractDataModule,
        batch_size: int,
        num_workers: int = 1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            root,
            batch_size,
            num_workers,
            pin_memory,
            persistent_workers,
            **kwargs,
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

    def train_dataloader(self) -> DataLoader:
        return self._data_loader(self.dm.get_train_set(), self.train_idx)

    def val_dataloader(self) -> DataLoader:
        return self._data_loader(self.dm.get_train_set(), self.val_idx)

    def test_dataloader(self) -> DataLoader:
        return self._data_loader(self.dm.get_train_set(), self.val_idx)

    def get_train_set(self) -> Dataset:
        return self.dm.train

    def get_test_set(self) -> Dataset:
        return self.dm.val

    def get_val_set(self) -> Dataset:
        return self.dm.val
