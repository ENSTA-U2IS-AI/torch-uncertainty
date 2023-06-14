# fmt: off
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Any, List, Optional, Union

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

from ..datasets.uci_regression import UCIRegression


# fmt: on
class UCIDataModule(LightningDataModule):
    def __init__(
        self,
        root: Union[str, Path],
        batch_size: int,
        dataset_name: str,
        val_split: int = 0,
        num_workers: int = 1,
        num_dataloaders: int = 1,
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
        self.num_dataloaders = num_dataloaders

        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.dataset = partial(UCIRegression, dataset_name=dataset_name)

    def prepare_data(self) -> None:
        self.dataset(root=self.root, train=True, download=True)
        self.dataset(root=self.root, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            full = self.dataset(
                self.root,
                train=True,
                download=False,
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
                )
        elif stage == "test":
            self.test = self.dataset(
                self.root,
                train=False,
                download=False,
            )

    def train_dataloader(self) -> DataLoader:
        return self._data_loader(self.train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._data_loader(self.val)

    def test_dataloader(self) -> List[DataLoader]:
        return self._data_loader(self.test)

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
        p.add_argument("--val_split", type=float, default=0)
        p.add_argument("--num_workers", type=int, default=4)
        return parent_parser
