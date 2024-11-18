import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

import torch
from torch import Generator
from torch.utils.data import Dataset
from torchvision.datasets.utils import (
    check_integrity,
    download_and_extract_archive,
)


class UCIClassificationDataset(ABC, Dataset):
    """The UCI classification dataset base class.

    Args:
        root (str): Root directory of the datasets.
        train (bool, optional): If True, creates dataset from training set,
            otherwise creates from test set.
        transform (callable, optional): A function/transform that takes in a
            numpy array and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        download (bool, optional): If true, downloads the dataset from the
            internet and puts it in root directory. If dataset is already
            downloaded, it is not downloaded again.
        binary (bool, optional): Whether to use binary classification. Defaults
            to ``True``.

    Note - License:
        The licenses of the datasets may differ from TorchUncertainty's
        license. Check before use.

    """

    md5_zip: str = ""
    url: str = ""
    filename: str = ""
    dataset_name: str = ""
    need_split = True
    apply_standardization = True

    def __init__(
        self,
        root: Path | str,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        binary: bool = True,
        download: bool = False,
        train: bool = True,
        test_split: float = 0.2,
        split_seed: int = 21893027,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        self._make_dataset()
        if self.apply_standardization:
            self._compute_statistics()
            self._standardize()

        if self.need_split:
            gen = Generator().manual_seed(split_seed)

            self.split_idx = torch.ones(len(self)).multinomial(
                num_samples=int((1 - test_split) * len(self)),
                replacement=False,
                generator=gen,
            )
            if not self.train:
                self.split_idx = torch.tensor(
                    [i for i in range(len(self)) if i not in self.split_idx]
                )
            self.data = self.data[self.split_idx]
            self.targets = self.targets[self.split_idx]
        if not binary:
            self.targets = torch.nn.functional.one_hot(
                self.targets, num_classes=2
            )

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return self.data.shape[0]

    def _check_integrity(self) -> bool:
        """Check the integrity of the dataset(s)."""
        return check_integrity(
            self.root / Path(self.dataset_name + ".zip"),
            self.md5_zip,
        )

    def _standardize(self) -> None:
        self.data = (self.data - self.data_mean) / self.data_std

    def _compute_statistics(self) -> None:
        self.data_mean = self.data.mean(dim=0)
        self.data_std = self.data.std(dim=0)
        self.data_std[self.data_std == 0] = 1

    def download(self) -> None:
        """Download and extract dataset."""
        if self._check_integrity():
            logging.info("Files already downloaded and verified")
            return
        download_and_extract_archive(
            self.url,
            download_root=self.root / self.dataset_name,
            filename=self.filename + ".zip",
            md5=self.md5_zip,
        )

    @abstractmethod
    def _make_dataset(self) -> None:
        """Create dataset from extracted files."""

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get sample and target for a given index."""
        data = self.data[index, :]
        if self.transform is not None:
            data = self.transform(data)
        target = self.targets[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, target
