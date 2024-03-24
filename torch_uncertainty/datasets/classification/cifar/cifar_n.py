from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import torch
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.datasets.utils import (
    check_integrity,
    download_and_extract_archive,
)


class CIFAR10N(CIFAR10):
    """`CIFAR-10N <https://github.com/UCSC-REAL/cifar-10-100n>`_ Dataset.

    Args:
        root (string): Root directory of dataset where file
            ``cifar-10h-probs.npy`` exists or will be saved to if download
            is set to True.
        train (bool, optional): For API consistency, not used.
        transform (callable, optional): A function/transform that takes in
            a PIL image and returns a transformed version. E.g,
            ``transforms.RandomCrop``. Defaults to None.
        target_transform (callable, optional): A function/transform that
            takes in the target and transforms it. Defaults to None.
        download (bool, optional): If True, downloads the dataset from the
            internet and puts it in root directory. If dataset is already
            downloaded, it is not downloaded again. Defaults to False.
    """

    n_test_list = ["CIFAR-N-1.zip", "666bf3cff3a944c245f2b6f62af4b919"]
    n_url = "http://www.yliuu.com/web-cifarN/files/CIFAR-N-1.zip"
    filename = "CIFAR-N/CIFAR-10_human.pt"
    file_arg = ""

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        file_arg: Literal[
            "aggre_label",
            "worse_label",
            "random_label1",
            "random_label2",
            "random_label3",
        ] = "aggre_label",
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
        download: bool = False,
    ) -> None:
        super().__init__(
            Path(root),
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        if download:
            self.download_n()

        if not self._check_specific_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True to "
                "download it."
            )

        self.targets = list(torch.load(self.root / self.filename)[file_arg])

    def _check_specific_integrity(self) -> bool:
        filename, md5 = self.n_test_list
        fpath = self.root / filename
        return check_integrity(fpath, md5)

    def download_n(self) -> None:
        download_and_extract_archive(
            self.n_url,
            self.root,
            filename=self.n_test_list[0],
            md5=self.n_test_list[1],
        )


class CIFAR100N(CIFAR100):
    n_test_list = ["CIFAR-N-1.zip", "666bf3cff3a944c245f2b6f62af4b919"]
    n_url = "http://www.yliuu.com/web-cifarN/files/CIFAR-N-1.zip"
    filename = "CIFAR-N/CIFAR-100_human.pt"
    file_arg = ""

    def __init__(
        self,
        root: str,
        train: bool = True,
        file_arg: Literal[
            "fine_label",
            "coarse_label",
        ] = "fine_label",
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
        download: bool = False,
    ) -> None:
        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self.root = Path(self.root)

        if download:
            self.download_n()

        if not self._check_specific_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True to "
                "download it."
            )

        self.targets = list(torch.load(self.root / self.filename)[file_arg])

    def _check_specific_integrity(self) -> bool:
        filename, md5 = self.n_test_list
        fpath = self.root / filename
        return check_integrity(fpath, md5)

    def download_n(self) -> None:
        download_and_extract_archive(
            self.n_url,
            self.root,
            filename=self.n_test_list[0],
            md5=self.n_test_list[1],
        )
