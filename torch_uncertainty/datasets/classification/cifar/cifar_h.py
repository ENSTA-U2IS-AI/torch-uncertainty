from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torchvision.datasets import CIFAR10
from torchvision.datasets.utils import check_integrity, download_url


class CIFAR10H(CIFAR10):
    """`CIFAR-10H <https://github.com/jcpeterson/cifar-10h>`_ Dataset.

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

    h_test_list = ["cifar-10h-probs.npy", "7b41f73eee90fdefc73bfc820ab29ba8"]
    h_url = (
        "https://github.com/jcpeterson/cifar-10h/raw/master/data/"
        "cifar10h-probs.npy"
    )

    def __init__(
        self,
        root: str | Path,
        train: bool | None = None,
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
        download: bool = False,
    ) -> None:
        if train:
            raise ValueError("CIFAR10H does not support training data.")
        print(
            "WARNING: CIFAR10H cannot be used with Classification routines "
            "for now."
        )
        super().__init__(
            Path(root),
            train=False,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        if download:
            self.download_h()

        if not self._check_specific_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True to "
                "download it."
            )

        self.targets = list(
            torch.as_tensor(np.load(self.root / self.h_test_list[0]))
        )

    def _check_specific_integrity(self) -> bool:
        filename, md5 = self.h_test_list
        fpath = self.root / filename
        return check_integrity(fpath, md5)

    def download_h(self) -> None:
        download_url(
            self.h_url,
            self.root,
            filename=self.h_test_list[0],
            md5=self.h_test_list[1],
        )
