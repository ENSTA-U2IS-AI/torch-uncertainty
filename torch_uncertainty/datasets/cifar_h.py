# fmt:off
import os
from typing import Any, Callable

import torch
from torchvision.datasets import CIFAR10
from torchvision.datasets.utils import check_integrity, download_url

import numpy as np


# fmt:on
class CIFAR10_H(CIFAR10):
    htest_list = [["cifar-10h-probs.npy", "7b41f73eee90fdefc73bfc820ab29ba8"]]
    cifarh_url = (
        "https://github.com/jcpeterson/cifar-10h/raw/master/data/"
        "cifar10h-probs.npy"
    )

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
        download: bool = False,
    ) -> None:
        """`CIFAR-10H <https://github.com/jcpeterson/cifar-10h>`_ Dataset.

        Args:
            root (string): Root directory of dataset where file
                ``cifar-10h-probs.npy`` exists or will be saved to if download
                is set to True.
            train (bool, optional): If True, creates dataset from training set,
                otherwise creates from test set.
            transform (callable, optional): A function/transform that takes in
                a PIL image and returns a transformed version. E.g,
                ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that
                takes in the target and transforms it.
            download (bool, optional): If True, downloads the dataset from the
                internet and puts it in root directory. If dataset is already
                downloaded, it is not downloaded again.
        """

        super().__init__(root, train, transform, target_transform, download)

        # Download the new targets
        if not self._check_specific_integrity() and download:
            download_url(
                self.cifarh_url,
                root,
                filename=self.htest_list[0][0],
                md5=self.htest_list[0][1],
            )
        elif not download:
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )
        self.targets = list(
            torch.as_tensor(
                np.load(os.path.join(self.root, self.htest_list[0][0]))
            )
        )

    def _check_specific_integrity(self) -> bool:
        for filename, md5 in self.htest_list:
            fpath = os.path.join(self.root, filename)
            if not check_integrity(fpath, md5):
                return False
        return True
