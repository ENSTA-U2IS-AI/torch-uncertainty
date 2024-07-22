import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import (
    check_integrity,
    download_and_extract_archive,
)


class MNISTC(VisionDataset):
    """The corrupted MNIST-C Dataset.

    Args:
        root (str): Root directory of the datasets.
        transform (callable, optional): A function/transform that takes in
            a PIL image and returns a transformed version. E.g,
            ``transforms.RandomCrop``. Defaults to None.
        target_transform (callable, optional): A function/transform that
            takes in the target and transforms it. Defaults to None.
        subset (str): The subset to use, one of ``all`` or the keys in
            ``mnistc_subsets``.
        download (bool, optional): If True, downloads the dataset from the
            internet and puts it in root directory. If dataset is already
            downloaded, it is not downloaded again. Defaults to False.

    References:
        Mu, Norman, and Justin Gilmer. "MNIST-C: A robustness benchmark for
        computer vision." In ICMLW 2019.

    License:
        The dataset is released by the dataset's authors under the Creative
        Commons Attribution 4.0.

    Note:
        This dataset does not contain severity levels. Raise an issue if you
        want someone to investigate this.
    """

    base_folder = "mnist_c"
    zip_md5 = "4b34b33045869ee6d424616cd3a65da3"
    mnistc_subsets = [
        "brightness",
        "canny_edges",
        "dotted_line",
        "fog",
        "glass_blur",
        "impulse_noise",
        "motion_blur",
        "rotate",
        "scale",
        "shear",
        "shot_noise",
        "spatter",
        "stripe",
        "translate",
        "zigzag",
    ]

    url = "https://zenodo.org/record/3239543/files/mnist_c.zip"
    filename = "mnist_c.zip"

    def __init__(
        self,
        root: str | Path,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        split: Literal["train", "test"] = "test",
        subset: str = "all",
        download: bool = False,
    ) -> None:
        self.root = Path(root)

        # Download the new targets
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it."
            )

        super().__init__(
            root=self.root / self.base_folder,
            transform=transform,
            target_transform=target_transform,
        )
        if subset not in ["all", *self.mnistc_subsets]:
            raise ValueError(
                f"The subset '{subset}' does not exist in MNIST-C."
            )
        self.subset = subset

        if split not in ["train", "test"]:
            raise ValueError(
                f"The split '{split}' should be either 'train' or 'test'."
            )
        self.split = split

        samples, labels = self.make_dataset(self.root, self.subset, self.split)

        self.samples = samples
        self.labels = labels

    def make_dataset(
        self,
        root: Path,
        subset: str,
        split: Literal["train", "test"],
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Build the corrupted dataset according to the chosen subset and
            severity. If the subset is 'all', gather all corruption types
            in the dataset.

        Args:
            root (Path):The path to the dataset.
            subset (str): The name of the corruption subset to be used. Choose
                `all` for the dataset to contain all subsets.
            split (str): The split to be used, either `train` or `test`.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The samples and labels of the chosen.
        """
        if subset == "all":
            # take any subset to get the labels
            labels: np.ndarray = np.load(root / f"identity/{split}_labels.npy")

            sample_arrays = [
                np.load(root / mnist_subset / f"{split}_images.npy")
                for mnist_subset in self.mnistc_subsets
            ]
            samples = np.concatenate(sample_arrays, axis=0)
            labels = np.tile(labels, len(self.mnistc_subsets))

        else:
            samples: np.ndarray = np.load(root / subset / f"{split}_images.npy")
            labels: np.ndarray = np.load(root / f"{split}_labels.npy")
        return samples, labels

    def __len__(self) -> int:
        """The number of samples in the dataset."""
        return self.labels.shape[0]

    def __getitem__(self, index: int) -> Any:
        """Get the samples and targets of the dataset.

        Args:
            index (int): The index of the sample to get.
        """
        sample, target = (
            self.samples[index],
            self.labels[index],
        )

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def _check_integrity(self) -> bool:
        """Check the integrity of the dataset."""
        fpath = self.root / self.filename
        return check_integrity(fpath, self.zip_md5)

    def download(self) -> None:
        """Download the dataset."""
        if self._check_integrity():
            logging.info("Files already downloaded and verified")
            return
        download_and_extract_archive(
            self.url, self.root, filename=self.filename, md5=self.zip_md5
        )
