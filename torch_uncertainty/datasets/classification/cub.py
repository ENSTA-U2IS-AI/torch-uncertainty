import logging
from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import check_integrity, download_and_extract_archive


class CUB(ImageFolder):
    base_folder = "CUB_200_2011/images"
    url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
    filename = "CUB_200_2011.tgz"
    tgz_md5 = "97eceeb196236b17998738112f37df78"

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ):
        """The Caltech-UCSD Birds-200-2011 dataset.

        Args:
            root (str): Root directory of the dataset.
            train (bool, optional): If True, creates dataset from training set, otherwise creates
                from test set. Defaults to True.
            transform (callable, optional): A function/transform that takes in an PIL image and
                returns a transformed version. E.g, transforms.RandomCrop. Defaults to None.
            target_transform (callable, optional): A function/transform that takes in the target
                and transforms it. Defaults to None.
            download (bool, optional): If True, downloads the dataset from the internet and puts it
                in root directory. If dataset is already downloaded, it is not downloaded again.
                Defaults to
        Reference:
            Wah, C. and Branson, S. and Welinder, P. and Perona, P. and Belongie, S. Caltech-UCSD
                Birds 200.
        """
        self.folder_root = Path(root)
        self.train = train
        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True to " "download it."
            )

        super().__init__(Path(root) / "CUB_200_2011" / "images", transform, target_transform)

        training_idx = self._load_train_idx()
        self.samples = [sample for i, sample in enumerate(self.samples) if training_idx[i] == train]
        self._labels = [label for i, label in enumerate(self.targets) if training_idx[i] == train]
        with Path(self.folder_root / "CUB_200_2011" / "classes.txt").open("r") as f:
            self.class_names = [
                line.split(" ")[1].split(".")[1].replace("\n", "").replace("_", " ") for line in f
            ]

    def _load_train_idx(self) -> Tensor:
        is_training_img = []
        with (self.folder_root / "CUB_200_2011" / "train_test_split.txt").open("r") as f:
            is_training_img = [int(line.split(" ")[1]) for line in f]
        return torch.as_tensor(is_training_img)

    def _check_integrity(self) -> bool:
        fpath = self.folder_root / self.filename
        return check_integrity(
            fpath,
            self.tgz_md5,
        )

    def _download(self):
        if self._check_integrity():
            logging.info("Files already downloaded and verified")
            return

        download_and_extract_archive(
            url=self.url, download_root=self.folder_root, filename=self.filename, md5=self.tgz_md5
        )
