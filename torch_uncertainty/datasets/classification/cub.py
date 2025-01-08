import logging
from collections.abc import Callable
from pathlib import Path

import torch
from einops import rearrange
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
        load_attributes: bool = False,
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
            load_attributes (bool, optional): If True, loads the attributes of the dataset and
                returns them instead of the images. Defaults to False.
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
        if load_attributes:
            self.samples = zip(
                self._load_attributes(), [sam[1] for sam in self.samples], strict=False
            )
            self.attribute_names = self._load_attribute_names()
            self.loader = torch.nn.Identity()

        self.samples = [sample for i, sample in enumerate(self.samples) if training_idx[i] == train]
        self._labels = [label for i, label in enumerate(self.targets) if training_idx[i] == train]

        self.classnames = self._load_classnames()

    def _load_classnames(self) -> list[str]:
        with Path(self.folder_root / "CUB_200_2011" / "classes.txt").open("r") as f:
            return [
                line.split(" ")[1].split(".")[1].replace("\n", "").replace("_", " ") for line in f
            ]

    def _load_train_idx(self) -> Tensor:
        with (self.folder_root / "CUB_200_2011" / "train_test_split.txt").open("r") as f:
            return torch.as_tensor([int(line.split(" ")[1]) for line in f])

    def _load_attributes(self) -> Tensor:
        attributes = []
        with (self.folder_root / "CUB_200_2011" / "attributes" / "image_attribute_labels.txt").open(
            "r"
        ) as f:
            attributes = [
                0.5 + 2 * (int(line.split(" ")[2]) - 0.5) * (int(line.split(" ")[3]) - 1) * 1 / 6
                for line in f
            ]
        return rearrange(torch.as_tensor(attributes), "(n c) -> n c", c=312)

    def _load_attribute_names(self) -> list[str]:
        with (self.folder_root / "attributes.txt").open("r") as f:
            return [line.split(" ")[1].replace("\n", "").replace("_", " ") for line in f]

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
