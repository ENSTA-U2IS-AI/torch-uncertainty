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
        return_attributes: bool = False,
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
            return_attributes (bool, optional): If True, returns the attributes instead of the images.
                Defaults to False.
            download (bool, optional): If True, downloads the dataset from the internet and puts it
                in root directory. If dataset is already downloaded, it is not downloaded again.
                Defaults to False.

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
                "Dataset not found or corrupted. You can use download=True to download it."
            )

        super().__init__(Path(root) / "CUB_200_2011" / "images", transform, target_transform)
        self.root = Path(root)

        training_idx = self._load_train_idx()
        self.attributes, self.uncertainties = self._load_attributes()
        self.attribute_names = self._load_attribute_names()
        self.classnames = self._load_classnames()

        self.samples = [sample for i, sample in enumerate(self.samples) if training_idx[i] == train]
        self._labels = [label for i, label in enumerate(self.targets) if training_idx[i] == train]
        self.attributes = rearrange(
            torch.masked_select(self.attributes, training_idx.unsqueeze(-1) == train),
            "(n c) -> n c",
            c=312,
        )
        self.uncertainties = rearrange(
            torch.masked_select(self.uncertainties, training_idx.unsqueeze(-1) == train),
            "(n c) -> n c",
            c=312,
        )

        if return_attributes:
            self.samples = zip(self.attributes, [sam[1] for sam in self.samples], strict=False)
            self.loader = torch.nn.Identity()

    def _load_classnames(self) -> list[str]:
        """Load the classnames of the dataset.

        Returns:
            list[str]: the list containing the names of the 200 classes.
        """
        with Path(self.folder_root / "CUB_200_2011" / "classes.txt").open("r") as f:
            return [
                line.split(" ")[1].split(".")[1].replace("\n", "").replace("_", " ") for line in f
            ]

    def _load_train_idx(self) -> Tensor:
        """Load the index of the training data to make the split.

        Returns:
            Tensor: whether the images belong to the training or test split.
        """
        with (self.folder_root / "CUB_200_2011" / "train_test_split.txt").open("r") as f:
            return torch.as_tensor([int(line.split(" ")[1]) for line in f])

    def _load_attributes(self) -> tuple[Tensor, Tensor]:
        """Load the attributes associated to each image.

        Returns:
            tuple[Tensor, Tensor]: The presence of the 312 attributes along with their uncertainty.
                The uncertainty is 0 for certain samples and 1 for non-visible attributes.
        """
        attributes, uncertainty = [], []
        with (self.folder_root / "CUB_200_2011" / "attributes" / "image_attribute_labels.txt").open(
            "r"
        ) as f:
            for line in f:
                attributes.append(int(line.split(" ")[2]))
                uncertainty.append(1 - (int(line.split(" ")[3]) - 1) / 3)
        return rearrange(torch.as_tensor(attributes), "(n c) -> n c", c=312), rearrange(
            torch.as_tensor(uncertainty), "(n c) -> n c", c=312
        )

    def _load_attribute_names(self) -> list[str]:
        """Load the names of the attributes.

        Returns:
            list[str]: The list of the names of the 312 attributes.
        """
        with (self.folder_root / "attributes.txt").open("r") as f:
            return [line.split(" ")[1].replace("\n", "").replace("_", " ") for line in f]

    def _check_integrity(self) -> bool:
        """Check the integrity of the dataset.

        Returns:
            bool: True when the md5 of the archive corresponds.
        """
        fpath = self.folder_root / self.filename
        return check_integrity(
            fpath,
            self.tgz_md5,
        )

    def _download(self):
        """Download the dataset from caltec.edu."""
        if self._check_integrity():
            logging.info("Files already downloaded and verified")
            return

        download_and_extract_archive(
            url=self.url, download_root=self.folder_root, filename=self.filename, md5=self.tgz_md5
        )
