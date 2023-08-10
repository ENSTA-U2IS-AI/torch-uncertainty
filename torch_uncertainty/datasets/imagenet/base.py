# fmt: off
from pathlib import Path
from typing import Callable, Optional

from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import (
    check_integrity,
    download_and_extract_archive,
    download_url,
)


# fmt:on
class ImageNetVariation(ImageFolder):
    """Virtual base class for ImageNet variations.

    Args:
        root (str): Root directory of the datasets.
        split (str, optional): For API consistency. Defaults to None.
        transform (callable, optional): A function/transform that takes in
                a PIL image and returns a transformed version. E.g,
                ``transforms.RandomCrop``. Defaults to None.
        target_transform (callable, optional): A function/transform that
            takes in the target and transforms it. Defaults to None.
        download (bool, optional): If True, downloads the dataset from the
            internet and puts it in root directory. If dataset is already
            downloaded, it is not downloaded again. Defaults to False.
    """

    url = None
    filename = None
    tgz_md5 = None
    dataset_name = None
    root_appendix = ""

    wnid_to_idx_url = (
        "https://raw.githubusercontent.com/torch-uncertainty/"
        "imagenet-classes/master/wnid_to_idx.txt"
    )
    wnid_to_idx_md5 = (
        "7fac43d97231a87a264a118fa76a13ad"  # avoid replacement attack
    )

    def __init__(
        self,
        root: str,
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        if isinstance(root, str):
            root = Path(root)

        self.root = root

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. "
                "You can use download=True to download it."
            )

        super().__init__(
            root=self.root / Path(self.dataset_name),
            transform=transform,
            target_transform=target_transform,
        )

        self._repair_dataset()

    def _check_integrity(self) -> bool:
        """Check the integrity of the dataset(s)."""
        if isinstance(self.filename, str):
            return check_integrity(
                self.root / Path(self.filename),
                self.tgz_md5,
            )
        elif isinstance(self.filename, list):  # ImageNet-C
            integrity: bool = True
            for filename, md5 in zip(self.filename, self.tgz_md5):
                integrity *= check_integrity(
                    self.root / self.root_appendix / Path(filename),
                    md5,
                )
            return integrity
        else:
            raise ValueError("filename must be str or list")

    def download(self) -> None:
        """Download and extract dataset."""
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        if isinstance(self.filename, str):
            download_and_extract_archive(
                self.url,
                self.root,
                extract_root=self.root / self.root_appendix,
                filename=self.filename,
                md5=self.tgz_md5,
            )
        elif isinstance(self.filename, list):  # ImageNet-C
            for url, filename, md5 in zip(
                self.url, self.filename, self.tgz_md5
            ):
                # Check that this particular file is not already downloaded
                if not check_integrity(
                    self.root / self.root_appendix / Path(filename), md5
                ):
                    download_and_extract_archive(
                        url,
                        self.root,
                        extract_root=self.root / self.root_appendix,
                        filename=filename,
                        md5=md5,
                    )

    def _repair_dataset(self) -> None:
        """Download the wnid_to_idx.txt file and to get the correct targets."""
        path = self.root / "imagenet_wnid_to_idx.txt"
        if not check_integrity(path, self.wnid_to_idx_md5):
            download_url(
                self.wnid_to_idx_url,
                self.root,
                "imagenet_wnid_to_idx.txt",
                self.wnid_to_idx_md5,
            )
        with open(self.root / "imagenet_wnid_to_idx.txt") as file:
            self.wnid_to_idx = eval(file.read())

        for i in range(len(self.samples)):
            wnid = Path(self.samples[i][0]).parts[-2]
            self.targets[i] = self.wnid_to_idx[wnid]
            self.samples[i] = (self.samples[i][0], self.targets[i])
