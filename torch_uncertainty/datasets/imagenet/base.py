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
    url = None
    filename = None
    tgz_md5 = None
    dataset_name = None

    wnid_to_idx_url = "https://raw.githubusercontent.com/torch-uncertainty/"
    "imagenet-classes/master/wnid_to_idx.txt"
    wnid_to_idx_md5 = (
        "7fac43d97231a87a264a118fa76a13ad"  # avoid replacement attack
    )

    def __init__(
        self,
        root: str,
        split: str = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        if isinstance(root, str):
            root = Path(root)

        self.root = root

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        super().__init__(
            root=self.root / Path(self.dataset_name),
            transform=transform,
            target_transform=target_transform,
        )

        self._repair_dataset()

    def _check_integrity(self) -> bool:
        return check_integrity(
            self.root / Path(self.filename),
            self.tgz_md5,
        )

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(
            self.url, self.root, filename=self.filename, md5=self.tgz_md5
        )

    def _repair_dataset(self) -> None:
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
            self.samples[i] = (self.samples[i][0], self.targets[i])
            self.targets[i] = self.wnid_to_idx[wnid]
