import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import (
    check_integrity,
    download_and_extract_archive,
)


class NotMNIST(ImageFolder):
    """The notMNIST dataset.

    Args:
        root (str): Root directory of the datasets.
        subset (str): The subset to use, one of ``small`` or ``large``.
        transform (callable, optional): A function/transform that takes in
            a PIL image and returns a transformed version. E.g,
            ``transforms.RandomCrop``. Defaults to None.
        target_transform (callable, optional): A function/transform that
            takes in the target and transforms it. Defaults to None.
        download (bool, optional): If True, downloads the dataset from the
            internet and puts it in root directory. If dataset is already
            downloaded, it is not downloaded again. Defaults to False.

    Note:
        There is no information on the license of the dataset. It may not
        be suitable for commercial use.
    """

    url_base = "https://zenodo.org/record/8274268/files/"
    filenames = ["notMNIST_small.zip", "notMNIST_large.zip"]
    tgz_md5s = [
        "3de91fb69221d9c2d5c57387101ebc6c",
        "c3f9e0862df000a897766593044e366a",
    ]
    subsets = ["small", "large"]

    def __init__(
        self,
        root: str | Path,
        subset: Literal["small", "large"] = "small",
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
        download: bool = False,
    ) -> None:
        self.root = Path(root)

        if subset not in self.subsets:
            raise ValueError(
                f"The subset '{subset}' does not exist for notMNIST."
            )
        ind = self.subsets.index(subset)
        self.url = self.url_base + "/" + self.filenames[ind]
        self.filename = self.filenames[ind]
        self.tgz_md5 = self.tgz_md5s[ind]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True to "
                "download it."
            )

        super().__init__(
            self.root / f"notMNIST_{subset}",
            transform=transform,
            target_transform=target_transform,
        )

    def _check_integrity(self) -> bool:
        fpath = self.root / self.filename
        return check_integrity(
            fpath,
            self.tgz_md5,
        )

    def download(self) -> None:
        if self._check_integrity():
            logging.info("Files already downloaded and verified")
            return

        download_and_extract_archive(
            self.url,
            download_root=self.root,
            filename=self.filename,
            md5=self.tgz_md5,
        )
        logging.info("Downloaded %s to %s.", self.filename, self.root)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """Get the samples and targets of the dataset.

        Args:
            index (int): The index of the sample to get.
        """
        return super().__getitem__(index)
