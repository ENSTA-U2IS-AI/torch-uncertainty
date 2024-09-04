import json
import logging
from collections.abc import Callable
from pathlib import Path

from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import (
    check_integrity,
    download_and_extract_archive,
    download_url,
)


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

    url: str | list[str]
    filename: str | list[str]
    tgz_md5: str | list[str]
    dataset_name: str
    root_appendix: str

    wnid_to_idx_url = "https://raw.githubusercontent.com/torch-uncertainty/dataset-metadata/main/classification/imagenet/classes.json"
    wnid_to_idx_md5 = (
        "1bcf467b49f735dbeb745249eae6b133"  # avoid replacement attack
    )

    def __init__(
        self,
        root: str | Path,
        split: str | None = None,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        if download:
            self.download()

        self.root = Path(root)
        self.split = split

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
        if isinstance(self.filename, list):  # ImageNet-C
            integrity: bool = True
            for filename, md5 in zip(self.filename, self.tgz_md5, strict=True):
                integrity *= check_integrity(
                    self.root / self.root_appendix / filename,
                    md5,
                )
            return integrity
        raise ValueError("filename must be str or list")

    def download(self) -> None:
        """Download and extract dataset."""
        if self._check_integrity():
            logging.info("Files already downloaded and verified")
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
                self.url, self.filename, self.tgz_md5, strict=True
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
        path = self.root / "classes.json"
        if not check_integrity(path, self.wnid_to_idx_md5):
            download_url(
                self.wnid_to_idx_url,
                self.root,
                "classes.json",
                self.wnid_to_idx_md5,
            )

        with (self.root / "classes.json").open() as file:
            self.wnid_to_idx = json.load(file)

        for i in range(len(self.samples)):
            wnid = Path(self.samples[i][0]).parts[-2]
            self.targets[i] = self.wnid_to_idx[wnid]
            self.samples[i] = (self.samples[i][0], self.targets[i])
