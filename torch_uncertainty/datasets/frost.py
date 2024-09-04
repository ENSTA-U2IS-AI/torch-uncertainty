import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import (
    check_integrity,
    download_and_extract_archive,
)


def pil_loader(path: Path) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with path.open("rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class FrostImages(VisionDataset):  # TODO: Use ImageFolder
    url = "https://zenodo.org/records/10438904/files/frost.zip"
    zip_md5 = "d82f29f620d43a68e71e34b28f7c35cb"
    filename = "frost.zip"
    samples = [
        "frost1.png",
        "frost2.png",
        "frost3.jpg",
        "frost4.jpg",
        "frost5.jpg",
    ]

    def __init__(
        self,
        root: str | Path,
        transform: Callable[..., Any] | None,
        target_transform: Callable[..., Any] | None = None,
        download: bool = False,
    ) -> None:
        self.root = Path(root)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True to "
                "download it."
            )

        super().__init__(
            self.root / "frost",
            transform=transform,
            target_transform=target_transform,
        )
        self.loader = pil_loader

    def _check_integrity(self) -> bool:
        fpath = self.root / self.filename
        return check_integrity(
            fpath,
            self.zip_md5,
        )

    def download(self) -> None:
        if self._check_integrity():
            logging.info("Files already downloaded and verified")
            return

        download_and_extract_archive(
            self.url,
            download_root=self.root,
            filename=self.filename,
            md5=self.zip_md5,
        )
        logging.info("Downloaded %s to %s.", self.filename, self.root)

    def __getitem__(self, index: int) -> Any:
        """Get the samples of the dataset.

        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.root / self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.samples)
