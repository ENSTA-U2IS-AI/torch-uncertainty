import logging
from collections.abc import Callable
from pathlib import Path

from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import (
    check_integrity,
    download_and_extract_archive,
)


class OpenImageO(ImageFolder):
    url = "https://zenodo.org/records/10540831/files/OpenImage-O.zip"
    filename = "OpenImage-O.zip"
    md5sum = "c0abd7cd4b6f218a7149adc718d70e6e"

    def __init__(
        self,
        root: str | Path,
        split: str | None = None,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        """OpenImage-O dataset.

        Args:
            root (str): Root directory of the datasets.
            split (str, optional): Unused, for API consistency. Defaults to
                None.
            transform (callable, optional): A function/transform that takes in
                a PIL image and returns a transformed version. E.g,
                ``transforms.RandomCrop``. Defaults to None.
            target_transform (callable, optional): A function/transform that
                takes in the target and transforms it. Defaults to None.
            download (bool, optional): If True, downloads the dataset from the
                internet and puts it in root directory. If dataset is already
                downloaded, it is not downloaded again. Defaults to False.

        References:
            Original dataset: The open images dataset v4: Unified image
            classification, object detection, and visual relationship detection
            at scale. Kuznetsova, A., et al. The International Journal of
            Computer Vision.

            Curation: ViM: Out-Of-Distribution with Virtual-logit Matching.
            Wang H., et al. In CVPR 2022.
        """
        self.root = Path(root)
        self.split = split

        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        super().__init__(
            self.root / "openimage-o/",
            transform=transform,
            target_transform=target_transform,
        )

    def _check_integrity(self) -> bool:
        fpath = self.root / self.filename
        return check_integrity(
            fpath,
            self.md5sum,
        )

    def download(self) -> None:
        if self._check_integrity():
            logging.info("Files already downloaded and verified")
            return

        download_and_extract_archive(
            self.url,
            download_root=self.root,
            extract_root=self.root / "openimage-o/ood/",
            filename=self.filename,
            md5=self.md5sum,
        )
