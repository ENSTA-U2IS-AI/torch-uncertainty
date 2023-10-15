# fmt: off
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import (
    check_integrity,
    download_file_from_google_drive,
    extract_archive,
)


# fmt: on
class Fractals(ImageFolder):
    """Dataset used for PixMix augmentations.

    Args:
        root (str): Root directory of dataset.

    Note:
        There is no information on the license of the dataset. It may not
        be suitable for commercial use.
    """

    file_id = "1qC2gIUx9ARU7zhgI4IwGD3YcFhm8J4cA"
    filename = "fractals_and_fvis.tar"
    tgz_md5 = "3619fb7e2c76130749d97913fdd3ab27"

    def __init__(
        self,
        root: str,
        transform: Optional[Callable[..., Any]] = None,
        target_transform: Optional[Callable[..., Any]] = None,
        download: bool = False,
    ):
        if isinstance(root, str):
            self.root = Path(root)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True to "
                "download it."
            )

        super().__init__(
            root, transform=transform, target_transform=target_transform
        )

    def _check_integrity(self) -> bool:
        fpath = self.root / self.filename
        return check_integrity(
            fpath,
            self.tgz_md5,
        )

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_file_from_google_drive(
            file_id=self.file_id,
            root=self.root,
            filename=self.filename,
            md5=self.tgz_md5,
        )
        extract_archive(self.root / self.filename, self.root)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return super().__getitem__(index)[0]
