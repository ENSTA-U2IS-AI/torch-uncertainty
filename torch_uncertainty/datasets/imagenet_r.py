from pathlib import Path, PurePath
from typing import Any, Callable, Optional, Tuple

from torchvision.datasets import ImageFolder, ImageNet
from torchvision.datasets.utils import download_and_extract_archive


class ImageNetR(ImageFolder):
    url = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar"
    filename = "imagenet-r.tar"
    tgz_md5 = "a61312130a589d0ca1a8fca1f2bd3337"

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        if download:
            self.download()

        super().__init__(
            root=root / Path("imagenet-r"),
            transform=transform,
            target_transform=target_transform,
        )
        imagenet = ImageNet(root, split="val")
        self.wnid_to_idx = imagenet.wnid_to_idx
        for id, sample in enumerate(self.samples):
            cls = PurePath(sample[0]).parts[-2]
            self.samples[id] = (sample[0], self.wnid_to_idx[cls])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return super().__getitem__(index)

    def __len__(self) -> int:
        return len(self.samples)

    def _check_integrity(self) -> bool:
        print("Warning: integrity check not implemented")
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(
            self.url, self.root, filename=self.filename, md5=self.tgz_md5
        )
