from pathlib import Path, PurePath
from typing import Any, Callable, Optional, Tuple

from torchvision.datasets import ImageFolder, ImageNet


class ImageNetR(ImageFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        split: str = None,
    ):
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
        path, target = self.samples[index]
        return super().__getitem__(index)

    def __len__(self) -> int:
        return len(self.samples)
