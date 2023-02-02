from pathlib import Path
from typing import Any, Callable, Optional, Tuple

from torchvision.datasets import ImageFolder


class ImageNetO(ImageFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        download: bool = False,
        split: str = None,
    ):
        super().__init__(root=root / Path("imagenet-o"), transform=transform)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return super().__getitem__(index)

    def __len__(self) -> int:
        return len(self.samples)
