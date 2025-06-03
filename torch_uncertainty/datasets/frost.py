from collections.abc import Callable
from importlib.abc import Traversable
from importlib.resources import files
from pathlib import Path
from typing import Any

from PIL import Image
from torchvision.datasets import VisionDataset


def pil_loader(path: Path | Traversable) -> Image.Image:
    with path.open("rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


FROST_ASSETS_MOD = "torch_uncertainty_assets.frost"


class FrostImages(VisionDataset):
    def __init__(
        self,
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__(
            FROST_ASSETS_MOD,
            transform=transform,
            target_transform=target_transform,
        )
        self.loader = pil_loader
        sample_path = files(FROST_ASSETS_MOD)
        self.samples = [sample_path.joinpath(f"frost{i}.jpg") for i in range(1, 6)]

    def __getitem__(self, index: int) -> Any:
        """Get the samples of the dataset.

        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        sample = self.loader(self.samples[index])
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.samples)
