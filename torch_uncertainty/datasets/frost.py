from collections.abc import Callable
from importlib import util
from importlib.abc import Traversable
from importlib.resources import files
from pathlib import Path
from typing import Any

from PIL import Image
from torchvision.datasets import VisionDataset

FROST_ASSETS_MOD = "torch_uncertainty_assets.frost"
tu_assets_installed = util.find_spec("torch_uncertainty_assets")


def pil_loader(path: Path | Traversable) -> Image.Image:
    with path.open("rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class FrostImages(VisionDataset):
    def __init__(
        self,
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
    ) -> None:
        if not tu_assets_installed:  # coverage: ignore
            raise ImportError(
                "The torch-uncertainty-assets library is not installed. Please install"
                "torch_uncertainty with the image option:"
                """pip install -U "torch_uncertainty[image]"."""
            )
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
