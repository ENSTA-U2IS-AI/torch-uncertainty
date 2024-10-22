from collections.abc import Callable
from typing import Any

import torch
from einops import rearrange
from PIL import Image
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
from torchvision import tv_tensors
from torchvision.datasets import Cityscapes as TVCityscapes
from torchvision.transforms.v2 import functional as F


class Cityscapes(TVCityscapes):
    def __init__(
        self,
        root: str,
        split: str = "train",
        mode: str = "fine",
        target_type: list[str] | str = "instance",
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
        transforms: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__(
            root,
            split,
            mode,
            target_type,
            transform,
            target_transform,
            transforms,
        )
        train_id_to_color = [
            c.color
            for c in self.classes
            if (c.train_id != -1 and c.train_id != 255)
        ]
        train_id_to_color.append([0, 0, 0])
        self.train_id_to_color = torch.tensor(train_id_to_color)

    @classmethod
    def encode_target(cls, target: Image.Image) -> Image.Image:
        """Encode target image to tensor.

        Args:
            target (Image.Image): Target PIL image.

        Returns:
            torch.Tensor: Encoded target.
        """
        colored_target = F.pil_to_tensor(target)
        colored_target = rearrange(colored_target, "c h w -> h w c")
        target = torch.zeros_like(colored_target[..., :1])
        # convert target color to index
        for cityscapes_class in cls.classes:
            target[
                (
                    colored_target
                    == torch.tensor(cityscapes_class.id, dtype=target.dtype)
                ).all(dim=-1)
            ] = cityscapes_class.train_id

        return F.to_pil_image(rearrange(target, "h w c -> c h w"))

    def decode_target(self, target: torch.Tensor) -> torch.Tensor:
        """Decode target tensor to RGB tensor.

        Args:
            target (torch.Tensor): Target RGB tensor.

        Returns:
            Image.Image: Decoded target.
        """
        target[target == 255] = -1
        return self.train_id_to_color[target]

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """Get the sample at the given index.

        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types
                if ``target_type`` is a list with more
            than one item. Otherwise, target is a json object if
                ``target_type="polygon"``, else the image segmentation.
        """
        image = tv_tensors.Image(Image.open(self.images[index]).convert("RGB"))

        targets: Any = []
        for i, t in enumerate(self.target_type):
            if t == "polygon":
                target = self._load_json(self.targets[index][i])
            elif t == "semantic":
                target = tv_tensors.Mask(
                    self.encode_target(Image.open(self.targets[index][i]))
                )
            else:
                target = Image.open(self.targets[index][i])

            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def plot_sample(
        self, index: int, ax: _AX_TYPE | None = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a sample from the dataset.

        Args:
            index: The index of the sample to plot.
            ax: Optional matplotlib axis to plot on.

        Returns:
            The axis on which the sample was plotted.
        """
        raise NotImplementedError("This method is not implemented yet.")

    @property
    def color_palette(self) -> list[tuple[int, int, int]]:
        """Return the color palette of the dataset."""
        return [c.color for c in self.classes]
