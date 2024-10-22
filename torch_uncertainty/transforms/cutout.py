import torch
from torch import nn


class Cutout(nn.Module):
    def __init__(self, length: int, value: int = 0) -> None:
        """Cutout augmentation class.

        Args:
            length (int): Length of the cutout square.
            value (int): Pixel value to be filled in the cutout square.
        """
        super().__init__()

        if length <= 0:
            raise ValueError(f"Cutout length must be positive. Got {length}.")
        self.length = length

        if value < 0 or value > 255:
            raise ValueError(
                f"Cutout value must be between 0 and 255. Got {value}."
            )
        self.value = value

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if len(img.shape) == 2:
            img = img.unsqueeze(0)
        h, w = img.size(1), img.size(2)
        mask = torch.ones(size=(h, w), dtype=torch.float32)
        y = torch.randint(high=h, size=(1,))
        x = torch.randint(high=w, size=(1,))

        y1 = torch.clip(y - self.length // 2, 0, h)
        y2 = torch.clip(y + self.length // 2, 0, h)
        x1 = torch.clip(x - self.length // 2, 0, w)
        x2 = torch.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = self.value
        return img * mask.expand_as(img)
