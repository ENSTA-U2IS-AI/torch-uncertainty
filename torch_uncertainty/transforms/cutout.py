# fmt: off
import torch
from torch import nn

import numpy as np


# fmt: on
class Cutout(nn.Module):
    """Cutout augmentation class.

    Args:
        length (int): Length of the cutout square.
        value (int): Pixel value to be filled in the cutout square.
    """

    def __init__(self, length: int, value: int = 0):
        super().__init__()

        if length <= 0:
            raise ValueError("Cutout length must be positive.")
        else:
            self.length = length

        if value < 0 or value > 255:
            raise ValueError("Cutout value must be between 0 and 255.")

        self.value = value

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if len(img.shape) == 2:
            img = img.unsqueeze(0)
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = self.value
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img
