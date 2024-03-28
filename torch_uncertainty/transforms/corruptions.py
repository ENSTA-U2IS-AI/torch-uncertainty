"""Adapted from https://github.com/hendrycks/robustness."""

from importlib import util
from io import BytesIO

if util.find_spec("cv2"):  # coverage: ignore
    import cv2
import numpy as np
import torch
from PIL import Image

if util.find_spec("skimage"):  # coverage: ignore
    from skimage.filters import gaussian
    from skimage.util import random_noise
from torch import Tensor, nn
from torchvision.transforms import (
    InterpolationMode,
    RandomResizedCrop,
    Resize,
    ToPILImage,
    ToTensor,
)

from torch_uncertainty.datasets import FrostImages

__all__ = [
    "DefocusBlur",
    "Frost",
    "GaussianBlur",
    "GaussianNoise",
    "GlassBlur",
    "ImpulseNoise",
    "JPEGCompression",
    "Pixelate",
    "ShotNoise",
    "SpeckleNoise",
]


class GaussianNoise(nn.Module):
    def __init__(self, severity: int) -> None:
        super().__init__()
        if not (0 <= severity <= 5):
            raise ValueError("Severity must be between 0 and 5.")
        if not isinstance(severity, int):
            raise TypeError("Severity must be an integer.")
        self.severity = severity
        self.scale = [0, 0.04, 0.06, 0.08, 0.09, 0.10][severity]

    def forward(self, img: Tensor) -> Tensor:
        if self.severity == 0:
            return img
        return torch.clip(torch.normal(img, self.scale), 0, 1)

    def __repr__(self) -> str:
        """Printable representation."""
        return self.__class__.__name__ + f"(severity={self.severity})"


class ShotNoise(nn.Module):
    def __init__(self, severity: int) -> None:
        super().__init__()
        if not (0 <= severity <= 5):
            raise ValueError("Severity must be between 0 and 5.")
        if not isinstance(severity, int):
            raise TypeError("Severity must be an integer.")
        self.severity = severity
        self.scale = [500, 250, 100, 75, 50][severity - 1]

    def forward(self, img: Tensor):
        if self.severity == 0:
            return img
        return torch.clip(torch.poisson(img * self.scale) / self.scale, 0, 1)

    def __repr__(self) -> str:
        """Printable representation."""
        return self.__class__.__name__ + f"(severity={self.severity})"


class ImpulseNoise(nn.Module):
    def __init__(self, severity: int) -> None:
        super().__init__()
        if not (0 <= severity <= 5):
            raise ValueError("Severity must be between 0 and 5.")
        if not isinstance(severity, int):
            raise TypeError("Severity must be an integer.")
        self.severity = severity
        self.scale = [0, 0.01, 0.02, 0.03, 0.05, 0.07][severity]

    def forward(self, img: Tensor) -> Tensor:
        if self.severity == 0:
            return img
        return torch.clip(
            torch.as_tensor(random_noise(img, mode="s&p", amount=self.scale)),
            0,
            1,
        )

    def __repr__(self) -> str:
        """Printable representation."""
        return self.__class__.__name__ + f"(severity={self.severity})"


class SpeckleNoise(nn.Module):
    def __init__(self, severity: int) -> None:
        super().__init__()
        if not (0 <= severity <= 5):
            raise ValueError("Severity must be between 0 and 5.")
        if not isinstance(severity, int):
            raise TypeError("Severity must be an integer.")
        self.severity = severity
        self.scale = [0.06, 0.1, 0.12, 0.16, 0.2][severity - 1]

    def forward(self, img: Tensor) -> Tensor:
        if self.severity == 0:
            return img
        return torch.clip(
            img + img * torch.normal(img, self.scale),
            0,
            1,
        )

    def __repr__(self) -> str:
        """Printable representation."""
        return self.__class__.__name__ + f"(severity={self.severity})"


class GaussianBlur(nn.Module):
    def __init__(self, severity: int) -> None:
        super().__init__()
        if not (0 <= severity <= 5):
            raise ValueError("Severity must be between 0 and 5.")
        if not isinstance(severity, int):
            raise TypeError("Severity must be an integer.")
        self.severity = severity
        self.sigma = [0.4, 0.6, 0.7, 0.8, 1.0][severity - 1]

    def forward(self, img: Tensor) -> Tensor:
        if self.severity == 0:
            return img
        return torch.clip(
            torch.as_tensor(gaussian(img, sigma=self.sigma)),
            0,
            1,
        )

    def __repr__(self) -> str:
        """Printable representation."""
        return self.__class__.__name__ + f"(severity={self.severity})"


class GlassBlur(nn.Module):  # TODO: batch
    def __init__(self, severity: int) -> None:
        super().__init__()
        if not (0 <= severity <= 5):
            raise ValueError("Severity must be between 0 and 5.")
        if not isinstance(severity, int):
            raise TypeError("Severity must be an integer.")
        self.severity = severity
        self.sigma = [0.05, 0.25, 0.4, 0.25, 0.4][severity - 1]
        self.max_delta = 1
        self.iterations = [1, 1, 1, 2, 2][severity - 1]

    def forward(self, img: Tensor) -> Tensor:
        img_size = img.shape
        img = torch.as_tensor(gaussian(img, sigma=self.sigma))
        for _ in range(self.iterations):
            for h in range(img_size[0] - self.max_delta, self.max_delta, -1):
                for w in range(
                    img_size[1] - self.max_delta, self.max_delta, -1
                ):
                    dx, dy = torch.randint(
                        -self.max_delta, self.max_delta, size=(2,)
                    )
                    h_prime, w_prime = h + dy, w + dx
                    img[h, w], img[h_prime, w_prime] = (
                        img[h_prime, w_prime],
                        img[h, w],
                    )
        return torch.clip(
            torch.as_tensor(gaussian(img, sigma=self.sigma)), 0, 1
        )

    def __repr__(self) -> str:
        """Printable representation."""
        return self.__class__.__name__ + f"(severity={self.severity})"


def disk(radius: int, alias_blur: float = 0.1, dtype=np.float32):
    if radius <= 8:
        size = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:  # coverage: ignore
        size = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    xs, ys = np.meshgrid(size, size)
    aliased_disk = np.array((xs**2 + ys**2) <= radius**2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


class DefocusBlur(nn.Module):
    def __init__(self, severity: int) -> None:
        super().__init__()
        if not (0 <= severity <= 5):
            raise ValueError("Severity must be between 0 and 5.")
        if not isinstance(severity, int):
            raise TypeError("Severity must be an integer.")
        self.severity = severity
        self.radius = [0.3, 0.4, 0.5, 1, 1.5][severity - 1]
        self.alias_blur = [0.4, 0.5, 0.6, 0.2, 0.1][severity - 1]

    def forward(self, img: Tensor) -> Tensor:
        if self.severity == 0:
            return img
        img = np.array(img)
        channels = [
            torch.as_tensor(
                cv2.filter2D(
                    img[d, :, :],
                    -1,
                    disk(self.radius, alias_blur=self.alias_blur),
                )
            )
            for d in range(3)
        ]
        return torch.clip(torch.stack(channels), 0, 1)

    def __repr__(self) -> str:
        """Printable representation."""
        return self.__class__.__name__ + f"(severity={self.severity})"


class JPEGCompression(nn.Module):  # TODO: batch
    def __init__(self, severity: int) -> None:
        super().__init__()
        if not (0 <= severity <= 5):
            raise ValueError("Severity must be between 0 and 5.")
        if not isinstance(severity, int):
            raise TypeError("Severity must be an integer.")
        self.severity = severity
        self.quality = [80, 65, 58, 50, 40][severity - 1]

    def forward(self, img: Tensor) -> Tensor:
        if self.severity == 0:
            return img
        output = BytesIO()
        ToPILImage()(img).save(output, "JPEG", quality=self.quality)
        return ToTensor()(Image.open(output))

    def __repr__(self) -> str:
        """Printable representation."""
        return self.__class__.__name__ + f"(severity={self.severity})"


class Pixelate(nn.Module):  # TODO: batch
    def __init__(self, severity: int) -> None:
        super().__init__()
        if not (0 <= severity <= 5):
            raise ValueError("Severity must be between 0 and 5.")
        if not isinstance(severity, int):
            raise TypeError("Severity must be an integer.")
        self.severity = severity
        self.quality = [0.95, 0.9, 0.85, 0.75, 0.65][severity - 1]

    def forward(self, img: Tensor) -> Tensor:
        if self.severity == 0:
            return img
        _, height, width = img.shape
        img = ToPILImage()(img)
        img = Resize(
            (int(height * self.quality), int(width * self.quality)),
            InterpolationMode.BOX,
        )(img)
        return ToTensor()(Resize((height, width), InterpolationMode.BOX)(img))

    def __repr__(self) -> str:
        """Printable representation."""
        return self.__class__.__name__ + f"(severity={self.severity})"


class Frost(nn.Module):
    def __init__(self, severity: int) -> None:
        super().__init__()
        if not (0 <= severity <= 5):
            raise ValueError("Severity must be between 0 and 5.")
        if not isinstance(severity, int):
            raise TypeError("Severity must be an integer.")
        self.severity = severity
        self.mix = [(1, 0.2), (1, 0.3), (0.9, 0.4), (0.85, 0.4), (0.75, 0.45)][
            severity - 1
        ]
        self.frost_ds = FrostImages(
            "./data", download=True, transform=ToTensor()
        )

    def forward(self, img: Tensor) -> Tensor:
        if self.severity == 0:
            return img
        _, height, width = img.shape
        frost_img = RandomResizedCrop((height, width))(
            self.frost_ds[np.random.randint(5)]
        )

        return torch.clip(self.mix[0] * img + self.mix[1] * frost_img, 0, 1)
