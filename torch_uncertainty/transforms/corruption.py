"""Adapted from https://github.com/hendrycks/robustness."""

from importlib import util
from io import BytesIO

if util.find_spec("cv2"):
    import cv2

    cv2_installed = True
else:  # coverage: ignore
    cv2_installed = False

import numpy as np
import torch
from PIL import Image

if util.find_spec("skimage"):
    from skimage.filters import gaussian
    from skimage.util import random_noise

    skimage_installed = True
else:  # coverage: ignore
    skimage_installed = False

from scipy.ndimage import map_coordinates
from scipy.ndimage import zoom as scizoom
from torch import Tensor, nn
from torchvision.transforms import (
    InterpolationMode,
    RandomResizedCrop,
    Resize,
    ToPILImage,
    ToTensor,
)
from wand.api import library as wandlibrary
from wand.image import Image as WandImage

from torch_uncertainty.datasets import FrostImages

from .image import Brightness as IBrightness
from .image import Contrast as IContrast
from .image import Saturation as ISaturation

__all__ = [
    "GaussianNoise",
    "ShotNoise",
    "ImpulseNoise",
    "DefocusBlur",
    "GlassBlur",
    "MotionBlur",
    "ZoomBlur",
    "Snow",
    "Frost",
    "Fog",
    "Brightness",
    "Contrast",
    "Elastic",
    "Pixelate",
    "JPEGCompression",
    "GaussianBlur",
    "SpeckleNoise",
]


class TUCorruption(nn.Module):
    def __init__(self, severity: int) -> None:
        """Base class for corruptions."""
        super().__init__()
        if not (0 <= severity <= 5):
            raise ValueError("Severity must be between 0 and 5.")
        if not isinstance(severity, int):
            raise TypeError("Severity must be an integer.")
        self.severity = severity

    def __repr__(self) -> str:
        """Printable representation."""
        return self.__class__.__name__ + f"(severity={self.severity})"


class GaussianNoise(TUCorruption):
    def __init__(self, severity: int) -> None:
        """Add Gaussian noise to an image.

        Args:
            severity (int): Severity level of the corruption.
        """
        super().__init__(severity)
        self.scale = [0, 0.04, 0.06, 0.08, 0.09, 0.10][severity]

    def forward(self, img: Tensor) -> Tensor:
        if self.severity == 0:
            return img
        return torch.clip(torch.normal(img, self.scale), 0, 1)


class ShotNoise(TUCorruption):
    def __init__(self, severity: int) -> None:
        """Add shot noise to an image.

        Args:
            severity (int): Severity level of the corruption.
        """
        super().__init__(severity)
        self.scale = [500, 250, 100, 75, 50][severity - 1]

    def forward(self, img: Tensor):
        if self.severity == 0:
            return img
        return torch.clip(torch.poisson(img * self.scale) / self.scale, 0, 1)


class ImpulseNoise(TUCorruption):
    def __init__(self, severity: int) -> None:
        """Add impulse noise to an image.

        Args:
            severity (int): Severity level of the corruption.
        """
        super().__init__(severity)
        if not skimage_installed:  # coverage: ignore
            raise ImportError(
                "Please install torch_uncertainty with the image option:"
                """pip install -U "torch_uncertainty[image]"."""
            )
        self.scale = [0, 0.01, 0.02, 0.03, 0.05, 0.07][severity]

    def forward(self, img: Tensor) -> Tensor:
        if self.severity == 0:
            return img
        return torch.clip(
            torch.as_tensor(random_noise(img, mode="s&p", amount=self.scale)),
            0,
            1,
        )


class DefocusBlur(TUCorruption):
    def __init__(self, severity: int) -> None:
        """Add defocus blur to an image.

        Args:
            severity (int): Severity level of the corruption.
        """
        super().__init__(severity)
        if not cv2_installed:  # coverage: ignore
            raise ImportError(
                "Please install torch_uncertainty with the image option:"
                """pip install -U "torch_uncertainty[image]"."""
            )
        self.radius = [0.3, 0.4, 0.5, 1, 1.5][severity - 1]
        self.alias_blur = [0.4, 0.5, 0.6, 0.2, 0.1][severity - 1]

    def forward(self, img: Tensor) -> Tensor:
        if self.severity == 0:
            return img
        img = img.numpy()
        channels = [
            torch.as_tensor(
                cv2.filter2D(
                    img[ch, :, :],
                    -1,
                    disk(self.radius, alias_blur=self.alias_blur),
                )
            )
            for ch in range(3)
        ]
        return torch.clip(torch.stack(channels), 0, 1)


class GlassBlur(TUCorruption):  # TODO: batch
    def __init__(self, severity: int) -> None:
        super().__init__(severity)
        if not skimage_installed or not cv2_installed:  # coverage: ignore
            raise ImportError(
                "Please install torch_uncertainty with the image option:"
                """pip install -U "torch_uncertainty[image]"."""
            )
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


class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


class MotionBlur(TUCorruption):
    def __init__(self, severity: int) -> None:
        super().__init__(severity)
        self.rng = self.rng.default_rng()

    def forward(self, img: Tensor) -> Tensor:
        if self.severity == 0:
            return img
        output = BytesIO()
        ToPILImage()(img).save(output, "PNG")
        x = MotionImage(blob=output.getvalue())
        x.motion_blur(radius=10, sigma=3, angle=self.rng.uniform(-45, 45))
        x = cv2.imdecode(
            np.fromstring(x.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED
        )
        return torch.clip(torch.as_tensor(x)[..., [2, 1, 0]], 0, 255) / 255


def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / zoom_factor))

    top = (h - ch) // 2
    img = scizoom(
        img[top : top + ch, top : top + ch],
        (zoom_factor, zoom_factor, 1),
        order=1,
    )
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2

    return img[trim_top : trim_top + h, trim_top : trim_top + h]


class ZoomBlur(TUCorruption):
    def __init__(self, severity: int) -> None:
        super().__init__(severity)
        self.zooms = [
            np.arange(1, 1.11, 0.01),
            np.arange(1, 1.16, 0.01),
            np.arange(1, 1.21, 0.02),
            np.arange(1, 1.26, 0.02),
            np.arange(1, 1.31, 0.03),
        ][severity - 1]

    def forward(self, img: Tensor) -> Tensor:
        if self.severity == 0:
            return img
        img = img.numpy()
        out = np.zeros_like(img)
        for zoom_factor in self.zooms:
            out += clipped_zoom(img, zoom_factor)
        img = (img + out) / (len(self.zooms) + 1)
        return torch.clip(torch.as_tensor(img), 0, 1)


class Snow(TUCorruption):
    def __init__(self, severity: int) -> None:
        super().__init__(severity)
        self.snow_layers = [
            (0.1, 0.3, 3, 0.5, 10, 4, 0.8),
            (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
            (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
            (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
            (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55),
        ][severity - 1]
        self.rng = self.rng.default_rng()

    def forward(self, img: Tensor) -> Tensor:
        if self.severity == 0:
            return img
        _, height, width = img.shape
        x = img.numpy()
        snow_layer = self.rng.normal(
            size=x.shape[:2], loc=self.snow_layers[0], scale=self.snow_layers[1]
        )
        snow_layer = clipped_zoom(
            snow_layer[..., np.newaxis], self.snow_layers[2]
        )
        snow_layer[snow_layer < self.snow_layers[3]] = 0
        snow_layer = Image.fromarray(
            (np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8),
            mode="L",
        )
        output = BytesIO()
        snow_layer.save(output, format="PNG")
        snow_layer = MotionImage(blob=output.getvalue())
        snow_layer.motion_blur(
            radius=self.snow_layers[4],
            sigma=self.snow_layers[5],
            angle=self.rng.uniform(-135, -45),
        )
        snow_layer = (
            cv2.imdecode(
                np.fromstring(snow_layer.make_blob(), np.uint8),
                cv2.IMREAD_UNCHANGED,
            )
            / 255.0
        )
        snow_layer = snow_layer[..., np.newaxis]
        x = self.snow_layers[6] * x + (1 - self.snow_layers[6]) * np.maximum(
            x,
            cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(height, width, 1) * 1.5
            + 0.5,
        )
        return torch.clip(
            torch.as_tensor(x + snow_layer + np.rot90(snow_layer, k=2)), 0, 1
        )


class Frost(TUCorruption):
    def __init__(self, severity: int) -> None:
        super().__init__(severity)
        self.rng = self.rng.default_rng()
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
            self.frost_ds[self.rng.integers(low=0, high=4)]
        )
        return torch.clip(self.mix[0] * img + self.mix[1] * frost_img, 0, 1)


class Fog(TUCorruption):
    def __init__(self, severity: int) -> None:
        super().__init__(severity)
        self.mix = [(1.5, 2), (2, 2), (2.5, 1.7), (2.5, 1.5), (3, 1.4)][
            severity - 1
        ]

    def forward(self, img: Tensor) -> Tensor:
        _, height, width = img.shape
        if self.severity == 0:
            return img
        max_val = img.max()
        img += (
            self.mix[0]
            * plasma_fractal(wibbledecay=self.mix[1])[:height, :width][
                ..., np.newaxis
            ]
        )
        return torch.clip(img * max_val / (max_val + self.mix[0]), 0, 1)


class Brightness(TUCorruption, IBrightness):
    def __init__(self, severity: int) -> None:
        super(TUCorruption, self).__init__(severity)
        super(IBrightness, self).__init__()
        self.level = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]

    def forward(self, img: Tensor) -> Tensor:
        if self.severity == 0:
            return img
        return super(IBrightness, self).forward(img, self.level)


class Contrast(TUCorruption, IContrast):
    def __init__(self, severity: int) -> None:
        super(TUCorruption, self).__init__(severity)
        super(IContrast, self).__init__()
        self.level = [0.4, 0.3, 0.2, 0.1, 0.05][severity - 1]

    def forward(self, img: Tensor) -> Tensor:
        if self.severity == 0:
            return img
        return super(IContrast, self).forward(img, self.level)


class Pixelate(TUCorruption):
    def __init__(self, severity: int) -> None:
        super().__init__(severity)
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


class JPEGCompression(TUCorruption):
    def __init__(self, severity: int) -> None:
        super().__init__(severity)
        self.quality = [80, 65, 58, 50, 40][severity - 1]

    def forward(self, img: Tensor) -> Tensor:
        if self.severity == 0:
            return img
        output = BytesIO()
        ToPILImage()(img).save(output, "JPEG", quality=self.quality)
        return ToTensor()(Image.open(output))


class Elastic(TUCorruption):
    def __init__(self, severity: int) -> None:
        super().__init__(severity)
        if not cv2_installed:
            raise ImportError(
                "Please install torch_uncertainty with the image option:"
                """pip install -U "torch_uncertainty[image]"."""
            )
        # Check with different sizes
        self.mixing = [
            (
                244 * 2,
                244 * 0.7,
                244 * 0.1,
            ),  # 244 should have been 224, but ultimately nothing is incorrect
            (244 * 2, 244 * 0.08, 244 * 0.2),
            (244 * 0.05, 244 * 0.01, 244 * 0.02),
            (244 * 0.07, 244 * 0.01, 244 * 0.02),
            (244 * 0.12, 244 * 0.01, 244 * 0.02),
        ][severity - 1]
        self.rng = self.rng.default_rng()

    def forward(self, img: Tensor) -> Tensor:
        shape = img.shape
        shape_size = shape[1:]
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32(
            [
                center_square + square_size,
                [
                    center_square[0] + square_size,
                    center_square[1] - square_size,
                ],
                center_square - square_size,
            ]
        )
        pts2 = pts1 + self.rng.uniform(
            -self.mixing[2], self.mixing[2], size=pts1.shape
        ).astype(np.float32)
        m = cv2.getAffineTransform(pts1, pts2)
        img = cv2.warpAffine(
            img, m, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101
        )

        dx = (
            gaussian(
                self.rng.uniform(-1, 1, size=shape[:2]),
                self.mixing[1],
                mode="reflect",
                truncate=3,
            )
            * self.mixing[0]
        ).astype(np.float32)
        dy = (
            gaussian(
                self.rng.uniform(-1, 1, size=shape[:2]),
                self.mixing[1],
                mode="reflect",
                truncate=3,
            )
            * self.mixing[0]
        ).astype(np.float32)
        dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]
        x, y, z = np.meshgrid(
            np.arange(shape[2]), np.arange(shape[1]), np.arange(shape[0])
        )
        indices = (
            np.reshape(y + dy, (-1, 1)),
            np.reshape(x + dx, (-1, 1)),
            np.reshape(z, (-1, 1)),
        )
        return torch.as_tensor(
            np.clip(
                map_coordinates(img, indices, order=1, mode="reflect").reshape(
                    shape
                ),
                0,
                1,
            )
        )


# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(mapsize=256, wibbledecay=3):
    """Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert mapsize & (mapsize - 1) == 0
    maparray = np.empty((mapsize, mapsize), dtype=np.float64)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100
    rng = np.random.default_rng()

    def wibbledmean(array):
        return array / 4 + wibble * rng.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart, calculate middle value as mean of points + wibble."""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[
            stepsize // 2 : mapsize : stepsize,
            stepsize // 2 : mapsize : stepsize,
        ] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart, calculate middle value as mean of points + wibble."""
        mapsize = maparray.shape[0]
        drgrid = maparray[
            stepsize // 2 : mapsize : stepsize,
            stepsize // 2 : mapsize : stepsize,
        ]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2 : mapsize : stepsize] = (
            wibbledmean(ltsum)
        )
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2 : mapsize : stepsize, 0:mapsize:stepsize] = (
            wibbledmean(ttsum)
        )

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


class SpeckleNoise(TUCorruption):
    def __init__(self, severity: int) -> None:
        super().__init__(severity)
        self.scale = [0.06, 0.1, 0.12, 0.16, 0.2][severity - 1]
        self.rng = self.rng.default_rng()

    def forward(self, img: Tensor) -> Tensor:
        if self.severity == 0:
            return img
        return torch.clip(
            img + img * self.rng.normal(img, self.scale),
            0,
            1,
        )


class GaussianBlur(TUCorruption):
    def __init__(self, severity: int) -> None:
        super().__init__(severity)
        if not skimage_installed:  # coverage: ignore
            raise ImportError(
                "Please install torch_uncertainty with the image option:"
                """pip install -U "torch_uncertainty[image]"."""
            )
        self.sigma = [0.4, 0.6, 0.7, 0.8, 1.0][severity - 1]

    def forward(self, img: Tensor) -> Tensor:
        if self.severity == 0:
            return img
        return torch.clip(
            torch.as_tensor(gaussian(img, sigma=self.sigma)),
            0,
            1,
        )


class Saturate(ISaturation):
    def __init__(self, severity: int) -> None:
        super().__init__()
        if not (0 <= severity <= 5):
            raise ValueError("Severity must be between 0 and 5.")
        if not isinstance(severity, int):
            raise TypeError("Severity must be an integer.")
        self.severity = severity
        self.level = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]

    def forward(self, img: Tensor) -> Tensor:
        if self.severity == 0:
            return img
        return super().forward(img, self.level)
