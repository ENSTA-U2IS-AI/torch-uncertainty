"""Adapted from https://github.com/hendrycks/robustness."""

from importlib import util
from io import BytesIO

if util.find_spec("cv2"):
    import cv2

    cv2_installed = True
else:  # coverage: ignore
    cv2_installed = False

import math as m

import numpy as np
import torch
from kornia.augmentation import RandomSaltAndPepperNoise
from PIL import Image

if util.find_spec("scipy"):
    from scipy.ndimage import map_coordinates
    from scipy.ndimage import zoom as scizoom

    scipy_installed = True
else:  # coverage: ignore
    scipy_installed = False

from torch import Tensor, nn
from torchvision.transforms import (
    InterpolationMode,
    RandomResizedCrop,
    Resize,
    ToPILImage,
    ToTensor,
)

if util.find_spec("kornia"):
    from kornia.filters import gaussian_blur2d, motion_blur

    kornia_installed = True
else:  # coverage: ignore
    kornia_installed = False

from torch_uncertainty.datasets import FrostImages

from .image import Brightness as IBrightness
from .image import Contrast as IContrast
from .image import Saturation as ISaturation

__all__ = [
    "Brightness",
    "Contrast",
    "DefocusBlur",
    "Elastic",
    "Fog",
    "Frost",
    "GaussianBlur",
    "GaussianNoise",
    "GlassBlur",
    "ImpulseNoise",
    "JPEGCompression",
    "MotionBlur",
    "Pixelate",
    "Saturation",
    "ShotNoise",
    "Snow",
    "SpeckleNoise",
    "ZoomBlur",
    "corruption_transforms",
]


class TUCorruption(nn.Module):
    def __init__(self, severity: int) -> None:
        """Base class for corruption transforms."""
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
    name = "gaussian_noise"

    def __init__(self, severity: int) -> None:
        """Apply a Gaussian noise corruption to unbatched tensor images.

        Args:
            severity (int): Severity level of the corruption.
        """
        super().__init__(severity)
        self.scale = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]

    def forward(self, img: Tensor) -> Tensor:
        if self.severity == 0:
            return img
        return torch.clamp(torch.normal(img, self.scale), 0, 1)


class ShotNoise(TUCorruption):
    name = "shot_noise"

    def __init__(self, severity: int) -> None:
        """Apply a shot (Poisson) noise corruption to unbatched tensor images.

        Args:
            severity (int): Severity level of the corruption.
        """
        super().__init__(severity)
        self.scale = [60, 25, 12, 5, 3][severity - 1]

    def forward(self, img: Tensor):
        if self.severity == 0:
            return img
        return torch.clamp(torch.poisson(img * self.scale) / self.scale, 0, 1)


class ImpulseNoise(TUCorruption):
    name = "impulse_noise"

    def __init__(self, severity: int, black_white: bool = False) -> None:
        """Apply an impulse (channel-independent Salt & Pepper) noise corruption to unbatched
        tensor images.

        Args:
            severity (int): Severity level of the corruption.
            black_white (bool): If black and white, set all pixel channel values to 0 or 1.
                Defaults to ``False`` (as in the original paper).
        """
        super().__init__(severity)
        if not kornia_installed:
            raise ImportError(
                "Please install torch_uncertainty with the image option:"
                """pip install -U "torch_uncertainty[image]"."""
            )
        self.aug = RandomSaltAndPepperNoise(
            amount=[0.03, 0.06, 0.09, 0.17, 0.27][severity - 1],
            salt_vs_pepper=0.5,
            p=1,
            same_on_batch=False,
        )
        self.black_white = black_white

    def forward(self, img: Tensor) -> Tensor:
        if self.severity == 0:
            return img
        img = img.unsqueeze(0) if self.black_white else img.unsqueeze(1)
        img = torch.clamp(
            input=torch.as_tensor(self.aug(img)),
            min=torch.zeros(1),
            max=torch.ones(1),
        )
        return img.squeeze(0) if self.black_white else img.squeeze(1)


class DefocusBlur(TUCorruption):
    name = "defocus_blur"

    def __init__(self, severity: int) -> None:
        """Apply a defocus blur corruption to unbatched tensor images.

        Args:
            severity (int): Severity level of the corruption.
        """
        super().__init__(severity)
        if not cv2_installed:
            raise ImportError(
                "Please install torch_uncertainty with the image option:"
                """pip install -U "torch_uncertainty[image]"."""
            )
        self.radius = [3, 4, 6, 8, 10][severity - 1]
        self.alias_blur = [0.1, 0.5, 0.5, 0.5, 0.5][severity - 1]

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
        return torch.clamp(torch.stack(channels), 0, 1)


class GlassBlur(TUCorruption):  # TODO: batch
    name = "glass_blur"

    def __init__(self, severity: int) -> None:
        """Apply a glass blur corruption to unbatched tensor images.

        Args:
            severity (int): Severity level of the corruption.
        """
        super().__init__(severity)
        if not kornia_installed:
            raise ImportError(
                "Please install torch_uncertainty with the image option:"
                """pip install -U "torch_uncertainty[image]"."""
            )
        sigma = [0.7, 0.9, 1, 1.1, 1.5][severity - 1]
        self.sigma = (sigma, sigma)
        self.kernel_size = m.ceil(sigma * 4)
        self.iterations = [1, 1, 1, 2, 2][severity - 1]
        self.max_delta = [1, 2, 2, 3, 4][severity - 1]

    def forward(self, img: Tensor) -> Tensor:
        img_size = img.shape
        img = gaussian_blur2d(
            img.unsqueeze(0), kernel_size=self.kernel_size, sigma=self.sigma
        ).squeeze(0)
        for _ in range(self.iterations):
            for h in range(img_size[1] - self.max_delta, self.max_delta, -1):
                for w in range(img_size[2] - self.max_delta, self.max_delta, -1):
                    dx, dy = torch.randint(-self.max_delta, self.max_delta, size=(2,))
                    h_prime, w_prime = h + dy, w + dx
                    img[:, h, w], img[:, h_prime, w_prime] = (
                        img[:, h_prime, w_prime],
                        img[:, h, w],
                    )
        return torch.clamp(
            gaussian_blur2d(
                img.unsqueeze(0), kernel_size=self.kernel_size, sigma=self.sigma
            ).squeeze(0),
            0,
            1,
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


class MotionBlur(TUCorruption):
    name = "motion_blur"

    def __init__(self, severity: int) -> None:
        """Apply a motion blur corruption to unbatched tensor images.

        Args:
            severity (int): Severity level of the corruption.

        Note:
            Originally, Hendrycks et al. used Gaussian motion blur. To remove the dependency with
            with `Wand` we changed the transform to a simpler motion blur and kept the values of
            sigma as the new kernel radius sizes.
        """
        super().__init__(severity)
        self.rng = np.random.default_rng()
        self.radius = [3, 5, 8, 12, 15][severity - 1]

        if not kornia_installed:
            raise ImportError(
                "Please install torch_uncertainty with the image option:"
                """pip install -U "torch_uncertainty[image]"."""
            )

    def forward(self, img: Tensor) -> Tensor:
        if self.severity == 0:
            return img
        no_batch = False
        if img.ndim == 3:
            no_batch = True
            img = img.unsqueeze(0)
        out = motion_blur(
            img, kernel_size=self.radius * 2 + 1, angle=self.rng.uniform(-45, 45), direction=0
        )
        if no_batch:
            out = out.squeeze(0)
        return out


def clipped_zoom(img, zoom_factor):
    h, w = img.shape[:2]
    # ceil crop height(= crop width)
    ceil_crop_height = int(np.ceil(h / zoom_factor))
    left_crop_width = int(np.ceil(w / zoom_factor))

    top = (h - ceil_crop_height) // 2
    left = (w - left_crop_width) // 2
    img = scizoom(
        img[top : top + ceil_crop_height, left : left + left_crop_width],
        (zoom_factor, zoom_factor, 1),
        order=1,
    )
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2
    trim_left = (img.shape[1] - w) // 2

    return img[trim_top : trim_top + h, trim_left : trim_left + w]


class ZoomBlur(TUCorruption):
    name = "zoom_blur"

    def __init__(self, severity: int) -> None:
        """Apply a zoom blur corruption to unbatched tensor images.

        Args:
            severity (int): Severity level of the corruption.
        """
        super().__init__(severity)
        self.zooms = [
            np.arange(1, 1.11, 0.01),
            np.arange(1, 1.16, 0.01),
            np.arange(1, 1.21, 0.02),
            np.arange(1, 1.26, 0.02),
            np.arange(1, 1.31, 0.03),
        ][severity - 1]

        if not scipy_installed:
            raise ImportError(
                "Please install torch_uncertainty with the all option:"
                """pip install -U "torch_uncertainty[all]"."""
            )

    def forward(self, img: Tensor) -> Tensor:
        if self.severity == 0:
            return img
        img = img.permute(1, 2, 0).numpy()
        out = np.zeros_like(img)
        for zoom_factor in self.zooms:
            out += clipped_zoom(img, zoom_factor)
        img = (img + out) / (len(self.zooms) + 1)
        return torch.clamp(torch.as_tensor(img).permute(2, 0, 1), 0, 1)


class Snow(TUCorruption):
    name = "snow"

    def __init__(self, severity: int) -> None:
        """Apply a snow effect on unbatched tensor images.

        Args:
            severity (int): Severity level of the corruption.

        Note:
            The transformation has been slightly modified, see MotionBlur for details.
        """
        super().__init__(severity)
        self.mix = [
            (0.1, 0.3, 3, 0.5, 4, 0.8),
            (0.2, 0.3, 2, 0.5, 4, 0.7),
            (0.55, 0.3, 4, 0.9, 8, 0.7),
            (0.55, 0.3, 4.5, 0.85, 8, 0.65),
            (0.55, 0.3, 2.5, 0.85, 12, 0.55),
        ][severity - 1]
        self.rng = np.random.default_rng()

        if not kornia_installed:
            raise ImportError(
                "Please install torch_uncertainty with the image option:"
                """pip install -U "torch_uncertainty[image]"."""
            )

    def forward(self, img: Tensor) -> Tensor:
        if self.severity == 0:
            return img
        _, height, width = img.shape
        x = img.numpy()
        snow_layer = self.rng.normal(size=x.shape[1:], loc=self.mix[0], scale=self.mix[1])[
            ..., np.newaxis
        ]
        snow_layer = clipped_zoom(snow_layer, self.mix[2])
        snow_layer[snow_layer < self.mix[3]] = 0
        snow_layer = np.clip(snow_layer.squeeze(), 0, 1)

        snow_layer = (
            motion_blur(
                torch.as_tensor(snow_layer).unsqueeze(0).unsqueeze(0),
                kernel_size=self.mix[4] * 2 + 1,
                angle=self.rng.uniform(-135, -45),
                direction=0,
            )
            .squeeze(0)
            .numpy()
        )

        x = self.mix[5] * x + (1 - self.mix[5]) * np.maximum(
            x,
            cv2.cvtColor(x.transpose([1, 2, 0]), cv2.COLOR_RGB2GRAY).reshape(1, height, width) * 1.5
            + 0.5,
        )
        return torch.clamp(torch.as_tensor(x + snow_layer + np.rot90(snow_layer, k=2)), 0, 1)


class Frost(TUCorruption):
    name = "frost"

    def __init__(self, severity: int) -> None:
        """Apply a frost corruption effect on unbatched tensor images.

        Args:
            severity (int): Severity level of the corruption.
        """
        super().__init__(severity)
        self.rng = np.random.default_rng()
        self.mix = [(1, 0.4), (0.8, 0.6), (0.7, 0.7), (0.65, 0.7), (0.6, 0.75)][severity - 1]
        self.frost_ds = FrostImages("./data", download=True, transform=ToTensor())

    def forward(self, img: Tensor) -> Tensor:
        if self.severity == 0:
            return img
        _, height, width = img.shape
        frost_img = RandomResizedCrop((height, width))(
            self.frost_ds[self.rng.integers(low=0, high=5)]
        )
        return torch.clamp(self.mix[0] * img + self.mix[1] * frost_img, 0, 1)


def plasma_fractal(height, width, wibbledecay=3):
    """Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-1.
    'mapsize' must be a power of two.
    """
    maparray = np.empty((height, width), dtype=np.float64)
    maparray[0, 0] = 0
    stepsize = height
    wibble = 100
    rng = np.random.default_rng()

    def wibbledmean(array):
        return array / 4 + wibble * rng.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart, calculate middle value as mean of points + wibble."""
        cornerref = maparray[0:height:stepsize, 0:height:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[
            stepsize // 2 : height : stepsize,
            stepsize // 2 : height : stepsize,
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
        maparray[0:mapsize:stepsize, stepsize // 2 : mapsize : stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2 : mapsize : stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


class Fog(TUCorruption):
    name = "fog"

    def __init__(self, severity: int) -> None:
        """Apply a fog corruption effect on unbatched tensor images.

        Args:
            severity (int): Severity level of the corruption.
        """
        super().__init__(severity)
        self.mix = [(1.5, 2), (2, 2), (2.5, 1.7), (2.5, 1.5), (3, 1.4)][severity - 1]

    def forward(self, img: Tensor) -> Tensor:
        if self.severity == 0:
            return img
        _, height, width = img.shape
        max_val = img.max()
        random_height_map_size = int(2 ** (m.ceil(m.log2(max(height, width) - 1))))
        fog = (
            self.mix[0]
            * plasma_fractal(
                height=random_height_map_size, width=random_height_map_size, wibbledecay=self.mix[1]
            )[:height, :width]
        )
        return torch.clamp((img + fog) * max_val / (max_val + self.mix[0]), 0, 1)


class Brightness(IBrightness, TUCorruption):
    name = "brightness"

    def __init__(self, severity: int) -> None:
        """Apply a brightness corruption to unbatched tensor images.

        Args:
            severity (int): Severity level of the corruption.
        """
        TUCorruption.__init__(self, severity)
        self.level = [1.1, 1.2, 1.3, 1.4, 1.5][severity - 1]

    def forward(self, img: Tensor) -> Tensor:
        if self.severity == 0:
            return img
        return IBrightness.forward(self, img, self.level)


class Contrast(IContrast, TUCorruption):
    name = "contrast"

    def __init__(self, severity: int) -> None:
        """Apply a contrast corruption to unbatched tensor images.

        Args:
            severity (int): Severity level of the corruption.
        """
        TUCorruption.__init__(self, severity)
        self.level = [0.4, 0.3, 0.2, 0.1, 0.05][severity - 1]

    def forward(self, img: Tensor) -> Tensor | Image.Image:
        if self.severity == 0:
            return img
        return IContrast.forward(self, img, self.level)


class Pixelate(TUCorruption):
    name = "pixelate"

    def __init__(self, severity: int) -> None:
        """Apply a pixelation corruption to unbatched tensor images.

        Args:
            severity (int): Severity level of the corruption.
        """
        super().__init__(severity)
        self.quality = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
        self.to_pil = ToPILImage()
        self.to_tensor = ToTensor()

    def forward(self, img: Tensor) -> Tensor:
        if self.severity == 0:
            return img
        _, height, width = img.shape
        img = self.to_pil(img)
        img = Resize(
            (int(height * self.quality), int(width * self.quality)),
            InterpolationMode.BOX,
        )(img)
        return self.to_tensor(Resize((height, width), InterpolationMode.BOX)(img))


class JPEGCompression(TUCorruption):
    name = "jpeg_compression"

    def __init__(self, severity: int) -> None:
        """Apply a JPEG compression corruption to unbatched tensor images.

        Args:
            severity (int): Severity level of the corruption.
        """
        super().__init__(severity)
        self.quality = [25, 18, 15, 10, 7][severity - 1]

    def forward(self, img: Tensor) -> Tensor:
        if self.severity == 0:
            return img
        output = BytesIO()
        ToPILImage()(img).save(output, "JPEG", quality=self.quality)
        return ToTensor()(Image.open(output))


class Elastic(TUCorruption):
    name = "elastic"

    def __init__(self, severity: int) -> None:
        """Apply an elastic corruption to unbatched tensor images.

        Args:
            severity (int): Severity level of the corruption.
        """
        super().__init__(severity)
        if not cv2_installed or not scipy_installed:
            raise ImportError(
                "Please install torch_uncertainty with the all option:"
                """pip install -U "torch_uncertainty[all]"."""
            )
        # The following pertubation values are based on the original repo but
        # are quite strange, notably for the severities 3 and 4
        self.mix = [
            (2, 0.7, 0.1),
            (2, 0.08, 0.2),
            (0.05, 0.01, 0.02),
            (0.07, 0.01, 0.02),
            (0.12, 0.01, 0.02),
        ][severity - 1]
        self.rng = np.random.default_rng()

    def forward(self, img: Tensor) -> Tensor:
        if self.severity == 0:
            return img
        image = np.array(img.permute(1, 2, 0), dtype=np.float32)
        shape = image.shape
        shape_size = shape[:2]

        # random affine
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
            -self.mix[2] * shape_size[0],
            self.mix[2] * shape_size[0],
            size=pts1.shape,
        ).astype(np.float32)
        affine_transform = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(
            image,
            affine_transform,
            shape_size[::-1],
            borderMode=cv2.BORDER_REFLECT_101,
        )

        sigma = self.mix[1] * shape_size[0]
        dx = (
            (
                gaussian_blur2d(
                    torch.as_tensor(self.rng.uniform(-1, 1, size=shape[:2])).unsqueeze(0),
                    kernel_size=sigma * 3,
                    sigma=(sigma, sigma),
                ).squeeze(0)
                * self.mix[0]
                * shape_size[0]
            )
            .numpy()
            .astype(np.float32)[..., np.newaxis]
        )
        dy = (
            (
                gaussian_blur2d(
                    torch.as_tensor(self.rng.uniform(-1, 1, size=shape[:2])).unsqueeze(0),
                    kernel_size=sigma * 3,
                    sigma=(sigma, sigma),
                ).squeeze(0)
                * self.mix[0]
                * shape_size[0]
            )
            .numpy()
            .astype(np.float32)[..., np.newaxis]
        )

        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = (
            np.reshape(y + dy, (-1, 1)),
            np.reshape(x + dx, (-1, 1)),
            np.reshape(z, (-1, 1)),
        )
        img = np.clip(
            map_coordinates(image, indices, order=1, mode="reflect").reshape(shape),
            0,
            1,
        )
        return torch.as_tensor(img).permute(2, 0, 1)


# Additional corruption transforms


class SpeckleNoise(TUCorruption):
    name = "speckle_noise"

    def __init__(self, severity: int) -> None:
        """Apply speckle noise to unbatched tensor images.

        Args:
            severity (int): Severity level of the corruption.
        """
        super().__init__(severity)
        self.scale = [0.15, 0.2, 0.35, 0.45, 0.6][severity - 1]
        self.rng = np.random.default_rng()

    def forward(self, img: Tensor) -> Tensor:
        if self.severity == 0:
            return img
        return torch.clamp(
            img + img * self.rng.normal(img, self.scale),
            0,
            1,
        )


class GaussianBlur(TUCorruption):
    name = "gaussian_blur"

    def __init__(self, severity: int) -> None:
        """Apply a Gaussian blur corruption to unbatched tensor images.

        Args:
            severity (int): Severity level of the corruption.
        """
        super().__init__(severity)
        if not kornia_installed:
            raise ImportError(
                "Please install torch_uncertainty with the image option:"
                """pip install -U "torch_uncertainty[image]"."""
            )
        self.sigma = [0.4, 0.6, 0.7, 0.8, 1.0][severity - 1]

    def forward(self, img: Tensor) -> Tensor:
        if self.severity == 0:
            return img
        return torch.clamp(
            gaussian_blur2d(img, kernel_size=self.sigma * 4, sigma=self.sigma),
            min=0,
            max=1,
        )


class Saturation(ISaturation, TUCorruption):
    name = "saturation"

    def __init__(self, severity: int) -> None:
        """Apply a saturation corruption to unbatched tensor images.

        Args:
            severity (int): Severity level of the corruption.
        """
        TUCorruption.__init__(self, severity)
        self.severity = severity
        self.level = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]

    def forward(self, img: Tensor) -> Tensor:
        if self.severity == 0:
            return img
        return ISaturation.forward(self, img, self.level)


corruption_transforms = (
    GaussianNoise,
    ShotNoise,
    ImpulseNoise,
    DefocusBlur,
    GlassBlur,
    MotionBlur,
    ZoomBlur,
    Snow,
    Frost,
    Fog,
    Brightness,
    Contrast,
    Elastic,
    Pixelate,
    JPEGCompression,
)
