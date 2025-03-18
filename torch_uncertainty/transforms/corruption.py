"""These corruptive transformations are mostly PyTorch portings of the originals provided by
Dan Hendrycks and Thomas Dietterich in "Benchmarking neural network robustness to common
corruptions and perturbations" published at ICLR 2019 through their GitHub repository
https://github.com/hendrycks/robustness.

However, please note that these transforms have been rewritten with more modern tools to improve
their efficiency as well as reduce the number of dependencies. As a result, some parameters had
to be modified to remain as close as possible to the original transforms.

The authors of the library advise avoiding using the stochastic transforms to generate your dataset
to avoid reproducibility issues. It may be preferable to first check if the corrupted dataset is
available on TorchUncertainty's Hugging Face https://huggingface.co/torch-uncertainty. File an
issue if you would like one specific and missing dataset to be published on this page.

In most of the cases, we have chosen to follow the hyperparameters used for ImageNet-C, which
differ from those of TinyImageNet-C, CIFAR-C or even the Inception version of ImageNet-C. However,
this may not be entirely suitable in the case of datasets with much smaller or bigger images.
"""

from importlib import util
from io import BytesIO

import torch.nn.functional as F
from einops import rearrange, repeat
from torch.distributions import Categorical

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
    from kornia.color import rgb_to_grayscale
    from kornia.filters import filter2d, gaussian_blur2d, motion_blur

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
    batched: bool = False

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
    batchable = True

    def __init__(self, severity: int) -> None:
        """Apply a Gaussian noise corruption to tensor images.

        Args:
            severity (int): Severity level of the corruption.
        """
        super().__init__(severity)
        self.scale = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]

    def forward(self, img: Tensor) -> Tensor:
        """Apply Gaussian noise on an input image.

        Args:
            img (Tensor): A potentially batched image of shape (C, H, W) or (B, C, H, W)
        """
        if self.severity == 0:
            return img
        return torch.clamp(torch.normal(img, self.scale), 0, 1)


class ShotNoise(TUCorruption):
    batchable = True

    def __init__(self, severity: int) -> None:
        """Apply a shot (Poisson) noise corruption to tensor images.

        Args:
            severity (int): Severity level of the corruption.
        """
        super().__init__(severity)
        self.scale = [60, 25, 12, 5, 3][severity - 1]

    def forward(self, img: Tensor):
        """Apply Poisson noise on an input image.

        Args:
            img (Tensor): A potentially batched image of shape (C, H, W) or (B, C, H, W)
        """
        if self.severity == 0:
            return img
        return torch.clamp(torch.poisson(img * self.scale) / self.scale, 0, 1)


class ImpulseNoise(TUCorruption):
    batchable = True

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
        no_batch = False
        if img.ndim == 3:
            no_batch = True
            img = img.unsqueeze(0)
        channels = img.shape[1]
        if not self.black_white:
            img = rearrange(img, "b c ... -> (b c) 1 ...")
        img = torch.clamp(
            input=self.aug(img),
            min=torch.zeros(1),
            max=torch.ones(1),
        )
        if not self.black_white:
            img = rearrange(img, "(b c) 1 ... -> b c ... ", c=channels)

        if no_batch:
            img = img.squeeze(0)
        return img.squeeze(0) if self.black_white else img.squeeze(1)


def disk(radius: int, alias_blur: float = 0.1, dtype=torch.float32):
    """Generate a Gaussian disk of shape (1, radius, radius) for filtering."""
    if radius <= 8:
        size = torch.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:  # coverage: ignore
        size = torch.arange(-radius, radius + 1)
        ksize = (5, 5)
    xs, ys = torch.meshgrid(size, size, indexing="xy")

    aliased_disk = ((xs**2 + ys**2) <= radius**2).to(dtype=dtype)
    aliased_disk /= aliased_disk.sum()
    return gaussian_blur2d(
        aliased_disk.unsqueeze(0).unsqueeze(0), kernel_size=ksize, sigma=(alias_blur, alias_blur)
    ).squeeze(0)


class DefocusBlur(TUCorruption):
    batchable = True

    def __init__(self, severity: int) -> None:
        """Apply a defocus blur corruption to unbatched tensor images.

        Args:
            severity (int): Severity level of the corruption.
        """
        super().__init__(severity)
        if not kornia_installed:
            raise ImportError(
                "Please install torch_uncertainty with the image option:"
                """pip install -U "torch_uncertainty[image]"."""
            )
        radius = [3, 4, 6, 8, 10][severity - 1]
        alias_blur = [0.1, 0.5, 0.5, 0.5, 0.5][severity - 1]
        self.disk = disk(radius, alias_blur=alias_blur)

    def forward(self, img: Tensor) -> Tensor:
        if self.severity == 0:
            return img
        no_batch = False
        if img.ndim == 3:
            no_batch = True
            img = img.unsqueeze(0)
        out = torch.clamp(filter2d(img, kernel=self.disk), 0, 1)
        if no_batch:
            out = out.squeeze(0)
        return out


def generate_offset_distribution(max_delta, iterations):
    """Symmetrized version of the glass blur swapping algorithm.

    The original implementation is sequential and extremely long on large images. This version should
    be statistically equivalent. The sketch of proof will be provided in TorchUncertainty's paper.
    """
    interval_length = 2 * max_delta + 1
    diagram_size = 12 * max_delta  # sufficient for a proper density estimation
    tab = torch.zeros((diagram_size, diagram_size), dtype=torch.float32)
    tab[0, max_delta] = 1
    for pivot, t in enumerate(range(1, diagram_size)):
        # the pivot gets 1/interval_length of all the accessible previous densities
        for i in range(-max_delta, max_delta + 1):
            if 0 <= pivot + i < diagram_size:
                tab[t, pivot] += tab[t - 1, pivot + i]

        # the other values keep (interval_length-1/interval_length of their previous densities
        # and 1/interval_length the value of the pivot
        for i in range(-max_delta, max_delta + 1):
            if i != 0 and 0 <= pivot + i < diagram_size:
                tab[t, pivot + i] += (interval_length - 1) * tab[t - 1, pivot + i] + tab[
                    t - 1, pivot
                ]
        tab[t, :] /= interval_length
    density = torch.diag(tab, -max_delta - 1)

    # reducing distribution dimention
    idx = torch.clamp(density, 1e-4).argmin()
    density = density[:idx]

    padded_density = F.pad(density, (len(density) - 2 * max_delta - 1, 0))
    sym_density = 1 / 2 * padded_density + 1 / 2 * padded_density.flip(-1)

    # Convolve the density in lieu of iterating
    sym_density = sym_density.unsqueeze(0).unsqueeze(0)
    sym_density_iter = sym_density.clone()
    for _ in range(iterations - 1):
        sym_density_iter = F.conv1d(
            sym_density_iter, torch.flip(sym_density, (-1,)), padding=sym_density.shape[-1] // 2
        )
    return Categorical(probs=sym_density_iter.squeeze(0, 1))


class GlassBlur(TUCorruption):
    def __init__(self, severity: int, seed: int | None = None) -> None:
        """Apply a glass blur corruption to unbatched tensor images.

        Faster implementation using a symetrized offset distribution.

        Args:
            severity (int): Severity level of the corruption.
            seed (int | None): Optional seed for the rng.

        Note:
            The hyperparameters have been adapted to output images qualitatively calibrated with
            the original implementation despite the changes in implementation that increase the
            power of the transformation. This is notably due to discarding the correlation between
            the offsets to simplify the derivation.
        """
        super().__init__(severity)
        if not kornia_installed:
            raise ImportError(
                "Please install torch_uncertainty with the image option:"
                """pip install -U "torch_uncertainty[image]"."""
            )
        sigma = [0.7, 0.9, 1, 1.1, 1.5][severity - 1]
        self.sigma = (sigma, sigma)
        self.kernel_size = int(sigma * 6 // 2 * 2 + 1)
        iterations = [1, 2, 3, 2, 3][severity - 1]
        max_delta = [1, 1, 1, 2, 3][severity - 1]
        self.max_delta = max_delta

        self.offset_dist = generate_offset_distribution(max_delta, iterations)

    def forward(self, img: Tensor) -> Tensor:
        if self.severity == 0:
            return img

        img = gaussian_blur2d(
            img.unsqueeze(0), kernel_size=self.kernel_size, sigma=self.sigma
        ).squeeze(0)

        img = rearrange(img, "c h w -> h w c")
        height, width, _ = img.shape
        max_d = self.max_delta

        valid_h = height - max_d
        valid_w = width - max_d

        # Generate random offsets
        rand_offsets = (
            self.offset_dist.sample(sample_shape=(valid_h, valid_w, 2))
            - self.offset_dist.param_shape[0] // 2
        )

        # Create base indices
        hs = repeat(torch.arange(max_d, height, device=img.device)[:valid_h], "h -> h w", w=valid_w)
        ws = repeat(torch.arange(max_d, width, device=img.device)[:valid_w], "w -> h w", h=valid_h)

        dy = rand_offsets[..., 0]
        dx = rand_offsets[..., 1]
        hs_prime = (hs + dy).clamp(0, height - 1)
        ws_prime = (ws + dx).clamp(0, width - 1)

        flat_idx = hs.flatten(), ws.flatten()
        flat_idx_prime = hs_prime.flatten(), ws_prime.flatten()

        tmp = img[flat_idx].clone()
        img[flat_idx] = img[flat_idx_prime]
        img[flat_idx_prime] = tmp

        img = rearrange(img, "h w c -> 1 c h w")  # Back to BCHW
        img = gaussian_blur2d(img, kernel_size=self.kernel_size, sigma=self.sigma).squeeze(0)
        return torch.clamp(img, 0, 1)


class OriginalGlassBlur(TUCorruption):
    def __init__(self, severity: int, seed: int | None = None) -> None:
        """Apply a glass blur corruption to unbatched tensor images.

        Original, likely incorrect and very slow implementation.

        Args:
            severity (int): Severity level of the corruption.
            seed (int | None): Optional seed for the rng.
        """
        super().__init__(severity)
        if not kornia_installed:
            raise ImportError(
                "Please install torch_uncertainty with the image option:"
                """pip install -U "torch_uncertainty[image]"."""
            )
        sigma = [0.7, 0.9, 1, 1.1, 1.5][severity - 1]
        self.sigma = (sigma, sigma)
        self.kernel_size = int(sigma * 4 // 2 * 2 + 1)
        self.iterations = [2, 1, 3, 2, 2][severity - 1]
        self.max_delta = [1, 2, 2, 3, 4][severity - 1]

        if seed is None:
            self.rng = None
        else:
            self.rng = torch.Generator(device="cpu").manual_seed(seed)

    def forward(self, img: Tensor) -> Tensor:
        if self.severity == 0:
            return img
        img_size = img.shape
        img = rearrange(
            gaussian_blur2d(img.unsqueeze(0), kernel_size=self.kernel_size, sigma=self.sigma),
            "1 c h w -> h w c",
        )

        rands = torch.randint(
            -self.max_delta,
            self.max_delta,
            size=(self.iterations, img_size[1] - self.max_delta, img_size[2] - self.max_delta, 2),
            generator=self.rng,
        )

        for iteration in range(self.iterations):
            for i, h in enumerate(range(img_size[1] - self.max_delta, self.max_delta, -1)):
                for j, w in enumerate(range(img_size[2] - self.max_delta, self.max_delta, -1)):
                    dx, dy = rands[iteration, i, j, :]
                    h_prime, w_prime = h + dy, w + dx
                    img[h, w, :], img[h_prime, w_prime, :] = img[h_prime, w_prime, :], img[h, w, :]

        return torch.clamp(
            gaussian_blur2d(
                rearrange(img, "h w c -> 1 c h w"), kernel_size=self.kernel_size, sigma=self.sigma
            ).squeeze(0),
            0,
            1,
        )


class MotionBlur(TUCorruption):
    batchable = True

    def __init__(self, severity: int, seed: int | None = None) -> None:
        """Apply a motion blur corruption to unbatched tensor images.

        Args:
            severity (int): Severity level of the corruption.
            seed (int | None): Optional seed for the rng.

        Note:
            Originally, Hendrycks et al. used Gaussian motion blur. To remove the dependency with
            with `Wand` we changed the transform to a simpler motion blur and kept the values of
            sigma as the new kernel radius sizes.
        """
        super().__init__(severity)
        self.rng = np.random.default_rng(seed)
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
        img = rearrange(img, "c h w -> h w c").numpy()
        out = np.zeros_like(img)
        for zoom_factor in self.zooms:
            out += clipped_zoom(img, zoom_factor)
        img = (img + out) / (len(self.zooms) + 1)
        return torch.clamp(rearrange(torch.as_tensor(img), "h w c -> c h w"), 0, 1)


class Snow(TUCorruption):
    def __init__(self, severity: int, seed: int | None = None) -> None:
        """Apply a snow effect on unbatched tensor images.

        Args:
            severity (int): Severity level of the corruption.
            seed (int | None): Optional seed for the rng.

        Note:
            The transformation has been slightly modified, see MotionBlur for details.
        """
        super().__init__(severity)
        self.mix = [
            (0.1, 3, 0.5, 4, 0.8),
            (0.2, 2, 0.5, 4, 0.7),
            (0.55, 4, 0.9, 8, 0.7),
            (0.55, 4.5, 0.85, 8, 0.65),
            (0.55, 2.5, 0.85, 12, 0.55),
        ][severity - 1]
        self.rng = np.random.default_rng(seed)

        if not kornia_installed or not scipy_installed:
            raise ImportError(
                "Please install torch_uncertainty with the image option:"
                """pip install -U "torch_uncertainty[image]"."""
            )

    def forward(self, img: Tensor) -> Tensor:
        if self.severity == 0:
            return img
        snow_layer = self.rng.normal(size=img.shape[1:], loc=self.mix[0], scale=0.3)[
            ..., np.newaxis
        ]
        snow_layer = clipped_zoom(snow_layer, self.mix[1])
        snow_layer[snow_layer < self.mix[2]] = 0
        snow_layer = np.clip(snow_layer.squeeze(), 0, 1)

        snow_layer = motion_blur(
            torch.as_tensor(snow_layer).unsqueeze(0).unsqueeze(0),
            kernel_size=self.mix[3] * 2 + 1,
            angle=self.rng.uniform(-135, -45),
            direction=0,
        ).squeeze(0)

        x = self.mix[4] * img + (1 - self.mix[4]) * torch.maximum(
            img,
            rgb_to_grayscale(img) * 1.5 + 0.5,
        )

        return torch.clamp(x + snow_layer + snow_layer.flip(dims=(1, 2)), 0, 1)


class Frost(TUCorruption):
    def __init__(self, severity: int, seed: int | None = None) -> None:
        """Apply a frost corruption effect on unbatched tensor images.

        Args:
            severity (int): Severity level of the corruption.
            seed (int | None): Optional seed for the rng.
        """
        super().__init__(severity)
        self.rng = np.random.default_rng(seed)
        self.mix = [(1, 0.4), (0.8, 0.6), (0.7, 0.7), (0.65, 0.7), (0.6, 0.75)][severity - 1]
        self.frost_ds = FrostImages("./data", download=True, transform=ToTensor())

    def forward(self, img: Tensor) -> Tensor:
        if self.severity == 0:
            return img
        frost_img = RandomResizedCrop(img.shape[1:])(
            self.frost_ds[self.rng.integers(low=0, high=5)]
        )
        return torch.clamp(self.mix[0] * img + self.mix[1] * frost_img, 0, 1)


def plasma_fractal(height, width, rng, wibbledecay):
    """Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-1.
    'mapsize' must be a power of two.
    """
    maparray = np.empty((height, width), dtype=np.float64)
    maparray[0, 0] = 0
    stepsize = height
    wibble = 100

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
    def __init__(self, severity: int, seed: int | None = None) -> None:
        """Apply a fog corruption effect on unbatched tensor images.

        Args:
            severity (int): Severity level of the corruption.
            seed (int | None): Optional seed for the rng.
        """
        super().__init__(severity)
        self.mix = [(1.5, 2), (2, 2), (2.5, 1.7), (2.5, 1.5), (3, 1.4)][severity - 1]
        self.rng = np.random.default_rng(seed)

    def forward(self, img: Tensor) -> Tensor:
        if self.severity == 0:
            return img
        _, height, width = img.shape
        max_val = img.max()
        random_height_map_size = int(2 ** (m.ceil(m.log2(max(height, width)))))
        fog = (
            self.mix[0]
            * plasma_fractal(
                height=random_height_map_size,
                width=random_height_map_size,
                wibbledecay=self.mix[1],
                rng=self.rng,
            )[:height, :width]
        )
        return torch.clamp((img + fog) * max_val / (max_val + self.mix[0]), 0, 1)


class Brightness(IBrightness, TUCorruption):
    batchable = True

    def __init__(self, severity: int) -> None:
        """Apply a brightness corruption to unbatched tensor images.

        Args:
            severity (int): Severity level of the corruption.

        Note:
            The values have been changed to better reflect the magnitude of the original
            transformation replaced with the more principled torchvision adjust_brightness.
        """
        TUCorruption.__init__(self, severity)
        self.level = [1.3, 1.6, 1.9, 2.2, 2.5][severity - 1]

    def forward(self, img: Tensor) -> Tensor:
        if self.severity == 0:
            return img
        return IBrightness.forward(self, img, self.level)


class Contrast(IContrast, TUCorruption):
    batchable = True

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
    def __init__(self, severity: int, seed: int | None = None) -> None:
        """Apply an elastic corruption to unbatched tensor images.

        Args:
            severity (int): Severity level of the corruption.
            seed (int | None): Optional seed for the rng.

        Note:
            mix[0][1] has been changed to 0.5 to avoid errors when dealing with small images.
        """
        super().__init__(severity)
        if not cv2_installed or not scipy_installed:
            raise ImportError(
                "Please install torch_uncertainty with the all option:"
                """pip install -U "torch_uncertainty[all]"."""
            )
        self.mix = [
            (2, 0.5, 0.1),
            (2, 0.08, 0.2),
            (0.05, 0.01, 0.02),
            (0.07, 0.01, 0.02),
            (0.12, 0.01, 0.02),
        ][severity - 1]
        self.rng = np.random.default_rng(seed)

    def forward(self, img: Tensor) -> Tensor:
        if self.severity == 0:
            return img
        image = np.array(rearrange(img, "c h w -> h w c"), dtype=np.float32)
        height, width, channels = image.shape
        shape_size = height, width
        min_shape_size = min(shape_size)

        # random affine
        center_square = np.float32(shape_size) // 2
        square_size = min_shape_size // 3
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
            -self.mix[2] * min_shape_size,
            self.mix[2] * min_shape_size,
            size=pts1.shape,
        ).astype(np.float32)
        affine_transform = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(
            image,
            affine_transform,
            shape_size[::-1],
            borderMode=cv2.BORDER_REFLECT_101,
        )

        sigma = self.mix[1] * min_shape_size
        ks = min(int((sigma * 3 // 2) * 2 + 1), min_shape_size // 2 * 2 - 1)
        dx = (
            (
                gaussian_blur2d(
                    torch.as_tensor(self.rng.uniform(-1, 1, size=(1, 1, *shape_size))),
                    kernel_size=ks,
                    sigma=(sigma, sigma),
                ).squeeze(0, 1)
                * self.mix[0]
                * shape_size[1]
            )
            .numpy()
            .astype(np.float32)[..., np.newaxis]
        )
        dy = (
            (
                gaussian_blur2d(
                    torch.as_tensor(self.rng.uniform(-1, 1, size=(1, 1, *shape_size))),
                    kernel_size=ks,
                    sigma=(sigma, sigma),
                ).squeeze(0, 1)
                * self.mix[0]
                * shape_size[0]
            )
            .numpy()
            .astype(np.float32)[..., np.newaxis]
        )

        x, y, z = np.meshgrid(np.arange(width), np.arange(height), np.arange(channels))
        indices = (
            np.reshape(y + dy, (-1, 1)),
            np.reshape(x + dx, (-1, 1)),
            np.reshape(z, (-1, 1)),
        )
        img = np.clip(
            map_coordinates(image, indices, order=1, mode="reflect").reshape(
                (height, width, channels)
            ),
            0,
            1,
        )
        return rearrange(torch.as_tensor(img), "h w c -> c h w")


# Additional corruption transforms


class SpeckleNoise(TUCorruption):
    batchable = True

    def __init__(self, severity: int, seed: int | None = None) -> None:
        """Apply speckle noise to tensor images.

        Args:
            severity (int): Severity level of the corruption.
            seed (int | None): Optional seed for the rng.
        """
        super().__init__(severity)
        self.scale = [0.15, 0.2, 0.35, 0.45, 0.6][severity - 1]
        self.rng = np.random.default_rng(seed)

    def forward(self, img: Tensor) -> Tensor:
        """Apply speckle noise on images.

        Args:
            img (Tensor): A potentially batched image of shape (C, H, W) or (B, C, H, W).
        """
        if self.severity == 0:
            return img
        return torch.clamp(
            img * self.rng.normal(1, self.scale, size=img.shape),
            0,
            1,
        )


class GaussianBlur(TUCorruption):
    batchable = True

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
        sigma = [1, 2, 3, 4, 6][severity - 1]
        self.sigma = (sigma, sigma)
        self.kernel_size = int(sigma // 2 * 2 * 4 + 1)

    def forward(self, img: Tensor) -> Tensor:
        if self.severity == 0:
            return img
        no_batch = False
        if img.ndim == 3:
            no_batch = True
            img = img.unsqueeze(0)
        out = torch.clamp(
            gaussian_blur2d(img, kernel_size=self.kernel_size, sigma=self.sigma),
            0,
            1,
        )
        if no_batch:
            out = out.squeeze(0)
        return out


class Saturation(ISaturation, TUCorruption):
    batchable = True

    def __init__(self, severity: int) -> None:
        """Apply a saturation corruption to unbatched tensor images.

        Args:
            severity (int): Severity level of the corruption.
        """
        TUCorruption.__init__(self, severity)
        self.severity = severity
        self.level = [0.8, 0.6, 0.4, 0.2, 0.1][severity - 1]

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
