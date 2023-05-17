# fmt: off
from typing import Any

from PIL import Image, ImageEnhance, ImageOps

import numpy as np


# fmt: on
def get_ab(beta):
    if np.random.random() < 0.5:
        a = np.float32(np.random.beta(beta, 1))
        b = np.float32(np.random.beta(1, beta))
    else:
        a = 1 + np.float32(np.random.beta(1, beta))
        b = -np.float32(np.random.beta(1, beta))
    return a, b


def add(img1: Image, img2: Image, beta: float) -> Image:
    a, b = get_ab(beta)
    img1, img2 = img1 * 2 - 1, img2 * 2 - 1
    out = a * img1 + b * img2
    return (out + 1) / 2


def multiply(img1: Image, img2: Image, beta: float) -> Image:
    a, b = get_ab(beta)
    img1, img2 = img1 * 2, img2 * 2
    out = (img1**a) * (img2.clip(1e-37) ** b)
    return out / 2


# Summarize mixing operations
mixings = [add, multiply]


def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval.

    Args:
        level: Level of the operation that will be between [0, `PARAMETER_MAX`]
        maxval: Maximum value that the operation can have. This will be scaled
        to level/PARAMETER_MAX.

    Returns:
        An int that results from scaling `maxval` according to `level`.
    """
    return int(level * maxval / 10)


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval.

    Args:
        level: Level of the operation that will be in the range
            [0, `PARAMETER_MAX`]
        maxval: Maximum value that the operation can have. This will be scaled
            to level/PARAMETER_MAX.

    Returns:
        A float that results from scaling `maxval` according to `level`.
    """
    return float(level) * maxval / 10.0


def sample_level(n: float) -> float:
    return np.random.uniform(low=0.1, high=n)


def autocontrast(img: Image, _: Any) -> Image:
    return ImageOps.autocontrast(img)


def equalize(img: Image, _: Any) -> Image:
    return ImageOps.equalize(img)


def posterize(img: Image, level: float) -> Image:
    level = int_parameter(sample_level(level), 4)
    return ImageOps.posterize(img, 4 - level)


def rotate(img: Image, level: float) -> Image:
    degrees = int_parameter(sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return img.rotate(degrees, resample=Image.BILINEAR)


def solarize(img: Image, level: float) -> Image:
    level = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(img, 256 - level)


def shear_x(img: Image, level: float) -> Image:
    h, w, _ = img.shape
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return img.transform(
        (h, w),
        Image.AFFINE,
        (1, level, 0, 0, 1, 0),
        resample=Image.BILINEAR,
    )


def shear_y(img: Image, level: float) -> Image:
    h, w, _ = img.shape
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return img.transform(
        (h, w),
        Image.AFFINE,
        (1, 0, 0, level, 1, 0),
        resample=Image.BILINEAR,
    )


def translate_x(img: Image, level: float) -> Image:
    h, w, _ = img.shape
    level = int_parameter(sample_level(level), min(h, w) / 3)
    if np.random.random() > 0.5:
        level = -level
    return img.transform(
        (h, w),
        Image.AFFINE,
        (1, 0, level, 0, 1, 0),
        resample=Image.BILINEAR,
    )


def translate_y(img: Image, level: float) -> Image:
    h, w, _ = img.shape
    level = int_parameter(sample_level(level), min(h, w) / 3)
    if np.random.random() > 0.5:
        level = -level
    return img.transform(
        (h, w),
        Image.AFFINE,
        (1, 0, 0, 0, 1, level),
        resample=Image.BILINEAR,
    )


# operation that overlaps with ImageNet-C's test set
def color(img: Image, level: float) -> Image:
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(img: Image, level: float) -> Image:
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(img: Image, level: float) -> Image:
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(img: Image, level: float) -> Image:
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(img).enhance(level)


# Summarize all operations
augmentations = [
    autocontrast,
    equalize,
    posterize,
    rotate,
    solarize,
    shear_x,
    shear_y,
    translate_x,
    translate_y,
]

augmentations_all = [
    autocontrast,
    equalize,
    posterize,
    rotate,
    solarize,
    shear_x,
    shear_y,
    translate_x,
    translate_y,
    color,
    contrast,
    brightness,
    sharpness,
]
