# flake8: noqa
from .cutout import Cutout
from .transforms import (
    AutoContrast,
    Brightness,
    Color,
    Contrast,
    Equalize,
    Posterize,
    Rotation,
    Sharpness,
    Shear,
    Solarize,
    Translate,
)

augmentations = [
    AutoContrast,
    Equalize,
    Posterize,
    Rotation,
    Solarize,
    Shear,
    Translate,
    Contrast,
    Brightness,
    Color,
    Sharpness,
]
