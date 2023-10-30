# ruff: noqa: F401
from .cutout import Cutout
from .transforms import (
    AutoContrast,
    Brightness,
    Color,
    Contrast,
    Equalize,
    MIMOBatchFormat,
    Posterize,
    RepeatTarget,
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
