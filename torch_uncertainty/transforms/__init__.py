# flake8: noqa
from .cutout import Cutout
from .mimo_format import MIMOBatchFormat
from .transforms import (
    AutoContrast,
    Brightness,
    Color,
    Contrast,
    Equalize,
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
