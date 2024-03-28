# ruff: noqa: F401
from .batch import MIMOBatchFormat, RepeatTarget
from .cutout import Cutout
from .image import (
    AutoContrast,
    Brightness,
    Color,
    Contrast,
    Equalize,
    Posterize,
    RandomRescale,
    Rotate,
    Sharpen,
    Shear,
    Solarize,
    Translate,
)
from .mixup import Mixup, MixupIO, RegMixup, WarpingMixup

augmentations = [
    AutoContrast,
    Equalize,
    Posterize,
    Rotate,
    Solarize,
    Shear,
    Translate,
    Contrast,
    Brightness,
    Color,
    Sharpen,
]
