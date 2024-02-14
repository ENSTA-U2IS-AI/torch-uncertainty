import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageEnhance
from torch import Tensor, nn


class AutoContrast(nn.Module):
    pixmix_max_level = None
    level_type = None
    corruption_overlap = False

    def forward(self, img: Tensor | Image.Image) -> Tensor | Image.Image:
        return F.autocontrast(img)


class Equalize(nn.Module):
    pixmix_max_level = None
    level_type = None
    corruption_overlap = False

    def forward(self, img: Tensor | Image.Image) -> Tensor | Image.Image:
        return F.equalize(img)


class Posterize(nn.Module):
    max_level = 4
    pixmix_max_level = 4
    level_type = int
    corruption_overlap = False

    def forward(
        self, img: Tensor | Image.Image, level: int
    ) -> Tensor | Image.Image:
        if level >= self.max_level:
            raise ValueError(f"Level must be less than {self.max_level}.")
        if level < 0:
            raise ValueError("Level must be greater than 0.")
        return F.posterize(img, self.max_level - level)


class Solarize(nn.Module):
    max_level = 256
    pixmix_max_level = 256
    level_type = int
    corruption_overlap = False

    def forward(
        self, img: Tensor | Image.Image, level: float
    ) -> Tensor | Image.Image:
        if level >= self.max_level:
            raise ValueError(f"Level must be less than {self.max_level}.")
        if level < 0:
            raise ValueError("Level must be greater than 0.")
        return F.solarize(img, self.max_level - level)


class Rotate(nn.Module):
    pixmix_max_level = 30
    level_type = float
    corruption_overlap = False

    def __init__(
        self,
        random_direction: bool = True,
        interpolation: F.InterpolationMode = F.InterpolationMode.NEAREST,
        expand: bool = False,
        center: list[int] | None = None,
        fill: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.random_direction = random_direction
        self.interpolation = interpolation
        self.expand = expand
        self.center = center
        self.fill = fill

    def forward(
        self, img: Tensor | Image.Image, level: float
    ) -> Tensor | Image.Image:
        if (
            self.random_direction and torch.rand(1).item() > 0.5
        ):  # coverage: ignore
            level = -level
        return F.rotate(
            img,
            level,
            interpolation=self.interpolation,
            expand=self.expand,
            center=self.center,
            fill=self.fill,
        )


class Shear(nn.Module):
    pixmix_max_level = 0.3
    level_type = float
    corruption_overlap = False

    def __init__(
        self,
        axis: int,
        random_direction: bool = True,
        interpolation: F.InterpolationMode = F.InterpolationMode.NEAREST,
        center: list[int] | None = None,
        fill: list[int] | None = None,
    ) -> None:
        super().__init__()
        if axis not in (0, 1):
            raise ValueError("Axis must be 0 or 1.")
        self.axis = axis
        self.random_direction = random_direction
        self.interpolation = interpolation
        self.center = center
        self.fill = fill

    def forward(
        self, img: Tensor | Image.Image, level: int
    ) -> Tensor | Image.Image:
        if (
            self.random_direction and torch.rand(1).item() > 0.5
        ):  # coverage: ignore
            level = -level
        shear = [0, 0]
        shear[self.axis] = level
        return F.affine(
            img,
            angle=0,
            scale=1.0,
            shear=shear,
            translate=[0, 0],
            interpolation=self.interpolation,
            center=self.center,
            fill=self.fill,
        )


class Translate(nn.Module):
    pixmix_max_level = 0.45
    level_type = float
    corruption_overlap = False

    def __init__(
        self,
        axis: int,
        random_direction: bool = True,
        interpolation: F.InterpolationMode = F.InterpolationMode.NEAREST,
        center: list[int] | None = None,
        fill: list[int] | None = None,
    ) -> None:
        super().__init__()
        if axis not in (0, 1):
            raise ValueError("Axis must be 0 or 1.")
        self.axis = axis
        self.random_direction = random_direction
        self.interpolation = interpolation
        self.center = center
        self.fill = fill

    def forward(
        self, img: Tensor | Image.Image, level: int
    ) -> Tensor | Image.Image:
        if (
            self.random_direction and torch.rand(1).item() > 0.5
        ):  # coverage: ignore
            level = -level
        translate = [0, 0]
        translate[self.axis] = level
        return F.affine(
            img,
            angle=0,
            scale=1.0,
            shear=[0, 0],
            translate=translate,
            interpolation=self.interpolation,
            center=self.center,
            fill=self.fill,
        )


class Contrast(nn.Module):
    pixmix_max_level = 1.8
    level_type = float
    corruption_overlap = True

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, img: Tensor | Image.Image, level: float
    ) -> Tensor | Image.Image:
        if level < 0:
            raise ValueError("Level must be greater than 0.")
        return F.adjust_contrast(img, level)


class Brightness(nn.Module):
    pixmix_max_level = 1.8
    level_type = float
    corruption_overlap = True

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, img: Tensor | Image.Image, level: float
    ) -> Tensor | Image.Image:
        if level < 0:
            raise ValueError("Level must be greater than 0.")
        return F.adjust_brightness(img, level)


class Sharpen(nn.Module):
    pixmix_max_level = 1.8
    level_type = float
    corruption_overlap = True

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, img: Tensor | Image.Image, level: float
    ) -> Tensor | Image.Image:
        if level < 0:
            raise ValueError("Level must be greater than 0.")
        return F.adjust_sharpness(img, level)


class Color(nn.Module):
    pixmix_max_level = 1.8
    level_type = float
    corruption_overlap = True

    def __init__(self) -> None:
        """Color augmentation class."""
        super().__init__()

    def forward(
        self, img: Tensor | Image.Image, level: float
    ) -> Tensor | Image.Image:
        if level < 0:
            raise ValueError("Level must be greater than 0.")
        if isinstance(img, Tensor):
            img: Image.Image = F.to_pil_image(img)
        return ImageEnhance.Color(img).enhance(level)
