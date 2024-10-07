from typing import Any

import torch
import torchvision.transforms.v2.functional as F
from PIL import Image, ImageEnhance
from torch import Tensor, nn
from torchvision.transforms.v2 import InterpolationMode, Transform
from torchvision.transforms.v2._utils import query_size


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
        center: list[float] | None = None,
        fill: list[float] | None = None,
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
        center: list[float] | None = None,
        fill: list[float] | None = None,
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
        shear = [0.0, 0.0]
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
        center: list[float] | None = None,
        fill: list[float] | None = None,
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
        self, img: Tensor | Image.Image, level: float
    ) -> Tensor | Image.Image:
        if (
            self.random_direction and torch.rand(1).item() > 0.5
        ):  # coverage: ignore
            level = -level
        translate = [0.0, 0.0]
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

    def forward(self, img: Tensor, level: float) -> Tensor:
        if level < 0:
            raise ValueError("Level must be greater than 0.")
        return F.adjust_brightness(img, level)


class Saturation(nn.Module):
    level_type = float
    corruption_overlap = True

    def __init__(self) -> None:
        super().__init__()

    def forward(self, img: Tensor, level: float) -> Tensor:
        if level < 0:
            raise ValueError("Level must be greater than 0.")
        return F.adjust_saturation_image(img, level)


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
        pil_img = F.to_pil_image(img) if isinstance(img, Tensor) else img
        pil_img = ImageEnhance.Color(pil_img).enhance(level)
        if isinstance(img, Tensor):
            return F.pil_to_tensor(pil_img)
        return pil_img


class RandomRescale(Transform):
    """Randomly rescale the input.

    This transformation can be used together with ``RandomCrop`` as data augmentations to train
    models on image segmentation task.

    Output spatial size is randomly sampled from the interval ``[min_size, max_size]``:

    .. code-block:: python

        scale = uniform_sample(min_scale, max_scale)
        output_width = input_width * scale
        output_height = input_height * scale

    If the input is a :class:`torch.Tensor` or a ``TVTensor`` (e.g. :class:`~torchvision.tv_tensors.Image`,
    :class:`~torchvision.tv_tensors.Video`, :class:`~torchvision.tv_tensors.BoundingBoxes` etc.)
    it can have arbitrary number of leading batch dimensions. For example,
    the image can have ``[..., C, H, W]`` shape. A bounding box can have ``[..., 4]`` shape.

    Args:
        min_scale (int): Minimum scale for random sampling
        max_scale (int): Maximum scale for random sampling
        interpolation (InterpolationMode, optional): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.NEAREST_EXACT``,
            ``InterpolationMode.BILINEAR`` and ``InterpolationMode.BICUBIC`` are supported.
            The corresponding Pillow integer constants, e.g. ``PIL.Image.BILINEAR`` are accepted as well.
        antialias (bool, optional): Whether to apply antialiasing.
            It only affects **tensors** with bilinear or bicubic modes and it is
            ignored otherwise: on PIL images, antialiasing is always applied on
            bilinear or bicubic modes; on other modes (for PIL images and
            tensors), antialiasing makes no sense and this parameter is ignored.
            Possible values are:

            - ``True`` (default): will apply antialiasing for bilinear or bicubic modes.
              Other mode aren't affected. This is probably what you want to use.
            - ``False``: will not apply antialiasing for tensors on any mode. PIL
              images are still antialiased on bilinear or bicubic modes, because
              PIL doesn't support no antialias.
            - ``None``: equivalent to ``False`` for tensors and ``True`` for
              PIL images. This value exists for legacy reasons and you probably
              don't want to use it unless you really know what you are doing.

            The default value changed from ``None`` to ``True`` in
            v0.17, for the PIL and Tensor backends to be consistent.
    """

    def __init__(
        self,
        min_scale: int,
        max_scale: int,
        interpolation: InterpolationMode | int = InterpolationMode.BILINEAR,
        antialias: bool | None = True,
    ) -> None:
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.interpolation = interpolation
        self.antialias = antialias

    def _get_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        height, width = query_size(flat_inputs)
        scale = torch.rand(1)
        scale = self.min_scale + scale * (self.max_scale - self.min_scale)
        return {"size": (int(height * scale), int(width * scale))}

    def _transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        return self._call_kernel(
            F.resize,
            inpt,
            params["size"],
            interpolation=self.interpolation,
            antialias=self.antialias,
        )
