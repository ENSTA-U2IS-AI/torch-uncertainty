import numpy as np
import torch
import torchvision.transforms.functional as F
from einops import rearrange
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


class Rotation(nn.Module):
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
            self.random_direction and np.random.uniform() > 0.5
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
            self.random_direction and np.random.uniform() > 0.5
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
            self.random_direction and np.random.uniform() > 0.5
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


class Sharpness(nn.Module):
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


class RepeatTarget(nn.Module):
    def __init__(self, num_repeats: int) -> None:
        """Repeat the targets for ensemble training.

        Args:
            num_repeats: Number of times to repeat the targets.
        """
        super().__init__()

        if not isinstance(num_repeats, int):
            raise TypeError(
                f"num_repeats must be an integer. Got {num_repeats}."
            )
        if num_repeats <= 0:
            raise ValueError(
                f"num_repeats must be greater than 0. Got {num_repeats}."
            )

        self.num_repeats = num_repeats

    def forward(self, batch: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        inputs, targets = batch
        return inputs, targets.repeat(self.num_repeats)


class MIMOBatchFormat(nn.Module):
    def __init__(
        self, num_estimators: int, rho: float = 0.0, batch_repeat: int = 1
    ) -> None:
        """Format the batch for MIMO training.

        Args:
            num_estimators: Number of estimators.
            rho: Ratio of the correlation between the images for MIMO.
            batch_repeat: Number of times to repeat the batch.

        Reference:
            Havasi, M., et al. Training independent subnetworks for robust
            prediction. In ICLR, 2021.
        """
        super().__init__()

        if num_estimators <= 0:
            raise ValueError("num_estimators must be greater than 0.")
        if not (0.0 <= rho <= 1.0):
            raise ValueError("rho must be between 0 and 1.")
        if batch_repeat <= 0:
            raise ValueError("batch_repeat must be greater than 0.")

        self.num_estimators = num_estimators
        self.rho = rho
        self.batch_repeat = batch_repeat

    def shuffle(self, inputs: Tensor) -> Tensor:
        idx = torch.randperm(inputs.nelement(), device=inputs.device)
        return inputs.view(-1)[idx].view(inputs.size())

    def forward(self, batch: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        inputs, targets = batch
        indexes = torch.arange(
            0, inputs.shape[0], device=inputs.device, dtype=torch.int64
        ).repeat(self.batch_repeat)
        main_shuffle = self.shuffle(indexes)
        threshold_shuffle = int(main_shuffle.shape[0] * (1.0 - self.rho))
        shuffle_indices = [
            torch.concat(
                [
                    self.shuffle(main_shuffle[:threshold_shuffle]),
                    main_shuffle[threshold_shuffle:],
                ],
                dim=0,
            )
            for _ in range(self.num_estimators)
        ]
        inputs = torch.stack(
            [
                torch.index_select(inputs, dim=0, index=indices)
                for indices in shuffle_indices
            ],
            dim=0,
        )
        targets = torch.stack(
            [
                torch.index_select(targets, dim=0, index=indices)
                for indices in shuffle_indices
            ],
            dim=0,
        )
        inputs = rearrange(
            inputs, "m b c h w -> (m b) c h w", m=self.num_estimators
        )
        targets = rearrange(targets, "m b -> (m b)", m=self.num_estimators)
        return inputs, targets
