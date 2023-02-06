""" Modified from https://github.com/nikitadurasov/masksembles/ """
# fmt: off
from typing import Any, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.common_types import _size_2_t

import numpy as np


# fmt: on
def _generate_masks(m: int, n: int, s: float) -> np.ndarray:
    """Generates set of binary masks with properties defined by n, m, s params.
    Results of this function are stochastic, that is, calls with the same sets
    of arguments might generate outputs of different shapes. Check
    generate_masks
    and generation_wrapper function for more deterministic behaviour.
    :param m: int, number of ones in each mask
    :param n: int, number of masks in the set
    :param s: float, scale param controls overlap of generated masks
    :return: np.ndarray, matrix of binary vectors
    """

    total_positions = int(m * s)
    masks = []

    for _ in range(n):
        new_vector = np.zeros([total_positions])
        idx = np.random.choice(range(total_positions), m, replace=False)
        new_vector[idx] = 1
        masks.append(new_vector)

    masks = np.array(masks)
    # drop useless positions
    masks = masks[:, ~np.all(masks == 0, axis=0)]
    return masks


def generate_masks(m: int, n: int, s: float) -> np.ndarray:
    """Generates set of binary masks with properties defined by n, m, s params
    Resulting masks are required to have fixed features size as it's described
    in [1].
    Since process of masks generation is stochastic therefore function
     evaluates
    _generate_masks multiple times till expected size is acquired.
    :param m: int, number of ones in each mask
    :param n: int, number of masks in the set
    :param s: float, scale param controls overlap of generated masks
    :return: np.ndarray, matrix of binary vectors
    References
    [1] `Masksembles for Uncertainty Estimation: Supplementary Material`,
    Nikita Durasov, Timur Bagautdinov, Pierre Baque, Pascal Fua
    """

    masks = _generate_masks(m, n, s)
    # hardcoded formula for expected size, check reference
    expected_size = int(m * s * (1 - (1 - 1 / s) ** n))
    while masks.shape[1] != expected_size:
        masks = _generate_masks(m, n, s)
    return masks


def generation_wrapper(c: int, n: int, scale: float) -> np.ndarray:
    """Generates set of binary masks with properties defined by c, n, scale
    params.
     Allows to generate masks sets with predefined features number c.
    Particularly
     convenient to use in torch-like layers where one need to define shapes
      inputs
     tensors beforehand.
    :param c: int, number of channels in generated masks
    :param n: int, number of masks in the set
    :param scale: float, scale param controls overlap of generated masks
    :return: np.ndarray, matrix of binary vectors
    """

    if c < 10:
        raise ValueError(
            "Masksembles approach couldn't be used in such setups where "
            "number of channels is less then 10. Current value is "
            f"(channels={c})."
            "Please increase number of features in your layer or remove this "
            "particular instance of Masksembles from your architecture."
        )

    if scale > 6.0:
        raise ValueError(
            "Masksembles approach couldn't be used in such setups where "
            f"scale parameter is larger then 6. Current value is  "
            f"(scale={scale})."
        )

    # inverse formula for number of active features in masks
    active_features = int(int(c) / (scale * (1 - (1 - 1 / scale) ** n)))

    masks = generate_masks(active_features, n, scale)
    up = 4 * scale
    down = max(0.2 * scale, 1.0)
    s = (down + up) / 2
    im_s = -1
    while im_s != c:
        masks = generate_masks(active_features, n, s)
        im_s = masks.shape[-1]
        if im_s < c:
            down = s
            s = (down + up) / 2
        elif im_s > c:
            up = s
            s = (down + up) / 2

    return masks


class Mask2D(nn.Module):
    def __init__(
        self, channels: int, num_masks: int, scale: float, **factory_kwargs
    ):
        super().__init__()

        self.channels = channels
        self.num_masks = num_masks
        self.scale = scale
        masks = generation_wrapper(channels, num_masks, scale)
        masks = torch.from_numpy(masks)
        self.masks = torch.nn.Parameter(masks, requires_grad=False).cuda()

    def forward(self, inputs):
        batch = inputs.shape[0]
        x = torch.split(inputs.unsqueeze(1), batch // self.num_masks, dim=0)
        x = torch.cat(x, dim=1).permute([1, 0, 2, 3, 4])
        x = x * self.masks.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        x = torch.cat(torch.split(x, 1, dim=0), dim=1)
        return torch.as_tensor(x, dtype=inputs.dtype).squeeze(0)


class Mask1D(nn.Module):
    def __init__(
        self, channels: int, num_masks: int, scale: float, **factory_kwargs
    ):
        super().__init__()

        self.channels = channels
        self.num_masks = num_masks
        self.scale = scale

        masks = generation_wrapper(channels, num_masks, scale)
        masks = torch.from_numpy(masks)
        self.masks = torch.nn.Parameter(masks, requires_grad=False).cuda()

    def forward(self, inputs):
        batch = inputs.shape[0]
        x = torch.split(inputs.unsqueeze(1), batch // self.num_masks, dim=0)
        x = torch.cat(x, dim=1).permute([1, 0, 2])
        x = x * self.masks.unsqueeze(1)
        x = torch.cat(torch.split(x, 1, dim=0), dim=1)
        return torch.as_tensor(x, dtype=inputs.dtype).squeeze(0)


class MaskedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_estimators: int,
        scale: float,
        bias: bool = True,
        groups: int = 1,
        device: Union[Any, None] = None,
        dtype: Union[Any, None] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.num_estimators = num_estimators

        self.mask = Mask1D(in_features, num_masks=num_estimators, scale=scale)
        self.conv1x1 = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            **factory_kwargs,
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.conv1x1(input)


class MaskedConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        num_estimators: int,
        scale: float,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        device: Union[Any, None] = None,
        dtype: Union[Any, None] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.mask = Mask2D(in_channels, num_masks=num_estimators, scale=scale)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode="zeros",
            **factory_kwargs,
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.conv(self.mask(input))
