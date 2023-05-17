# fmt: off

import torch
import torch.nn as nn
from torch import Tensor

# fmt: on
__all__ = [
    "dummy_model",
]


class _Dummy(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
    ) -> None:
        super().__init__()

        self.linear = nn.Linear(
            1,
            num_classes,
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.linear(torch.as_tensor(1))
        return out


def dummy_model(
    in_channels: int,
    num_classes: int,
    imagenet_structure: bool = True,
) -> _Dummy:
    """Dummy model for testing purposes.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        imagenet_structure (bool, optional): Whether to use the ImageNet
            structure. Defaults to True.

    Returns:
        _Dummy: Dummy model.
    """
    return _Dummy(
        in_channels=in_channels,
        num_classes=num_classes,
        imagenet_structure=imagenet_structure,
    )
