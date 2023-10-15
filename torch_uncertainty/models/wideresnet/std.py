# fmt: off
from typing import Type

import torch.nn.functional as F
from torch import nn

from ..utils import enable_dropout

# fmt: on
__all__ = [
    "wideresnet28x10",
]


class WideBasicBlock(nn.Module):
    def __init__(
        self,
        in_planes,
        planes,
        dropout_rate,
        stride=1,
        groups=1,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    groups=groups,
                    bias=True,
                ),
            )
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.dropout(self.conv1(x))))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(self.bn2(out))
        return out


class _Wide(nn.Module):
    def __init__(
        self,
        depth: int,
        widen_factor: int,
        in_channels: int,
        num_classes: int,
        dropout_rate: float,
        groups: int = 1,
        style: str = "imagenet",
        num_estimators: int = None,
        enable_last_layer_dropout: bool = False,
    ):
        super().__init__()
        self.in_planes = 16
        self.num_estimators = num_estimators
        self.enable_last_layer_dropout = enable_last_layer_dropout

        assert (depth - 4) % 6 == 0, "Wide-resnet depth should be 6n+4."
        num_blocks = int((depth - 4) / 6)
        k = widen_factor

        nStages = [16, 16 * k, 32 * k, 64 * k]

        if style == "imagenet":
            self.conv1 = nn.Conv2d(
                in_channels,
                nStages[0],
                kernel_size=7,
                stride=2,
                padding=3,
                groups=groups,
                bias=True,
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels,
                nStages[0],
                kernel_size=3,
                stride=1,
                padding=1,
                groups=groups,
                bias=True,
            )

        self.bn1 = nn.BatchNorm2d(nStages[0])

        if style == "imagenet":
            self.optional_pool = nn.MaxPool2d(
                kernel_size=3, stride=2, padding=1
            )
        else:
            self.optional_pool = nn.Identity()

        self.layer1 = self._wide_layer(
            WideBasicBlock,
            nStages[1],
            num_blocks=num_blocks,
            dropout_rate=dropout_rate,
            stride=1,
            groups=groups,
        )
        self.layer2 = self._wide_layer(
            WideBasicBlock,
            nStages[2],
            num_blocks=num_blocks,
            dropout_rate=dropout_rate,
            stride=2,
            groups=groups,
        )
        self.layer3 = self._wide_layer(
            WideBasicBlock,
            nStages[3],
            num_blocks=num_blocks,
            dropout_rate=dropout_rate,
            stride=2,
            groups=groups,
        )

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten(1)

        self.linear = nn.Linear(
            nStages[3],
            num_classes,
        )

    def _wide_layer(
        self,
        block: Type[WideBasicBlock],
        planes: int,
        num_blocks: int,
        dropout_rate: float,
        stride: int,
        groups,
    ):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []

        for stride in strides:
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    dropout_rate,
                    stride,
                    groups,
                )
            )
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.num_estimators is not None:
            if not self.training:
                if self.enable_last_layer_dropout is not None:
                    enable_dropout(self, self.enable_last_layer_dropout)
            x = x.repeat(self.num_estimators, 1, 1, 1)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.optional_pool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out)
        out = self.flatten(out)
        out = self.linear(out)
        return out


def wideresnet28x10(
    in_channels: int,
    num_classes: int,
    groups: int = 1,
    dropout_rate: float = 0.0,
    style: str = "imagenet",
    num_estimators: int = None,
    enable_last_layer_dropout: bool = False,
) -> nn.Module:
    """Wide-ResNet-28x10 from `Wide Residual Networks
    <https://arxiv.org/pdf/1605.07146.pdf>`_.

    Args:
        in_channels (int): Number of input channels
        num_classes (int): Number of classes to predict.
        groups (int, optional): Number of groups in convolutions. Defaults to
            ``1``.
        style (bool, optional): Whether to use the ImageNet
            structure. Defaults to ``True``.

    Returns:
        _Wide: A Wide-ResNet-28x10.
    """
    return _Wide(
        depth=28,
        widen_factor=10,
        in_channels=in_channels,
        dropout_rate=dropout_rate,
        num_classes=num_classes,
        groups=groups,
        style=style,
        num_estimators=num_estimators,
        enable_last_layer_dropout=enable_last_layer_dropout,
    )
