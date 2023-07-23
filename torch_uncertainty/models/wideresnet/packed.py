# fmt: off
from typing import Type

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ...layers import PackedConv2d, PackedLinear

# fmt: on
__all__ = [
    "packed_wideresnet28x10",
]


class WideBasicBlock(nn.Module):
    def __init__(
        self,
        in_planes: int,
        planes: int,
        dropout_rate: float,
        stride: int = 1,
        alpha: float = 2,
        num_estimators: int = 4,
        gamma: int = 1,
        groups: int = 1,
    ):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(alpha * in_planes)
        self.conv1 = PackedConv2d(
            in_planes,
            planes,
            kernel_size=3,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
            groups=groups,
            padding=1,
            bias=False,
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(alpha * planes)
        self.conv2 = PackedConv2d(
            planes,
            planes,
            kernel_size=3,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
            groups=groups,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                PackedConv2d(
                    in_planes,
                    planes,
                    kernel_size=1,
                    alpha=alpha,
                    num_estimators=num_estimators,
                    gamma=gamma,
                    groups=groups,
                    stride=stride,
                    bias=True,
                ),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class _PackedWide(nn.Module):
    def __init__(
        self,
        depth: int,
        widen_factor: int,
        in_channels: int,
        num_classes: int,
        num_estimators: int = 4,
        alpha: int = 2,
        gamma: int = 1,
        groups: int = 1,
        dropout_rate: float = 0,
        style: str = "imagenet",
    ):
        super().__init__()
        self.num_estimators = num_estimators
        self.in_planes = 16

        assert (depth - 4) % 6 == 0, "Wide-resnet depth should be 6n+4."
        num_blocks = int((depth - 4) / 6)
        k = widen_factor

        nStages = [16, 16 * k, 32 * k, 64 * k]

        if style == "imagenet":
            self.conv1 = PackedConv2d(
                in_channels,
                nStages[0],
                kernel_size=7,
                alpha=alpha,
                num_estimators=self.num_estimators,
                stride=2,
                padding=3,
                gamma=1,  # No groups for the first layer
                groups=groups,
                bias=True,
                first=True,
            )
        else:
            self.conv1 = PackedConv2d(
                in_channels,
                nStages[0],
                kernel_size=3,
                alpha=alpha,
                num_estimators=self.num_estimators,
                stride=1,
                padding=1,
                gamma=gamma,
                groups=groups,
                bias=True,
                first=True,
            )

        if style == "imagenet":
            self.optional_pool = nn.MaxPool2d(
                kernel_size=3, stride=2, padding=1
            )
        else:
            self.optional_pool = nn.Identity()

        self.layer1 = self._wide_layer(
            WideBasicBlock,
            nStages[1],
            num_blocks,
            dropout_rate,
            stride=1,
            alpha=alpha,
            num_estimators=self.num_estimators,
            gamma=gamma,
            groups=groups,
        )
        self.layer2 = self._wide_layer(
            WideBasicBlock,
            nStages[2],
            num_blocks,
            dropout_rate,
            stride=2,
            alpha=alpha,
            num_estimators=self.num_estimators,
            gamma=gamma,
            groups=groups,
        )
        self.layer3 = self._wide_layer(
            WideBasicBlock,
            nStages[3],
            num_blocks,
            dropout_rate,
            stride=2,
            alpha=alpha,
            num_estimators=self.num_estimators,
            gamma=gamma,
            groups=groups,
        )
        self.bn1 = nn.BatchNorm2d(nStages[3] * alpha, momentum=0.9)

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten(1)

        self.linear = PackedLinear(
            nStages[3],
            num_classes,
            alpha=alpha,
            num_estimators=num_estimators,
            last=True,
        )

    def _wide_layer(
        self,
        block: Type[WideBasicBlock],
        planes: int,
        num_blocks: int,
        dropout_rate: float,
        stride: int,
        alpha: float,
        num_estimators: int,
        gamma: int,
        groups: int,
    ):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []

        for stride in strides:
            layers.append(
                block(
                    in_planes=self.in_planes,
                    planes=planes,
                    dropout_rate=dropout_rate,
                    stride=stride,
                    alpha=alpha,
                    num_estimators=num_estimators,
                    gamma=gamma,
                    groups=groups,
                )
            )
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.optional_pool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = rearrange(
            out, "e (m c) h w -> (m e) c h w", m=self.num_estimators
        )
        out = self.pool(out)
        out = self.flatten(out)
        out = self.linear(out)

        return out


def packed_wideresnet28x10(
    in_channels: int,
    num_estimators: int,
    alpha: int,
    gamma: int,
    groups: int,
    num_classes: int,
    style: str = "imagenet",
) -> _PackedWide:
    """Packed-Ensembles of Wide-ResNet-28x10 from `Wide Residual Networks
    <https://arxiv.org/pdf/1605.07146.pdf>`_.

    Args:
        in_channels (int): Number of input channels.
        num_estimators (int): Number of estimators in the ensemble.
        alpha (int): Expansion factor affecting the width of the estimators.
        gamma (int): Number of groups within each estimator.
        num_classes (int): Number of classes to predict.
        style (bool, optional): Whether to use the ImageNet
            structure. Defaults to ``True``.

    Returns:
        _PackedWide: A Packed-Ensembles Wide-ResNet-28x10.
    """
    return _PackedWide(
        in_channels=in_channels,
        depth=28,
        widen_factor=10,
        num_classes=num_classes,
        dropout_rate=0.3,
        num_estimators=num_estimators,
        alpha=alpha,
        gamma=gamma,
        groups=groups,
        style=style,
    )
