# fmt: off
from typing import Type

import torch.nn as nn
import torch.nn.functional as F

from ...layers import BatchConv2d, BatchLinear

# fmt: on
__all__ = [
    "batched_wideresnet28x10",
]


class WideBasicBlock(nn.Module):
    def __init__(
        self,
        in_planes,
        planes,
        dropout_rate,
        stride=1,
        num_estimators=4,
        groups: int = 1,
    ):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = BatchConv2d(
            in_planes,
            planes,
            kernel_size=3,
            num_estimators=num_estimators,
            groups=groups,
            padding=1,
            bias=False,
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = BatchConv2d(
            planes,
            planes,
            kernel_size=3,
            num_estimators=num_estimators,
            groups=groups,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                BatchConv2d(
                    in_planes,
                    planes,
                    kernel_size=1,
                    num_estimators=num_estimators,
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


class _BatchedWide(nn.Module):
    def __init__(
        self,
        depth: int,
        widen_factor: int,
        in_channels: int,
        num_classes: int,
        num_estimators: int,
        groups: int = 1,
        dropout_rate: float = 0.0,
        style: str = "imagenet",
    ) -> None:
        super().__init__()
        self.num_estimators = num_estimators
        self.in_planes = 16

        assert (depth - 4) % 6 == 0, "Wide-resnet depth should be 6n+4."
        n = (depth - 4) // 6
        k = widen_factor

        nStages = [16, 16 * k, 32 * k, 64 * k]

        if style == "imagenet":
            self.conv1 = BatchConv2d(
                in_channels,
                nStages[0],
                num_estimators=self.num_estimators,
                groups=groups,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=True,
            )
        else:
            self.conv1 = BatchConv2d(
                in_channels,
                nStages[0],
                num_estimators=self.num_estimators,
                groups=groups,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
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
            n,
            dropout_rate,
            stride=1,
            num_estimators=self.num_estimators,
            groups=groups,
        )
        self.layer2 = self._wide_layer(
            WideBasicBlock,
            nStages[2],
            n,
            dropout_rate,
            stride=2,
            num_estimators=self.num_estimators,
            groups=groups,
        )
        self.layer3 = self._wide_layer(
            WideBasicBlock,
            nStages[3],
            n,
            dropout_rate,
            stride=2,
            num_estimators=self.num_estimators,
            groups=groups,
        )
        self.bn1 = nn.BatchNorm2d(nStages[3])

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten(1)

        self.linear = BatchLinear(
            nStages[3],
            num_classes,
            num_estimators,
        )

    def _wide_layer(
        self,
        block: Type[nn.Module],
        planes: int,
        num_blocks: int,
        dropout_rate: float,
        stride: int,
        num_estimators: int,
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
                    num_estimators=num_estimators,
                    groups=groups,
                )
            )
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = x.repeat(self.num_estimators, 1, 1, 1)
        out = self.conv1(out)
        out = self.optional_pool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))

        out = self.pool(out)
        out = self.flatten(out)
        out = self.linear(out)

        return out


def batched_wideresnet28x10(
    in_channels: int,
    num_estimators: int,
    groups: int,
    num_classes: int,
    style: str = "imagenet",
) -> _BatchedWide:
    """BatchEnsemble of Wide-ResNet-28x10 from `Wide Residual Networks
    <https://arxiv.org/pdf/1605.07146.pdf>`_.

    Args:
        in_channels (int): Number of input channels.
        num_estimators (int): Number of estimators in the ensemble.
        num_classes (int): Number of classes to predict.
        style (bool, optional): Whether to use the ImageNet
            structure. Defaults to ``True``.

    Returns:
        _BatchedWide: A BatchEnsemble-style Wide-ResNet-28x10.
    """
    return _BatchedWide(
        in_channels=in_channels,
        depth=28,
        widen_factor=10,
        dropout_rate=0.3,
        num_classes=num_classes,
        num_estimators=num_estimators,
        groups=groups,
        style=style,
    )
