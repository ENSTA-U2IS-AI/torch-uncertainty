# fmt: off
from typing import Type

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# fmt: on
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
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
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

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
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
    ):
        super().__init__()
        self.in_planes = 16

        assert (depth - 4) % 6 == 0, "Wide-resnet depth should be 6n+4"
        num_blocks = int((depth - 4) / 6)
        k = widen_factor

        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = nn.Conv2d(
            in_channels,
            nStages[0],
            kernel_size=3,
            stride=1,
            padding=1,
            groups=groups,
            bias=True,
        )

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
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)

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


def wideresnet28x10(
    in_channels: int, groups: int, num_classes: int
) -> nn.Module:
    return _Wide(
        depth=28,
        widen_factor=10,
        in_channels=in_channels,
        dropout_rate=0.3,
        num_classes=num_classes,
        groups=groups,
    )
