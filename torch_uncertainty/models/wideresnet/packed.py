# fmt: off
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ...layers import PackedConv2d, PackedLinear


# fmt: on
class WideBasicBlock(nn.Module):
    def __init__(
        self,
        in_planes,
        planes,
        dropout_rate,
        stride=1,
        num_estimators=4,
        gamma=1,
    ):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = PackedConv2d(
            in_planes,
            planes,
            kernel_size=3,
            num_estimators=num_estimators,
            padding=1,
            groups=gamma,
            bias=False,
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = PackedConv2d(
            planes,
            planes,
            kernel_size=3,
            num_estimators=num_estimators,
            stride=stride,
            padding=1,
            groups=gamma,
            bias=False,
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                PackedConv2d(
                    in_planes,
                    planes,
                    kernel_size=1,
                    num_estimators=num_estimators,
                    stride=stride,
                    groups=gamma,
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
        num_estimators: int,
        dropout_rate: float,
        alpha: int = 2,
        gamma: int = 1,
    ):
        super().__init__()
        self.num_estimators = num_estimators
        self.in_planes = 16 * alpha

        assert (depth - 4) % 6 == 0, "Wide-resnet depth should be 6n+4"
        n = (depth - 4) / 6
        k = widen_factor

        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = PackedConv2d(
            in_channels * self.num_estimators,
            nStages[0] * alpha,
            kernel_size=3,
            num_estimators=self.num_estimators,
            stride=1,
            padding=1,
            groups=gamma,
            bias=True,
        )

        self.optional_pool = nn.Identity()

        self.layer1 = self._wide_layer(
            WideBasicBlock,
            nStages[1] * alpha,
            n,
            dropout_rate,
            stride=1,
            num_estimators=self.num_estimators,
            gamma=gamma,
        )
        self.layer2 = self._wide_layer(
            WideBasicBlock,
            nStages[2] * alpha,
            n,
            dropout_rate,
            stride=2,
            num_estimators=self.num_estimators,
            gamma=gamma,
        )
        self.layer3 = self._wide_layer(
            WideBasicBlock,
            nStages[3] * alpha,
            n,
            dropout_rate,
            stride=2,
            num_estimators=self.num_estimators,
            gamma=gamma,
        )
        self.bn1 = nn.BatchNorm2d(nStages[3] * alpha, momentum=0.9)

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten(1)

        self.linear = PackedLinear(
            nStages[3] * alpha,
            num_classes * num_estimators,
            num_estimators,
        )

    def _wide_layer(
        self,
        block: nn.Module,
        planes: int,
        num_blocks: int,
        dropout_rate: float,
        stride: int,
        num_estimators: int,
        gamma,
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
                    num_estimators,
                    gamma,
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
    num_estimators: int, alpha: int, gamma: int, num_classes: int
) -> nn.Module:
    return _PackedWide(
        depth=28,
        widen_factor=10,
        dropout_rate=0.3,
        num_classes=num_classes,
        num_estimators=num_estimators,
        alpha=alpha,
        gamma=gamma,
    )
