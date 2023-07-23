# fmt: off
import torch.nn as nn
import torch.nn.functional as F

from ...layers import MaskedConv2d, MaskedLinear

# fmt: on
__all__ = [
    "masked_wideresnet28x10",
]


class WideBasicBlock(nn.Module):
    def __init__(
        self,
        in_planes: int,
        planes: int,
        dropout_rate: float,
        stride: int = 1,
        num_estimators: int = 4,
        scale: float = 2.0,
        groups: int = 1,
    ):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = MaskedConv2d(
            in_planes,
            planes,
            kernel_size=3,
            num_estimators=num_estimators,
            padding=1,
            bias=False,
            scale=scale,
            groups=groups,
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = MaskedConv2d(
            planes,
            planes,
            kernel_size=3,
            num_estimators=num_estimators,
            stride=stride,
            padding=1,
            bias=False,
            scale=scale,
            groups=groups,
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                MaskedConv2d(
                    in_planes,
                    planes,
                    kernel_size=1,
                    num_estimators=num_estimators,
                    stride=stride,
                    bias=True,
                    scale=scale,
                    groups=groups,
                ),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class _MaskedWide(nn.Module):
    def __init__(
        self,
        depth: int,
        widen_factor: int,
        in_channels: int,
        num_classes: int,
        num_estimators: int,
        scale: float = 2.0,
        groups: int = 1,
        dropout_rate: float = 0.0,
        style: str = "imagenet",
    ):
        super().__init__()
        self.num_estimators = num_estimators
        self.in_planes = 16

        assert (depth - 4) % 6 == 0, "Wide-resnet depth should be 6n+4."
        n = (depth - 4) / 6
        k = widen_factor

        nStages = [16, 16 * k, 32 * k, 64 * k]

        if style == "imagenet":
            self.conv1 = nn.Conv2d(
                in_channels,
                nStages[0],
                kernel_size=7,
                stride=2,
                padding=3,
                bias=True,
                groups=1,
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels,
                nStages[0],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                groups=1,
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
            scale=scale,
            groups=groups,
        )
        self.layer2 = self._wide_layer(
            WideBasicBlock,
            nStages[2],
            n,
            dropout_rate,
            stride=2,
            num_estimators=self.num_estimators,
            scale=scale,
            groups=groups,
        )
        self.layer3 = self._wide_layer(
            WideBasicBlock,
            nStages[3],
            n,
            dropout_rate,
            stride=2,
            num_estimators=self.num_estimators,
            scale=scale,
            groups=groups,
        )
        self.bn1 = nn.BatchNorm2d(nStages[3])

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten(1)

        self.linear = MaskedLinear(
            nStages[3], num_classes, num_estimators, scale=scale
        )

    def _wide_layer(
        self,
        block: nn.Module,
        planes: int,
        num_blocks: int,
        dropout_rate: float,
        stride: int,
        num_estimators: int,
        scale: float = 2.0,
        groups: int = 1,
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
                    scale=scale,
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


def masked_wideresnet28x10(
    in_channels: int,
    num_estimators: int,
    scale: float,
    groups: int,
    num_classes: int,
    style: str = "imagenet",
) -> _MaskedWide:
    """Masksembles of Wide-ResNet-28x10 from `Wide Residual Networks
    <https://arxiv.org/pdf/1605.07146.pdf>`_.

    Args:
        in_channels (int): Number of input channels.
        num_estimators (int): Number of estimators in the ensemble.
        scale (float): Expansion factor affecting the width of the estimators.
        groups (int): Number of groups within each estimator.
        num_classes (int): Number of classes to predict.
        style (bool, optional): Whether to use the ImageNet
            structure. Defaults to ``True``.

    Returns:
        _MaskedWide: A Masksembles-style Wide-ResNet-28x10.
    """
    return _MaskedWide(
        in_channels=in_channels,
        depth=28,
        widen_factor=10,
        dropout_rate=0.3,
        num_classes=num_classes,
        num_estimators=num_estimators,
        scale=scale,
        groups=groups,
        style=style,
    )
