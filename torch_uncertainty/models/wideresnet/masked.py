from typing import Literal

import torch.nn.functional as F
from torch import Tensor, nn

from torch_uncertainty.layers import MaskedConv2d, MaskedLinear

__all__ = [
    "masked_wideresnet28x10",
]


class _WideBasicBlock(nn.Module):
    def __init__(
        self,
        in_planes: int,
        planes: int,
        conv_bias: bool,
        dropout_rate: float,
        stride: int,
        num_estimators: int,
        scale: float,
        groups: int,
    ) -> None:
        super().__init__()
        self.conv1 = MaskedConv2d(
            in_planes,
            planes,
            kernel_size=3,
            num_estimators=num_estimators,
            padding=1,
            bias=conv_bias,
            scale=scale,
            groups=groups,
        )
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = MaskedConv2d(
            planes,
            planes,
            kernel_size=3,
            num_estimators=num_estimators,
            stride=stride,
            padding=1,
            bias=conv_bias,
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
                    bias=conv_bias,
                    scale=scale,
                    groups=groups,
                ),
            )
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.dropout(self.conv1(x))))
        out = self.conv2(out)
        out += self.shortcut(x)
        return F.relu(self.bn2(out))


class _MaskedWideResNet(nn.Module):
    def __init__(
        self,
        depth: int,
        widen_factor: int,
        in_channels: int,
        num_classes: int,
        num_estimators: int,
        conv_bias: bool,
        dropout_rate: float,
        scale: float = 2.0,
        groups: int = 1,
        style: Literal["imagenet", "cifar"] = "imagenet",
    ) -> None:
        super().__init__()
        self.num_estimators = num_estimators
        self.in_planes = 16

        if (depth - 4) % 6 != 0:
            raise ValueError("Wide-resnet depth should be 6n+4.")
        num_blocks = (depth - 4) // 6
        k = widen_factor

        num_stages = [16, 16 * k, 32 * k, 64 * k]

        if style == "imagenet":
            self.conv1 = nn.Conv2d(
                in_channels,
                num_stages[0],
                kernel_size=7,
                stride=2,
                padding=3,
                bias=conv_bias,
                groups=1,
            )
        elif style == "cifar":
            self.conv1 = nn.Conv2d(
                in_channels,
                num_stages[0],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=conv_bias,
                groups=1,
            )
        else:
            raise ValueError(f"Unknown WideResNet style: {style}. ")

        self.bn1 = nn.BatchNorm2d(num_stages[0])

        if style == "imagenet":
            self.optional_pool = nn.MaxPool2d(
                kernel_size=3, stride=2, padding=1
            )
        else:
            self.optional_pool = nn.Identity()

        self.layer1 = self._wide_layer(
            _WideBasicBlock,
            num_stages[1],
            num_blocks=num_blocks,
            conv_bias=conv_bias,
            dropout_rate=dropout_rate,
            stride=1,
            num_estimators=self.num_estimators,
            scale=scale,
            groups=groups,
        )
        self.layer2 = self._wide_layer(
            _WideBasicBlock,
            num_stages[2],
            num_blocks=num_blocks,
            conv_bias=conv_bias,
            dropout_rate=dropout_rate,
            stride=2,
            num_estimators=self.num_estimators,
            scale=scale,
            groups=groups,
        )
        self.layer3 = self._wide_layer(
            _WideBasicBlock,
            num_stages[3],
            num_blocks=num_blocks,
            conv_bias=conv_bias,
            dropout_rate=dropout_rate,
            stride=2,
            num_estimators=self.num_estimators,
            scale=scale,
            groups=groups,
        )

        self.dropout = nn.Dropout(p=dropout_rate)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten(1)

        self.linear = MaskedLinear(
            num_stages[3], num_classes, num_estimators, scale=scale
        )

    def _wide_layer(
        self,
        block: type[nn.Module],
        planes: int,
        num_blocks: int,
        conv_bias: bool,
        dropout_rate: float,
        stride: int,
        num_estimators: int,
        scale: float = 2.0,
        groups: int = 1,
    ) -> nn.Module:
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []

        for stride in strides:
            layers.append(
                block(
                    in_planes=self.in_planes,
                    planes=planes,
                    stride=stride,
                    num_estimators=num_estimators,
                    conv_bias=conv_bias,
                    dropout_rate=dropout_rate,
                    scale=scale,
                    groups=groups,
                )
            )
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = x.repeat(self.num_estimators, 1, 1, 1)
        out = F.relu(self.bn1(self.conv1(out)))
        out = self.optional_pool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out)
        out = self.dropout(self.flatten(out))
        return self.linear(out)


def masked_wideresnet28x10(
    in_channels: int,
    num_classes: int,
    num_estimators: int,
    scale: float,
    groups: int,
    conv_bias: bool = True,
    dropout_rate: float = 0.3,
    style: Literal["imagenet", "cifar"] = "imagenet",
) -> _MaskedWideResNet:
    """Masksembles of Wide-ResNet-28x10.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of classes to predict.
        num_estimators (int): Number of estimators in the ensemble.
        scale (float): Expansion factor affecting the width of the estimators.
        groups (int): Number of groups within each estimator.
        conv_bias (bool): Whether to use bias in convolutions. Defaults to
            ``True``.
        dropout_rate (float, optional): Dropout rate. Defaults to ``0.3``.
        style (bool, optional): Whether to use the ImageNet
            structure. Defaults to ``True``.

    Returns:
        _MaskedWideResNet: A Masksembles-style Wide-ResNet-28x10.
    """
    return _MaskedWideResNet(
        in_channels=in_channels,
        num_classes=num_classes,
        depth=28,
        widen_factor=10,
        conv_bias=conv_bias,
        dropout_rate=dropout_rate,
        num_estimators=num_estimators,
        scale=scale,
        groups=groups,
        style=style,
    )
