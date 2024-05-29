from typing import Literal

import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "wideresnet28x10",
]


class WideBasicBlock(nn.Module):
    def __init__(
        self,
        in_planes: int,
        planes: int,
        dropout_rate: float,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.dropout = nn.Dropout2d(p=dropout_rate)
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

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.dropout(self.conv1(x))))
        out = self.conv2(out)
        out += self.shortcut(x)
        return F.relu(self.bn2(out))


class _WideResNet(nn.Module):
    """WideResNet from `Wide Residual Networks`.

    Note:
        if `dropout_rate` and `num_estimators` are set, the model will sample
        from the dropout distribution during inference. If `last_layer_dropout`
        is set, only the last layer will be sampled from the dropout
        distribution during inference.
    """

    def __init__(
        self,
        depth: int,
        widen_factor: int,
        in_channels: int,
        num_classes: int,
        conv_bias: bool,
        dropout_rate: float,
        groups: int = 1,
        style: Literal["imagenet", "cifar"] = "imagenet",
    ) -> None:
        super().__init__()
        self.in_planes = 16
        self.dropout_rate = dropout_rate

        if (depth - 4) % 6 != 0:
            raise ValueError(f"Wide-resnet depth should be 6n+4. Got {depth}.")
        num_blocks = int((depth - 4) / 6)
        k = widen_factor

        num_stages = [16, 16 * k, 32 * k, 64 * k]

        if style == "imagenet":
            self.conv1 = nn.Conv2d(
                in_channels,
                num_stages[0],
                kernel_size=7,
                stride=2,
                padding=3,
                groups=groups,
                bias=conv_bias,
            )
        elif style == "cifar":
            self.conv1 = nn.Conv2d(
                in_channels,
                num_stages[0],
                kernel_size=3,
                stride=1,
                padding=1,
                groups=groups,
                bias=conv_bias,
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
            WideBasicBlock,
            num_stages[1],
            num_blocks=num_blocks,
            dropout_rate=dropout_rate,
            stride=1,
            groups=groups,
        )
        self.layer2 = self._wide_layer(
            WideBasicBlock,
            num_stages[2],
            num_blocks=num_blocks,
            dropout_rate=dropout_rate,
            stride=2,
            groups=groups,
        )
        self.layer3 = self._wide_layer(
            WideBasicBlock,
            num_stages[3],
            num_blocks=num_blocks,
            dropout_rate=dropout_rate,
            stride=2,
            groups=groups,
        )

        self.dropout = nn.Dropout(p=dropout_rate)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten(1)

        self.linear = nn.Linear(
            num_stages[3],
            num_classes,
        )

    def _wide_layer(
        self,
        block: type[WideBasicBlock],
        planes: int,
        num_blocks: int,
        dropout_rate: float,
        stride: int,
        groups: int,
    ) -> nn.Module:
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

    def feats_forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.optional_pool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out)
        return self.flatten(out)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(self.feats_forward(x))


def wideresnet28x10(
    in_channels: int,
    num_classes: int,
    groups: int = 1,
    conv_bias: bool = True,
    dropout_rate: float = 0.3,
    style: Literal["imagenet", "cifar"] = "imagenet",
) -> _WideResNet:
    """Wide-ResNet-28x10 from `Wide Residual Networks
    <https://arxiv.org/pdf/1605.07146.pdf>`_.

    Args:
        in_channels (int): Number of input channels
        num_classes (int): Number of classes to predict.
        groups (int, optional): Number of groups in convolutions. Defaults to
            ``1``.
        conv_bias (bool): Whether to use bias in convolutions. Defaults to
            ``True``.
        dropout_rate (float, optional): Dropout rate. Defaults to ``0.3``.
        style (bool, optional): Whether to use the ImageNet
            structure. Defaults to ``True``.

    Returns:
        _Wide: A Wide-ResNet-28x10.
    """
    return _WideResNet(
        depth=28,
        widen_factor=10,
        in_channels=in_channels,
        conv_bias=conv_bias,
        dropout_rate=dropout_rate,
        num_classes=num_classes,
        groups=groups,
        style=style,
    )
