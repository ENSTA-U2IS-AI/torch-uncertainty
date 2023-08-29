# fmt: off
from typing import List, Type, Union

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# fmt: on
__all__ = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        dropout_rate: float = 0,
        groups: int = 1,
    ):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)

        # As in timm
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    groups=groups,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.dropout(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        dropout_rate: float = 0,
        groups: int = 1,
    ):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=1,
            groups=groups,
            bias=False,
        )
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
        self.bn2 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv3 = nn.Conv2d(
            planes,
            self.expansion * planes,
            kernel_size=1,
            groups=groups,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    groups=groups,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.dropout(self.bn2(self.conv2(out))))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# class RobustBottleneck(nn.Module):
#     """Robust Bottleneck from "Can CNNs be more robust than transformers?"
#     This corresponds to ResNet-Up-Inverted-DW in the paper.
#     """

#     expansion = 4

#     def __init__(
#         self,
#         in_planes: int,
#         planes: int,
#         stride: int = 1,
#         dropout_rate: float = 0,
#         groups: int = 1,
#     ):
#         super().__init__()
#         self.conv1 = nn.Conv2d(
#             in_planes,
#             planes,
#             kernel_size=11,
#             padding=5,
#             groups=in_planes,
#             stride=stride,
#             bias=False,
#         )
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(
#             planes,
#             self.expansion * planes,
#             kernel_size=1,
#             groups=groups,
#             bias=True,
#         )
#         self.conv3 = nn.Conv2d(
#             self.expansion * planes,
#             planes,
#             kernel_size=1,
#             groups=groups,
#             bias=True,
#         )
#         self.shortcut = nn.Sequential()

#     def forward(self, x: Tensor) -> Tensor:
#         out = self.bn1(self.conv1(x))
#         out = F.relu(self.conv2(out))
#         out = self.conv3(out)
#         out += self.shortcut(x)
#         return out


class _ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        num_blocks: List[int],
        in_channels: int,
        num_classes: int,
        dropout_rate: float,
        groups: int,
        style: str = "imagenet",
    ) -> None:
        super().__init__()

        self.in_planes = 64
        block_planes = self.in_planes

        if style == "imagenet":
            self.conv1 = nn.Conv2d(
                in_channels,
                block_planes,
                kernel_size=7,
                stride=2,
                padding=3,
                groups=1,  # No groups in the first layer
                bias=False,
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels,
                block_planes,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=1,  # No groups in the first layer
                bias=False,
            )

        self.bn1 = nn.BatchNorm2d(block_planes)

        if style == "imagenet":
            self.optional_pool = nn.MaxPool2d(
                kernel_size=3, stride=2, padding=1
            )
        else:
            self.optional_pool = nn.Identity()

        self.layer1 = self._make_layer(
            block,
            block_planes,
            num_blocks[0],
            stride=1,
            dropout_rate=dropout_rate,
            groups=groups,
        )
        self.layer2 = self._make_layer(
            block,
            block_planes * 2,
            num_blocks[1],
            stride=2,
            dropout_rate=dropout_rate,
            groups=groups,
        )
        self.layer3 = self._make_layer(
            block,
            block_planes * 4,
            num_blocks[2],
            stride=2,
            dropout_rate=dropout_rate,
            groups=groups,
        )
        self.layer4 = self._make_layer(
            block,
            block_planes * 8,
            num_blocks[3],
            stride=2,
            dropout_rate=dropout_rate,
            groups=groups,
        )

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten(1)

        self.linear = nn.Linear(
            block_planes * 8 * block.expansion,
            num_classes,
        )

    def _make_layer(
        self,
        block: Union[Type[BasicBlock], Type[Bottleneck]],
        planes: int,
        num_blocks: int,
        stride: int,
        dropout_rate: float,
        groups: int,
    ) -> nn.Module:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    in_planes=self.in_planes,
                    planes=planes,
                    stride=stride,
                    dropout_rate=dropout_rate,
                    groups=groups,
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.optional_pool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
        out = self.flatten(out)
        out = self.linear(out)
        return out


def resnet18(
    in_channels: int,
    num_classes: int,
    dropout_rate: float = 0,
    groups: int = 1,
    style: str = "imagenet",
) -> _ResNet:
    """ResNet-18 from `Deep Residual Learning for Image Recognition
    <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of classes to predict.
        dropout_rate (float): Dropout rate. Defaults to 0.
        groups (int): Number of groups in convolutions. Defaults to 1.
        style (bool, optional): Whether to use the ImageNet
            structure. Defaults to ``True``.

    Returns:
        _ResNet: A ResNet-18.
    """
    return _ResNet(
        block=BasicBlock,
        num_blocks=[2, 2, 2, 2],
        in_channels=in_channels,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        groups=groups,
        style=style,
    )


def resnet34(
    in_channels: int,
    num_classes: int,
    dropout_rate: float = 0,
    groups: int = 1,
    style: str = "imagenet",
) -> _ResNet:
    """ResNet-34 from `Deep Residual Learning for Image Recognition
    <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of classes to predict.
        dropout_rate (float): Dropout rate. Defaults to 0.
        groups (int): Number of groups in convolutions. Defaults to 1.
        style (bool, optional): Whether to use the ImageNet
            structure. Defaults to ``True``.

    Returns:
        _ResNet: A ResNet-34.
    """
    return _ResNet(
        block=BasicBlock,
        num_blocks=[3, 4, 6, 3],
        in_channels=in_channels,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        groups=groups,
        style=style,
    )


def resnet50(
    in_channels: int,
    num_classes: int,
    dropout_rate: float = 0,
    groups: int = 1,
    style: str = "imagenet",
) -> _ResNet:
    """ResNet-50 from `Deep Residual Learning for Image Recognition
    <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of classes to predict.
        dropout_rate (float): Dropout rate. Defaults to 0.
        groups (int): Number of groups in convolutions. Defaults to 1.
        style (bool, optional): Whether to use the ImageNet
            structure. Defaults to ``True``.

    Returns:
        _ResNet: A ResNet-50.
    """
    return _ResNet(
        block=Bottleneck,
        num_blocks=[3, 4, 6, 3],
        in_channels=in_channels,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        groups=groups,
        style=style,
    )


def resnet101(
    in_channels: int,
    num_classes: int,
    dropout_rate: float = 0,
    groups: int = 1,
    style: str = "imagenet",
) -> _ResNet:
    """ResNet-101 from `Deep Residual Learning for Image Recognition
    <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of classes to predict.
        dropout_rate (float): Dropout rate. Defaults to 0.
        groups (int): Number of groups in convolutions. Defaults to 1.
        style (bool, optional): Whether to use the ImageNet
            structure. Defaults to ``True``.

    Returns:
        _ResNet: A ResNet-101.
    """
    return _ResNet(
        block=Bottleneck,
        num_blocks=[3, 4, 23, 3],
        in_channels=in_channels,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        groups=groups,
        style=style,
    )


def resnet152(
    in_channels: int,
    num_classes: int,
    dropout_rate: float = 0,
    groups: int = 1,
    style: str = "imagenet",
) -> _ResNet:
    """ResNet-152 from `Deep Residual Learning for Image Recognition
    <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of classes to predict.
        dropout_rate (float): Dropout rate. Defaults to 0.
        groups (int, optional): Number of groups in convolutions. Defaults to
            ``1``.
        style (bool, optional): Whether to use the ImageNet
            structure. Defaults to ``True``.

    Returns:
        _ResNet: A ResNet-152.
    """
    return _ResNet(
        block=Bottleneck,
        num_blocks=[3, 8, 36, 3],
        in_channels=in_channels,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        groups=groups,
        style=style,
    )
