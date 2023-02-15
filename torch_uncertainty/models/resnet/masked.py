# fmt: off
from typing import List, Type, Union

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...layers import MaskedConv2d, MaskedLinear

# fmt: on
__all__ = [
    "masked_resnet18",
    "masked_resnet34",
    "masked_resnet50",
    "masked_resnet101",
    "masked_resnet152",
]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        num_estimators: int = 4,
        scale: float = 2.0,
        groups: int = 1,
    ):
        super(BasicBlock, self).__init__()

        # No subgroups for the first layer
        self.conv1 = MaskedConv2d(
            in_planes,
            planes,
            kernel_size=3,
            num_estimators=num_estimators,
            scale=scale,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = MaskedConv2d(
            planes,
            planes,
            kernel_size=3,
            num_estimators=num_estimators,
            scale=scale,
            stride=1,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                MaskedConv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    num_estimators=num_estimators,
                    scale=scale,
                    stride=stride,
                    groups=groups,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
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
        num_estimators: int = 4,
        scale: float = 2.0,
        groups: int = 1,
    ):
        super(Bottleneck, self).__init__()

        # No subgroups for the first layer
        self.conv1 = MaskedConv2d(
            in_planes,
            planes,
            kernel_size=1,
            num_estimators=num_estimators,
            scale=scale,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = MaskedConv2d(
            planes,
            planes,
            kernel_size=3,
            num_estimators=num_estimators,
            scale=scale,
            stride=stride,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = MaskedConv2d(
            planes,
            self.expansion * planes,
            kernel_size=1,
            num_estimators=num_estimators,
            scale=scale,
            groups=groups,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                MaskedConv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    num_estimators=num_estimators,
                    scale=scale,
                    stride=stride,
                    groups=groups,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class _MaskedResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        num_blocks: List[int],
        in_channels: int,
        num_classes: int,
        num_estimators: int,
        scale: float = 2.0,
        groups: int = 1,
        # dataset: str = "cifar",
    ) -> None:
        super().__init__()
        # assert dataset in [
        #     "cifar",
        #     "mnist",
        #     "tinyimagenet",
        #     "imagenet",
        # ], "The dataset is not taken in charge by this implementation."
        # self.dataset = dataset

        self.in_channels = in_channels
        self.num_estimators = num_estimators
        self.in_planes = 64
        if self.in_planes % self.num_estimators:
            self.in_planes += (
                self.num_estimators - self.in_planes % self.num_estimators
            )
        block_planes = self.in_planes

        # No subgroups in the first layer
        # if self.dataset == "imagenet":
        #     self.conv1 = MaskedConv2d(
        #         3 * self.num_estimators,
        #         block_planes,
        #         kernel_size=7,
        #         stride=2,
        #         padding=3,
        #         groups=1,
        #         num_estimators=num_estimators,
        #         bias=False,
        #     )
        # elif self.dataset == "mnist":
        #     self.conv1 = MaskedConv2d(
        #         1 * self.num_estimators,
        #         block_planes,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1,
        #         groups=1,
        #         num_estimators=num_estimators,
        #         bias=False,
        #     )
        # else:
        self.conv1 = nn.Conv2d(
            self.in_channels,
            block_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(block_planes)

        self.optional_pool = nn.Identity()

        self.layer1 = self._make_layer(
            block,
            block_planes,
            num_blocks[0],
            stride=1,
            num_estimators=num_estimators,
            scale=scale,
            groups=groups,
        )
        self.layer2 = self._make_layer(
            block,
            block_planes * 2,
            num_blocks[1],
            stride=2,
            num_estimators=num_estimators,
            scale=scale,
            groups=groups,
        )
        self.layer3 = self._make_layer(
            block,
            block_planes * 4,
            num_blocks[2],
            stride=2,
            num_estimators=num_estimators,
            scale=scale,
            groups=groups,
        )
        self.layer4 = self._make_layer(
            block,
            block_planes * 8,
            num_blocks[3],
            stride=2,
            num_estimators=num_estimators,
            scale=scale,
            groups=groups,
        )

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten(1)

        self.linear = MaskedLinear(
            block_planes * 8 * block.expansion,
            num_classes,
            num_estimators,
            scale=scale,
        )

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        num_blocks: int,
        stride: int,
        num_estimators: int,
        scale: float,
        groups: int,
    ) -> nn.Module:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    stride,
                    num_estimators,
                    scale=scale,
                    groups=groups,
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        # if self.dataset == "imagenet" or self.dataset == "tinyimagenet":
        # out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)
        out = self.optional_pool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.pool(out)
        out = self.flatten(out)
        out = self.linear(out)
        return out


def masked_resnet18(
    in_channels: int,
    num_estimators: int,
    scale: float,
    groups: int,
    num_classes: int,
) -> _MaskedResNet:
    """Masksembles of ResNet-18 from `Deep Residual Learning for Image
    Recognition <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        in_channels (int): Number of input channels.
        num_estimators (int): Number of estimators in the ensemble.
        groups (int): Number of groups within each estimator.
        num_classes (int): Number of classes to predict.

    Returns:
        _MaskedResNet: A Masksembles-style ResNet-18.
    """
    return _MaskedResNet(
        block=BasicBlock,
        num_blocks=[2, 2, 2, 2],
        in_channels=in_channels,
        num_estimators=num_estimators,
        scale=scale,
        groups=groups,
        num_classes=num_classes,
    )


def masked_resnet34(
    in_channels: int,
    num_estimators: int,
    scale: float,
    groups: int,
    num_classes: int,
) -> _MaskedResNet:
    """Masksembles of ResNet-34 from `Deep Residual Learning for Image
    Recognition <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        in_channels (int): Number of input channels.
        num_estimators (int): Number of estimators in the ensemble.
        groups (int): Number of groups within each estimator.
        num_classes (int): Number of classes to predict.

    Returns:
        _MaskedResNet: A Masksembles-style ResNet-34.
    """
    return _MaskedResNet(
        block=BasicBlock,
        num_blocks=[3, 4, 6, 3],
        in_channels=in_channels,
        num_estimators=num_estimators,
        scale=scale,
        groups=groups,
        num_classes=num_classes,
    )


def masked_resnet50(
    in_channels: int,
    num_estimators: int,
    scale: float,
    groups: int,
    num_classes: int,
) -> _MaskedResNet:
    """Masksembles of ResNet-50 from `Deep Residual Learning for Image
    Recognition <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        in_channels (int): Number of input channels.
        num_estimators (int): Number of estimators in the ensemble.
        groups (int): Number of groups within each estimator.
        num_classes (int): Number of classes to predict.

    Returns:
        _MaskedResNet: A Masksembles-style ResNet-50.
    """
    return _MaskedResNet(
        block=Bottleneck,
        num_blocks=[3, 4, 6, 3],
        in_channels=in_channels,
        num_estimators=num_estimators,
        scale=scale,
        groups=groups,
        num_classes=num_classes,
    )


def masked_resnet101(
    in_channels: int,
    num_estimators: int,
    scale: float,
    groups: int,
    num_classes: int,
) -> _MaskedResNet:
    """Masksembles of ResNet-101 from `Deep Residual Learning for Image
    Recognition <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        in_channels (int): Number of input channels.
        num_estimators (int): Number of estimators in the ensemble.
        groups (int): Number of groups within each estimator.
        num_classes (int): Number of classes to predict.

    Returns:
        _MaskedResNet: A Masksembles-style ResNet-101.
    """
    return _MaskedResNet(
        block=Bottleneck,
        num_blocks=[3, 4, 23, 3],
        in_channels=in_channels,
        num_estimators=num_estimators,
        scale=scale,
        groups=groups,
        num_classes=num_classes,
    )


def masked_resnet152(
    in_channels: int,
    num_estimators: int,
    scale: float,
    groups: int,
    num_classes: int,
) -> _MaskedResNet:
    """Masksembles of ResNet-152 from `Deep Residual Learning for Image
    Recognition <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        in_channels (int): Number of input channels.
        num_estimators (int): Number of estimators in the ensemble.
        groups (int): Number of groups within each estimator.
        num_classes (int): Number of classes to predict.

    Returns:
        _MaskedResNet: A Masksembles-style ResNet-152.
    """
    return _MaskedResNet(
        block=Bottleneck,
        num_blocks=[3, 8, 36, 3],
        in_channels=in_channels,
        num_estimators=num_estimators,
        scale=scale,
        groups=groups,
        num_classes=num_classes,
    )
