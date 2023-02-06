# fmt: off
from typing import List, Type, Union

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from ...layers import PackedConv2d, PackedLinear

# fmt: on
__all__ = [
    "packed_resnet18",
    "packed_resnet34",
    "packed_resnet50",
    "packed_resnet101",
    "packed_resnet152",
]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        num_estimators: int = 4,
        gamma: int = 1,
    ):
        super(BasicBlock, self).__init__()

        # No subgroups for the first layer
        self.conv1 = PackedConv2d(
            in_planes,
            planes,
            kernel_size=3,
            num_estimators=num_estimators,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = PackedConv2d(
            planes,
            planes,
            kernel_size=3,
            num_estimators=num_estimators,
            stride=1,
            padding=1,
            groups=gamma,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                PackedConv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    num_estimators=num_estimators,
                    stride=stride,
                    groups=gamma,
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

    def __init__(self, in_planes, planes, stride=1, num_estimators=4, gamma=1):
        super(Bottleneck, self).__init__()

        # No subgroups for the first layer
        self.conv1 = PackedConv2d(
            in_planes,
            planes,
            kernel_size=1,
            num_estimators=num_estimators,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
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
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = PackedConv2d(
            planes,
            self.expansion * planes,
            kernel_size=1,
            num_estimators=num_estimators,
            groups=gamma,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                PackedConv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    num_estimators=num_estimators,
                    stride=stride,
                    groups=gamma,
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


class _PackedResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        num_blocks: List[int],
        in_channels: int,
        num_classes: int,
        num_estimators: int,
        alpha: int = 2,
        gamma: int = 1,
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
        self.in_planes = int(64 * alpha)
        if self.in_planes % self.num_estimators:
            self.in_planes += (
                self.num_estimators - self.in_planes % self.num_estimators
            )
        block_planes = self.in_planes

        # No subgroups in the first layer
        # if self.dataset == "imagenet":
        #     self.conv1 = PackedConv2d(
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
        #     self.conv1 = PackedConv2d(
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
        self.conv1 = PackedConv2d(
            self.in_channels * self.num_estimators,
            block_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            num_estimators=num_estimators,
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
            gamma=gamma,
        )
        self.layer2 = self._make_layer(
            block,
            block_planes * 2,
            num_blocks[1],
            stride=2,
            num_estimators=num_estimators,
            gamma=gamma,
        )
        self.layer3 = self._make_layer(
            block,
            block_planes * 4,
            num_blocks[2],
            stride=2,
            num_estimators=num_estimators,
            gamma=gamma,
        )
        self.layer4 = self._make_layer(
            block,
            block_planes * 8,
            num_blocks[3],
            stride=2,
            num_estimators=num_estimators,
            gamma=gamma,
        )

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten(1)

        self.linear = PackedLinear(
            block_planes * 8 * block.expansion,
            num_classes * num_estimators,
            num_estimators,
        )

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        num_blocks: int,
        stride: int,
        num_estimators: int,
        gamma: int,
    ) -> nn.Module:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(self.in_planes, planes, stride, num_estimators, gamma)
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
        out = rearrange(
            out, "e (m c) h w -> (m e) c h w", m=self.num_estimators
        )
        out = self.pool(out)
        out = self.flatten(out)
        out = self.linear(out)
        return out


def packed_resnet18(
    in_channels: int,
    num_estimators: int,
    alpha: int,
    gamma: int,
    num_classes: int,
) -> _PackedResNet:
    """Packed-Ensembles of ResNet-18 from `Deep Residual Learning for Image
    Recognition <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        in_channels (int): Number of input channels.
        num_estimators (int): Number of estimators in the ensemble.
        alpha (int): Expansion factor affecting the width of the estimators.
        gamma (int): Number of groups within each estimator.
        num_classes (int): Number of classes to predict.

    Returns:
        _PackedResNet: A Packed-Ensembles ResNet-18.
    """
    return _PackedResNet(
        block=BasicBlock,
        num_blocks=[2, 2, 2, 2],
        in_channels=in_channels,
        num_estimators=num_estimators,
        alpha=alpha,
        gamma=gamma,
        num_classes=num_classes,
    )


def packed_resnet34(
    in_channels: int,
    num_estimators: int,
    alpha: int,
    gamma: int,
    num_classes: int,
) -> _PackedResNet:
    """Packed-Ensembles of ResNet-34 from `Deep Residual Learning for Image
    Recognition <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        in_channels (int): Number of input channels.
        num_estimators (int): Number of estimators in the ensemble.
        alpha (int): Expansion factor affecting the width of the estimators.
        gamma (int): Number of groups within each estimator.
        num_classes (int): Number of classes to predict.

    Returns:
        _PackedResNet: A Packed-Ensembles ResNet-34.
    """
    return _PackedResNet(
        block=BasicBlock,
        num_blocks=[3, 4, 6, 3],
        in_channels=in_channels,
        num_estimators=num_estimators,
        alpha=alpha,
        gamma=gamma,
        num_classes=num_classes,
    )


def packed_resnet50(
    in_channels: int,
    num_estimators: int,
    alpha: int,
    gamma: int,
    num_classes: int,
) -> _PackedResNet:
    """Packed-Ensembles of ResNet-50 from `Deep Residual Learning for Image
    Recognition <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        in_channels (int): Number of input channels.
        num_estimators (int): Number of estimators in the ensemble.
        alpha (int): Expansion factor affecting the width of the estimators.
        gamma (int): Number of groups within each estimator.
        num_classes (int): Number of classes to predict.

    Returns:
        _PackedResNet: A Packed-Ensembles ResNet-50.
    """
    return _PackedResNet(
        block=Bottleneck,
        num_blocks=[3, 4, 6, 3],
        in_channels=in_channels,
        num_estimators=num_estimators,
        alpha=alpha,
        gamma=gamma,
        num_classes=num_classes,
    )


def packed_resnet101(
    in_channels: int,
    num_estimators: int,
    alpha: int,
    gamma: int,
    num_classes: int,
) -> _PackedResNet:
    """Packed-Ensembles of ResNet-101 from `Deep Residual Learning for Image
    Recognition <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        in_channels (int): Number of input channels.
        num_estimators (int): Number of estimators in the ensemble.
        alpha (int): Expansion factor affecting the width of the estimators.
        gamma (int): Number of groups within each estimator.
        num_classes (int): Number of classes to predict.

    Returns:
        _PackedResNet: A Packed-Ensembles ResNet-101.
    """
    return _PackedResNet(
        block=Bottleneck,
        num_blocks=[3, 4, 23, 3],
        in_channels=in_channels,
        num_estimators=num_estimators,
        alpha=alpha,
        gamma=gamma,
        num_classes=num_classes,
    )


def packed_resnet152(
    in_channels: int,
    num_estimators: int,
    alpha: int,
    gamma: int,
    num_classes: int,
) -> _PackedResNet:
    """Packed-Ensembles of ResNet-152 from `Deep Residual Learning for Image
    Recognition <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        in_channels (int): Number of input channels.
        num_estimators (int): Number of estimators in the ensemble.
        alpha (int): Expansion factor affecting the width of the estimators.
        gamma (int): Number of groups within each estimator.
        num_classes (int): Number of classes to predict.

    Returns:
        _PackedResNet: A Packed-Ensembles ResNet-152.
    """
    return _PackedResNet(
        block=Bottleneck,
        num_blocks=[3, 8, 36, 3],
        in_channels=in_channels,
        num_estimators=num_estimators,
        alpha=alpha,
        gamma=gamma,
        num_classes=num_classes,
    )
