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
        alpha: float = 2,
        num_estimators: int = 4,
        gamma: int = 1,
    ):
        super(BasicBlock, self).__init__()

        # No subgroups for the first layer
        self.conv1 = PackedConv2d(
            in_planes,
            planes,
            kernel_size=3,
            alpha=alpha,
            num_estimators=num_estimators,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes * alpha)
        self.conv2 = PackedConv2d(
            planes,
            planes,
            kernel_size=3,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes * alpha)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                PackedConv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    alpha=alpha,
                    num_estimators=num_estimators,
                    gamma=gamma,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes * alpha),
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
        alpha: float = 2,
        num_estimators: int = 4,
        gamma: int = 1,
    ):
        super(Bottleneck, self).__init__()

        # No subgroups for the first layer
        self.conv1 = PackedConv2d(
            in_planes,
            planes,
            kernel_size=1,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=1,  # No groups in the first layer
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes * alpha)
        self.conv2 = PackedConv2d(
            planes,
            planes,
            kernel_size=3,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
            stride=stride,
            padding=1,
            groups=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes * alpha)
        self.conv3 = PackedConv2d(
            planes,
            self.expansion * planes,
            kernel_size=1,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes * alpha)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                PackedConv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    alpha=alpha,
                    num_estimators=num_estimators,
                    gamma=gamma,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes * alpha),
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
        imagenet_structure: bool = True,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.num_estimators = num_estimators
        self.in_planes = 64
        block_planes = self.in_planes

        if imagenet_structure:
            self.conv1 = PackedConv2d(
                self.in_channels,
                block_planes,
                kernel_size=7,
                stride=2,
                padding=3,
                alpha=alpha,
                num_estimators=num_estimators,
                gamma=1,  # No groups for the first layer
                groups=1,
                bias=False,
                first=True,
            )
        else:
            self.conv1 = PackedConv2d(
                self.in_channels,
                block_planes,
                kernel_size=3,
                stride=1,
                padding=1,
                alpha=alpha,
                num_estimators=num_estimators,
                gamma=1,  # No groups for the first layer
                groups=1,
                bias=False,
                first=True,
            )

        self.bn1 = nn.BatchNorm2d(block_planes * alpha)

        if imagenet_structure:
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
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
        )
        self.layer2 = self._make_layer(
            block,
            block_planes * 2,
            num_blocks[1],
            stride=2,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
        )
        self.layer3 = self._make_layer(
            block,
            block_planes * 4,
            num_blocks[2],
            stride=2,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
        )
        self.layer4 = self._make_layer(
            block,
            block_planes * 8,
            num_blocks[3],
            stride=2,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
        )

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten(1)

        self.linear = PackedLinear(
            block_planes * 8 * block.expansion,
            num_classes,
            alpha=alpha,
            num_estimators=num_estimators,
            last=True,
        )

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        num_blocks: int,
        stride: int,
        alpha: float,
        num_estimators: int,
        gamma: int,
    ) -> nn.Module:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    stride,
                    alpha=alpha,
                    num_estimators=num_estimators,
                    gamma=gamma,
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
    imagenet_structure: bool = True,
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
        imagenet_structure=imagenet_structure,
    )


def packed_resnet34(
    in_channels: int,
    num_estimators: int,
    alpha: int,
    gamma: int,
    num_classes: int,
    imagenet_structure: bool = True,
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
        imagenet_structure=imagenet_structure,
    )


def packed_resnet50(
    in_channels: int,
    num_estimators: int,
    alpha: int,
    gamma: int,
    num_classes: int,
    imagenet_structure: bool = True,
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
        imagenet_structure=imagenet_structure,
    )


def packed_resnet101(
    in_channels: int,
    num_estimators: int,
    alpha: int,
    gamma: int,
    num_classes: int,
    imagenet_structure: bool = True,
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
        imagenet_structure=imagenet_structure,
    )


def packed_resnet152(
    in_channels: int,
    num_estimators: int,
    alpha: int,
    gamma: int,
    num_classes: int,
    imagenet_structure: bool = True,
) -> _PackedResNet:
    """Packed-Ensembles of ResNet-152 from `Deep Residual Learning for Image
    Recognition <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        in_channels (int): Number of input channels.
        num_estimators (int): Number of estimators in the ensemble.
        alpha (int): Expansion factor affecting the width of the estimators.
        gamma (int): Number of groups within each estimator.
        num_classes (int): Number of classes to predict.
        imagenet_structure (bool, optional): Whether to use the ImageNet
            structure. Defaults to ``True``.

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
        imagenet_structure=imagenet_structure,
    )
