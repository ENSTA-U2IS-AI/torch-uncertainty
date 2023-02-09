"""_BatchedResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
# fmt: off
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...layers import BatchConv2d, BatchLinear

# fmt: on
__all__ = [
    "batched_resnet18",
    "batched_resnet34",
    "batched_resnet50",
    "batched_resnet101",
    "batched_resnet152",
]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        num_estimators: int = 4,
    ) -> None:
        super().__init__()
        self.conv1 = BatchConv2d(
            in_planes,
            planes,
            kernel_size=3,
            num_estimators=num_estimators,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = BatchConv2d(
            planes,
            planes,
            kernel_size=3,
            num_estimators=num_estimators,
            stride=1,
            padding=1,
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
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, input: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(input)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(input)
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
    ) -> None:
        super(Bottleneck, self).__init__()
        self.conv1 = BatchConv2d(
            in_planes,
            planes,
            kernel_size=1,
            num_estimators=num_estimators,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = BatchConv2d(
            planes,
            planes,
            kernel_size=3,
            num_estimators=num_estimators,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = BatchConv2d(
            planes,
            self.expansion * planes,
            num_estimators=num_estimators,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                BatchConv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    num_estimators=num_estimators,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, input: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(input)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(input)
        out = F.relu(out)
        return out


class _BatchedResNet(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        in_channels: int,
        num_estimators,
        num_classes=10,
        width_multiplier: int = 1,
        # dataset: str = "cifar",
    ):
        super().__init__()
        self.in_planes = 64 * width_multiplier
        # self.dataset = dataset

        self.width_multiplier = width_multiplier
        # if dataset == "imagenet":
        #     self.conv1 = BatchConv2d(
        #         3,
        #         64 * self.width_multiplier,
        #         kernel_size=7,
        #         stride=2,
        #         padding=3,
        #         bias=False,
        #         num_estimators=num_estimators,
        #     )

        # elif dataset == "mnist":
        #     self.conv1 = BatchConv2d(
        #         1,
        #         64 * self.width_multiplier,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1,
        #         bias=False,
        #         num_estimators=num_estimators,
        #     )
        # else:
        self.conv1 = BatchConv2d(
            in_channels,
            64 * self.width_multiplier,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            num_estimators=num_estimators,
        )
        self.bn1 = nn.BatchNorm2d(64 * self.width_multiplier)

        self.optional_pool = nn.Identity()

        self.layer1 = self._make_layer(
            block,
            64 * width_multiplier,
            num_blocks[0],
            stride=1,
            num_estimators=num_estimators,
        )
        self.layer2 = self._make_layer(
            block,
            128 * width_multiplier,
            num_blocks[1],
            stride=2,
            num_estimators=num_estimators,
        )
        self.layer3 = self._make_layer(
            block,
            256 * width_multiplier,
            num_blocks[2],
            stride=2,
            num_estimators=num_estimators,
        )
        self.layer4 = self._make_layer(
            block,
            512 * width_multiplier,
            num_blocks[3],
            stride=2,
            num_estimators=num_estimators,
        )
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten(1)

        self.linear = BatchLinear(
            512 * width_multiplier * block.expansion,
            num_classes,
            num_estimators=num_estimators,
        )

    def _make_layer(self, block, planes, num_blocks, stride, num_estimators):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, num_estimators))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # if self.dataset == "imagenet" or self.dataset == "tinyimagenet":
        #     out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)
        out = self.optional_pool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
        out = self.flatten(out)
        out = self.linear(out)
        return out


def batched_resnet18(
    in_channels: int, num_estimators: int, num_classes: int
) -> _BatchedResNet:
    """BatchEnsembles of ResNet-18 from `Deep Residual Learning for Image
    Recognition <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        in_channels (int): Number of input channels.
        num_estimators (int): Number of estimators in the ensemble.
        groups (int): Number of groups within each estimator.
        num_classes (int): Number of classes to predict.

    Returns:
        _BatchedResNet: A BatchEnsembles-style ResNet-18.
    """
    return _BatchedResNet(
        BasicBlock,
        [2, 2, 2, 2],
        in_channels=in_channels,
        num_estimators=num_estimators,
        num_classes=num_classes,
    )


def batched_resnet34(
    in_channels: int, num_estimators: int, num_classes: int
) -> _BatchedResNet:
    """BatchEnsembles of ResNet-18 from `Deep Residual Learning for Image
    Recognition <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        in_channels (int): Number of input channels.
        num_estimators (int): Number of estimators in the ensemble.
        groups (int): Number of groups within each estimator.
        num_classes (int): Number of classes to predict.

    Returns:
        _BatchedResNet: A BatchEnsembles-style ResNet-34.
    """
    return _BatchedResNet(
        BasicBlock,
        [3, 4, 6, 3],
        in_channels=in_channels,
        num_estimators=num_estimators,
        num_classes=num_classes,
    )


def batched_resnet50(
    in_channels: int,
    num_estimators: int,
    num_classes: int,
    width_multiplier: int = 1,
) -> _BatchedResNet:
    """BatchEnsembles of ResNet-18 from `Deep Residual Learning for Image
    Recognition <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        in_channels (int): Number of input channels.
        num_estimators (int): Number of estimators in the ensemble.
        groups (int): Number of groups within each estimator.
        num_classes (int): Number of classes to predict.

    Returns:
        _BatchedResNet: A BatchEnsembles-style ResNet-50.
    """
    return _BatchedResNet(
        Bottleneck,
        [3, 4, 6, 3],
        in_channels=in_channels,
        num_estimators=num_estimators,
        num_classes=num_classes,
        width_multiplier=width_multiplier,
    )


def batched_resnet101(
    in_channels: int, num_estimators: int, num_classes: int
) -> _BatchedResNet:
    """BatchEnsembles of ResNet-18 from `Deep Residual Learning for Image
    Recognition <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        in_channels (int): Number of input channels.
        num_estimators (int): Number of estimators in the ensemble.
        groups (int): Number of groups within each estimator.
        num_classes (int): Number of classes to predict.

    Returns:
        _BatchedResNet: A BatchEnsembles-style ResNet-101.
    """
    return _BatchedResNet(
        Bottleneck,
        [3, 4, 23, 3],
        in_channels=in_channels,
        num_estimators=num_estimators,
        num_classes=num_classes,
    )


def batched_resnet152(
    in_channels: int, num_estimators: int, num_classes: int
) -> _BatchedResNet:
    """BatchEnsembles of ResNet-18 from `Deep Residual Learning for Image
    Recognition <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        in_channels (int): Number of input channels.
        num_estimators (int): Number of estimators in the ensemble.
        groups (int): Number of groups within each estimator.
        num_classes (int): Number of classes to predict.

    Returns:
        _BatchedResNet: A BatchEnsembles-style ResNet-152.
    """
    return _BatchedResNet(
        Bottleneck,
        [3, 8, 36, 3],
        in_channels=in_channels,
        num_estimators=num_estimators,
        num_classes=num_classes,
    )
