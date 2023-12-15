"""_BatchedResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""

import torch.nn.functional as F
from torch import Tensor, nn

from torch_uncertainty.layers import BatchConv2d, BatchLinear

__all__ = [
    "batched_resnet18",
    "batched_resnet34",
    "batched_resnet50",
    "batched_resnet101",
    "batched_resnet152",
]


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int,
        num_estimators: int,
        dropout_rate: float,
        groups: int,
    ) -> None:
        super().__init__()
        self.conv1 = BatchConv2d(
            in_planes,
            planes,
            kernel_size=3,
            num_estimators=num_estimators,
            groups=groups,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)

        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv2 = BatchConv2d(
            planes,
            planes,
            kernel_size=3,
            num_estimators=num_estimators,
            groups=groups,
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
                    groups=groups,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, inputs: Tensor) -> Tensor:
        out = F.relu(self.dropout(self.bn1(self.conv1(inputs))))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(inputs)
        return F.relu(out)


class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int,
        num_estimators: int,
        dropout_rate: float,
        groups: int,
    ) -> None:
        super().__init__()
        self.conv1 = BatchConv2d(
            in_planes,
            planes,
            kernel_size=1,
            num_estimators=num_estimators,
            groups=groups,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = BatchConv2d(
            planes,
            planes,
            kernel_size=3,
            num_estimators=num_estimators,
            groups=groups,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv3 = BatchConv2d(
            planes,
            self.expansion * planes,
            num_estimators=num_estimators,
            groups=groups,
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
                    groups=groups,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, inputs: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(inputs)))
        out = F.relu(self.dropout(self.bn2(self.conv2(out))))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(inputs)
        return F.relu(out)


class _BatchedResNet(nn.Module):
    def __init__(
        self,
        block: type[_BasicBlock | _Bottleneck],
        num_blocks: list[int],
        in_channels: int,
        num_classes: int,
        num_estimators: int,
        dropout_rate: float,
        groups: int = 1,
        width_multiplier: int = 1,
        style: str = "imagenet",
    ) -> None:
        super().__init__()
        self.in_planes = 64 * width_multiplier
        self.num_estimators = num_estimators
        self.dropout_rate = dropout_rate

        self.width_multiplier = width_multiplier
        if style == "imagenet":
            self.conv1 = BatchConv2d(
                3,
                64 * self.width_multiplier,
                kernel_size=7,
                stride=2,
                padding=3,
                num_estimators=num_estimators,
                groups=groups,
                bias=False,
            )
        else:
            self.conv1 = BatchConv2d(
                in_channels,
                64 * self.width_multiplier,
                kernel_size=3,
                stride=1,
                padding=1,
                num_estimators=num_estimators,
                groups=groups,
                bias=False,
            )
        self.bn1 = nn.BatchNorm2d(64 * self.width_multiplier)

        if style == "imagenet":
            self.optional_pool = nn.MaxPool2d(
                kernel_size=3, stride=2, padding=1
            )
        else:
            self.optional_pool = nn.Identity()

        self.layer1 = self._make_layer(
            block,
            64 * width_multiplier,
            num_blocks[0],
            stride=1,
            num_estimators=num_estimators,
            dropout_rate=dropout_rate,
            groups=groups,
        )
        self.layer2 = self._make_layer(
            block,
            128 * width_multiplier,
            num_blocks[1],
            stride=2,
            num_estimators=num_estimators,
            dropout_rate=dropout_rate,
            groups=groups,
        )
        self.layer3 = self._make_layer(
            block,
            256 * width_multiplier,
            num_blocks[2],
            stride=2,
            num_estimators=num_estimators,
            dropout_rate=dropout_rate,
            groups=groups,
        )
        self.layer4 = self._make_layer(
            block,
            512 * width_multiplier,
            num_blocks[3],
            stride=2,
            num_estimators=num_estimators,
            dropout_rate=dropout_rate,
            groups=groups,
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten(1)

        self.linear = BatchLinear(
            512 * width_multiplier * block.expansion,
            num_classes,
            num_estimators=num_estimators,
        )

    def _make_layer(
        self,
        block: type[_BasicBlock | _Bottleneck],
        planes: int,
        num_blocks: int,
        stride: int,
        num_estimators: int,
        dropout_rate: float,
        groups: int = 1,
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
                    num_estimators=num_estimators,
                    groups=groups,
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = x.repeat(self.num_estimators, 1, 1, 1)
        out = F.relu(self.bn1(self.conv1(out)))
        out = self.optional_pool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
        out = self.dropout(self.flatten(out))
        return self.linear(out)


def batched_resnet18(
    in_channels: int,
    num_classes: int,
    num_estimators: int,
    dropout_rate: float = 0,
    groups: int = 1,
    style: str = "imagenet",
) -> _BatchedResNet:
    """BatchEnsemble of ResNet-18 from `Deep Residual Learning for Image
    Recognition <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        in_channels (int): Number of input channels.
        num_estimators (int): Number of estimators in the ensemble.
        dropout_rate (float): Dropout rate. Defaults to 0.
        groups (int): Number of groups within each estimator.
        num_classes (int): Number of classes to predict.
        style (bool, optional): Whether to use the ImageNet
            structure. Defaults to ``True``.

    Returns:
        _BatchedResNet: A BatchEnsemble-style ResNet-18.
    """
    return _BatchedResNet(
        _BasicBlock,
        [2, 2, 2, 2],
        in_channels=in_channels,
        num_classes=num_classes,
        num_estimators=num_estimators,
        dropout_rate=dropout_rate,
        groups=groups,
        style=style,
    )


def batched_resnet34(
    in_channels: int,
    num_classes: int,
    num_estimators: int,
    dropout_rate: float = 0,
    groups: int = 1,
    style: str = "imagenet",
) -> _BatchedResNet:
    """BatchEnsemble of ResNet-34 from `Deep Residual Learning for Image
    Recognition <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        in_channels (int): Number of input channels.
        num_estimators (int): Number of estimators in the ensemble.
        dropout_rate (float): Dropout rate. Defaults to 0.
        groups (int): Number of groups within each estimator.
        num_classes (int): Number of classes to predict.
        style (bool, optional): Whether to use the ImageNet
            structure. Defaults to ``True``.

    Returns:
        _BatchedResNet: A BatchEnsemble-style ResNet-34.
    """
    return _BatchedResNet(
        _BasicBlock,
        [3, 4, 6, 3],
        in_channels=in_channels,
        num_classes=num_classes,
        num_estimators=num_estimators,
        dropout_rate=dropout_rate,
        groups=groups,
        style=style,
    )


def batched_resnet50(
    in_channels: int,
    num_classes: int,
    num_estimators: int,
    dropout_rate: float = 0,
    groups: int = 1,
    width_multiplier: int = 1,
    style: str = "imagenet",
) -> _BatchedResNet:
    """BatchEnsemble of ResNet-50 from `Deep Residual Learning for Image
    Recognition <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        in_channels (int): Number of input channels.
        num_estimators (int): Number of estimators in the ensemble.
        dropout_rate (float): Dropout rate. Defaults to 0.
        groups (int): Number of groups within each estimator.
        num_classes (int): Number of classes to predict.
        width_multiplier (int, optional): Expansion factor affecting the width
            of the estimators. Defaults to ``1``.
        style (bool, optional): Whether to use the ImageNet
            structure. Defaults to ``True``.

    Returns:
        _BatchedResNet: A BatchEnsemble-style ResNet-50.
    """
    return _BatchedResNet(
        _Bottleneck,
        [3, 4, 6, 3],
        in_channels=in_channels,
        num_classes=num_classes,
        num_estimators=num_estimators,
        width_multiplier=width_multiplier,
        dropout_rate=dropout_rate,
        groups=groups,
        style=style,
    )


def batched_resnet101(
    in_channels: int,
    num_classes: int,
    num_estimators: int,
    dropout_rate: float = 0,
    groups: int = 1,
    style: str = "imagenet",
) -> _BatchedResNet:
    """BatchEnsemble of ResNet-101 from `Deep Residual Learning for Image
    Recognition <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        in_channels (int): Number of input channels.
        num_estimators (int): Number of estimators in the ensemble.
        dropout_rate (float): Dropout rate. Defaults to 0.
        groups (int): Number of groups within each estimator.
        num_classes (int): Number of classes to predict.
        style (bool, optional): Whether to use the ImageNet
            structure. Defaults to ``True``.

    Returns:
        _BatchedResNet: A BatchEnsemble-style ResNet-101.
    """
    return _BatchedResNet(
        _Bottleneck,
        [3, 4, 23, 3],
        in_channels=in_channels,
        num_classes=num_classes,
        num_estimators=num_estimators,
        dropout_rate=dropout_rate,
        groups=groups,
        style=style,
    )


def batched_resnet152(
    in_channels: int,
    num_classes: int,
    num_estimators: int,
    dropout_rate: float = 0,
    groups: int = 1,
    style: str = "imagenet",
) -> _BatchedResNet:
    """BatchEnsemble of ResNet-152 from `Deep Residual Learning for Image
    Recognition <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        in_channels (int): Number of input channels.
        num_estimators (int): Number of estimators in the ensemble.
        dropout_rate (float): Dropout rate. Defaults to 0.
        groups (int): Number of groups within each estimator.
        num_classes (int): Number of classes to predict.
        style (bool, optional): Whether to use the ImageNet
            structure. Defaults to ``True``.

    Returns:
        _BatchedResNet: A BatchEnsemble-style ResNet-152.
    """
    return _BatchedResNet(
        _Bottleneck,
        [3, 8, 36, 3],
        in_channels=in_channels,
        num_classes=num_classes,
        num_estimators=num_estimators,
        dropout_rate=dropout_rate,
        groups=groups,
        style=style,
    )
