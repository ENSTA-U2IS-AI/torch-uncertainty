"""_BatchedResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""

from typing import Literal

import torch.nn.functional as F
from torch import Tensor, nn

from torch_uncertainty.layers import BatchConv2d, BatchLinear

__all__ = [
    "batched_resnet18",
    "batched_resnet20",
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
        conv_bias: bool,
        dropout_rate: float,
        groups: int,
        normalization_layer: nn.Module,
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
            bias=conv_bias,
        )
        self.bn1 = normalization_layer(planes)

        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv2 = BatchConv2d(
            planes,
            planes,
            kernel_size=3,
            num_estimators=num_estimators,
            groups=groups,
            stride=1,
            padding=1,
            bias=conv_bias,
        )
        self.bn2 = normalization_layer(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    groups=groups,
                    kernel_size=1,
                    stride=stride,
                    bias=conv_bias,
                ),
                normalization_layer(self.expansion * planes),
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
        conv_bias: bool,
        dropout_rate: float,
        groups: int,
        normalization_layer: nn.Module,
    ) -> None:
        super().__init__()
        self.conv1 = BatchConv2d(
            in_planes,
            planes,
            kernel_size=1,
            num_estimators=num_estimators,
            groups=groups,
            bias=conv_bias,
        )
        self.bn1 = normalization_layer(planes)
        self.conv2 = BatchConv2d(
            planes,
            planes,
            kernel_size=3,
            num_estimators=num_estimators,
            groups=groups,
            stride=stride,
            padding=1,
            bias=conv_bias,
        )
        self.bn2 = normalization_layer(planes)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv3 = BatchConv2d(
            planes,
            self.expansion * planes,
            num_estimators=num_estimators,
            groups=groups,
            kernel_size=1,
            bias=conv_bias,
        )
        self.bn3 = normalization_layer(self.expansion * planes)

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
                    bias=conv_bias,
                ),
                normalization_layer(self.expansion * planes),
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
        conv_bias: bool,
        dropout_rate: float,
        groups: int = 1,
        width_multiplier: int = 1,
        style: Literal["imagenet", "cifar"] = "imagenet",
        in_planes: int = 64,
        normalization_layer: nn.Module = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        self.in_planes = in_planes * width_multiplier
        block_planes = in_planes * width_multiplier

        self.num_estimators = num_estimators
        self.dropout_rate = dropout_rate
        self.width_multiplier = width_multiplier

        if style == "imagenet":
            self.conv1 = BatchConv2d(
                3,
                block_planes,
                kernel_size=7,
                stride=2,
                padding=3,
                num_estimators=num_estimators,
                groups=groups,
                bias=False,
            )
        elif style == "cifar":
            self.conv1 = BatchConv2d(
                in_channels,
                block_planes,
                kernel_size=3,
                stride=1,
                padding=1,
                num_estimators=num_estimators,
                groups=groups,
                bias=False,
            )
        else:
            raise ValueError(f"Unknown style. Got {style}.")

        self.bn1 = normalization_layer(block_planes)

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
            num_estimators=num_estimators,
            conv_bias=conv_bias,
            dropout_rate=dropout_rate,
            groups=groups,
            normalization_layer=normalization_layer,
        )
        self.layer2 = self._make_layer(
            block,
            block_planes * 2,
            num_blocks[1],
            stride=2,
            num_estimators=num_estimators,
            conv_bias=conv_bias,
            dropout_rate=dropout_rate,
            groups=groups,
            normalization_layer=normalization_layer,
        )
        self.layer3 = self._make_layer(
            block,
            block_planes * 4,
            num_blocks[2],
            stride=2,
            num_estimators=num_estimators,
            conv_bias=conv_bias,
            dropout_rate=dropout_rate,
            groups=groups,
            normalization_layer=normalization_layer,
        )
        if len(num_blocks) == 4:
            self.layer4 = self._make_layer(
                block,
                block_planes * 8,
                num_blocks[3],
                stride=2,
                num_estimators=num_estimators,
                conv_bias=conv_bias,
                dropout_rate=dropout_rate,
                groups=groups,
                normalization_layer=normalization_layer,
            )
            linear_multiplier = 8
        else:
            self.layer4 = nn.Identity()
            linear_multiplier = 4

        self.dropout = nn.Dropout(p=dropout_rate)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten(1)

        self.linear = BatchLinear(
            block_planes * linear_multiplier * block.expansion,
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
        conv_bias: bool,
        dropout_rate: float,
        groups: int,
        normalization_layer: nn.Module,
    ) -> nn.Module:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    in_planes=self.in_planes,
                    planes=planes,
                    stride=stride,
                    conv_bias=conv_bias,
                    dropout_rate=dropout_rate,
                    num_estimators=num_estimators,
                    groups=groups,
                    normalization_layer=normalization_layer,
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
    conv_bias: bool = True,
    dropout_rate: float = 0,
    groups: int = 1,
    style: Literal["imagenet", "cifar"] = "imagenet",
    normalization_layer: nn.Module = nn.BatchNorm2d,
) -> _BatchedResNet:
    """BatchEnsemble of ResNet-18.

    Args:
        in_channels (int): Number of input channels.
        num_estimators (int): Number of estimators in the ensemble.
        conv_bias (bool): Whether to use bias in convolutions. Defaults to
            ``True``.
        dropout_rate (float): Dropout rate. Defaults to 0.
        groups (int): Number of groups within each estimator.
        num_classes (int): Number of classes to predict.
        style (bool, optional): Whether to use the ImageNet
            structure. Defaults to ``True``.
        normalization_layer (nn.Module, optional): Normalization layer.

    Returns:
        _BatchedResNet: A BatchEnsemble-style ResNet-18.
    """
    return _BatchedResNet(
        _BasicBlock,
        [2, 2, 2, 2],
        in_channels=in_channels,
        num_classes=num_classes,
        num_estimators=num_estimators,
        conv_bias=conv_bias,
        dropout_rate=dropout_rate,
        groups=groups,
        style=style,
        in_planes=64,
        normalization_layer=normalization_layer,
    )


def batched_resnet20(
    in_channels: int,
    num_classes: int,
    num_estimators: int,
    conv_bias: bool = True,
    dropout_rate: float = 0,
    groups: int = 1,
    style: Literal["imagenet", "cifar"] = "imagenet",
    normalization_layer: nn.Module = nn.BatchNorm2d,
) -> _BatchedResNet:
    """BatchEnsemble of ResNet-20.

    Args:
        in_channels (int): Number of input channels.
        num_estimators (int): Number of estimators in the ensemble.
        conv_bias (bool): Whether to use bias in convolutions. Defaults to
            ``True``.
        dropout_rate (float): Dropout rate. Defaults to 0.
        groups (int): Number of groups within each estimator.
        num_classes (int): Number of classes to predict.
        style (bool, optional): Whether to use the ImageNet
            structure. Defaults to ``True``.
        normalization_layer (nn.Module, optional): Normalization layer.

    Returns:
        _BatchedResNet: A BatchEnsemble-style ResNet-20.
    """
    return _BatchedResNet(
        _BasicBlock,
        [3, 3, 3],
        in_channels=in_channels,
        num_classes=num_classes,
        num_estimators=num_estimators,
        conv_bias=conv_bias,
        dropout_rate=dropout_rate,
        groups=groups,
        style=style,
        in_planes=16,
        normalization_layer=normalization_layer,
    )


def batched_resnet34(
    in_channels: int,
    num_classes: int,
    num_estimators: int,
    conv_bias: bool = True,
    dropout_rate: float = 0,
    groups: int = 1,
    style: Literal["imagenet", "cifar"] = "imagenet",
    normalization_layer: nn.Module = nn.BatchNorm2d,
) -> _BatchedResNet:
    """BatchEnsemble of ResNet-34.

    Args:
        in_channels (int): Number of input channels.
        num_estimators (int): Number of estimators in the ensemble.
        conv_bias (bool): Whether to use bias in convolutions. Defaults to
            ``True``.
        dropout_rate (float): Dropout rate. Defaults to 0.
        groups (int): Number of groups within each estimator.
        num_classes (int): Number of classes to predict.
        style (bool, optional): Whether to use the ImageNet
            structure. Defaults to ``True``.
        normalization_layer (nn.Module, optional): Normalization layer.

    Returns:
        _BatchedResNet: A BatchEnsemble-style ResNet-34.
    """
    return _BatchedResNet(
        _BasicBlock,
        [3, 4, 6, 3],
        in_channels=in_channels,
        num_classes=num_classes,
        num_estimators=num_estimators,
        conv_bias=conv_bias,
        dropout_rate=dropout_rate,
        groups=groups,
        style=style,
        in_planes=64,
        normalization_layer=normalization_layer,
    )


def batched_resnet50(
    in_channels: int,
    num_classes: int,
    num_estimators: int,
    conv_bias: bool = True,
    dropout_rate: float = 0,
    groups: int = 1,
    width_multiplier: int = 1,
    style: Literal["imagenet", "cifar"] = "imagenet",
    normalization_layer: nn.Module = nn.BatchNorm2d,
) -> _BatchedResNet:
    """BatchEnsemble of ResNet-50.

    Args:
        in_channels (int): Number of input channels.
        num_estimators (int): Number of estimators in the ensemble.
        conv_bias (bool): Whether to use bias in convolutions. Defaults to
            ``True``.
        dropout_rate (float): Dropout rate. Defaults to 0.
        groups (int): Number of groups within each estimator.
        num_classes (int): Number of classes to predict.
        width_multiplier (int, optional): Expansion factor affecting the width
            of the estimators. Defaults to ``1``.
        style (bool, optional): Whether to use the ImageNet
            structure. Defaults to ``True``.
        normalization_layer (nn.Module, optional): Normalization layer.

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
        conv_bias=conv_bias,
        dropout_rate=dropout_rate,
        groups=groups,
        style=style,
        in_planes=64,
        normalization_layer=normalization_layer,
    )


def batched_resnet101(
    in_channels: int,
    num_classes: int,
    num_estimators: int,
    conv_bias: bool = True,
    dropout_rate: float = 0,
    groups: int = 1,
    style: Literal["imagenet", "cifar"] = "imagenet",
    normalization_layer: nn.Module = nn.BatchNorm2d,
) -> _BatchedResNet:
    """BatchEnsemble of ResNet-101.

    Args:
        in_channels (int): Number of input channels.
        num_estimators (int): Number of estimators in the ensemble.
        conv_bias (bool): Whether to use bias in convolutions. Defaults to
            ``True``.
        dropout_rate (float): Dropout rate. Defaults to 0.
        groups (int): Number of groups within each estimator.
        num_classes (int): Number of classes to predict.
        style (bool, optional): Whether to use the ImageNet
            structure. Defaults to ``True``.
        normalization_layer (nn.Module, optional): Normalization layer.

    Returns:
        _BatchedResNet: A BatchEnsemble-style ResNet-101.
    """
    return _BatchedResNet(
        _Bottleneck,
        [3, 4, 23, 3],
        in_channels=in_channels,
        num_classes=num_classes,
        num_estimators=num_estimators,
        conv_bias=conv_bias,
        dropout_rate=dropout_rate,
        groups=groups,
        style=style,
        in_planes=64,
        normalization_layer=normalization_layer,
    )


def batched_resnet152(
    in_channels: int,
    num_classes: int,
    num_estimators: int,
    conv_bias: bool = True,
    dropout_rate: float = 0,
    groups: int = 1,
    style: Literal["imagenet", "cifar"] = "imagenet",
    normalization_layer: nn.Module = nn.BatchNorm2d,
) -> _BatchedResNet:
    """BatchEnsemble of ResNet-152.

    Args:
        in_channels (int): Number of input channels.
        num_estimators (int): Number of estimators in the ensemble.
        conv_bias (bool): Whether to use bias in convolutions. Defaults to
            ``True``.
        dropout_rate (float): Dropout rate. Defaults to 0.
        groups (int): Number of groups within each estimator.
        num_classes (int): Number of classes to predict.
        style (bool, optional): Whether to use the ImageNet
            structure. Defaults to ``True``.
        normalization_layer (nn.Module, optional): Normalization layer.

    Returns:
        _BatchedResNet: A BatchEnsemble-style ResNet-152.
    """
    return _BatchedResNet(
        _Bottleneck,
        [3, 8, 36, 3],
        in_channels=in_channels,
        num_classes=num_classes,
        num_estimators=num_estimators,
        conv_bias=conv_bias,
        dropout_rate=dropout_rate,
        groups=groups,
        style=style,
        in_planes=64,
        normalization_layer=normalization_layer,
    )
