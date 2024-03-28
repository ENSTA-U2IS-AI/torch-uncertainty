from collections.abc import Callable
from typing import Literal

from torch import Tensor, nn
from torch.nn.functional import relu

__all__ = [
    "resnet18",
    "resnet20",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
]


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int,
        dropout_rate: float,
        groups: int,
        activation_fn: Callable,
        normalization_layer: nn.Module,
        conv_bias: bool,
    ) -> None:
        super().__init__()
        self.activation_fn = activation_fn

        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=groups,
            bias=conv_bias,
        )
        self.bn1 = normalization_layer(planes)

        # As in timm
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=groups,
            bias=conv_bias,
        )
        self.bn2 = normalization_layer(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    groups=groups,
                    bias=conv_bias,
                ),
                normalization_layer(self.expansion * planes),
            )

    def forward(self, x: Tensor) -> Tensor:
        out = self.activation_fn(self.dropout(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.activation_fn(out)


class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int,
        dropout_rate: float,
        groups: int,
        activation_fn: Callable,
        normalization_layer: nn.Module,
        conv_bias: bool,
    ) -> None:
        super().__init__()
        self.activation_fn = activation_fn

        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=1,
            groups=groups,
            bias=conv_bias,
        )
        self.bn1 = normalization_layer(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=groups,
            bias=conv_bias,
        )
        self.bn2 = normalization_layer(planes)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv3 = nn.Conv2d(
            planes,
            self.expansion * planes,
            kernel_size=1,
            groups=groups,
            bias=conv_bias,
        )
        self.bn3 = normalization_layer(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    groups=groups,
                    bias=conv_bias,
                ),
                normalization_layer(self.expansion * planes),
            )

    def forward(self, x: Tensor) -> Tensor:
        out = self.activation_fn(self.bn1(self.conv1(x)))
        out = self.activation_fn(self.dropout(self.bn2(self.conv2(out))))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return self.activation_fn(out)


# ruff: noqa: ERA001
# class Robust_Bottleneck(nn.Module):
#     """Robust _Bottleneck from "Can CNNs be more robust than transformers?"
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
#             bias=self.conv_bias,
#         )
#         self.bn1 = normalization_layer(planes)
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
#         out = relu(self.conv2(out))
#         out = self.conv3(out)
#         out += self.shortcut(x)
#         return out


class _ResNet(nn.Module):
    def __init__(
        self,
        block: type[_BasicBlock | _Bottleneck],
        num_blocks: list[int],
        in_channels: int,
        num_classes: int,
        conv_bias: bool,
        dropout_rate: float,
        groups: int,
        style: Literal["imagenet", "cifar"] = "imagenet",
        in_planes: int = 64,
        activation_fn: Callable = relu,
        normalization_layer: nn.Module = nn.BatchNorm2d,
    ) -> None:
        """ResNet from `Deep Residual Learning for Image Recognition`."""
        super().__init__()

        self.in_planes = in_planes
        block_planes = in_planes
        self.dropout_rate = dropout_rate
        self.activation_fn = activation_fn

        if style == "imagenet":
            self.conv1 = nn.Conv2d(
                in_channels,
                block_planes,
                kernel_size=7,
                stride=2,
                padding=3,
                groups=1,  # No groups in the first layer
                bias=conv_bias,
            )
        elif style == "cifar":
            self.conv1 = nn.Conv2d(
                in_channels,
                block_planes,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=1,  # No groups in the first layer
                bias=conv_bias,
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
            dropout_rate=dropout_rate,
            groups=groups,
            activation_fn=activation_fn,
            normalization_layer=normalization_layer,
            conv_bias=conv_bias,
        )
        self.layer2 = self._make_layer(
            block,
            block_planes * 2,
            num_blocks[1],
            stride=2,
            dropout_rate=dropout_rate,
            groups=groups,
            activation_fn=activation_fn,
            normalization_layer=normalization_layer,
            conv_bias=conv_bias,
        )
        self.layer3 = self._make_layer(
            block,
            block_planes * 4,
            num_blocks[2],
            stride=2,
            dropout_rate=dropout_rate,
            groups=groups,
            activation_fn=activation_fn,
            normalization_layer=normalization_layer,
            conv_bias=conv_bias,
        )
        if len(num_blocks) == 4:
            self.layer4 = self._make_layer(
                block,
                block_planes * 8,
                num_blocks[3],
                stride=2,
                dropout_rate=dropout_rate,
                groups=groups,
                activation_fn=activation_fn,
                normalization_layer=normalization_layer,
                conv_bias=conv_bias,
            )
            linear_multiplier = 8
        else:
            self.layer4 = nn.Identity()
            linear_multiplier = 4

        self.dropout = nn.Dropout(p=dropout_rate)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten(1)

        self.linear = nn.Linear(
            block_planes * linear_multiplier * block.expansion,
            num_classes,
        )

    def _make_layer(
        self,
        block: type[_BasicBlock] | type[_Bottleneck],
        planes: int,
        num_blocks: int,
        stride: int,
        dropout_rate: float,
        groups: int,
        activation_fn: Callable,
        normalization_layer: nn.Module,
        conv_bias: bool,
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
                    activation_fn=activation_fn,
                    normalization_layer=normalization_layer,
                    conv_bias=conv_bias,
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def feats_forward(self, x: Tensor) -> Tensor:
        out = self.activation_fn(self.bn1(self.conv1(x)))
        out = self.optional_pool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
        return self.dropout(self.flatten(out))

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(self.feats_forward(x))


def resnet18(
    in_channels: int,
    num_classes: int,
    conv_bias: bool = True,
    dropout_rate: float = 0.0,
    groups: int = 1,
    style: Literal["imagenet", "cifar"] = "imagenet",
    activation_fn: Callable = relu,
    normalization_layer: nn.Module = nn.BatchNorm2d,
) -> _ResNet:
    """ResNet-18 model.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of classes to predict.
        conv_bias (bool): Whether to use bias in convolutions. Defaults to
            ``True``.
        conv_bias (bool): Whether to use bias in convolutions. Defaults to
            ``True``.
        dropout_rate (float): Dropout rate. Defaults to 0.
        groups (int): Number of groups in convolutions. Defaults to 1.
        style (bool, optional): Whether to use the ImageNet
            structure. Defaults to ``True``.
        activation_fn (Callable, optional): Activation function.
        normalization_layer (nn.Module, optional): Normalization layer.

    Returns:
        _ResNet: A ResNet-18.
    """
    return _ResNet(
        block=_BasicBlock,
        num_blocks=[2, 2, 2, 2],
        in_channels=in_channels,
        num_classes=num_classes,
        conv_bias=conv_bias,
        dropout_rate=dropout_rate,
        groups=groups,
        style=style,
        in_planes=64,
        activation_fn=activation_fn,
        normalization_layer=normalization_layer,
    )


def resnet20(
    in_channels: int,
    num_classes: int,
    conv_bias: bool = True,
    dropout_rate: float = 0.0,
    groups: int = 1,
    style: Literal["imagenet", "cifar"] = "imagenet",
    activation_fn: Callable = relu,
    normalization_layer: nn.Module = nn.BatchNorm2d,
) -> _ResNet:
    """ResNet-18 model.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of classes to predict.
        conv_bias (bool): Whether to use bias in convolutions. Defaults to
            ``True``.
        dropout_rate (float): Dropout rate. Defaults to 0.
        groups (int): Number of groups in convolutions. Defaults to 1.
        style (bool, optional): Whether to use the ImageNet
            structure. Defaults to ``True``.
        activation_fn (Callable, optional): Activation function.
        normalization_layer (nn.Module, optional): Normalization layer.

    Returns:
        _ResNet: A ResNet-20.
    """
    return _ResNet(
        block=_BasicBlock,
        num_blocks=[3, 3, 3],
        in_channels=in_channels,
        num_classes=num_classes,
        conv_bias=conv_bias,
        dropout_rate=dropout_rate,
        groups=groups,
        style=style,
        in_planes=16,
        activation_fn=activation_fn,
        normalization_layer=normalization_layer,
    )


def resnet34(
    in_channels: int,
    num_classes: int,
    conv_bias: bool = True,
    dropout_rate: float = 0,
    groups: int = 1,
    style: Literal["imagenet", "cifar"] = "imagenet",
    activation_fn: Callable = relu,
    normalization_layer: nn.Module = nn.BatchNorm2d,
) -> _ResNet:
    """ResNet-34 model.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of classes to predict.
        conv_bias (bool): Whether to use bias in convolutions. Defaults to
            ``True``.
        dropout_rate (float): Dropout rate. Defaults to 0.
        groups (int): Number of groups in convolutions. Defaults to 1.
        style (bool, optional): Whether to use the ImageNet
            structure. Defaults to ``True``.
        activation_fn (Callable, optional): Activation function.
        normalization_layer (nn.Module, optional): Normalization layer.

    Returns:
        _ResNet: A ResNet-34.
    """
    return _ResNet(
        block=_BasicBlock,
        num_blocks=[3, 4, 6, 3],
        in_channels=in_channels,
        num_classes=num_classes,
        conv_bias=conv_bias,
        dropout_rate=dropout_rate,
        groups=groups,
        style=style,
        in_planes=64,
        activation_fn=activation_fn,
        normalization_layer=normalization_layer,
    )


def resnet50(
    in_channels: int,
    num_classes: int,
    conv_bias: bool = True,
    dropout_rate: float = 0,
    groups: int = 1,
    style: Literal["imagenet", "cifar"] = "imagenet",
    activation_fn: Callable = relu,
    normalization_layer: nn.Module = nn.BatchNorm2d,
) -> _ResNet:
    """ResNet-50 model.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of classes to predict.
        conv_bias (bool): Whether to use bias in convolutions. Defaults to
            ``True``.
        dropout_rate (float): Dropout rate. Defaults to 0.
        groups (int): Number of groups in convolutions. Defaults to 1.
        style (bool, optional): Whether to use the ImageNet
            structure. Defaults to ``True``.
        activation_fn (Callable, optional): Activation function.
        normalization_layer (nn.Module, optional): Normalization layer.

    Returns:
        _ResNet: A ResNet-50.
    """
    return _ResNet(
        block=_Bottleneck,
        num_blocks=[3, 4, 6, 3],
        in_channels=in_channels,
        num_classes=num_classes,
        conv_bias=conv_bias,
        dropout_rate=dropout_rate,
        groups=groups,
        style=style,
        in_planes=64,
        activation_fn=activation_fn,
        normalization_layer=normalization_layer,
    )


def resnet101(
    in_channels: int,
    num_classes: int,
    conv_bias: bool = True,
    dropout_rate: float = 0,
    groups: int = 1,
    style: Literal["imagenet", "cifar"] = "imagenet",
    activation_fn: Callable = relu,
    normalization_layer: nn.Module = nn.BatchNorm2d,
) -> _ResNet:
    """ResNet-101 model.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of classes to predict.
        conv_bias (bool): Whether to use bias in convolutions. Defaults to
            ``True``.
        dropout_rate (float): Dropout rate. Defaults to 0.
        groups (int): Number of groups in convolutions. Defaults to 1.
        style (bool, optional): Whether to use the ImageNet
            structure. Defaults to ``True``.
        activation_fn (Callable, optional): Activation function.
        normalization_layer (nn.Module, optional): Normalization layer.

    Returns:
        _ResNet: A ResNet-101.
    """
    return _ResNet(
        block=_Bottleneck,
        num_blocks=[3, 4, 23, 3],
        in_channels=in_channels,
        num_classes=num_classes,
        conv_bias=conv_bias,
        dropout_rate=dropout_rate,
        groups=groups,
        style=style,
        in_planes=64,
        activation_fn=activation_fn,
        normalization_layer=normalization_layer,
    )


def resnet152(
    in_channels: int,
    num_classes: int,
    conv_bias: bool = True,
    dropout_rate: float = 0,
    groups: int = 1,
    style: Literal["imagenet", "cifar"] = "imagenet",
    activation_fn: Callable = relu,
    normalization_layer: nn.Module = nn.BatchNorm2d,
) -> _ResNet:
    """ResNet-152 model.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of classes to predict.
        conv_bias (bool): Whether to use bias in convolutions. Defaults to
            ``True``.
        dropout_rate (float): Dropout rate. Defaults to 0.
        groups (int, optional): Number of groups in convolutions. Defaults to
            ``1``.
        style (bool, optional): Whether to use the ImageNet
            structure. Defaults to ``True``.
        activation_fn (Callable, optional): Activation function.
        normalization_layer (nn.Module, optional): Normalization layer.

    Returns:
        _ResNet: A ResNet-152.
    """
    return _ResNet(
        block=_Bottleneck,
        num_blocks=[3, 8, 36, 3],
        in_channels=in_channels,
        num_classes=num_classes,
        conv_bias=conv_bias,
        dropout_rate=dropout_rate,
        groups=groups,
        style=style,
        in_planes=64,
        activation_fn=activation_fn,
        normalization_layer=normalization_layer,
    )
