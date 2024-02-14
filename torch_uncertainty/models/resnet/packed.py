from typing import Any, Literal

import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from torch_uncertainty.layers import PackedConv2d, PackedLinear
from torch_uncertainty.utils import load_hf

__all__ = [
    "packed_resnet18",
    "packed_resnet20",
    "packed_resnet34",
    "packed_resnet50",
    "packed_resnet101",
    "packed_resnet152",
]

weight_ids = {
    "10": {
        "18": "pe_resnet18_c10",
        "32": None,
        "50": "pe_resnet50_c10",
        "101": None,
        "152": None,
    },
    "100": {
        "18": "pe_resnet18_c100",
        "32": None,
        "50": "pe_resnet50_c100",
        "101": None,
        "152": None,
    },
    "1000": {
        "18": None,
        "32": None,
        "50": "pe_resnet50_in1k",
        "101": None,
        "152": None,
    },
}


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int,
        alpha: float,
        num_estimators: int,
        gamma: int,
        conv_bias: bool,
        dropout_rate: float,
        groups: int,
        normalization_layer: nn.Module,
    ) -> None:
        super().__init__()

        # No subgroups for the first layer
        self.conv1 = PackedConv2d(
            in_planes,
            planes,
            kernel_size=3,
            alpha=alpha,
            num_estimators=num_estimators,
            groups=groups,
            stride=stride,
            padding=1,
            bias=conv_bias,
        )
        self.bn1 = normalization_layer(planes * alpha)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv2 = PackedConv2d(
            planes,
            planes,
            kernel_size=3,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
            groups=groups,
            stride=1,
            padding=1,
            bias=conv_bias,
        )
        self.bn2 = normalization_layer(planes * alpha)

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
                    groups=groups,
                    stride=stride,
                    bias=conv_bias,
                ),
                normalization_layer(self.expansion * planes * alpha),
            )

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.dropout(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int,
        alpha: float,
        num_estimators: int,
        gamma: int,
        conv_bias: bool,
        dropout_rate: float,
        groups: int,
        normalization_layer: nn.Module,
    ) -> None:
        super().__init__()

        # No subgroups for the first layer
        self.conv1 = PackedConv2d(
            in_planes,
            planes,
            kernel_size=1,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=1,  # No groups from gamma in the first layer
            groups=groups,
            bias=conv_bias,
        )
        self.bn1 = normalization_layer(planes * alpha)
        self.conv2 = PackedConv2d(
            planes,
            planes,
            kernel_size=3,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
            stride=stride,
            padding=1,
            groups=groups,
            bias=conv_bias,
        )
        self.bn2 = normalization_layer(planes * alpha)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv3 = PackedConv2d(
            planes,
            self.expansion * planes,
            kernel_size=1,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
            groups=groups,
            bias=conv_bias,
        )
        self.bn3 = normalization_layer(self.expansion * planes * alpha)

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
                    groups=groups,
                    stride=stride,
                    bias=conv_bias,
                ),
                normalization_layer(self.expansion * planes * alpha),
            )

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.dropout(self.bn2(self.conv2(out))))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)


class _PackedResNet(nn.Module):
    def __init__(
        self,
        block: type[_BasicBlock | _Bottleneck],
        num_blocks: list[int],
        in_channels: int,
        num_classes: int,
        conv_bias: bool,
        num_estimators: int,
        dropout_rate: float,
        alpha: int = 2,
        gamma: int = 1,
        groups: int = 1,
        style: Literal["imagenet", "cifar"] = "imagenet",
        in_planes: int = 64,
        normalization_layer: nn.Module = nn.BatchNorm2d,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.alpha = alpha
        self.gamma = gamma
        self.groups = groups
        self.num_estimators = num_estimators

        self.in_planes = in_planes
        block_planes = in_planes

        if style == "imagenet":
            self.conv1 = PackedConv2d(
                self.in_channels,
                block_planes,
                kernel_size=7,
                stride=2,
                padding=3,
                alpha=alpha,
                num_estimators=num_estimators,
                gamma=1,  # No groups for the first layer
                groups=groups,
                bias=conv_bias,
                first=True,
            )
        elif style == "cifar":
            self.conv1 = PackedConv2d(
                self.in_channels,
                block_planes,
                kernel_size=3,
                stride=1,
                padding=1,
                alpha=alpha,
                num_estimators=num_estimators,
                gamma=1,  # No groups for the first layer
                groups=groups,
                bias=conv_bias,
                first=True,
            )
        else:
            raise ValueError(f"Unknown style. Got {style}.")

        self.bn1 = normalization_layer(block_planes * alpha)

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
            alpha=alpha,
            num_estimators=num_estimators,
            conv_bias=conv_bias,
            dropout_rate=dropout_rate,
            gamma=gamma,
            groups=groups,
            normalization_layer=normalization_layer,
        )
        self.layer2 = self._make_layer(
            block,
            block_planes * 2,
            num_blocks[1],
            stride=2,
            alpha=alpha,
            num_estimators=num_estimators,
            conv_bias=conv_bias,
            dropout_rate=dropout_rate,
            gamma=gamma,
            groups=groups,
            normalization_layer=normalization_layer,
        )
        self.layer3 = self._make_layer(
            block,
            block_planes * 4,
            num_blocks[2],
            stride=2,
            alpha=alpha,
            num_estimators=num_estimators,
            conv_bias=conv_bias,
            dropout_rate=dropout_rate,
            gamma=gamma,
            groups=groups,
            normalization_layer=normalization_layer,
        )
        if len(num_blocks) == 4:
            self.layer4 = self._make_layer(
                block,
                block_planes * 8,
                num_blocks[3],
                stride=2,
                alpha=alpha,
                num_estimators=num_estimators,
                conv_bias=conv_bias,
                dropout_rate=dropout_rate,
                gamma=gamma,
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

        self.linear = PackedLinear(
            block_planes * linear_multiplier * block.expansion,
            num_classes,
            alpha=alpha,
            num_estimators=num_estimators,
            last=True,
        )

    def _make_layer(
        self,
        block: type[_BasicBlock | _Bottleneck],
        planes: int,
        num_blocks: int,
        stride: int,
        alpha: float,
        num_estimators: int,
        conv_bias: bool,
        dropout_rate: float,
        gamma: int,
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
                    alpha=alpha,
                    num_estimators=num_estimators,
                    conv_bias=conv_bias,
                    dropout_rate=dropout_rate,
                    gamma=gamma,
                    groups=groups,
                    normalization_layer=normalization_layer,
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
        out = self.dropout(self.flatten(out))
        return self.linear(out)

    def check_config(self, config: dict[str, Any]) -> bool:
        """Check if the pretrained configuration matches the current model."""
        return (
            (config["alpha"] == self.alpha)
            * (config["gamma"] == self.gamma)
            * (config["groups"] == self.groups)
            * (config["num_estimators"] == self.num_estimators)
        )


def packed_resnet18(
    in_channels: int,
    num_classes: int,
    num_estimators: int,
    alpha: int,
    gamma: int,
    conv_bias: bool = True,
    groups: int = 1,
    dropout_rate: float = 0,
    style: Literal["imagenet", "cifar"] = "imagenet",
    normalization_layer: nn.Module = nn.BatchNorm2d,
    pretrained: bool = False,
) -> _PackedResNet:
    """Packed-Ensembles of ResNet-18.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of classes to predict.
        conv_bias (bool): Whether to use bias in convolutions. Defaults to
            ``True``.
        dropout_rate (float): Dropout rate. Defaults to 0.
        num_estimators (int): Number of estimators in the ensemble.
        alpha (int): Expansion factor affecting the width of the estimators.
        gamma (int): Number of groups within each estimator.
        groups (int): Number of groups within each estimator.
        style (bool, optional): Whether to use the ImageNet
            structure. Defaults to ``True``.
        normalization_layer (nn.Module, optional): Normalization layer.
        pretrained (bool, optional): Whether to load pretrained weights.
            Defaults to ``False``.

    Returns:
        _PackedResNet: A Packed-Ensembles ResNet-18.
    """
    net = _PackedResNet(
        block=_BasicBlock,
        num_blocks=[2, 2, 2, 2],
        in_channels=in_channels,
        num_estimators=num_estimators,
        alpha=alpha,
        gamma=gamma,
        conv_bias=conv_bias,
        dropout_rate=dropout_rate,
        groups=groups,
        num_classes=num_classes,
        style=style,
        in_planes=64,
        normalization_layer=normalization_layer,
    )
    if pretrained:  # coverage: ignore
        weights = weight_ids[str(num_classes)]["18"]
        if weights is None:
            raise ValueError("No pretrained weights for this configuration")
        state_dict, config = load_hf(weights)
        if not net.check_config(config):
            raise ValueError(
                "Pretrained weights do not match current configuration."
            )
        net.load_state_dict(state_dict)
    return net


def packed_resnet20(
    in_channels: int,
    num_classes: int,
    num_estimators: int,
    alpha: int,
    gamma: int,
    groups: int = 1,
    conv_bias: bool = True,
    dropout_rate: float = 0,
    style: Literal["imagenet", "cifar"] = "imagenet",
    normalization_layer: nn.Module = nn.BatchNorm2d,
    pretrained: bool = False,
) -> _PackedResNet:
    """Packed-Ensembles of ResNet-20.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of classes to predict.
        num_estimators (int): Number of estimators in the ensemble.
        alpha (int): Expansion factor affecting the width of the estimators.
        gamma (int): Number of groups within each estimator.
        groups (int): Number of groups within each estimator. Defaults to 1.
        conv_bias (bool): Whether to use bias in convolutions. Defaults to
            ``True``.
        dropout_rate (float): Dropout rate. Defaults to 0.
        style (bool, optional): Whether to use the ImageNet
            structure. Defaults to ``True``.
        normalization_layer (nn.Module, optional): Normalization layer.
        pretrained (bool, optional): Whether to load pretrained weights.
            Defaults to ``False``.

    Returns:
        _PackedResNet: A Packed-Ensembles ResNet-20.
    """
    net = _PackedResNet(
        block=_BasicBlock,
        num_blocks=[3, 3, 3],
        in_channels=in_channels,
        num_estimators=num_estimators,
        alpha=alpha,
        gamma=gamma,
        conv_bias=conv_bias,
        dropout_rate=dropout_rate,
        groups=groups,
        num_classes=num_classes,
        style=style,
        in_planes=16,
        normalization_layer=normalization_layer,
    )
    if pretrained:  # coverage: ignore
        weights = weight_ids[str(num_classes)]["18"]
        if weights is None:
            raise ValueError("No pretrained weights for this configuration")
        state_dict, config = load_hf(weights)
        if not net.check_config(config):
            raise ValueError(
                "Pretrained weights do not match current configuration."
            )
        net.load_state_dict(state_dict)
    return net


def packed_resnet34(
    in_channels: int,
    num_classes: int,
    num_estimators: int,
    alpha: int,
    gamma: int,
    groups: int = 1,
    conv_bias: bool = True,
    dropout_rate: float = 0,
    style: Literal["imagenet", "cifar"] = "imagenet",
    normalization_layer: nn.Module = nn.BatchNorm2d,
    pretrained: bool = False,
) -> _PackedResNet:
    """Packed-Ensembles of ResNet-34.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of classes to predict.
        num_estimators (int): Number of estimators in the ensemble.
        alpha (int): Expansion factor affecting the width of the estimators.
        gamma (int): Number of groups within each estimator.
        groups (int): Number of groups within each estimator. Defaults to 1.
        conv_bias (bool): Whether to use bias in convolutions. Defaults to
            ``True``.
        dropout_rate (float): Dropout rate. Defaults to 0.
        style (bool, optional): Whether to use the ImageNet
            structure. Defaults to ``True``.
        normalization_layer (nn.Module, optional): Normalization layer.
        pretrained (bool, optional): Whether to load pretrained weights.
            Defaults to ``False``.

    Returns:
        _PackedResNet: A Packed-Ensembles ResNet-34.
    """
    net = _PackedResNet(
        block=_BasicBlock,
        num_blocks=[3, 4, 6, 3],
        in_channels=in_channels,
        num_estimators=num_estimators,
        alpha=alpha,
        gamma=gamma,
        groups=groups,
        conv_bias=conv_bias,
        dropout_rate=dropout_rate,
        num_classes=num_classes,
        style=style,
        in_planes=64,
        normalization_layer=normalization_layer,
    )
    if pretrained:  # coverage: ignore
        weights = weight_ids[str(num_classes)]["34"]
        if weights is None:
            raise ValueError("No pretrained weights for this configuration")
        state_dict, config = load_hf(weights)
        if not net.check_config(config):
            raise ValueError(
                "Pretrained weights do not match current configuration."
            )
        net.load_state_dict(state_dict)
    return net


def packed_resnet50(
    in_channels: int,
    num_classes: int,
    num_estimators: int,
    alpha: int,
    gamma: int,
    groups: int = 1,
    conv_bias: bool = True,
    dropout_rate: float = 0,
    style: Literal["imagenet", "cifar"] = "imagenet",
    normalization_layer: nn.Module = nn.BatchNorm2d,
    pretrained: bool = False,
) -> _PackedResNet:
    """Packed-Ensembles of ResNet-50.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of classes to predict.
        num_estimators (int): Number of estimators in the ensemble.
        alpha (int): Expansion factor affecting the width of the estimators.
        gamma (int): Number of groups within each estimator.
        groups (int): Number of groups within each estimator. Defaults to 1.
        conv_bias (bool): Whether to use bias in convolutions. Defaults to
            ``True``.
        dropout_rate (float): Dropout rate. Defaults to 0.
        style (bool, optional): Whether to use the ImageNet
            structure. Defaults to ``True``.
        normalization_layer (nn.Module, optional): Normalization layer.
        pretrained (bool, optional): Whether to load pretrained weights.
            Defaults to ``False``.

    Returns:
        _PackedResNet: A Packed-Ensembles ResNet-50.
    """
    net = _PackedResNet(
        block=_Bottleneck,
        num_blocks=[3, 4, 6, 3],
        in_channels=in_channels,
        num_estimators=num_estimators,
        alpha=alpha,
        gamma=gamma,
        groups=groups,
        conv_bias=conv_bias,
        dropout_rate=dropout_rate,
        num_classes=num_classes,
        style=style,
        in_planes=64,
        normalization_layer=normalization_layer,
    )
    if pretrained:  # coverage: ignore
        weights = weight_ids[str(num_classes)]["50"]
        if weights is None:
            raise ValueError("No pretrained weights for this configuration")
        state_dict, config = load_hf(weights)
        if not net.check_config(config):
            raise ValueError(
                "Pretrained weights do not match current configuration."
            )
        net.load_state_dict(state_dict)
    return net


def packed_resnet101(
    in_channels: int,
    num_classes: int,
    num_estimators: int,
    alpha: int,
    gamma: int,
    groups: int = 1,
    conv_bias: bool = True,
    dropout_rate: float = 0,
    style: Literal["imagenet", "cifar"] = "imagenet",
    normalization_layer: nn.Module = nn.BatchNorm2d,
    pretrained: bool = False,
) -> _PackedResNet:
    """Packed-Ensembles of ResNet-101.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of classes to predict.
        num_estimators (int): Number of estimators in the ensemble.
        alpha (int): Expansion factor affecting the width of the estimators.
        gamma (int): Number of groups within each estimator.
        groups (int): Number of groups within each estimator. Defaults to 1.
        conv_bias (bool): Whether to use bias in convolutions. Defaults to
            ``True``.
        dropout_rate (float): Dropout rate. Defaults to 0.
        style (bool, optional): Whether to use the ImageNet
            structure. Defaults to ``True``.
        normalization_layer (nn.Module, optional): Normalization layer.
        pretrained (bool, optional): Whether to load pretrained weights.
            Defaults to ``False``.

    Returns:
        _PackedResNet: A Packed-Ensembles ResNet-101.
    """
    net = _PackedResNet(
        block=_Bottleneck,
        num_blocks=[3, 4, 23, 3],
        in_channels=in_channels,
        num_estimators=num_estimators,
        alpha=alpha,
        gamma=gamma,
        groups=groups,
        conv_bias=conv_bias,
        dropout_rate=dropout_rate,
        num_classes=num_classes,
        style=style,
        in_planes=64,
        normalization_layer=normalization_layer,
    )
    if pretrained:  # coverage: ignore
        weights = weight_ids[str(num_classes)]["101"]
        if weights is None:
            raise ValueError("No pretrained weights for this configuration")
        state_dict, config = load_hf(weights)
        if not net.check_config(config):
            raise ValueError(
                "Pretrained weights do not match current configuration."
            )
        net.load_state_dict(state_dict)
    return net


def packed_resnet152(
    in_channels: int,
    num_classes: int,
    num_estimators: int,
    alpha: int,
    gamma: int,
    groups: int = 1,
    conv_bias: bool = True,
    dropout_rate: float = 0,
    style: Literal["imagenet", "cifar"] = "imagenet",
    normalization_layer: nn.Module = nn.BatchNorm2d,
    pretrained: bool = False,
) -> _PackedResNet:
    """Packed-Ensembles of ResNet-152.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of classes to predict.
        num_estimators (int): Number of estimators in the ensemble.
        alpha (int): Expansion factor affecting the width of the estimators.
        gamma (int): Number of groups within each estimator.
        groups (int): Number of groups within each estimator. Defaults to 1.
        conv_bias (bool): Whether to use bias in convolutions. Defaults to
            ``True``.
        dropout_rate (float): Dropout rate. Defaults to 0.
        style (bool, optional): Whether to use the ImageNet
            structure. Defaults to ``True``.
        normalization_layer (nn.Module, optional): Normalization layer.
        pretrained (bool, optional): Whether to load pretrained weights.
            Defaults to ``False``.

    Returns:
        _PackedResNet: A Packed-Ensembles ResNet-152.
    """
    net = _PackedResNet(
        block=_Bottleneck,
        num_blocks=[3, 8, 36, 3],
        in_channels=in_channels,
        num_estimators=num_estimators,
        alpha=alpha,
        gamma=gamma,
        groups=groups,
        conv_bias=conv_bias,
        dropout_rate=dropout_rate,
        num_classes=num_classes,
        style=style,
        in_planes=64,
        normalization_layer=normalization_layer,
    )
    if pretrained:  # coverage: ignore
        weights = weight_ids[str(num_classes)]["152"]
        if weights is None:
            raise ValueError("No pretrained weights for this configuration")
        state_dict, config = load_hf(weights)
        if not net.check_config(config):
            raise ValueError(
                "Pretrained weights do not match current configuration."
            )
        net.load_state_dict(state_dict)
    return net
