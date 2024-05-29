from typing import Literal

import torch
import torchvision.models as tv_models
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t
from torchvision.models.resnet import ResNet50_Weights, ResNet101_Weights

from torch_uncertainty.models.utils import Backbone, set_bn_momentum


class SeparableConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        bias=True,
    ) -> None:
        """Separable Convolution with dilation.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (_size_2_t): Kernel size.
            stride (_size_2_t, optional): Stride. Defaults to 1.
            padding (_size_2_t, optional): Padding. Defaults to 0.
            dilation (_size_2_t, optional): Dilation. Defaults to 1.
            bias (bool, optional): Use biases. Defaults to True.
        """
        super().__init__()
        self.separable = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.separable(x))


class InnerConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: _size_2_t,
        separable: bool,
    ) -> None:
        """Inner convolution block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            dilation (_size_2_t): Dilation.
            separable (bool): Use separable convolutions to reduce the number
                of parameters.
        """
        super().__init__()
        if not separable:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            )
        else:
            self.conv = SeparableConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.bn(self.conv(x)))


class InnerPooling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Inner pooling block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        size = x.shape[-2:]
        x = F.relu(self.bn(self.conv(self.pool(x))))
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        atrous_rates: list[int],
        separable: bool,
        dropout_rate: float,
    ) -> None:
        """Atrous Spatial Pyramid Pooling.

        Args:
            in_channels (int): Number of input channels.
            atrous_rates (list[int]): Atrous rates for the ASPP module.
            separable (bool): Use separable convolutions to reduce the number
                of parameters.
            dropout_rate (float): Dropout rate of the ASPP.
        """
        super().__init__()
        out_channels = 256
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )
        modules += [
            InnerConv(in_channels, out_channels, dilation, separable)
            for dilation in atrous_rates
        ]
        modules.append(InnerPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)

        self.projection = nn.Sequential(
            nn.Conv2d(
                5 * out_channels, out_channels, kernel_size=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: Tensor) -> Tensor:
        res = torch.cat([conv(x) for conv in self.convs], dim=1)
        return self.projection(res)


class DeepLabV3Backbone(Backbone):
    def __init__(
        self,
        backbone_name: Literal["resnet50", "resnet101"],
        style: str,
        pretrained: bool,
        norm_momentum: float,
    ) -> None:
        """DeepLab V3(+) backbone.

        Args:
            backbone_name (str): Backbone name.
            style (str): Whether to use a DeepLab V3 or V3+ model.
            pretrained (bool): Use pretrained backbone.
            norm_momentum (float): BatchNorm momentum.
        """
        # TODO: handle dilations
        if backbone_name == "resnet50":
            base_model = tv_models.resnet50(
                weights=ResNet50_Weights.DEFAULT if pretrained else None
            )
        elif backbone_name == "resnet101":
            base_model = tv_models.resnet101(
                weights=ResNet101_Weights.DEFAULT if pretrained else None
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}.")
        base_model.avgpool = nn.Identity()
        base_model.fc = nn.Identity()
        set_bn_momentum(base_model, norm_momentum)

        feat_names = ["layer1", "layer4"] if style == "v3+" else ["layer4"]
        super().__init__(base_model, feat_names)


class DeepLabV3Decoder(nn.Module):
    """Decoder for the DeepLabV3 model.

    Args:
        in_channels (int): Number of channels of the input latent space.
        num_classes (int): Number of classes.
        aspp_dilate (list[int], optional): Atrous rates for the ASPP module.
        separable (bool, optional): Use separable convolutions to reduce the number
            of parameters. Defaults to False.
        dropout_rate (float, optional): Dropout rate of the ASPP. Defaults to 0.1.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        aspp_dilate: list[int],
        separable: bool = False,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.aspp = ASPP(in_channels, aspp_dilate, separable, dropout_rate)
        if not separable:
            self.conv = nn.Conv2d(256, 256, 3, padding=1, bias=False)
        else:
            self.conv = SeparableConv2d(256, 256, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(256)
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, features: list[Tensor]) -> Tensor:
        out = F.relu(self.bn(self.conv(self.aspp(features[0]))))
        return self.classifier(out)


class DeepLabV3PlusDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        low_level_channels: int,
        num_classes: int,
        aspp_dilate: list[int],
        separable: bool,
        dropout_rate: float = 0.1,
    ) -> None:
        """Decoder for the DeepLabV3+ model.

        Args:
            in_channels (int): Number of channels of the input latent space.
            low_level_channels (int): Number of low-level features channels.
            num_classes (int): Number of classes.
            aspp_dilate (list[int]): Atrous rates for the ASPP module.
            separable (bool): Use separable convolutions to reduce the number
                of parameters.
            dropout_rate (float, optional): Dropout rate of the ASPP. Defaults
                to 0.1.
        """
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.atrous_spatial_pyramid_pool = ASPP(
            in_channels, aspp_dilate, separable, dropout_rate
        )
        if separable:
            self.conv = SeparableConv2d(304, 256, 3, padding=1, bias=False)
        else:
            self.conv = nn.Conv2d(304, 256, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(256)
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, features: list[Tensor]) -> Tensor:
        low_level_features = self.project(features[0])
        output_features = self.atrous_spatial_pyramid_pool(features[1])
        output_features = F.interpolate(
            output_features,
            size=low_level_features.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        output_features = torch.cat(
            [low_level_features, output_features], dim=1
        )
        out = F.relu(self.bn(self.conv(output_features)))
        return self.classifier(out)


class _DeepLabV3(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone_name: str,
        style: Literal["v3", "v3+"],
        output_stride: int = 16,
        separable: bool = False,
        pretrained_backbone: bool = True,
        norm_momentum: float = 0.01,
    ) -> None:
        """DeepLab V3(+) model.

        Args:
            num_classes (int): Number of classes.
            backbone_name (str): Backbone name.
            style (Literal["v3", "v3+"]):  Whether to use a DeepLab V3 or
                V3+ model.
            output_stride (int, optional): Output stride. Defaults to 16.
            separable (bool, optional): Use separable convolutions. Defaults
                to False.
            pretrained_backbone (bool, optional): Use pretrained backbone.
                Defaults to True.
            norm_momentum (float, optional): BatchNorm momentum. Defaults to
                0.01.

        References:
            - Rethinking atrous convolution for semantic image segmentation.
            Chen, L. C., Papandreou, G., Schroff, F., & Adam, H. (2018).
            - Encoder-decoder with atrous separable convolution for semantic image segmentation.
            Chen, L. C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. In ECCV 2018.
        """
        super().__init__()
        if output_stride == 16:
            dilations = [6, 12, 18]
        elif output_stride == 8:
            dilations = [12, 24, 36]
        else:
            raise ValueError(
                f"output_stride: {output_stride} is not supported."
            )

        self.backbone = DeepLabV3Backbone(
            backbone_name, style, pretrained_backbone, norm_momentum
        )
        if style == "v3":
            self.decoder = DeepLabV3Decoder(
                in_channels=2048,
                num_classes=num_classes,
                aspp_dilate=dilations,
                separable=separable,
                dropout_rate=0.1,
            )
        elif style == "v3+":
            self.decoder = DeepLabV3PlusDecoder(
                in_channels=2048,
                low_level_channels=256,
                num_classes=num_classes,
                aspp_dilate=dilations,
                separable=separable,
                dropout_rate=0.1,
            )
        else:
            raise ValueError(f"Unknown style: {style}.")

    def forward(self, x: Tensor) -> Tensor:
        input_shape = x.shape[-2:]
        return F.interpolate(
            self.decoder(self.backbone(x)),
            size=input_shape,
            mode="bilinear",
            align_corners=False,
        )


def deep_lab_v3_resnet50(
    num_classes: int,
    style: Literal["v3", "v3+"],
    output_stride: int = 16,
    separable: bool = False,
    pretrained_backbone: bool = True,
) -> _DeepLabV3:
    """DeepLab V3(+) model with ResNet-50 backbone.

    Args:
        num_classes (int): Number of classes.
        style (Literal["v3", "v3+"]): Whether to use a DeepLab V3 or V3+ model.
        output_stride (int, optional): Output stride. Defaults to 16.
        separable (bool, optional): Use separable convolutions. Defaults to
            False.
        pretrained_backbone (bool, optional): Use pretrained backbone. Defaults
            to True.
    """
    return _DeepLabV3(
        num_classes,
        "resnet50",
        style,
        output_stride=output_stride,
        separable=separable,
        pretrained_backbone=pretrained_backbone,
    )


def deep_lab_v3_resnet101(
    num_classes: int,
    style: Literal["v3", "v3+"],
    output_stride: int = 16,
    separable: bool = False,
    pretrained_backbone: bool = True,
) -> _DeepLabV3:
    """DeepLab V3(+) model with ResNet-50 backbone.

    Args:
        num_classes (int): Number of classes.
        style (Literal["v3", "v3+"]): Whether to use a DeepLab V3 or V3+ model.
        output_stride (int, optional): Output stride. Defaults to 16.
        separable (bool, optional): Use separable convolutions. Defaults to False.
        pretrained_backbone (bool, optional): Use pretrained backbone. Defaults to True.
    """
    return _DeepLabV3(
        num_classes,
        "resnet101",
        style,
        output_stride=output_stride,
        separable=separable,
        pretrained_backbone=pretrained_backbone,
    )
