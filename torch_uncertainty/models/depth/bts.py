import math
from typing import Literal

import torch
import torchvision.models as tv_models
from torch import Tensor, nn
from torch.distributions import Distribution
from torch.nn import functional as F
from torchvision.models.densenet import DenseNet121_Weights, DenseNet161_Weights
from torchvision.models.resnet import (
    ResNet50_Weights,
    ResNet101_Weights,
    ResNeXt50_32X4D_Weights,
    ResNeXt101_32X8D_Weights,
)

from torch_uncertainty.layers.distributions import LaplaceLayer, NormalLayer
from torch_uncertainty.models.utils import Backbone

resnet_feat_out_channels = [64, 256, 512, 1024, 2048]
resnet_feat_names = ["relu", "layer1", "layer2", "layer3", "layer4"]
densenet_feat_names = [
    "relu0",
    "pool0",
    "transition1",
    "transition2",
    "norm5",
]

bts_backbones = [
    "densenet121",
    "densenet161",
    "resnet50",
    "resnet101",
    "resnext50",
    "resnext101",
]


class AtrousBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int,
        norm_first: bool = True,
        norm_momentum: float = 0.1,
        **factory_kwargs,
    ):
        """Atrous block with 1x1 and 3x3 convolutions.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            dilation (int): Dilation rate for the 3x3 convolution.
            norm_first (bool): Whether to apply normalization before the 1x1 convolution.
                Defaults to True.
            norm_momentum (float): Momentum for the normalization layer. Defaults to 0.1.
            factory_kwargs: Additional arguments for the PyTorch layers.
        """
        super().__init__()

        self.norm_first = norm_first
        if norm_first:
            self.first_norm = nn.BatchNorm2d(
                in_channels, momentum=norm_momentum, **factory_kwargs
            )

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels * 2,
            bias=False,
            kernel_size=1,
            stride=1,
            padding=0,
            **factory_kwargs,
        )
        self.norm = nn.BatchNorm2d(
            out_channels * 2, momentum=norm_momentum, **factory_kwargs
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels * 2,
            out_channels=out_channels,
            bias=False,
            kernel_size=3,
            stride=1,
            padding=(dilation, dilation),
            dilation=dilation,
            **factory_kwargs,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.norm_first:
            x = self.first_norm(x)
        out = F.relu(self.conv1(x))
        out = F.relu(self.norm(out))
        return self.conv2(out)


class UpConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ratio: int = 2,
        **factory_kwargs,
    ):
        """Upsampling convolution.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            ratio (int): Upsampling ratio.
            factory_kwargs: Additional arguments for the convolution layer.
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=False,
            kernel_size=3,
            stride=1,
            padding=1,
            **factory_kwargs,
        )
        self.ratio = ratio

    def forward(self, x: Tensor) -> Tensor:
        out = F.interpolate(x, scale_factor=self.ratio, mode="nearest")
        return F.elu(self.conv(out))


class Reduction1x1(nn.Module):
    def __init__(
        self,
        num_in_filters: int,
        num_out_filters: int,
        max_depth: float,
        is_final: bool = False,
        **factory_kwargs,
    ):
        super().__init__()
        self.max_depth = max_depth
        self.is_final = is_final
        self.reduc = torch.nn.Sequential()

        while num_out_filters >= 4:
            if num_out_filters < 8:
                if self.is_final:
                    self.reduc.add_module(
                        "final",
                        torch.nn.Sequential(
                            nn.Conv2d(
                                num_in_filters,
                                out_channels=1,
                                bias=False,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                **factory_kwargs,
                            ),
                            nn.Sigmoid(),
                        ),
                    )
                else:
                    self.reduc.add_module(
                        "plane_params",
                        torch.nn.Conv2d(
                            num_in_filters,
                            out_channels=3,
                            bias=False,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            **factory_kwargs,
                        ),
                    )
                break

            self.reduc.add_module(
                f"inter_{num_in_filters}_{num_out_filters}",
                torch.nn.Sequential(
                    nn.Conv2d(
                        in_channels=num_in_filters,
                        out_channels=num_out_filters,
                        bias=False,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        **factory_kwargs,
                    ),
                    nn.ELU(),
                ),
            )

            num_in_filters = num_out_filters
            num_out_filters = num_out_filters // 2

    def forward(self, x: Tensor) -> Tensor:
        x = self.reduc.forward(x)
        if not self.is_final:
            theta = F.sigmoid(x[:, 0, :, :]) * math.pi / 3
            phi = F.sigmoid(x[:, 1, :, :]) * math.pi * 2
            dist = F.sigmoid(x[:, 2, :, :]) * self.max_depth
            x = torch.cat(
                [
                    torch.mul(torch.sin(theta), torch.cos(phi)).unsqueeze(1),
                    torch.mul(torch.sin(theta), torch.sin(phi)).unsqueeze(1),
                    torch.cos(theta).unsqueeze(1),
                    dist.unsqueeze(1),
                ],
                dim=1,
            )
        return x


class LocalPlanarGuidance(nn.Module):
    def __init__(self, up_ratio: int) -> None:
        super().__init__()
        self.register_buffer(
            "u", torch.arange(up_ratio).reshape([1, 1, up_ratio])
        )
        self.register_buffer(
            "v", torch.arange(up_ratio).reshape([1, up_ratio, 1])
        )
        self.up_ratio = up_ratio

    def forward(self, x: Tensor) -> Tensor:
        x_expanded = torch.repeat_interleave(
            torch.repeat_interleave(x, self.up_ratio, 2), self.up_ratio, 3
        )

        u = self.u.repeat(
            x.size(0),
            x.size(2) * self.up_ratio,
            x.size(3),
        )
        u = (u - (self.up_ratio - 1) * 0.5) / self.up_ratio

        v = self.v.repeat(
            x.size(0),
            x.size(2),
            x.size(3) * self.up_ratio,
        )
        v = (v - (self.up_ratio - 1) * 0.5) / self.up_ratio

        return x_expanded[:, 3, :, :] / (
            x_expanded[:, 0, :, :] * u
            + x_expanded[:, 1, :, :] * v
            + x_expanded[:, 2, :, :]
        )


class BTSBackbone(Backbone):
    def __init__(self, backbone_name: str, pretrained: bool) -> None:
        """BTS backbone.

        Args:
            backbone_name (str): Name of the backbone.
            pretrained (bool): Use a pretrained backbone.
        """
        if backbone_name == "densenet121":
            model = tv_models.densenet121(
                weights=DenseNet121_Weights.DEFAULT if pretrained else None
            ).features
            feat_names = densenet_feat_names
            self.feat_out_channels = [64, 64, 128, 256, 1024]
        elif backbone_name == "densenet161":
            model = tv_models.densenet161(
                weights=DenseNet161_Weights.DEFAULT if pretrained else None
            ).features
            feat_names = densenet_feat_names
            self.feat_out_channels = [96, 96, 192, 384, 2208]
        elif backbone_name == "resnet50":
            model = tv_models.resnet50(
                weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            )
        elif backbone_name == "resnet101":
            model = tv_models.resnet101(
                weights=ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
            )
        elif backbone_name == "resnext50":
            model = tv_models.resnext50_32x4d(
                weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V2
                if pretrained
                else None
            )
        else:  # backbone_name == "resnext101":
            model = tv_models.resnext101_32x8d(
                weights=ResNeXt101_32X8D_Weights.IMAGENET1K_V2
                if pretrained
                else None
            )
        if "res" in backbone_name:  # remove classification heads from ResNets
            feat_names = resnet_feat_names
            self.feat_out_channels = resnet_feat_out_channels
            model.avgpool = nn.Identity()
            model.fc = nn.Identity()
        super().__init__(model=model, feat_names=feat_names)


class BTSDecoder(nn.Module):
    def __init__(
        self,
        max_depth: float,
        feat_out_channels: list[int],
        num_features: int,
        dist_layer: type[nn.Module],
    ):
        super().__init__()
        self.max_depth = max_depth

        self.upconv5 = UpConv2d(
            in_channels=feat_out_channels[4], out_channels=num_features
        )
        self.bn5 = nn.BatchNorm2d(num_features, momentum=0.01, affine=True)

        self.conv5 = nn.Conv2d(
            num_features + feat_out_channels[3],
            num_features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.upconv4 = UpConv2d(
            in_channels=num_features, out_channels=num_features // 2
        )
        self.bn4 = nn.BatchNorm2d(num_features // 2, momentum=0.01, affine=True)
        self.conv4 = nn.Conv2d(
            num_features // 2 + feat_out_channels[2],
            num_features // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn4_2 = nn.BatchNorm2d(
            num_features // 2, momentum=0.01, affine=True
        )

        self.daspp_3 = AtrousBlock2d(
            num_features // 2,
            num_features // 4,
            3,
            norm_first=False,
            norm_momentum=0.01,
        )
        self.daspp_6 = AtrousBlock2d(
            num_features // 2 + num_features // 4 + feat_out_channels[2],
            num_features // 4,
            6,
            norm_momentum=0.01,
        )
        self.daspp_12 = AtrousBlock2d(
            num_features + feat_out_channels[2],
            num_features // 4,
            12,
            norm_momentum=0.01,
        )
        self.daspp_18 = AtrousBlock2d(
            num_features + num_features // 4 + feat_out_channels[2],
            num_features // 4,
            18,
            norm_momentum=0.01,
        )
        self.daspp_24 = AtrousBlock2d(
            num_features + num_features // 2 + feat_out_channels[2],
            num_features // 4,
            24,
            norm_momentum=0.01,
        )
        self.daspp_conv = torch.nn.Sequential(
            nn.Conv2d(
                num_features + num_features // 2 + num_features // 4,
                num_features // 4,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ELU(),
        )
        self.reduc8x8 = Reduction1x1(
            num_features // 4, num_features // 4, self.max_depth
        )
        self.lpg8x8 = LocalPlanarGuidance(8)

        self.upconv3 = UpConv2d(
            in_channels=num_features // 4, out_channels=num_features // 4
        )
        self.bn3 = nn.BatchNorm2d(num_features // 4, momentum=0.01, affine=True)
        self.conv3 = nn.Conv2d(
            num_features // 4 + feat_out_channels[1] + 1,
            num_features // 4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.reduc4x4 = Reduction1x1(
            num_features // 4, num_features // 8, self.max_depth
        )
        self.lpg4x4 = LocalPlanarGuidance(4)

        self.upconv2 = UpConv2d(
            in_channels=num_features // 4, out_channels=num_features // 8
        )
        self.bn2 = nn.BatchNorm2d(num_features // 8, momentum=0.01, affine=True)
        self.conv2 = nn.Conv2d(
            num_features // 8 + feat_out_channels[0] + 1,
            num_features // 8,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.reduc2x2 = Reduction1x1(
            num_features // 8, num_features // 16, self.max_depth
        )
        self.lpg2x2 = LocalPlanarGuidance(2)

        self.upconv1 = UpConv2d(
            in_channels=num_features // 8, out_channels=num_features // 16
        )
        self.reduc1x1 = Reduction1x1(
            num_features // 16,
            num_features // 32,
            self.max_depth,
            is_final=True,
        )
        self.conv1 = nn.Conv2d(
            num_features // 16 + 4, num_features // 16, 3, 1, 1, bias=False
        )
        self.output_channels = 1
        if dist_layer in (NormalLayer, LaplaceLayer):
            self.output_channels = 2
        elif dist_layer != nn.Identity:
            raise ValueError(
                f"Unsupported distribution layer. Got {dist_layer}."
            )
        self.depth = nn.Conv2d(
            num_features // 16, self.output_channels, 3, 1, 1, bias=False
        )
        self.dist_layer = dist_layer(dim=1)

    def feat_forward(self, features: list[Tensor]) -> Tensor:
        dense_features = F.relu(features[4])
        upconv5 = self.bn5(self.upconv5(dense_features))  # H/16
        iconv5 = F.elu(self.conv5(torch.cat([upconv5, features[3]], dim=1)))

        upconv4 = self.bn4(self.upconv4(iconv5))  # H/8
        concat4 = torch.cat([upconv4, features[2]], dim=1)
        iconv4 = self.bn4_2(F.elu(self.conv4(concat4)))

        daspp_3 = self.daspp_3(iconv4)
        concat4_2 = torch.cat([concat4, daspp_3], dim=1)
        daspp_6 = self.daspp_6(concat4_2)
        concat4_3 = torch.cat([concat4_2, daspp_6], dim=1)
        daspp_12 = self.daspp_12(concat4_3)
        concat4_4 = torch.cat([concat4_3, daspp_12], dim=1)
        daspp_18 = self.daspp_18(concat4_4)
        daspp_24 = self.daspp_24(torch.cat([concat4_4, daspp_18], dim=1))
        concat4_daspp = torch.cat(
            [iconv4, daspp_3, daspp_6, daspp_12, daspp_18, daspp_24], dim=1
        )
        daspp_feat = self.daspp_conv(concat4_daspp)

        reduc8x8 = self.reduc8x8(daspp_feat)
        plane_normal_8x8 = reduc8x8[:, :3, :, :]
        plane_normal_8x8 = F.normalize(plane_normal_8x8, p=2, dim=1)
        plane_dist_8x8 = reduc8x8[:, 3, :, :]
        plane_eq_8x8 = torch.cat(
            [plane_normal_8x8, plane_dist_8x8.unsqueeze(1)], 1
        )
        depth_8x8 = self.lpg8x8(plane_eq_8x8)
        depth_8x8_scaled = depth_8x8.unsqueeze(1) / self.max_depth
        depth_8x8_scaled_ds = F.interpolate(
            depth_8x8_scaled, scale_factor=0.25, mode="nearest"
        )

        upconv3 = self.bn3(self.upconv3(daspp_feat))  # H/4
        concat3 = torch.cat([upconv3, features[1], depth_8x8_scaled_ds], dim=1)
        iconv3 = F.elu(self.conv3(concat3))

        reduc4x4 = self.reduc4x4(iconv3)
        plane_normal_4x4 = reduc4x4[:, :3, :, :]
        plane_normal_4x4 = F.normalize(plane_normal_4x4, p=2, dim=1)
        plane_dist_4x4 = reduc4x4[:, 3, :, :]
        plane_eq_4x4 = torch.cat(
            [plane_normal_4x4, plane_dist_4x4.unsqueeze(1)], 1
        )
        depth_4x4 = self.lpg4x4(plane_eq_4x4)
        depth_4x4_scaled = depth_4x4.unsqueeze(1) / self.max_depth
        depth_4x4_scaled_ds = F.interpolate(
            depth_4x4_scaled, scale_factor=0.5, mode="nearest"
        )

        upconv2 = self.bn2(self.upconv2(iconv3))  # H/2
        iconv2 = F.elu(
            self.conv2(
                torch.cat([upconv2, features[0], depth_4x4_scaled_ds], dim=1)
            )
        )

        reduc2x2 = self.reduc2x2(iconv2)
        plane_normal_2x2 = reduc2x2[:, :3, :, :]
        plane_normal_2x2 = F.normalize(plane_normal_2x2, p=2, dim=1)
        plane_dist_2x2 = reduc2x2[:, 3, :, :]
        plane_eq_2x2 = torch.cat(
            [plane_normal_2x2, plane_dist_2x2.unsqueeze(1)], 1
        )
        depth_2x2 = self.lpg2x2(plane_eq_2x2)
        depth_2x2_scaled = depth_2x2.unsqueeze(1) / self.max_depth

        upconv1 = self.upconv1(iconv2)
        reduc1x1 = self.reduc1x1(upconv1)
        concat1 = torch.cat(
            [
                upconv1,
                reduc1x1,
                depth_2x2_scaled,
                depth_4x4_scaled,
                depth_8x8_scaled,
            ],
            dim=1,
        )
        return F.elu(self.conv1(concat1))

    def forward(self, features: list[Tensor]) -> Tensor | Distribution:
        """Forward pass.

        Args:
            features (list[Tensor]): list of the features from the backbone.

        Note:
            Depending of the :attr:`dist_layer` of the backbone, the output can
            be a distribution or a single tensor.
        """
        # TODO: handle focal
        out = self.depth(self.feat_forward(features))
        if self.output_channels != 1:
            loc = self.max_depth * F.sigmoid(out[:, 0, :, :])
            scale = self.max_depth * out[:, 1, :, :]
            out = self.dist_layer(torch.stack([loc, scale], -1))
        else:
            out = self.max_depth * F.sigmoid(out)
        return out


class _BTS(nn.Module):
    def __init__(
        self,
        backbone_name: Literal[
            "densenet121",
            "densenet161",
            "resnet50",
            "resnet101",
            "resnext50",
            "resnext101",
        ],
        max_depth: float,
        bts_size: int = 512,
        dist_layer: type[nn.Module] = nn.Identity,
        pretrained_backbone: bool = True,
    ) -> None:
        """BTS model.

        Args:
            backbone_name (str): Name of the encoding backbone.
            max_depth (float): Maximum predicted depth.
            bts_size (int): BTS feature size. Defaults to 512.
            dist_layer (nn.Module): Distribution layer for probabilistic depth
            estimation. Defaults to nn.Identity.
            pretrained_backbone (bool): Use a pretrained backbone. Defaults to True.

        Reference:
            From Big to Small: Multi-Scale Local Planar Guidance for Monocular Depth Estimation.
            Jin Han Lee, Myung-Kyu Han, Dong Wook Ko, Il Hong Suh. ArXiv.
        """
        super().__init__()
        self.max_depth = max_depth

        self.backbone = BTSBackbone(backbone_name, pretrained_backbone)
        self.decoder = BTSDecoder(
            max_depth, self.backbone.feat_out_channels, bts_size, dist_layer
        )

    # TODO: Handle focal
    def forward(self, x: Tensor, focal: float | None = None) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): Input tensor.
            focal (float): Focal length for API consistency.
        """
        return self.decoder(self.backbone(x))


def _bts(
    backbone_name: str,
    max_depth: float,
    bts_size: int = 512,
    dist_layer: type[nn.Module] = nn.Identity,
    pretrained_backbone: bool = True,
) -> _BTS:
    if backbone_name not in bts_backbones:
        raise ValueError(f"Unsupported backbone. Got {backbone_name}.")
    return _BTS(
        backbone_name, max_depth, bts_size, dist_layer, pretrained_backbone
    )


def bts_resnet50(
    max_depth: float,
    bts_size: int = 512,
    dist_layer: type[nn.Module] = nn.Identity,
    pretrained_backbone: bool = True,
) -> _BTS:
    """BTS model with ResNet-50 backbone.

    Args:
        max_depth (float): Maximum predicted depth.
        bts_size (int): BTS feature size. Defaults to 512.
        dist_layer (nn.Module): Distribution layer for probabilistic depth
            estimation. Defaults to nn.Identity.
        pretrained_backbone (bool): Use a pretrained backbone. Defaults to True.
    """
    return _bts(
        "resnet50",
        max_depth,
        bts_size=bts_size,
        dist_layer=dist_layer,
        pretrained_backbone=pretrained_backbone,
    )


def bts_resnet101(
    max_depth: float,
    bts_size: int = 512,
    dist_layer: type[nn.Module] = nn.Identity,
    pretrained_backbone: bool = True,
) -> _BTS:
    """BTS model with ResNet-101 backbone.

    Args:
        max_depth (float): Maximum predicted depth.
        bts_size (int): BTS feature size. Defaults to 512.
        dist_layer (nn.Module): Distribution layer for probabilistic depth
            estimation. Defaults to nn.Identity.
        pretrained_backbone (bool): Use a pretrained backbone. Defaults to True.
    """
    return _bts(
        "resnet101",
        max_depth,
        bts_size=bts_size,
        dist_layer=dist_layer,
        pretrained_backbone=pretrained_backbone,
    )
