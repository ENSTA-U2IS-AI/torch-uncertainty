import logging
import math
from functools import partial

import torch
import torch.nn.functional as F
from timm.layers import DropPath, to_2tuple, trunc_normal_
from torch import Tensor, nn


class DWConv(nn.Module):
    def __init__(self, dim: int = 768) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, inputs: Tensor, h: int, w: int) -> Tensor:
        b, _, c = inputs.shape
        inputs = self.dwconv(inputs.transpose(1, 2).view(b, c, h, w))
        return inputs.flatten(2).transpose(1, 2)


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout_rate)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            nn.init.constant_(m.bias, 0)

    def forward(self, x, h, w):
        x = self.fc1(x)
        x = self.dwconv(x, h, w)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        sr_ratio: int = 1,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor, h: int, w: int):
        b, n, c = x.shape
        q = (
            self.q(x)
            .reshape(b, n, self.num_heads, c // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(b, c, h, w)
            x_ = self.sr(x_).reshape(b, c, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = (
                self.kv(x_)
                .reshape(b, -1, 2, self.num_heads, c // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        else:
            kv = (
                self.kv(x)
                .reshape(b, -1, 2, self.num_heads, c // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        return self.proj_drop(x)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: None | float = None,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        sr_ratio: int = 1,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=dropout,
            sr_ratio=sr_ratio,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better
        #   than dropout here
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            dropout_rate=dropout,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            nn.init.constant_(m.bias, 0)

    def forward(self, x, h, w):
        x = x + self.drop_path(self.attn(self.norm1(x), h, w))
        return x + self.drop_path(self.mlp(self.norm2(x), h, w))


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(
        self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.h, self.w = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        self.num_patches = self.h * self.w
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.proj(x)
        _, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, h, w


class MixVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int,
        in_channels: int,
        num_classes: int,
        embed_dims: list[int],
        num_heads: list[int],
        mlp_ratios: list[int],
        qkv_bias: bool,
        qk_scale: float | None,
        drop_rate: float,
        attn_drop_rate: float,
        drop_path_rate: float,
        norm_layer,
        depths,
        sr_ratios: list[int],
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(
            img_size=img_size,
            patch_size=7,
            stride=4,
            in_chans=in_channels,
            embed_dim=embed_dims[0],
        )
        self.patch_embed2 = OverlapPatchEmbed(
            img_size=img_size // 4,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1],
        )
        self.patch_embed3 = OverlapPatchEmbed(
            img_size=img_size // 8,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2],
        )
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 16,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[2],
            embed_dim=embed_dims[3],
        )

        # transformer encoder
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[0],
                    num_heads=num_heads[0],
                    mlp_ratio=mlp_ratios[0],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    dropout=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[0],
                )
                for i in range(depths[0])
            ]
        )
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[1],
                    num_heads=num_heads[1],
                    mlp_ratio=mlp_ratios[1],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    dropout=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[1],
                )
                for i in range(depths[1])
            ]
        )
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[2],
                    num_heads=num_heads[2],
                    mlp_ratio=mlp_ratios[2],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    dropout=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[2],
                )
                for i in range(depths[2])
            ]
        )
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[3],
                    num_heads=num_heads[3],
                    mlp_ratio=mlp_ratios[3],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    dropout=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[3],
                )
                for i in range(depths[3])
            ]
        )
        self.norm4 = norm_layer(embed_dims[3])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        b = x.shape[0]
        outs = []

        # stage 1
        x, h, w = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, h, w)
        x = self.norm1(x)
        x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, h, w = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, h, w)
        x = self.norm2(x)
        x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, h, w = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, h, w)
        x = self.norm3(x)
        x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, h, w = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, h, w)
        x = self.norm4(x)
        x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        return self.forward_features(x)


def _get_embed_dims(arch: int) -> list[int]:
    if arch == 0:
        return [32, 64, 160, 256]
    return [64, 128, 320, 512]


def _get_depths(arch: int) -> list[int]:
    if arch == 0 or arch == 1:
        return [2, 2, 2, 2]
    if arch == 2:
        return [3, 4, 6, 3]
    if arch == 3:
        return [3, 4, 18, 3]
    if arch == 4:
        return [3, 8, 27, 3]
    # arch == 5:
    return [3, 6, 40, 3]


class Mit(MixVisionTransformer):
    def __init__(self, arch: int):
        embed_dims = _get_embed_dims(arch)
        depths = _get_depths(arch)
        super().__init__(
            img_size=224,
            in_channels=3,
            num_classes=1000,
            qk_scale=None,
            embed_dims=embed_dims,
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=depths,
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.1,
            attn_drop_rate=0.0,
        )


class MLPHead(nn.Module):
    """Linear Embedding with transposition."""

    def __init__(self, input_dim: int = 2048, embed_dim: int = 768) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.proj(inputs.flatten(2).transpose(1, 2))


def resize(
    inputs: Tensor,
    size: torch.Size | None = None,
    scale_factor=None,
    mode: str = "nearest",
    align_corners: bool | None = None,
    warning: bool = True,
) -> Tensor:
    if warning and size is not None and align_corners:  # coverage: ignore
        input_h, input_w = tuple(int(x) for x in inputs.shape[2:])
        output_h, output_w = tuple(int(x) for x in size)
        if (output_h > input_h or output_w > output_h) and (
            (output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1)
            and (output_h - 1) % (input_h - 1)
            and (output_w - 1) % (input_w - 1)
        ):
            logging.info(
                "When align_corners=%s, "
                "the output would more aligned if "
                "input size %s is `x+1` and "
                "out size %s is `nx+1`",
                align_corners,
                (input_h, input_w),
                (output_h, output_w),
            )
    return F.interpolate(inputs, size, scale_factor, mode, align_corners)


class SegFormerHead(nn.Module):
    """Head for SegFormer.

    Reference:
        SegFormer: Simple and Efficient Design for Semantic Segmentation with
        Transformers.
    """

    def __init__(
        self,
        in_channels: list[int],
        feature_strides: list[int],
        embed_dim: int,
        num_classes: int,
        dropout_ratio: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]

        self.linear_c4 = MLPHead(input_dim=in_channels[3], embed_dim=embed_dim)
        self.linear_c3 = MLPHead(input_dim=in_channels[2], embed_dim=embed_dim)
        self.linear_c2 = MLPHead(input_dim=in_channels[1], embed_dim=embed_dim)
        self.linear_c1 = MLPHead(input_dim=in_channels[0], embed_dim=embed_dim)

        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(embed_dim),
        )
        self.classifier = nn.Conv2d(embed_dim, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio)

    def forward(self, inputs: Tensor) -> Tensor:
        # x [inputs[i] for i in self.in_index] # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = inputs[0], inputs[1], inputs[2], inputs[3]

        n, _, _, _ = c4.shape

        _c4 = (
            self.linear_c4(c4)
            .permute(0, 2, 1)
            .reshape(n, -1, c4.shape[2], c4.shape[3])
        )
        _c4 = resize(
            _c4, size=c1.size()[2:], mode="bilinear", align_corners=False
        )

        _c3 = (
            self.linear_c3(c3)
            .permute(0, 2, 1)
            .reshape(n, -1, c3.shape[2], c3.shape[3])
        )
        _c3 = resize(
            _c3, size=c1.size()[2:], mode="bilinear", align_corners=False
        )

        _c2 = (
            self.linear_c2(c2)
            .permute(0, 2, 1)
            .reshape(n, -1, c2.shape[2], c2.shape[3])
        )
        _c2 = resize(
            _c2, size=c1.size()[2:], mode="bilinear", align_corners=False
        )

        _c1 = (
            self.linear_c1(c1)
            .permute(0, 2, 1)
            .reshape(n, -1, c1.shape[2], c1.shape[3])
        )

        _c = self.fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        return self.classifier(x)


class _SegFormer(nn.Module):
    def __init__(
        self,
        in_channels: list[int],
        feature_strides: list[int],
        embed_dim: int,
        num_classes: int,
        dropout_ratio: float,
        mit: nn.Module,
    ):
        super().__init__()

        self.encoder = mit
        self.head = SegFormerHead(
            in_channels,
            feature_strides,
            embed_dim,
            num_classes,
            dropout_ratio,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.encoder(inputs)
        return self.head(features)


def seg_former(num_classes: int, arch: int) -> _SegFormer:
    in_channels = _get_embed_dims(arch)
    return _SegFormer(
        in_channels=in_channels,
        feature_strides=[4, 8, 16, 32],
        embed_dim=256 if arch == 0 else 512,
        num_classes=num_classes,
        dropout_ratio=0.1,
        mit=Mit(arch),
    )
