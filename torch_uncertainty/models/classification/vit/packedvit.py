
import torch
import torch.nn as nn

import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional
from torch_uncertainty.layers import PackedLayerNorm,PackedLinear,PackedMultiheadAttention,PackedConv2d
from einops import rearrange


class ConvStemConfig(NamedTuple):
    out_channels: int
    kernel_size: int
    stride: int
    norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d
    activation_layer: Callable[..., nn.Module] = nn.ReLU

class MLPBlock(nn.Module):
    """Transformer MLP block."""
    def __init__(self, in_dim: int, mlp_dim: int, dropout: float,num_estimators=1, alpha=1):
        super().__init__()
        self.layers = nn.Sequential(
            PackedLinear(in_dim, mlp_dim,num_estimators=num_estimators,alpha=alpha,implementation="einsum"),
            nn.GELU(),
            nn.Dropout(dropout),
            PackedLinear(mlp_dim, in_dim,num_estimators=num_estimators,alpha=alpha,implementation="einsum"),
            nn.Dropout(dropout)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x):
        return self.layers(x)

class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        num_estimators=1, 
        alpha=1,
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = PackedLayerNorm(hidden_dim,num_estimators,alpha)
        self.self_attention = PackedMultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout,num_estimators=num_estimators,alpha=alpha,batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = PackedLayerNorm(hidden_dim,num_estimators,alpha)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout,num_estimators, alpha)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        num_estimators=1, 
        alpha=1,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, int(hidden_dim*alpha)).normal_(std=0.02))
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
                num_estimators, 
                alpha,
            )
        self.layers = nn.Sequential(layers)
        self.ln = PackedLayerNorm(hidden_dim,num_estimators,alpha)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))

class PackedVit(nn.Module):

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        conv_stem_configs: Optional[List[ConvStemConfig]] = None,
        num_estimators=1, 
        alpha=1,
    ):
        super().__init__()
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        if conv_stem_configs is not None:
            seq_proj = nn.Sequential()
            prev_channels = 3
            for i, conv_stem_layer_config in enumerate(conv_stem_configs):
                seq_proj.add_module(f"conv_{i}", nn.Conv2d(
                    in_channels=prev_channels,
                    out_channels=conv_stem_layer_config.out_channels,
                    kernel_size=conv_stem_layer_config.kernel_size,
                    stride=conv_stem_layer_config.stride
                ))
                seq_proj.add_module(f"bn_{i}", conv_stem_layer_config.norm_layer(conv_stem_layer_config.out_channels))
                seq_proj.add_module(f"relu_{i}", conv_stem_layer_config.activation_layer())
                prev_channels = conv_stem_layer_config.out_channels
            seq_proj.add_module("conv_last", nn.Conv2d(in_channels=prev_channels, out_channels=hidden_dim, kernel_size=1))
            self.conv_proj = seq_proj
        else:
            self.conv_proj = PackedConv2d(
                in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size,first=True,num_estimators=num_estimators,alpha=alpha
            )

        seq_length = (image_size // patch_size) ** 2
        self.class_token = nn.Parameter(torch.zeros(1, 1, int(hidden_dim*alpha)))
        seq_length += 1

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
            num_estimators,
            alpha
        )

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = PackedLinear(hidden_dim, num_classes,num_estimators=num_estimators,alpha=alpha,implementation="einsum",last=True)
        else:
            heads_layers["pre_logits"] = PackedLinear(hidden_dim, representation_size,num_estimators=num_estimators,alpha=alpha,implementation="einsum")
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = PackedLinear(representation_size, num_classes,num_estimators=num_estimators,alpha=alpha,implementation="einsum",last=True)

        self.heads = nn.Sequential(heads_layers)
        self.alpha = alpha
        self.num_estimators = num_estimators

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p
        x = self.conv_proj(x)
        x = x.reshape(n, int(self.hidden_dim * self.alpha), n_h * n_w)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, x: torch.Tensor):
        x = self._process_input(x)
        n = x.shape[0]
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.encoder(x)
        x = x[:, 0]
        x = self.heads(x)
        out = rearrange(x, 'b (m c) -> (m b) c', m=self.num_estimators)
        return out