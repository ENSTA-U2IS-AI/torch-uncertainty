# Code inspired by https://github.com/timeseriesAI/tsai/blob/main/tsai/models/InceptionTime.py

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from torch_uncertainty.layers import BayesConv1d, BayesLinear
from torch_uncertainty.models.wrappers import StochasticModel


class _BayesianInceptionBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, bottleneck: bool = True
    ):
        super().__init__()
        kernel_sizes = [kernel_size // (2**i) for i in range(3)]
        kernel_sizes = [k if k % 2 != 0 else k - 1 for k in kernel_sizes]  # ensure odd kernel sizes

        bottleneck = bottleneck if in_channels > out_channels else False

        self.bottleneck = (
            BayesConv1d(in_channels, out_channels, 1, padding="same", bias=False)
            if bottleneck
            else None
        )
        self.convs = nn.ModuleList(
            [
                BayesConv1d(
                    out_channels if bottleneck else in_channels,
                    out_channels,
                    k,
                    padding="same",
                    bias=False,
                )
                for k in kernel_sizes
            ]
        )
        self.maxconvpool = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            BayesConv1d(in_channels, out_channels, 1, padding="same", bias=False),
        )
        self.batch_norm = nn.BatchNorm1d(out_channels * 4)

    def forward(self, x: Tensor) -> Tensor:
        out = self.bottleneck(x) if self.bottleneck is not None else x
        out = torch.cat(
            [conv(out) for conv in self.convs] + [self.maxconvpool(x)],
            dim=1,
        )
        return F.relu(self.batch_norm(out))


class _BayesianInceptionTime(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        kernel_size: int = 40,
        embed_dim: int = 32,
        num_blocks: int = 6,
        dropout: float = 0.0,
        residual: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.residual = residual

        self.layers = nn.ModuleList()
        self.shortcut = nn.ModuleList() if residual else None

        for i in range(num_blocks):
            self.layers.append(
                nn.Sequential(
                    _BayesianInceptionBlock(
                        in_channels if i == 0 else embed_dim * 4,
                        embed_dim,
                        kernel_size,
                    ),
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                )
            )
            if self.shortcut is not None and i % 3 == 2:
                n_in = in_channels if i == 2 else embed_dim * 4
                n_out = embed_dim * 4
                self.shortcut.append(
                    nn.BatchNorm1d(n_out)
                    if n_in == n_out
                    else nn.Sequential(
                        BayesConv1d(n_in, n_out, 1, bias=False),
                        nn.BatchNorm1d(n_out),
                    )
                )

        self.adaptive_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.last_layer = BayesLinear(embed_dim * 4, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, in_channels, seq_len).

        Returns:
            Tensor: Output tensor of shape (batch_size, num_classes).
        """
        res = x
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.shortcut is not None and i % 3 == 2:
                shortcut = self.shortcut[i // 3](res)
                x = F.relu(x + shortcut)
                res = x

        x = self.adaptive_avg_pool(x)
        x = x.flatten(1)
        return self.last_layer(x)


def bayesian_inception_time(
    in_channels: int,
    num_classes: int,
    num_samples: int = 16,
    kernel_size: int = 40,
    embed_dim: int = 32,
    num_blocks: int = 6,
    dropout: float = 0.0,
    residual: bool = True,
) -> _BayesianInceptionTime:
    """Create an InceptionTime model.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        num_samples (int): Number of samples for stochastic modeling.
        kernel_size (int): Size of the convolutional kernels.
        embed_dim (int): Dimension of the embedding.
        num_blocks (int): Number of inception blocks.
        dropout (float): Dropout rate.
        residual (bool): Whether to use residual connections.

    Returns:
        _InceptionTime: An instance of the InceptionTime model.
    """
    return StochasticModel(
        _BayesianInceptionTime(
            in_channels, num_classes, kernel_size, embed_dim, num_blocks, dropout, residual
        ),
        num_samples=num_samples,
    )
