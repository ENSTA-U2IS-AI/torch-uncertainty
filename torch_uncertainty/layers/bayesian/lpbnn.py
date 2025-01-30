"""These layers are still work in progress."""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.common_types import _size_2_t


def check_lpbnn_parameters_consistency(
    hidden_size: int, std_factor: float, num_estimators: int
) -> None:
    if hidden_size < 1:
        raise ValueError(f"hidden_size must be greater than 0. Got {hidden_size}.")
    if std_factor < 0:
        raise ValueError(f"std_factor must be greater than 0. Got {std_factor}.")
    if num_estimators < 1:
        raise ValueError(f"num_estimators must be greater than 0. Got {num_estimators}.")


def _sample(mu: Tensor, logvar: Tensor, std_factor: float) -> Tensor:
    eps = torch.randn_like(mu)
    return eps * std_factor * torch.exp(logvar * 0.5) + mu


class LPBNNLinear(nn.Module):
    __constants__ = [
        "in_features",
        "out_features",
        "num_estimators",
        "hidden_size",
    ]
    in_features: int
    out_features: int
    num_estimators: int
    bias: torch.Tensor | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_estimators: int,
        hidden_size: int = 32,
        std_factor: float = 1e-2,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        """LPBNN-style linear layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            num_estimators (int): Number of models to sample from.
            hidden_size (int): Size of the hidden layer. Defaults to ``32``.
            std_factor (float): Factor to multiply the standard deviation of the latent noise. Defaults to ``1e-2``.
            bias (bool): If ``True``, adds a learnable bias to the output. Defaults to ``True``.
            device (torch.device): Device on which the layer is stored. Defaults to ``None``.
            dtype (torch.dtype): Data type of the layer. Defaults to ``None``.

        References:
            [1] `Encoding the latent posterior of Bayesian Neural Networks for uncertainty quantification
            <https://arxiv.org/abs/2012.02818>`_.
        """
        check_lpbnn_parameters_consistency(hidden_size, std_factor, num_estimators)
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_size = hidden_size
        self.std_factor = std_factor
        self.num_estimators = num_estimators

        # for the KL Loss
        self.lprior = 0

        self.linear = nn.Linear(in_features, out_features, bias=False, **factory_kwargs)
        self.alpha = nn.Parameter(
            torch.empty((num_estimators, in_features), **factory_kwargs),
            requires_grad=False,
        )
        self.gamma = nn.Parameter(torch.empty((num_estimators, out_features), **factory_kwargs))
        self.encoder = nn.Linear(in_features, self.hidden_size, **factory_kwargs)
        self.latent_mean = nn.Linear(self.hidden_size, self.hidden_size, **factory_kwargs)
        self.latent_logvar = nn.Linear(self.hidden_size, self.hidden_size, **factory_kwargs)
        self.decoder = nn.Linear(self.hidden_size, in_features, **factory_kwargs)
        self.latent_loss = torch.zeros(1, **factory_kwargs)
        if bias:
            self.bias = nn.Parameter(torch.empty((num_estimators, out_features), **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.alpha, mean=1.0, std=0.1)
        nn.init.normal_(self.gamma, mean=1.0, std=0.1)
        self.linear.reset_parameters()
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()
        self.latent_mean.reset_parameters()
        self.latent_logvar.reset_parameters()
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.linear.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        # Draw a sample from the dist generated by the noise self.alpha
        latent = F.relu(self.encoder(self.alpha))
        latent_mean, latent_logvar = (
            self.latent_mean(latent),
            self.latent_logvar(latent),
        )
        z_latent = _sample(latent_mean, latent_logvar, self.std_factor)

        # one sample per "model" with as many features as x
        alpha_sample = self.decoder(z_latent)

        # Compute the latent loss
        if self.training:
            mse = F.mse_loss(alpha_sample, self.alpha)
            kld = -0.5 * torch.sum(1 + latent_logvar - latent_mean**2 - torch.exp(latent_logvar))
            # For the KL Loss
            self.lvposterior = mse + kld

        # Compute the output
        num_examples_per_model = int(x.size(0) / self.num_estimators)
        alpha = alpha_sample.repeat((num_examples_per_model, 1))
        gamma = self.gamma.repeat((num_examples_per_model, 1))
        out = self.linear(x * alpha) * gamma

        if self.bias is not None:
            bias = self.bias.repeat((num_examples_per_model, 1))
            out += bias
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"num_estimators={self.num_estimators}, "
            f"hidden_size={self.hidden_size}, bias={self.bias is not None}"
        )


class LPBNNConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_estimators: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: str | _size_2_t = 0,
        groups: int = 1,
        hidden_size: int = 32,
        std_factor: float = 1e-2,
        gamma: bool = True,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ):
        """LPBNN-style 2D convolutional layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            num_estimators (int): Number of models to sample from.
            kernel_size (int or tuple): Size of the convolving kernel.
            stride (int or tuple, optional): Stride of the convolution. Default: ``1``.
            padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: ``0``.
            groups (int, optional): Number of blocked connections from input channels to output channels. Default: ``1``.
            hidden_size (int): Size of the hidden layer. Defaults to ``32``.
            std_factor (float): Factor to multiply the standard deviation of the latent noise. Defaults to ``1e-2``.
            gamma (bool): If ``True``, adds a learnable gamma to the output. Defaults to ``True``.
            bias (bool): If ``True``, adds a learnable bias to the output. Defaults to ``True``.
            padding_mode (str): 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'.
            device (torch.device): Device on which the layer is stored. Defaults to ``None``.
            dtype (torch.dtype): Data type of the layer. Defaults to ``None``.

        References:
            [1] `Encoding the latent posterior of Bayesian Neural Networks for uncertainty quantification
            <https://arxiv.org/abs/2012.02818>`_.
        """
        check_lpbnn_parameters_consistency(hidden_size, std_factor, num_estimators)
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.std_factor = std_factor
        self.num_estimators = num_estimators

        # for the KL Loss
        self.lprior = 0

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
            padding_mode=padding_mode,
        )
        self.alpha = nn.Parameter(
            torch.empty(num_estimators, in_channels, **factory_kwargs),
            requires_grad=False,
        )

        self.encoder = nn.Linear(in_channels, self.hidden_size, **factory_kwargs)
        self.decoder = nn.Linear(self.hidden_size, in_channels, **factory_kwargs)
        self.latent_mean = nn.Linear(self.hidden_size, self.hidden_size, **factory_kwargs)
        self.latent_logvar = nn.Linear(self.hidden_size, self.hidden_size, **factory_kwargs)

        self.latent_loss = torch.zeros(1, **factory_kwargs)
        if gamma:
            self.gamma = nn.Parameter(torch.empty((num_estimators, out_channels), **factory_kwargs))
        else:
            self.register_parameter("gamma", None)

        if bias:
            self.bias = nn.Parameter(torch.empty((num_estimators, out_channels), **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.alpha, mean=1.0, std=0.1)
        if self.gamma is not None:
            nn.init.normal_(self.gamma, mean=1.0, std=0.1)
        self.conv.reset_parameters()
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()
        self.latent_mean.reset_parameters()
        self.latent_logvar.reset_parameters()
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        # Draw a sample from the dist generated by the latent noise self.alpha
        latent = F.relu(self.encoder(self.alpha))
        latent_mean, latent_logvar = (
            self.latent_mean(latent),
            self.latent_logvar(latent),
        )
        z_latent = _sample(latent_mean, latent_logvar, self.std_factor)

        # one sample per "model" with as many features as x
        alpha_sample = self.decoder(z_latent)

        # Compute the latent loss
        if self.training:
            mse = F.mse_loss(alpha_sample, self.alpha)
            kld = -0.5 * torch.sum(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())
            # for the KL Loss
            self.lvposterior = mse + kld

        num_examples_per_model = int(x.size(0) / self.num_estimators)

        # Compute the output
        alpha = alpha_sample.repeat((num_examples_per_model, 1)).unsqueeze(-1).unsqueeze(-1)
        if self.gamma is not None:
            gamma = self.gamma.repeat((num_examples_per_model, 1)).unsqueeze(-1).unsqueeze(-1)
            out = self.conv(x * alpha) * gamma
        else:
            out = self.conv(x * alpha)

        if self.bias is not None:
            bias = self.bias.repeat((num_examples_per_model, 1)).unsqueeze(-1).unsqueeze(-1)
            out += bias

        return out

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"num_estimators={self.num_estimators}, "
            f"hidden_size={self.hidden_size}, "
            f"gamma={self.gamma is not None}, "
            f"bias={self.bias is not None}"
        )
