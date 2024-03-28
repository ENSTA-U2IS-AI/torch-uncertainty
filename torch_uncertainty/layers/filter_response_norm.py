import torch
from torch import Tensor, nn


class _FilterResponseNormNd(nn.Module):
    def __init__(
        self,
        dimension: int,
        num_channels: int,
        eps: float = 1e-6,
        device=None,
        dtype=None,
    ) -> None:
        """N-dimensional Filter Response Normalization layer.

        Args:
            dimension (int): Dimension of the input tensor.
            num_channels (int): Number of channels.
            eps (float, optional): Epsilon. Defaults to 1e-6.
            device (optional): Device. Defaults to None.
            dtype (optional): Data type. Defaults to None.
        """
        super().__init__()
        if dimension < 1 or not isinstance(dimension, int):
            raise ValueError(
                "dimension should be an integer greater or equal than 1. "
                f"got {dimension}."
            )
        self.dimension = dimension

        if num_channels < 1 or not isinstance(num_channels, int):
            raise ValueError(
                "num_channels should be an integer greater or equal than 1. "
                f"got {num_channels}."
            )
        shape = (1, num_channels) + (1,) * dimension
        self.eps = eps

        self.tau = nn.Parameter(torch.zeros(shape, device=device, dtype=dtype))
        self.beta = nn.Parameter(torch.zeros(shape, device=device, dtype=dtype))
        self.gamma = nn.Parameter(torch.ones(shape, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        nu2 = torch.mean(
            x**2, dim=list(range(-self.dimension, 0)), keepdim=True
        )
        x = x * torch.rsqrt(nu2 + self.eps)
        y = self.gamma * x + self.beta
        return torch.max(y, self.tau)


class FilterResponseNorm1d(_FilterResponseNormNd):
    def __init__(
        self, num_channels: int, eps: float = 1e-6, device=None, dtype=None
    ) -> None:
        """1-dimensional Filter Response Normalization layer.

        Args:
            num_channels (int): Number of channels.
            eps (float, optional): Epsilon. Defaults to 1e-6.
            device (optional): Device. Defaults to None.
            dtype (optional): Data type. Defaults to None.
        """
        super().__init__(
            dimension=1,
            num_channels=num_channels,
            eps=eps,
            device=device,
            dtype=dtype,
        )


class FilterResponseNorm2d(_FilterResponseNormNd):
    def __init__(
        self, num_channels: int, eps: float = 1e-6, device=None, dtype=None
    ) -> None:
        """2-dimensional Filter Response Normalization layer.

        Args:
            num_channels (int): Number of channels.
            eps (float, optional): Epsilon. Defaults to 1e-6.
            device (optional): Device. Defaults to None.
            dtype (optional): Data type. Defaults to None.
        """
        super().__init__(
            dimension=2,
            num_channels=num_channels,
            eps=eps,
            device=device,
            dtype=dtype,
        )


class FilterResponseNorm3d(_FilterResponseNormNd):
    def __init__(
        self, num_channels: int, eps: float = 1e-6, device=None, dtype=None
    ) -> None:
        """3-dimensional Filter Response Normalization layer.

        Args:
            num_channels (int): Number of channels.
            eps (float, optional): Epsilon. Defaults to 1e-6.
            device (optional): Device. Defaults to None.
            dtype (optional): Data type. Defaults to None.
        """
        super().__init__(
            dimension=3,
            num_channels=num_channels,
            eps=eps,
            device=device,
            dtype=dtype,
        )
