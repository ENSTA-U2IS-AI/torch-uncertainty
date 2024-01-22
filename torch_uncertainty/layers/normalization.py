import torch
from torch import Tensor, nn


class FilterResponseNorm(nn.Module):
    def __init__(self, num_features, eps=1e-6, device=None, dtype=None) -> None:
        """Filter Response Normalization layer.

        Args:
            num_features (int): Number of channels.
            eps (float, optional): Epsilon. Defaults to 1e-6.
            device (optional): Device. Defaults to None.
            dtype (optional): Data type. Defaults to None.
        """
        super().__init__()
        self.eps = eps
        self.tau = nn.Parameter(
            torch.zeros((1, num_features, 1, 1), device=device, dtype=dtype)
        )
        self.beta = nn.Parameter(
            torch.zeros((1, num_features, 1, 1), device=device, dtype=dtype)
        )
        self.gamma = nn.Parameter(
            torch.ones((1, num_features, 1, 1), device=device, dtype=dtype)
        )

    def forward(self, x: Tensor) -> Tensor:
        nu2 = torch.mean(x**2, dim=[-2, -1], keepdim=True)
        x = x * torch.rsqrt(nu2 + self.eps)
        y = self.gamma * x + self.beta
        return torch.max(y, self.tau)
