import torch
from torch import Tensor, nn


class FilterResponseNorm2d(nn.Module):
    def __init__(
        self, num_channels: int, eps: float = 1e-6, device=None, dtype=None
    ) -> None:
        """Filter Response Normalization layer.

        Args:
            num_channels (int): Number of channels.
            eps (float, optional): Epsilon. Defaults to 1e-6.
            device (optional): Device. Defaults to None.
            dtype (optional): Data type. Defaults to None.
        """
        super().__init__()
        self.eps = eps
        self.tau = nn.Parameter(
            torch.zeros((1, num_channels, 1, 1), device=device, dtype=dtype)
        )
        self.beta = nn.Parameter(
            torch.zeros((1, num_channels, 1, 1), device=device, dtype=dtype)
        )
        self.gamma = nn.Parameter(
            torch.ones((1, num_channels, 1, 1), device=device, dtype=dtype)
        )

    def forward(self, x: Tensor) -> Tensor:
        nu2 = torch.mean(x**2, dim=[-2, -1], keepdim=True)
        x = x * torch.rsqrt(nu2 + self.eps)
        y = self.gamma * x + self.beta
        return torch.max(y, self.tau)


class MCBatchNorm2d(nn.BatchNorm2d):
    counter: int
    mean: list
    var: list

    def __init__(
        self,
        num_features: int,
        num_estimators: int,
        eps: float = 0.00001,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            num_features,
            eps,
            momentum,
            affine,
            track_running_stats,
            device,
            dtype,
        )

        if num_estimators < 1 or not isinstance(num_estimators, int):
            raise ValueError(
                "num_estimators should be an integer greater or equal than 1. "
                f"got {num_estimators}."
            )
        self.accumulate = True
        self.num_estimators = num_estimators
        self.reset_mc_statistics()

    def forward(self, x: Tensor) -> Tensor:
        if self.accumulate:
            self.mean.append(x.mean((0, -2, -1)))
            self.var.append(x.var((0, -2, -1)))
            self.accumulate = self.counter < self.num_estimators
        if not self.training:
            if len(self.mean) != self.num_estimators:
                raise ValueError("The statistics are not yet filled.")
            self.running_mean = self.mean[self.counter]
            self.running_var = self.var[self.counter]
            self.counter += 1 if self.counter < self.num_estimators else 0
        return super().forward(x)

    def reset_mc_statistics(self):
        self.counter = 0
        self.mean = []
        self.std = []
