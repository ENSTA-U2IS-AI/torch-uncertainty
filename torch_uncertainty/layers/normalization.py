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
        if num_channels < 1 or not isinstance(num_channels, int):
            raise ValueError(
                "num_channels should be an integer greater or equal than 1. "
                f"got {num_channels}."
            )

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
        """Monte Carlo Stochastic Batch Normalization.

        Args:
            num_features (int): Number of features.
            num_estimators (int): Number of estimators.
            eps (float, optional): Epsilon. Defaults to 0.00001.
            momentum (float, optional): Momentum. Defaults to 0.1.
            affine (bool, optional): Affine. Defaults to True.
            track_running_stats (bool, optional): Track running statistics.
                Defaults to True.
            device (optional): Device. Defaults to None.
            dtype (optional): Data type. Defaults to None.

        Note:
            These layers should not be used out of the corresponding wrapper.
            Check MCBatchNorm in torch_uncertainty/post_processing/.
        """
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

        self.register_buffer(
            "means",
            torch.zeros(
                num_estimators, num_features, device=device, dtype=dtype
            ),
        )
        self.register_buffer(
            "vars",
            torch.zeros(
                num_estimators, num_features, device=device, dtype=dtype
            ),
        )

        self.accumulate = True
        self.num_estimators = num_estimators
        self.reset_mc_statistics()

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): Input tensor.
        """
        if not self.training:
            if self.accumulate:
                mean = x.mean((0, -2, -1))
                var = x.var((0, -2, -1), unbiased=True)
                self.means[self.counter] = mean
                self.vars[self.counter] = var
            self.running_mean = self.means[self.counter]
            self.running_var = self.vars[self.counter]
        return super().forward(x)

    def set_counter(self, counter: int) -> None:
        """Set the counter.

        Args:
            counter (int): new value for the counter.
        """
        self.counter = counter % self.num_estimators

    def reset_mc_statistics(self) -> None:
        """Reset the batch statistics."""
        self.counter = 0
        self.means = torch.zeros_like(self.means)
        self.vars = torch.zeros_like(self.vars)
