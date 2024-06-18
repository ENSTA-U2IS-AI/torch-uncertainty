import torch
from torch import Tensor, nn
from torch.nn import functional as F


class BatchNormAdapter2d(nn.Module):
    def __init__(
        self,
        num_features: int,
        alpha: float = 0.1,
        momentum: float = 0.1,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(num_features, device=device, dtype=dtype),
            requires_grad=True,
        )
        self.bias = nn.Parameter(
            torch.zeros(num_features, device=device, dtype=dtype),
            requires_grad=True,
        )

        self.register_buffer(
            "running_mean",
            torch.zeros(num_features, device=device, dtype=dtype),
        )
        self.register_buffer(
            "running_var",
            torch.zeros(num_features, device=device, dtype=dtype),
        )
        self.register_buffer(
            "num_batches_tracked",
            torch.tensor(0, dtype=torch.long, device=device),
        )
        self.alpha = alpha
        self.momentum = momentum
        self.eps = eps
        self.frozen = False

    def forward(self, x: Tensor) -> Tensor:
        if self.frozen:
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                self.training,
                self.momentum,
                self.eps,
            )
        out = F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            None,
            None,
            self.training,
            self.momentum,
            self.eps,
        )
        return self.weight.unsqueeze(-1).unsqueeze(-1) * out * (
            torch.randn_like(x) * self.alpha + 1
        ) + self.bias.unsqueeze(-1).unsqueeze(-1)
