import torch
from torch import Tensor, nn


class _MCBatchNorm(nn.Module):
    def __init__(self, model: nn.Module, num_estimators: int) -> None:
        super().__init__()

        self.model = model
        self.num_estimators = num_estimators

    def forward(
        self,
        x: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.training:
            return self.model(x)
        return torch.cat(
            [self.model(x) for _ in range(self.num_estimators)], dim=0
        )


def mc_batch_norm(model: nn.Module, num_estimators: int) -> _MCBatchNorm:
    return _MCBatchNorm(model=model, num_estimators=num_estimators)
