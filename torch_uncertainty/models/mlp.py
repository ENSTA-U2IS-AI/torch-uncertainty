# fmt: off
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# fmt: on
__all__ = [
    "mlp",
]


class _ResNet(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_outputs: int,
    ) -> None:
        super().__init__()

        self.linear1 = nn.Linear(
            in_features,
            50,
        )
        self.linear2 = nn.Linear(
            50,
            num_outputs,
        )

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.linear1(x))
        out = self.linear2(out)
        return out


def mlp(
    in_features: int,
    num_outputs: int,
) -> _ResNet:
    return _ResNet(
        in_features=in_features,
        num_outputs=num_outputs,
    )
