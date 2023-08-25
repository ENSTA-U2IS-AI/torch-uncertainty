# fmt: off
from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor


# fmt: on
class MIMOBatchFormat(nn.Module):
    def __init__(
        self, num_estimators: int, rho: float = 0.0, batch_repeat: int = 1
    ) -> None:
        super().__init__()
        self.num_estimators = num_estimators
        self.rho = rho
        self.batch_repeat = batch_repeat

    def shuffle(self, inputs: torch.Tensor):
        idx = torch.randperm(inputs.nelement(), device=inputs.device)
        return inputs.view(-1)[idx].view(inputs.size())

    def forward(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        inputs, targets = batch
        indexes = torch.arange(
            0, inputs.shape[0], device=inputs.device, dtype=torch.int64
        ).repeat(self.batch_repeat)
        main_shuffle = self.shuffle(indexes)
        threshold_shuffle = int(main_shuffle.shape[0] * (1.0 - self.rho))
        shuffle_indices = [
            torch.concat(
                [
                    self.shuffle(main_shuffle[:threshold_shuffle]),
                    main_shuffle[threshold_shuffle:],
                ],
                axis=0,
            )
            for _ in range(self.num_estimators)
        ]
        inputs = torch.stack(
            [
                torch.index_select(inputs, dim=0, index=indices)
                for indices in shuffle_indices
            ],
            axis=0,
        )
        targets = torch.stack(
            [
                torch.index_select(targets, dim=0, index=indices)
                for indices in shuffle_indices
            ],
            axis=0,
        )
        inputs = rearrange(
            inputs, "m b c h w -> (m b) c h w", m=self.num_estimators
        )
        targets = rearrange(targets, "m b -> (m b)", m=self.num_estimators)
        return inputs, targets
