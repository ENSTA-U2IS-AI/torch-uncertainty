import copy

import torch
from torch import nn
from torch.utils.data import DataLoader


class SWA(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        cycle_start: int,
        cycle_length: int,
    ) -> None:
        super().__init__()
        _swa_checks(cycle_start, cycle_length)
        self.model = model
        self.cycle_start = cycle_start
        self.cycle_length = cycle_length
        self.num_averaged = 0
        self.swa_model = None
        self.need_bn_update = False

    @torch.no_grad()
    def update_model(self, epoch: int) -> None:
        if (
            epoch >= self.cycle_start
            and (epoch - self.cycle_start) % self.cycle_length == 0
        ):
            if self.swa_model is None:
                self.swa_model = copy.deepcopy(self.model)
                self.num_averaged = 1
            else:
                for swa_param, param in zip(
                    self.swa_model.parameters(),
                    self.model.parameters(),
                    strict=False,
                ):
                    swa_param.data += (param.data - swa_param.data) / (
                        self.num_averaged + 1
                    )
            self.num_averaged += 1
            self.need_bn_update = True

    def eval_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.swa_model is None:
            return self.model.forward(x)
        return self.swa_model.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.model.forward(x)
        return self.eval_forward(x)

    def update_bn(self, loader: DataLoader, device) -> None:
        if self.need_bn_update:
            torch.optim.swa_utils.update_bn(
                loader, self.swag_model, device=device
            )
            self.need_bn_update = False


def _swa_checks(cycle_start: int, cycle_length: int) -> None:
    if cycle_start < 0:
        raise ValueError(
            f"`cycle_start` must be non-negative. Got {cycle_start}."
        )
    if cycle_length <= 0:
        raise ValueError(
            f"`cycle_length` must be strictly positive. Got {cycle_length}."
        )
