from copy import deepcopy
from typing import Literal

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from torch_uncertainty.layers.normalization import MCBatchNorm2d


class MCBatchNorm(nn.Module):
    counter: int = 0
    mc_batch_norm_layers: list[MCBatchNorm2d]
    populated = False

    def __init__(
        self,
        model: nn.Module,
        num_estimators: int,
        convert: bool,
        mc_batch_size: int = 32,
        device: Literal["cpu", "cuda"] | torch.device | None = None,
    ) -> None:
        super().__init__()

        self.mc_batch_size = mc_batch_size
        if num_estimators < 1 or not isinstance(num_estimators, int):
            raise ValueError(
                f"num_estimators must be a positive integer, got {num_estimators}."
            )
        self.num_estimators = num_estimators

        if not convert and not self._has_mcbn():
            raise ValueError(
                "model does not contain any MCBatchNorm2d nor is not to be "
                "converted."
            )
        self.model = deepcopy(model)
        self.device = device
        self.model = self.model.eval()
        if convert:
            self._convert()

    def fit(self, dataset: Dataset):
        self.dl = DataLoader(
            dataset, batch_size=self.mc_batch_size, shuffle=True
        )
        self.counter = 0
        self.reset_counters()
        for x, _ in self.dl:
            self.model(x.to(self.device))
            self.raise_counters()
            if self.counter == self.num_estimators:
                return

    def _est_forward(self, x: Tensor) -> Tensor:
        logit = self.model(x)
        self.raise_counters()
        return logit

    def forward(
        self,
        x: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.training:
            return self.model(x)
        self.reset_counters()
        preds = torch.cat(
            [self._est_forward(x) for _ in range(self.num_estimators)], dim=0
        )
        self.reset_counters()
        return preds

    def _has_mcbn(self) -> bool:
        for module in self.model.modules():
            if isinstance(module, MCBatchNorm2d):
                return True
        return False

    def _convert(self) -> None:
        self.replace_layers(self.model)

    def reset_counters(self) -> None:
        self.counter = 0
        for layer in self.mc_batch_norm_layers:
            layer.set_counter(0)

    def raise_counters(self) -> None:
        self.counter += 1
        for layer in self.mc_batch_norm_layers:
            layer.set_counter(self.counter)

    def replace_layers(self, model: nn.Module) -> None:
        for name, module in model.named_children():
            if len(list(module.children())) > 0:
                self.replace_layers(module)

            if isinstance(module, nn.BatchNorm2d):
                mc_layer = MCBatchNorm2d(
                    num_features=module.num_features,
                    num_estimators=self.num_estimators,
                    eps=module.eps,
                    momentum=module.momentum,
                    affine=module.affine,
                    track_running_stats=module.track_running_stats,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                mc_layer.training = module.training
                mc_layer.weight = module.weight
                mc_layer.bias = module.bias
                setattr(model, name, mc_layer)

                # Save pointers to the MC BatchNorm layers
                self.mc_batch_norm_layers.append(mc_layer)
