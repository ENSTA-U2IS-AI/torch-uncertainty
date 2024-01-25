from copy import deepcopy

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from torch_uncertainty.layers.normalization import MCBatchNorm2d


class _MCBatchNorm(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        num_estimators: int,
        convert: bool,
        dataset: Dataset,
        mc_batch_size: int = 32,
        device=None,
    ) -> None:
        super().__init__()

        if num_estimators < 1 or not isinstance(num_estimators, int):
            raise ValueError(
                f"num_estimators must be a positive integer, got {num_estimators}."
            )
        self.num_estimators = num_estimators

        if convert and dataset is None:
            raise ValueError(
                "dataset must be provided when converting a model to populate the statistics."
            )

        if not convert and not self._has_mcbn():
            raise ValueError(
                "model does not contain any MCBatchNorm2d nor is not to be "
                "converted."
            )
        self.dl = DataLoader(dataset, batch_size=mc_batch_size, shuffle=True)
        self.model = deepcopy(model)
        self.device = device
        self.model = self.model.eval()
        if convert:
            self._convert()
            self._populate()

    def forward(
        self,
        x: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.training:
            return self.model(x)
        return torch.cat(
            [self.model(x) for _ in range(self.num_estimators)], dim=0
        )

    def _has_mcbn(self) -> bool:
        for module in self.model.modules():
            if isinstance(module, MCBatchNorm2d):
                return True
        return False

    def _convert(self) -> None:
        replace_layers(self.model, self.num_estimators)

    def _populate(self) -> None:
        i = 0
        for x, _ in self.dl:
            self.model(x.to(self.device))
            i += 1
            if i == self.num_estimators:
                break


def mc_batch_norm(
    model: nn.Module,
    num_estimators: int,
    convert: bool = False,
    dataset: Dataset = None,
    mc_batch_size: int = 32,
    device=None,
) -> _MCBatchNorm:
    return _MCBatchNorm(
        model=model,
        num_estimators=num_estimators,
        convert=convert,
        dataset=dataset,
        mc_batch_size=mc_batch_size,
        device=device,
    )


def replace_layers(model: nn.Module, num_estimators: int):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_layers(module, num_estimators)

        if isinstance(module, nn.BatchNorm2d):
            new_layer = MCBatchNorm2d(
                num_features=module.num_features,
                num_estimators=num_estimators,
                eps=module.eps,
                momentum=module.momentum,
                affine=module.affine,
                track_running_stats=module.track_running_stats,
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
            new_layer.training = module.training
            new_layer.weight = module.weight
            new_layer.bias = module.bias
            setattr(model, name, new_layer)
