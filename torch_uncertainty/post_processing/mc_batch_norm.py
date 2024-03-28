from copy import deepcopy
from typing import Literal

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from torch_uncertainty.layers.mc_batch_norm import MCBatchNorm2d


class MCBatchNorm(nn.Module):
    counter: int = 0
    mc_batch_norm_layers: list[MCBatchNorm2d] = []
    trained = False

    def __init__(
        self,
        model: nn.Module,
        num_estimators: int,
        convert: bool,
        mc_batch_size: int = 32,
        device: Literal["cpu", "cuda"] | torch.device | None = None,
    ) -> None:
        """Monte Carlo Batch Normalization wrapper.

        Args:
            model (nn.Module): model to be converted.
            num_estimators (int): number of estimators.
            convert (bool): whether to convert the model.
            mc_batch_size (int, optional): Monte Carlo batch size. Defaults to 32.
            device (Literal["cpu", "cuda"] | torch.device | None, optional): device.
                Defaults to None.

        Note:
            This wrapper will be stochastic in eval mode only.

        Reference:
            Teye M, Azizpour H, Smith K. Bayesian uncertainty estimation for
            batch normalized deep networks. In ICML 2018.
        """
        super().__init__()

        self.mc_batch_size = mc_batch_size
        if num_estimators < 1 or not isinstance(num_estimators, int):
            raise ValueError(
                f"num_estimators must be a positive integer, got {num_estimators}."
            )
        self.num_estimators = num_estimators

        self.model = deepcopy(model)
        if not convert and not self._has_mcbn():
            raise ValueError(
                "model does not contain any MCBatchNorm2d nor is not to be "
                "converted."
            )
        self.device = device
        self.model = self.model.eval()
        if convert:
            self._convert()
            if not self._has_mcbn():
                raise ValueError(
                    "model does not contain any MCBatchNorm2d after conversion."
                )

    def fit(self, dataset: Dataset) -> None:
        """Fit the model on the dataset.

        Args:
            dataset (Dataset): dataset to be used for fitting.

        Note:
            This method is used to populate the MC BatchNorm layers.
            Use the training dataset.
        """
        self.dl = DataLoader(
            dataset, batch_size=self.mc_batch_size, shuffle=True
        )
        self.counter = 0
        self.reset_counters()
        self.set_accumulate(True)
        self.eval()
        for x, _ in self.dl:
            self.model(x.to(self.device))
            self.raise_counters()
            if self.counter == self.num_estimators:
                self.set_accumulate(False)
                self.trained = True
                return
        raise ValueError(
            "The dataset is too small to populate the MC BatchNorm statistics."
        )

    def _est_forward(self, x: Tensor) -> Tensor:
        """Forward pass of a single estimator."""
        logit = self.model(x)
        self.raise_counters()
        return logit

    def forward(
        self,
        x: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.training:
            return self.model(x)
        if not self.trained:
            raise RuntimeError(
                "MCBatchNorm has not been trained. Call .fit() first."
            )
        self.reset_counters()
        return torch.cat(
            [self._est_forward(x) for _ in range(self.num_estimators)], dim=0
        )

    def _has_mcbn(self) -> bool:
        """Check if the model contains any MCBatchNorm2d layers."""
        for module in self.model.modules():
            if isinstance(module, MCBatchNorm2d):
                return True
        return False

    def _convert(self) -> None:
        """Convert all BatchNorm2d layers to MCBatchNorm2d layers."""
        self.replace_layers(self.model)

    def reset_counters(self) -> None:
        """Reset all counters to 0."""
        self.counter = 0
        for layer in self.mc_batch_norm_layers:
            layer.set_counter(0)

    def raise_counters(self) -> None:
        """Raise all counters by 1."""
        self.counter += 1
        for layer in self.mc_batch_norm_layers:
            layer.set_counter(self.counter)

    def set_accumulate(self, accumulate: bool) -> None:
        """Set the accumulate flag for all MCBatchNorm2d layers.

        Args:
            accumulate (bool): accumulate flag.
        """
        for layer in self.mc_batch_norm_layers:
            layer.accumulate = accumulate

    def replace_layers(self, model: nn.Module) -> None:
        """Replace all BatchNorm2d layers with MCBatchNorm2d layers.

        Args:
            model (nn.Module): model to be converted.
        """
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
