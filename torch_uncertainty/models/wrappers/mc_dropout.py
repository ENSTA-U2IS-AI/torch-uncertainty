import torch
from torch import Tensor, nn
from torch.nn.modules.dropout import _DropoutNd


class MCDropout(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        num_estimators: int,
        last_layer: bool,
        on_batch: bool,
    ) -> None:
        """MC Dropout wrapper for a model containing nn.Dropout modules.

        Args:
            model (nn.Module): model to wrap
            num_estimators (int): number of estimators to use during the
                evaluation
            last_layer (bool): whether to apply dropout to the last layer only.
            on_batch (bool): Perform the MC-Dropout on the batch-size.
                Otherwise in a for loop. Useful when constrained in memory.

        Warning:
            This module will work only if you apply dropout through modules
            declared in the constructor (__init__).

        Warning:
            The `last-layer` option disables the lastly initialized dropout
            during evaluation: make sure that the last dropout is either
            functional or a module of its own.
        """
        super().__init__()
        filtered_modules = list(
            filter(
                lambda m: isinstance(m, _DropoutNd),
                model.modules(),
            )
        )
        if last_layer:
            filtered_modules = filtered_modules[-1:]

        _dropout_checks(filtered_modules, num_estimators)
        self.last_layer = last_layer
        self.on_batch = on_batch
        self.core_model = model
        self.num_estimators = num_estimators
        self.filtered_modules = filtered_modules

    def train(self, mode: bool = True) -> nn.Module:
        """Override the default train method to set the training mode of
        each submodule to be the same as the module itself except for the
        selected dropout modules.

        Args:
            mode (bool, optional): whether to set the module to training
                mode. Defaults to True.
        """
        if not isinstance(mode, bool):
            raise TypeError("Training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        for module in self.filtered_modules:
            module.train()
        return self

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Forward pass of the model.

        During training, the forward pass is the same as of the core model.
        During evaluation, the forward pass is repeated `num_estimators` times
        either on the batch size or in a for loop depending on
        :attr:`last_layer`.

        Args:
            x (Tensor): input tensor of shape (B, ...)

        Returns:
            Tensor: output tensor of shape (:attr:`num_estimators` * B, ...)
        """
        if self.training:
            return self.core_model(x)
        if self.on_batch:
            x = x.repeat(self.num_estimators, 1, 1, 1)
            return self.core_model(x)
        # Else, for loop
        return torch.cat(
            [self.core_model(x) for _ in range(self.num_estimators)], dim=0
        )


def mc_dropout(
    model: nn.Module,
    num_estimators: int,
    last_layer: bool = False,
    on_batch: bool = True,
) -> MCDropout:
    """MC Dropout wrapper for a model.

    Args:
        model (nn.Module): model to wrap
        num_estimators (int): number of estimators to use
        last_layer (bool, optional): whether to apply dropout to the last
            layer only. Defaults to False.
        on_batch (bool): Increase the batch_size to perform MC-Dropout.
            Otherwise in a for loop to reduce memory footprint. Defaults
            to true.
    """
    return MCDropout(
        model=model,
        num_estimators=num_estimators,
        last_layer=last_layer,
        on_batch=on_batch,
    )


def _dropout_checks(
    filtered_modules: list[nn.Module], num_estimators: int
) -> None:
    if not filtered_modules:
        raise ValueError(
            "No dropout module found in the model. "
            "Please use `nn.Dropout`-like modules to apply dropout."
        )
    # Check that at least one module has > 0.0 dropout rate
    if not any(mod.p > 0.0 for mod in filtered_modules):
        raise ValueError(
            "At least one dropout module must have a dropout rate > 0.0."
        )
    if num_estimators <= 0:
        raise ValueError(
            "`num_estimators` must be strictly positive to use MC Dropout."
        )
