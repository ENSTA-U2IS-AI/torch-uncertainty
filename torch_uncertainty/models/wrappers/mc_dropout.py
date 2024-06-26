import torch
from torch import Tensor, nn


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
            num_estimators (int): number of estimators to use
            last_layer (bool): whether to apply dropout to the last layer only.
            on_batch (bool): Increase the batch_size to perform MC-Dropout.
                Otherwise in a for loop.

        Warning:
            Apply dropout using modules and not functional for this wrapper to
            work as intended.

        Warning:
            The underlying models must have a non-zero :attr:`dropout_rate`
            attribute.

        Warning:
            For the `last-layer` option to work properly, the model must
            declare the last dropout at the end of the initialization
            (i.e. after all the other dropout layers).
        """
        super().__init__()
        _dropout_checks(model, num_estimators)
        self.last_layer = last_layer
        self.on_batch = on_batch
        self.core_model = model
        self.num_estimators = num_estimators

        self.filtered_modules = list(
            filter(
                lambda m: isinstance(m, nn.Dropout | nn.Dropout2d),
                model.modules(),
            )
        )
        if last_layer:
            self.filtered_modules = self.filtered_modules[-1:]

    def train(self, mode: bool = True) -> nn.Module:
        """Override the default train method to set the training mode of
        each submodule to be the same as the module itself.

        Args:
            mode (bool, optional): whether to set the module to training
                mode. Defaults to True.
        """
        if not isinstance(mode, bool):  # coverage: ignore
            raise TypeError("training mode is expected to be boolean")
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


def _dropout_checks(model: nn.Module, num_estimators: int) -> None:
    if not hasattr(model, "dropout_rate"):
        raise ValueError(
            "`dropout_rate` must be set in the model to use MC Dropout."
        )
    if model.dropout_rate <= 0.0:
        raise ValueError(
            "`dropout_rate` must be strictly positive to use MC Dropout."
        )
    if num_estimators is None:
        raise ValueError("`num_estimators` must be set to use MC Dropout.")
    if num_estimators <= 0:
        raise ValueError(
            "`num_estimators` must be strictly positive to use MC Dropout."
        )
