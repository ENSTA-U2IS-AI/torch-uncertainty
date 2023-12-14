from torch import Tensor, nn


class _MCDropout(nn.Module):
    def __init__(
        self, model: nn.Module, num_estimators: int, last_layer: bool
    ) -> None:
        """MC Dropout wrapper for a model.

        Args:
            model (nn.Module): model to wrap
            num_estimators (int): number of estimators to use
            last_layer (bool): whether to apply dropout to the last layer only.

        Warning:
            The underlying models must have a `dropout_rate` attribute.

        Warning:
            For the `last-layer` option to work properly, the model must
            declare the last dropout at the end of the initialization
            (i.e. after all the other dropout layers).
        """
        super().__init__()
        self.last_layer = last_layer

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

        self.model = model
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
    ) -> tuple[Tensor, Tensor]:
        if not self.training:
            x = x.repeat(self.num_estimators, 1, 1, 1)
        return self.model(x)


def mc_dropout(
    model: nn.Module, num_estimators: int, last_layer: bool = False
) -> _MCDropout:
    """MC Dropout wrapper for a model.

    Args:
        model (nn.Module): model to wrap
        num_estimators (int): number of estimators to use
        last_layer (bool, optional): whether to apply dropout to the last
            layer. Defaults to False.
    """
    return _MCDropout(
        model=model, num_estimators=num_estimators, last_layer=last_layer
    )
