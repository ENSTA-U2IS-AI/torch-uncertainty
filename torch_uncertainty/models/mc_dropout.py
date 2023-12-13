from torch import Tensor, nn


class _MCDropout(nn.Module):
    def __init__(
        self, model: nn.Module, num_passes: int, last_layer: bool = False
    ) -> None:
        super().__init__()
        self.last_layer = last_layer

        if model.dropout_rate is None:
            raise ValueError("dropout_rate must be set to use MC Dropout")
        if model.dropout_rate <= 0:
            raise ValueError("dropout_rate must be positive to use MC Dropout")
        if num_passes <= 0:
            raise ValueError(
                "num_passes must be strictly positive to use MC Dropout"
            )

        self.model = model
        self.num_passes = num_passes

        self.filtered_modules = filter(
            lambda m: isinstance(m, nn.Dropout), model.modules()
        )
        if last_layer:
            self.filtered_modules = list(self.filtered_modules)[-1:]

    def train(self: nn.Module, mode: bool = True) -> nn.Module:
        if not isinstance(mode, bool):
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
            x = x.repeat(self.num_passes, 1, 1, 1)
        return self.model(x)


def mc_dropout(model: nn.Module, num_passes: int) -> _MCDropout:
    return _MCDropout(model=model, num_passes=num_passes)
