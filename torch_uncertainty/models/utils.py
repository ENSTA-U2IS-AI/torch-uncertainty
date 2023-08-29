# fmt: off
from torch import nn

from ..layers.bayesian_layers import bayesian_modules


# fmt: on
def StochasticModel(Model: nn.Module):
    """Decorator for stochastic models. When applied to a model, it adds the
    freeze and unfreeze methods to the model. Use freeze to obtain
    deterministic outputs. Use unfreeze to obtain stochastic outputs.
    """

    def freeze(self) -> None:
        for module in self.modules():
            if isinstance(module, bayesian_modules):
                module.freeze()

    setattr(Model, "freeze", freeze)

    def unfreeze(self) -> None:
        for module in self.modules():
            if isinstance(module, bayesian_modules):
                module.unfreeze()

    setattr(Model, "unfreeze", unfreeze)

    return Model
