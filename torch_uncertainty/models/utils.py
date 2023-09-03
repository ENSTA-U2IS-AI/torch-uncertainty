# fmt: off
from typing import Dict, List

from torch import nn

from ..layers.bayesian import bayesian_modules


# fmt: on
def StochasticModel(model: nn.Module) -> nn.Module:
    """Decorator for stochastic models. When applied to a model, it adds the
    sample, freeze and unfreeze methods to the model. Use freeze to obtain
    deterministic outputs. Use unfreeze to obtain stochastic outputs. Samples
    provide samples of the estimated posterior distribution.
    """

    def sample(self, num_samples: int = 1) -> List[Dict]:
        sampled_models = [{}] * num_samples
        for module_name in self._modules:
            module = self._modules[module_name]
            if isinstance(module, bayesian_modules):
                for model in sampled_models:
                    weight, bias = module.sample()
                    model[module_name + ".weight"] = weight
                    if bias is not None:
                        model[module_name + ".bias"] = bias
            else:
                for model in sampled_models:
                    state = module.state_dict()
                    if not len(state):  # no parameter
                        break
                    model[module_name] |= module.state_dict()
        return sampled_models

    setattr(model, "sample", sample)

    def freeze(self) -> None:
        for module_name in self._modules:
            module = self._modules[module_name]
            if isinstance(module, bayesian_modules):
                module.freeze()

    setattr(model, "freeze", freeze)

    def unfreeze(self) -> None:
        for module_name in self._modules:
            module = self._modules[module_name]
            if isinstance(module, bayesian_modules):
                module.unfreeze()

    setattr(model, "unfreeze", unfreeze)

    return model
