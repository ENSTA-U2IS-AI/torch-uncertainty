# fmt: off
from typing import Dict, List

from torch import nn

from ..layers.bayesian import bayesian_modules


# fmt: on
def enable_dropout(model: nn.Module, last_layer_dropout: bool = False) -> None:
    """Function to enable or disable dropout layers during inference-time.

    Args:
        model (nn.Module, required): PyTorch model.
        last_layer_dropout (bool, optional): If set to True, only the
            last dropout layer will be in `eval` mode, otherwise, if set to
            False, all dropout layers in the model will be set
            to `eval` mode.
    """

    # filter all modules whose class name starts with `Dropout`
    filtered_modules = []
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            filtered_modules += [m]

    if last_layer_dropout:
        # set only the last filtered module to training mode
        filtered_modules[-1].train()
    else:
        # set all filtered modules to training mode
        for m in filtered_modules:
            m.train()


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
                    model |= {
                        module_name + "." + key: val
                        for key, val in module.state_dict().items()
                    }
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
