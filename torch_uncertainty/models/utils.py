from torch import nn

from torch_uncertainty.layers.bayesian import bayesian_modules


def stochastic_model(model: nn.Module) -> nn.Module:
    """Decorator for stochastic models.

    When applied to a model, it adds the `sample`, `freeze` and `unfreeze`
    methods. Use `freeze` to obtain deterministic outputs. Use unfreeze to
    obtain stochastic outputs. `sample` get samples of the estimated posterior
    distribution.

    Args:
        model (nn.Module): PyTorch model.
    """

    def sample(self, num_samples: int = 1) -> list[dict]:
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
                    # TODO: fix this
                    model |= {
                        module_name + "." + key: val
                        for key, val in module.state_dict().items()
                    }
        return sampled_models

    model.sample = sample

    def freeze(self) -> None:
        for module_name in self._modules:
            module = self._modules[module_name]
            if isinstance(module, bayesian_modules):
                module.freeze()

    model.freeze = freeze

    def unfreeze(self) -> None:
        for module_name in self._modules:
            module = self._modules[module_name]
            if isinstance(module, bayesian_modules):
                module.unfreeze()

    model.unfreeze = unfreeze

    return model
