from torch import Tensor, nn

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


class Backbone(nn.Module):
    def __init__(self, model: nn.Module, feat_names: list[str]) -> None:
        """Encoder backbone.

        Return the skip features of the :attr:`model` corresponding to the
        :attr:`feat_names`.

        Args:
            model (nn.Module): Base model.
            feat_names (list[str]): List of the feature names.
        """
        super().__init__()
        self.model = model
        self.feat_names = feat_names

    def forward(self, x: Tensor) -> list[Tensor]:
        """Encoder forward pass.

        Args:
            x (Tensor): Input tensor.

        Returns:
            list[Tensor]: List of the features.
        """
        feature = x
        features = []
        for k, v in self.model._modules.items():
            feature = v(feature)
            if k in self.feat_names:
                features.append(feature)
        return features


def set_bn_momentum(model: nn.Module, momentum: float) -> None:
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum
