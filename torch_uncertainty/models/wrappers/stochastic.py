import torch
from torch import Tensor, nn

from torch_uncertainty.layers.bayesian import bayesian_modules


class StochasticModel(nn.Module):
    def __init__(self, model: nn.Module, num_samples: int) -> None:
        super().__init__()
        self.model = model
        self.num_samples = num_samples

    def eval_forward(self, x: Tensor) -> Tensor:
        return torch.cat(
            [self.model.forward(x) for _ in range(self.num_samples)], dim=0
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            return self.model.forward(x)
        return self.eval_forward(x)

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

    def freeze(self) -> None:
        for module_name in self._modules:
            module = self._modules[module_name]
            if isinstance(module, bayesian_modules):
                module.freeze()

    def unfreeze(self) -> None:
        for module_name in self._modules:
            module = self._modules[module_name]
            if isinstance(module, bayesian_modules):
                module.unfreeze()
