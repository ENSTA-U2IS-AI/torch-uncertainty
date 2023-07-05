from ..layers.bayesian_layers import bayesian_modules


def StochasticModel(Model):
    """Decorator for stochastic models. When applied to a model, it adds the
    freeze and unfreeze methods to the model. Use freeze to obtain
    deterministic outputs. Use unfreeze to obtain stochastic outputs.
    """

    def freeze(self):
        for module in self.modules():
            if isinstance(module, bayesian_modules):
                module.freeze = True

    setattr(Model, "freeze", freeze)

    def unfreeze(self):
        for module in self.modules():
            if isinstance(module, bayesian_modules):
                module.freeze = False

    setattr(Model, "unfreeze", unfreeze)

    return Model
