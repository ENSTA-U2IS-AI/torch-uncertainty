from ..layers.bayesian_layers import bayesian_modules


def Variational(Model):
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
