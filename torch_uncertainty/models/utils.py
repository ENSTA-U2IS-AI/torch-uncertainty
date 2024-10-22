from torch import Tensor, nn
from torch.nn.modules.batchnorm import _BatchNorm


class Backbone(nn.Module):
    def __init__(self, model: nn.Module, feat_names: list[str]) -> None:
        """Encoder backbone.

        Return the skip features of the :attr:`model` corresponding to the
        :attr:`feat_names`.

        Args:
            model (nn.Module): Base model.
            feat_names (list[str]): list of the feature names.
        """
        super().__init__()
        self.model = model
        self.feat_names = feat_names

    def forward(self, x: Tensor) -> list[Tensor]:
        """Encoder forward pass.

        Args:
            x (Tensor): Input tensor.

        Returns:
            list[Tensor]: list of the features.
        """
        feature = x
        features = []
        for key, layer in self.model._modules.items():
            feature = layer(feature)
            if key in self.feat_names:
                features.append(feature)
        return features


def set_bn_momentum(model: nn.Module, momentum: float) -> None:
    """Set the momentum of all batch normalization layers in the model.

    Args:
        model (nn.Module): Model.
        momentum (float): Momentum of the batch normalization layers.
    """
    for m in model.modules():
        if isinstance(m, _BatchNorm):
            m.momentum = momentum
