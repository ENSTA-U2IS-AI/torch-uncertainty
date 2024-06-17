from torch import Tensor, nn


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
