from torch_uncertainty.models.resnet.std import resnet34
from torch_uncertainty.models.utils import enable_dropout
from torch_uncertainty.models.vgg.std import vgg11
from torch_uncertainty.models.wideresnet.std import wideresnet28x10


class TestMonteCarloDropout:
    """Testing the ResNet std class."""

    def test_resnet(self):
        resnet34(1, 10, dropout_rate=0.5, num_estimators=10)

        model = resnet34(1, 10, 1)
        model.eval()

        enable_dropout(model)

        for m in model.modules():
            if m.__class__.__name__.startswith("Dropout"):
                assert m.training

    def test_vgg(self):
        vgg11(1, 10, dropout_rate=0.5, num_estimators=10)

        model = vgg11(1, 10, dropout_rate=0.5, num_estimators=10)
        model.eval()

        enable_dropout(model)

        for m in model.modules():
            if m.__class__.__name__.startswith("Dropout"):
                assert m.training

    def test_wideresnet(self):
        wideresnet28x10(1, 10, dropout_rate=0.5, num_estimators=10)

        model = wideresnet28x10(1, 10, dropout_rate=0.5, num_estimators=10)
        model.eval()

        enable_dropout(model)

        for m in model.modules():
            if m.__class__.__name__.startswith("Dropout"):
                assert m.training
