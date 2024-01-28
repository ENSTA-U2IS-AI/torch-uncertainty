import pytest
import torch
import torchvision.transforms as T
from torch import nn

from tests._dummies.dataset import DummyClassificationDataset
from tests._dummies.model import Identity
from torch_uncertainty.models.lenet import lenet
from torch_uncertainty.post_processing import MCBatchNorm


class TestMCBatchNorm:
    """Testing the MCBatchNorm wrapper."""

    def test_main(self):
        """Test initialization."""
        model = lenet(1, 1, norm=nn.BatchNorm2d)
        stoch_model = MCBatchNorm(
            model, num_estimators=2, convert=True, mc_batch_size=1
        )
        dataset = DummyClassificationDataset(
            "./",
            num_channels=1,
            image_size=16,
            num_classes=1,
            num_images=2,
            transform=T.ToTensor(),
        )
        stoch_model.fit(dataset=dataset)
        stoch_model.train()
        stoch_model(torch.randn(1, 1, 20, 20))
        stoch_model.eval()
        stoch_model(torch.randn(1, 1, 20, 20))

    def test_errors(self):
        """Test errors."""
        model = Identity()
        with pytest.raises(ValueError):
            MCBatchNorm(model, num_estimators=0, convert=True)
        with pytest.raises(ValueError):
            MCBatchNorm(model, num_estimators=1, convert=False)
        with pytest.raises(ValueError):
            MCBatchNorm(model, num_estimators=1, convert=True)
