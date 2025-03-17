from functools import partial

import pytest
import torch
import torchvision.transforms as T
from torch import nn
from torch.utils.data import DataLoader

from tests._dummies.dataset import DummyClassificationDataset
from torch_uncertainty.layers.mc_batch_norm import MCBatchNorm2d
from torch_uncertainty.models.lenet import lenet
from torch_uncertainty.post_processing import MCBatchNorm


class TestMCBatchNorm:
    """Testing the MCBatchNorm wrapper."""

    def test_main(self):
        """Test initialization."""
        mc_model = lenet(1, 1, norm=partial(MCBatchNorm2d, num_estimators=2))
        stoch_model = MCBatchNorm(mc_model, num_estimators=2, convert=False)

        model = lenet(1, 1, norm=nn.BatchNorm2d)
        stoch_model = MCBatchNorm(
            nn.Sequential(model),
            num_estimators=2,
            convert=True,
        )
        dataset = DummyClassificationDataset(
            "./",
            num_channels=1,
            image_size=16,
            num_classes=1,
            num_images=2,
            transform=T.ToTensor(),
        )
        dl = DataLoader(dataset, batch_size=1, shuffle=True)
        stoch_model.fit(dataloader=dl)
        stoch_model.train()
        stoch_model(torch.randn(1, 1, 20, 20))
        stoch_model.eval()
        stoch_model(torch.randn(1, 1, 20, 20))

        stoch_model = MCBatchNorm(num_estimators=2, convert=False)
        stoch_model.set_model(mc_model)

    def test_errors(self):
        """Test errors."""
        model = nn.Identity()
        with pytest.raises(ValueError):
            MCBatchNorm(model, num_estimators=0, convert=True)
        with pytest.raises(ValueError):
            MCBatchNorm(model, num_estimators=1, convert=False)
        with pytest.raises(ValueError):
            MCBatchNorm(model, num_estimators=1, convert=True)
        model = lenet(1, 1, norm=nn.BatchNorm2d)
        stoch_model = MCBatchNorm(model, num_estimators=4, convert=True)
        dataset = DummyClassificationDataset(
            "./",
            num_channels=1,
            image_size=16,
            num_classes=1,
            num_images=2,
            transform=T.ToTensor(),
        )
        dl = DataLoader(dataset, batch_size=2, shuffle=True)
        stoch_model.eval()
        with pytest.raises(RuntimeError):
            stoch_model(torch.randn(1, 1, 20, 20))

        with pytest.raises(ValueError):
            stoch_model.fit(dataloader=dl)
