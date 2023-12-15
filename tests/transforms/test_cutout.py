import pytest
import torch

from torch_uncertainty.transforms import Cutout


class TestCutout:
    """Testing the Cutout transform."""

    def test_cutout_allchannels(self):
        inputs = torch.rand(32, 32) + 10  # no zeros
        transform = Cutout(16)
        output = transform(inputs)

        assert (output == 0).sum() > 0

        inputs = torch.rand(1, 32, 32) + 10  # no zeros
        output = transform(inputs)

        assert (output == 0).sum() > 0

        inputs = torch.rand(3, 32, 32) + 10  # no zeros
        output = transform(inputs)

        assert (output == 0).sum() > 0

    def test_cutout_negative_length(self):
        with pytest.raises(ValueError):
            _ = Cutout(-1)

    def test_cutout_negative_value(self):
        with pytest.raises(ValueError):
            _ = Cutout(42, -16)
