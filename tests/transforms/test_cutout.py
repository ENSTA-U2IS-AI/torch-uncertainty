# fmt:off
import pytest
import torch

from torch_uncertainty.transforms import Cutout


# fmt:on
class TestCutout:
    """Testing the Cutout transform."""

    def test_cutout_allchannels(self):
        input = torch.rand(32, 32) + 10  # no zeros
        transform = Cutout(16)
        output = transform(input)

        assert (output == 0).sum() > 0

        input = torch.rand(1, 32, 32) + 10  # no zeros
        transform = Cutout(16)
        output = transform(input)

        assert (output == 0).sum() > 0

        input = torch.rand(3, 32, 32) + 10  # no zeros
        transform = Cutout(16)
        output = transform(input)

        assert (output == 0).sum() > 0

    def test_cutout_negative_length(self):
        with pytest.raises(ValueError):
            _ = Cutout(-1)

    def test_cutout_negative_value(self):
        with pytest.raises(ValueError):
            _ = Cutout(42, -16)
