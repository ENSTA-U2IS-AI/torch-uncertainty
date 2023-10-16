# fmt: off
from pathlib import Path

import pytest
import torch

import torch_uncertainty.utils as utils
from torch_uncertainty.plotting_utils import CalibrationPlot


# fmt:on
class TestUtils:
    """Testing utils methods."""

    def test_getversion_log_success(self):
        _ = utils.get_version("tests/testlog", version=42)
        _ = utils.get_version(Path("tests/testlog"), version=42)

    def test_getversion_log_success_with_checkpoint(self):
        _ = utils.get_version("tests/testlog", version=42, checkpoint=45)

    def test_getversion_log_failure(self):
        with pytest.raises(Exception):
            _ = utils.get_version("tests/testlog", version=52)


class TestHub:
    """Testing hub methods."""

    def test_hub_exists(self):
        _ = utils.hub.load_hf("test")

    def test_hub_notexists(self):
        with pytest.raises(Exception):
            _ = utils.hub.load_hf("tests")


class TestCalibrationPlot:
    """Testing calibration plot class."""

    def test_failures(self):
        with pytest.raises(Exception):
            _ = CalibrationPlot(mode="full")
        with pytest.raises(Exception):
            _ = CalibrationPlot(adaptive=True)
        with pytest.raises(Exception):
            _ = CalibrationPlot(num_bins=0)
        with pytest.raises(Exception):
            _ = CalibrationPlot(num_bins=0.5)

    def test_forward(self):
        cal_plot = CalibrationPlot()
        cal_plot(
            torch.tensor([[0.5, 0.2, 0.3], [0.5, 0.5, 0.0]]),
            torch.tensor([0, 1]),
        )
