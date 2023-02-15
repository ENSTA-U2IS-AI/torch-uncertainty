# fmt: off
from pathlib import Path

import pytest

import torch_uncertainty.utils as utils

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
