# fmt: off
import pytest

import torch_uncertainty.utils as utils

# fmt:on


class TestUtils:
    """Testing utils methods."""

    def test_getversion_log_success(self):
        _ = utils.get_version("tests/testlog", version=42)

    def test_getversion_log_failure(self):
        with pytest.raises(Exception):
            _ = utils.get_version("tests/testlog", version=52)
