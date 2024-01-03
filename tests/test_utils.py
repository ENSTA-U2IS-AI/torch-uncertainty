from pathlib import Path

import pytest

from torch_uncertainty import utils


class TestUtils:
    """Testing utils methods."""

    def test_getversion_log_success(self):
        _ = utils.get_version("tests/testlog", version=42)
        _ = utils.get_version(Path("tests/testlog"), version=42)

        _ = utils.get_version("tests/testlog", version=42, checkpoint=45)

    def test_getversion_log_failure(self):
        with pytest.raises(Exception):
            _ = utils.get_version("tests/testlog", version=52)


class TestHub:
    """Testing hub methods."""

    def test_hub_exists(self):
        _ = utils.hub.load_hf("test")
        _ = utils.hub.load_hf("test", version=1)

    def test_hub_notexists(self):
        with pytest.raises(Exception):
            _ = utils.hub.load_hf("tests")
