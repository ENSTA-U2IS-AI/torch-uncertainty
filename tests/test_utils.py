from pathlib import Path

import pytest

from torch_uncertainty import utils


class TestUtils:
    """Testing utils methods."""

    def test_getversion_log_success(self):
        utils.get_version("tests/testlog", version=42)
        utils.get_version(Path("tests/testlog"), version=42)

        utils.get_version("tests/testlog", version=42, checkpoint=45)

    def test_getversion_log_failure(self):
        with pytest.raises(Exception):
            utils.get_version("tests/testlog", version=52)


class TestHub:
    """Testing hub methods."""

    def test_hub_exists(self):
        utils.hub.load_hf("test")
        utils.hub.load_hf("test", version=1)
        utils.hub.load_hf("test", version=2)

    def test_hub_notexists(self):
        with pytest.raises(Exception):
            utils.hub.load_hf("tests")

        with pytest.raises(ValueError):
            utils.hub.load_hf("test", version=42)
