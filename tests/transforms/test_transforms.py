# fmt:off
import pytest
import torch
from PIL import Image

import numpy
from torch_uncertainty.transforms import (
    AutoContrast,
    Brightness,
    Color,
    Contrast,
    Equalize,
    Posterize,
    Rotation,
    Sharpness,
    Shear,
    Solarize,
    Translate,
)


# fmt:on
@pytest.fixture
def img_input() -> torch.Tensor:
    imarray = numpy.random.rand(28, 28, 3) * 255
    im = Image.fromarray(imarray.astype("uint8")).convert("RGB")
    return im


class TestAutoContrast:
    """Testing the AutoContrast transform."""

    def test_PIL(self, img_input):
        aug = AutoContrast()
        _ = aug(img_input)


class TestEqualize:
    """Testing the Equalize transform."""

    def test_PIL(self, img_input):
        aug = Equalize()
        _ = aug(img_input)


class TestPosterize:
    """Testing the Posterize transform."""

    def test_PIL(self, img_input):
        aug = Posterize()
        _ = aug(img_input, 2)

    def test_failures(self, img_input):
        aug = Posterize()
        with pytest.raises(ValueError):
            _ = aug(img_input, 5)
        with pytest.raises(ValueError):
            _ = aug(img_input, -1)


class TestSolarize:
    """Testing the Solarize transform."""

    def test_PIL(self, img_input):
        aug = Solarize()
        _ = aug(img_input, 2)

    def test_failures(self, img_input):
        aug = Solarize()
        with pytest.raises(ValueError):
            _ = aug(img_input, 300)
        with pytest.raises(ValueError):
            _ = aug(img_input, -1)


class TestRotation:
    """Testing the Rotation transform."""

    def test_PIL(self, img_input):
        aug = Rotation(random_direction=True)
        _ = aug(img_input, 10)


class TestShear:
    """Testing the Shear transform."""

    def test_PIL(self, img_input):
        aug = Shear(axis=0, random_direction=True)
        _ = aug(img_input, 10)

    def test_failures(self):
        with pytest.raises(ValueError):
            _ = Shear(axis=2, random_direction=True)


class TestTranslate:
    """Testing the Translate transform."""

    def test_PIL(self, img_input):
        aug = Translate(axis=0, random_direction=True)
        _ = aug(img_input, 10)

    def test_failures(self):
        with pytest.raises(ValueError):
            _ = Translate(axis=2, random_direction=True)


class TestContrast:
    """Testing the Contrast transform."""

    def test_PIL(self, img_input):
        aug = Contrast()
        _ = aug(img_input, 2)

    def test_failures(self, img_input):
        aug = Contrast()
        with pytest.raises(ValueError):
            _ = aug(img_input, -1)


class TestBrightness:
    """Testing the Brightness transform."""

    def test_PIL(self, img_input):
        aug = Brightness()
        _ = aug(img_input, 2)

    def test_failures(self, img_input):
        aug = Brightness()
        with pytest.raises(ValueError):
            _ = aug(img_input, -1)


class TestSharpness:
    """Testing the Sharpness transform."""

    def test_PIL(self, img_input):
        aug = Sharpness()
        _ = aug(img_input, 2)

    def test_failures(self, img_input):
        aug = Sharpness()
        with pytest.raises(ValueError):
            _ = aug(img_input, -1)


class TestColor:
    """Testing the Color transform."""

    def test_PIL(self, img_input):
        aug = Color()
        _ = aug(img_input, 2)

    def test_failures(self, img_input):
        aug = Color()
        with pytest.raises(ValueError):
            _ = aug(img_input, -1)
