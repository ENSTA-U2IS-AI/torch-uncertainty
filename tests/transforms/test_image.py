import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import tv_tensors

from torch_uncertainty.transforms import (
    AutoContrast,
    Brightness,
    Color,
    Contrast,
    Equalize,
    MIMOBatchFormat,
    Posterize,
    RandomRescale,
    RepeatTarget,
    Rotate,
    Sharpen,
    Shear,
    Solarize,
    Translate,
)


@pytest.fixture()
def img_input() -> torch.Tensor:
    rng = np.random.default_rng()
    imarray = rng.uniform(low=0, high=255, size=(28, 28, 3))
    return Image.fromarray(imarray.astype("uint8")).convert("RGB")


@pytest.fixture()
def tv_tensors_input() -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng()
    imarray1 = rng.uniform(low=0, high=255, size=(3, 28, 28))
    imarray2 = rng.uniform(low=0, high=255, size=(1, 28, 28))
    return (
        tv_tensors.Image(imarray1.astype("uint8")),
        tv_tensors.Mask(imarray2.astype("uint8")),
    )


@pytest.fixture()
def batch_input() -> tuple[torch.Tensor, torch.Tensor]:
    imgs = torch.rand(2, 3, 28, 28)
    return imgs, torch.tensor([0, 1])


class TestAutoContrast:
    """Testing the AutoContrast transform."""

    def test_pil(self, img_input):
        aug = AutoContrast()
        _ = aug(img_input)


class TestEqualize:
    """Testing the Equalize transform."""

    def test_pil(self, img_input):
        aug = Equalize()
        _ = aug(img_input)


class TestPosterize:
    """Testing the Posterize transform."""

    def test_pil(self, img_input):
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

    def test_pil(self, img_input):
        aug = Solarize()
        _ = aug(img_input, 2)

    def test_failures(self, img_input):
        aug = Solarize()
        with pytest.raises(ValueError):
            _ = aug(img_input, 300)
        with pytest.raises(ValueError):
            _ = aug(img_input, -1)


class TestRotate:
    """Testing the Rotate transform."""

    def test_pil(self, img_input):
        aug = Rotate(random_direction=True)
        _ = aug(img_input, 10)


class TestShear:
    """Testing the Shear transform."""

    def test_pil(self, img_input):
        aug = Shear(axis=0, random_direction=True)
        _ = aug(img_input, 10)

    def test_failures(self):
        with pytest.raises(ValueError):
            _ = Shear(axis=2, random_direction=True)


class TestTranslate:
    """Testing the Translate transform."""

    def test_pil(self, img_input):
        aug = Translate(axis=0, random_direction=True)
        _ = aug(img_input, 10)

    def test_failures(self):
        with pytest.raises(ValueError):
            _ = Translate(axis=2, random_direction=True)


class TestContrast:
    """Testing the Contrast transform."""

    def test_pil(self, img_input):
        aug = Contrast()
        _ = aug(img_input, 2)

    def test_failures(self, img_input):
        aug = Contrast()
        with pytest.raises(ValueError):
            _ = aug(img_input, -1)


class TestBrightness:
    """Testing the Brightness transform."""

    def test_pil(self, img_input):
        aug = Brightness()
        _ = aug(img_input, 2)

    def test_failures(self, img_input):
        aug = Brightness()
        with pytest.raises(ValueError):
            _ = aug(img_input, -1)


class TestSharpen:
    """Testing the Sharpen transform."""

    def test_pil(self, img_input):
        aug = Sharpen()
        _ = aug(img_input, 2)

    def test_failures(self, img_input):
        aug = Sharpen()
        with pytest.raises(ValueError):
            _ = aug(img_input, -1)


class TestColor:
    """Testing the Color transform."""

    def test_pil(self, img_input):
        aug = Color()
        aug(img_input, 2)

    def test_tensor(self):
        aug = Color()
        aug(torch.rand(3, 28, 28), 2)

    def test_failures(self, img_input):
        aug = Color()
        with pytest.raises(ValueError):
            _ = aug(img_input, -1)


class TestRepeatTarget:
    """Testing the RepeatTarget transform."""

    def test_batch(self, batch_input):
        fn = RepeatTarget(3)
        _, target = fn(batch_input)
        assert target.shape == (6,)

    def test_failures(self):
        with pytest.raises(TypeError):
            _ = RepeatTarget(1.2)

        with pytest.raises(ValueError):
            _ = RepeatTarget(0)


class TestMIMOBatchFormat:
    """Testing the MIMOBatchFormat transform."""

    def test_batch(self, batch_input):
        b, c, h, w = batch_input[0].shape

        fn = MIMOBatchFormat(1, 0, 1)
        imgs, target = fn(batch_input)
        assert imgs.shape == (b, c, h, w)
        assert target.shape == (b,)

        fn = MIMOBatchFormat(4, 0, 2)
        imgs, target = fn(batch_input)
        assert imgs.shape == (b * 4 * 2, 3, 28, 28)
        assert target.shape == (b * 4 * 2,)

    def test_failures(self):
        with pytest.raises(ValueError):
            _ = MIMOBatchFormat(0, 0, 1)

        with pytest.raises(ValueError):
            _ = MIMOBatchFormat(1, -1, 1)

        with pytest.raises(ValueError):
            _ = MIMOBatchFormat(1, 1.2, 1)

        with pytest.raises(ValueError):
            _ = MIMOBatchFormat(1, 0, 0)


class TestRandomRescale:
    """Testing the RandomRescale transform."""

    def test_tv_tensors(self, tv_tensors_input):
        aug = RandomRescale(0.5, 2.0)
        _ = aug(tv_tensors_input)
