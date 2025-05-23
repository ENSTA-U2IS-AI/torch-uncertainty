import pytest
import torch
from requests.exceptions import HTTPError

from torch_uncertainty.transforms.corruption import (
    Brightness,
    Contrast,
    DefocusBlur,
    Elastic,
    Fog,
    Frost,
    GaussianBlur,
    GaussianNoise,
    GlassBlur,
    ImpulseNoise,
    JPEGCompression,
    MotionBlur,
    OriginalGlassBlur,
    Pixelate,
    Saturation,
    ShotNoise,
    Snow,
    SpeckleNoise,
    ZoomBlur,
)


class TestCorruptions:
    """Testing the Corruptions transform."""

    def test_gaussian_noise(self) -> None:
        with pytest.raises(ValueError):
            _ = GaussianNoise(-1)
        with pytest.raises(TypeError):
            _ = GaussianNoise(0.1)
        inputs = torch.rand(3, 32, 32)
        transform = GaussianNoise(1)
        assert transform(inputs).ndim == 3
        transform = GaussianNoise(0)
        assert transform(inputs).ndim == 3

        inputs = torch.rand(3, 3, 32, 32)
        assert transform(inputs).ndim == 4

        print(transform)  # noqa: T201

    def test_shot_noise(self) -> None:
        inputs = torch.rand(3, 32, 32)
        transform = ShotNoise(1)
        assert transform(inputs).ndim == 3
        transform = ShotNoise(0)
        assert transform(inputs).ndim == 3

        inputs = torch.rand(3, 3, 32, 32)
        assert transform(inputs).ndim == 4

    def test_impulse_noise(self) -> None:
        inputs = torch.rand(3, 32, 32)
        transform = ImpulseNoise(1, black_white=True)
        assert transform(inputs).ndim == 3
        transform = ImpulseNoise(0)
        assert transform(inputs).ndim == 3

        transform = ImpulseNoise(1, black_white=False)
        inputs = torch.rand(3, 3, 32, 32)
        assert transform(inputs).ndim == 4

    def test_speckle_noise(self) -> None:
        inputs = torch.rand(3, 32, 32)
        transform = SpeckleNoise(1)
        assert transform(inputs).ndim == 3
        transform = SpeckleNoise(0)
        assert transform(inputs).ndim == 3

        inputs = torch.rand(3, 3, 32, 32)
        transform = MotionBlur(1)
        assert transform(inputs).ndim == 4

    def test_gaussian_blur(self) -> None:
        inputs = torch.rand(3, 32, 32)
        transform = GaussianBlur(1)
        assert transform(inputs).ndim == 3
        transform = GaussianBlur(0)
        assert transform(inputs).ndim == 3

        inputs = torch.rand(3, 3, 32, 32)
        transform = GaussianBlur(1)
        assert transform(inputs).ndim == 4

    def test_glass_blur(self) -> None:
        inputs = torch.rand(3, 32, 32)
        transform = GlassBlur(1)
        assert transform(inputs).ndim == 3
        transform = GlassBlur(0)
        assert transform(inputs).ndim == 3

        inputs = torch.rand(3, 32, 32)
        transform = OriginalGlassBlur(1, seed=1)
        assert transform(inputs).ndim == 3
        transform = OriginalGlassBlur(0)
        assert transform(inputs).ndim == 3

    def test_defocus_blur(self) -> None:
        inputs = torch.rand(3, 32, 32)
        transform = DefocusBlur(1)
        assert transform(inputs).ndim == 3
        transform = DefocusBlur(0)
        assert transform(inputs).ndim == 3

        inputs = torch.rand(3, 3, 32, 32)
        transform = DefocusBlur(1)
        assert transform(inputs).ndim == 4

    def test_motion_blur(self) -> None:
        inputs = torch.rand(3, 32, 32)
        transform = MotionBlur(1)
        assert transform(inputs).ndim == 3
        transform = MotionBlur(0)
        assert transform(inputs).ndim == 3

        inputs = torch.rand(3, 3, 32, 32)
        transform = MotionBlur(1)
        assert transform(inputs).ndim == 4

    def test_zoom_blur(self) -> None:
        inputs = torch.rand(3, 32, 32)
        transform = ZoomBlur(1)
        assert transform(inputs).ndim == 3
        transform = ZoomBlur(0)
        assert transform(inputs).ndim == 3

    def test_jpeg_compression(self) -> None:
        inputs = torch.rand(3, 32, 32)
        transform = JPEGCompression(1)
        assert transform(inputs).ndim == 3
        transform = JPEGCompression(0)
        assert transform(inputs).ndim == 3

    def test_pixelate(self) -> None:
        inputs = torch.rand(3, 32, 32)
        transform = Pixelate(1)
        assert transform(inputs).ndim == 3
        transform = Pixelate(0)
        assert transform(inputs).ndim == 3

    def test_frost(self) -> None:
        try:
            Frost(1)
            frost_ok = True
        except HTTPError:
            frost_ok = False
        if frost_ok:
            inputs = torch.rand(3, 32, 32)
            transform = Frost(1)
            transform(inputs)
            transform = Frost(0)
            transform(inputs)

    def test_snow(self) -> None:
        inputs = torch.rand(3, 32, 32)
        transform = Snow(1)
        transform(inputs)
        transform = Snow(0)
        transform(inputs)

    def test_fog(self) -> None:
        inputs = torch.rand(3, 32, 32)
        transform = Fog(1)
        transform(inputs)
        transform = Fog(0)
        transform(inputs)

    def test_brightness(self) -> None:
        inputs = torch.rand(3, 32, 32)
        transform = Brightness(1)
        transform(inputs)
        transform = Brightness(0)
        transform(inputs)

    def test_contrast(self) -> None:
        inputs = torch.rand(3, 32, 32)
        transform = Contrast(1)
        transform(inputs)
        transform = Contrast(0)
        transform(inputs)

    def test_elastic(self) -> None:
        inputs = torch.rand(3, 32, 32)
        transform = Elastic(1)
        transform(inputs)
        transform = Elastic(0)
        transform(inputs)

    def test_saturation(self) -> None:
        inputs = torch.rand(3, 32, 32)
        transform = Saturation(1)
        transform(inputs)
        transform = Saturation(0)
        transform(inputs)
