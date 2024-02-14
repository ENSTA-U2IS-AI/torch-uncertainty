import pytest
import torch

from torch_uncertainty.transforms.corruptions import (
    DefocusBlur,
    Frost,
    GaussianBlur,
    GaussianNoise,
    GlassBlur,
    ImpulseNoise,
    JPEGCompression,
    Pixelate,
    ShotNoise,
    SpeckleNoise,
)


class TestCorruptions:
    """Testing the Corruptions transform."""

    def test_gaussian_noise(self):
        with pytest.raises(ValueError):
            _ = GaussianNoise(-1)
        with pytest.raises(TypeError):
            _ = GaussianNoise(0.1)
        inputs = torch.rand(3, 32, 32)
        transform = GaussianNoise(1)
        transform(inputs)
        transform = GaussianNoise(0)
        transform(inputs)
        print(transform)

    def test_shot_noise(self):
        with pytest.raises(ValueError):
            _ = ShotNoise(-1)
        with pytest.raises(TypeError):
            _ = ShotNoise(0.1)
        inputs = torch.rand(3, 32, 32)
        transform = ShotNoise(1)
        transform(inputs)
        transform = ShotNoise(0)
        transform(inputs)
        print(transform)

    def test_impulse_noise(self):
        with pytest.raises(ValueError):
            _ = ImpulseNoise(-1)
        with pytest.raises(TypeError):
            _ = ImpulseNoise(0.1)
        inputs = torch.rand(3, 32, 32)
        transform = ImpulseNoise(1)
        transform(inputs)
        transform = ImpulseNoise(0)
        transform(inputs)
        print(transform)

    def test_speckle_noise(self):
        with pytest.raises(ValueError):
            _ = SpeckleNoise(-1)
        with pytest.raises(TypeError):
            _ = SpeckleNoise(0.1)
        inputs = torch.rand(3, 32, 32)
        transform = SpeckleNoise(1)
        transform(inputs)
        transform = SpeckleNoise(0)
        transform(inputs)
        print(transform)

    def test_gaussian_blur(self):
        with pytest.raises(ValueError):
            _ = GaussianBlur(-1)
        with pytest.raises(TypeError):
            _ = GaussianBlur(0.1)
        inputs = torch.rand(3, 32, 32)
        transform = GaussianBlur(1)
        transform(inputs)
        transform = GaussianBlur(0)
        transform(inputs)
        print(transform)

    def test_glass_blur(self):
        with pytest.raises(ValueError):
            _ = GlassBlur(-1)
        with pytest.raises(TypeError):
            _ = GlassBlur(0.1)
        inputs = torch.rand(3, 32, 32)
        transform = GlassBlur(1)
        transform(inputs)
        transform = GlassBlur(0)
        transform(inputs)
        print(transform)

    def test_defocus_blur(self):
        with pytest.raises(ValueError):
            _ = DefocusBlur(-1)
        with pytest.raises(TypeError):
            _ = DefocusBlur(0.1)
        inputs = torch.rand(3, 32, 32)
        transform = DefocusBlur(1)
        transform(inputs)
        transform = DefocusBlur(0)
        transform(inputs)
        print(transform)

    def test_jpeg_compression(self):
        with pytest.raises(ValueError):
            _ = JPEGCompression(-1)
        with pytest.raises(TypeError):
            _ = JPEGCompression(0.1)
        inputs = torch.rand(3, 32, 32)
        transform = JPEGCompression(1)
        transform(inputs)
        transform = JPEGCompression(0)
        transform(inputs)
        print(transform)

    def test_pixelate(self):
        with pytest.raises(ValueError):
            _ = Pixelate(-1)
        with pytest.raises(TypeError):
            _ = Pixelate(0.1)
        inputs = torch.rand(3, 32, 32)
        transform = Pixelate(1)
        transform(inputs)
        transform = Pixelate(0)
        transform(inputs)
        print(transform)

    def test_frost(self):
        with pytest.raises(ValueError):
            _ = Frost(-1)
        with pytest.raises(TypeError):
            _ = Frost(0.1)
        inputs = torch.rand(3, 32, 32)
        transform = Frost(1)
        transform(inputs)
        transform = Frost(0)
        transform(inputs)
        print(transform)
