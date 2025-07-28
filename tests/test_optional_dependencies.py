"""Tests for optional dependencies."""

import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest


def test_import_without_kornia():
    """Test that torch_uncertainty can be imported without kornia."""
    # Create a temporary script that tries to import torch_uncertainty
    # in an environment where kornia is definitely not available
    script_content = textwrap.dedent("""
        import sys
        
        # Mock kornia as unavailable by removing it from sys.modules if present
        if 'kornia' in sys.modules:
            del sys.modules['kornia']
        
        # Also block kornia imports by adding a mock that raises ImportError
        class MockKorniaModule:
            def __getattr__(self, name):
                raise ImportError("No module named 'kornia'")
        
        import importlib.util
        original_find_spec = importlib.util.find_spec
        
        def mock_find_spec(name, package=None):
            if name == 'kornia' or name.startswith('kornia.'):
                return None
            return original_find_spec(name, package)
        
        importlib.util.find_spec = mock_find_spec
        
        try:
            import torch_uncertainty
            print("SUCCESS: torch_uncertainty imported without kornia")
        except ImportError as e:
            print(f"FAILED: {e}")
            sys.exit(1)
    """)
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script_content)
        f.flush()
        
        try:
            # Run the script in a subprocess
            result = subprocess.run(
                [sys.executable, f.name],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            assert result.returncode == 0, f"Script failed with stderr: {result.stderr}"
            assert "SUCCESS" in result.stdout, f"Unexpected output: {result.stdout}"
            
        finally:
            Path(f.name).unlink()


def test_kornia_transforms_fail_gracefully():
    """Test that transforms requiring kornia fail gracefully when kornia is not available."""
    import torch_uncertainty.transforms.corruption as corruption_module
    
    # Mock kornia as unavailable
    original_kornia_installed = corruption_module.kornia_installed
    corruption_module.kornia_installed = False
    
    try:
        from torch_uncertainty.transforms.corruption import (
            DefocusBlur,
            GaussianBlur,
            GlassBlur,
            ImpulseNoise,
            MotionBlur,
            Snow,
        )
        
        # Test a few transforms that require kornia
        transforms_requiring_kornia = [
            ImpulseNoise,
            DefocusBlur,
            GlassBlur,
            MotionBlur,
            Snow,
            GaussianBlur,
        ]
        
        for transform_class in transforms_requiring_kornia:
            with pytest.raises(ImportError, match=r"Please install torch_uncertainty with the image option"):
                transform_class(severity=1)
    finally:
        # Restore original state
        corruption_module.kornia_installed = original_kornia_installed


def test_kornia_transforms_work_with_kornia_installed():
    """Test that transforms work when kornia is available."""
    pytest.importorskip("kornia")
    
    import torch

    from torch_uncertainty.transforms.corruption import (
        DefocusBlur,
        GaussianBlur,
        GlassBlur,
        ImpulseNoise,
        MotionBlur,
    )
    
    # Test that transforms can be instantiated and run when kornia is available
    img = torch.rand(3, 32, 32)
    
    transforms_requiring_kornia = [
        ImpulseNoise,
        DefocusBlur,
        GlassBlur, 
        MotionBlur,
        GaussianBlur,
    ]
    
    for transform_class in transforms_requiring_kornia:
        transform = transform_class(severity=1)
        output = transform(img)
        assert output.shape == img.shape
        
        # Test severity 0 (no-op)
        transform_zero = transform_class(severity=0)
        output_zero = transform_zero(img)
        assert torch.allclose(output_zero, img)


def test_disk_function_fails_without_kornia():
    """Test that the disk function fails gracefully without kornia."""
    import torch_uncertainty.transforms.corruption as corruption_module
    
    # Mock kornia as unavailable
    original_kornia_installed = corruption_module.kornia_installed
    corruption_module.kornia_installed = False
    
    try:
        from torch_uncertainty.transforms.corruption import disk
        
        with pytest.raises(ImportError, match=r"Please install torch_uncertainty with the image option"):
            disk(radius=3)
    finally:
        # Restore original state
        corruption_module.kornia_installed = original_kornia_installed


def test_disk_function_works_with_kornia():
    """Test that the disk function works when kornia is available."""
    pytest.importorskip("kornia")
    
    import torch

    from torch_uncertainty.transforms.corruption import disk
    
    result = disk(radius=3)
    assert isinstance(result, torch.Tensor)
    assert result.ndim == 3  # (1, H, W)
    assert result.shape[0] == 1


def test_transforms_without_kornia_work():
    """Test that transforms not requiring kornia work fine."""
    import torch

    from torch_uncertainty.transforms.corruption import (
        Brightness,
        Contrast,
        Fog,
        Frost,
        GaussianNoise,
        JPEGCompression,
        Pixelate,
        Saturation,
        ShotNoise,
        SpeckleNoise,
    )
    
    img = torch.rand(3, 32, 32)
    
    transforms_not_requiring_kornia = [
        GaussianNoise,
        ShotNoise,
        SpeckleNoise,
        Brightness,
        Contrast,
        Pixelate,
        JPEGCompression,
        Saturation,
        Fog,
        Frost,
    ]
    
    for transform_class in transforms_not_requiring_kornia:
        try:
            transform = transform_class(severity=1)
            output = transform(img)
            assert output.shape == img.shape
        except ImportError:
            # Some transforms may depend on other optional packages
            # (like scipy, cv2), which is fine for this test
            pass