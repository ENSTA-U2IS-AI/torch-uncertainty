import importlib

import numpy as np
import torch
from torch import nn

from torch_uncertainty.ood.nets import ASHNet, ReactNet, ScaleNet


class _TinyBackbone(nn.Module):
    """Minimal CNN backbone for testing."""

    def __init__(self, in_ch=3, num_classes=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # -> (B, 8, 1, 1)
        )
        self.feature_size = 8
        self.fc = nn.Linear(self.feature_size, num_classes)

    def forward(self, x, return_feature: bool = False, return_feature_list: bool = False):
        feat = self.conv(x).flatten(1)  # (B, 8)
        logits = self.fc(feat)  # (B, C)
        if return_feature:
            return logits, feat
        return logits

    def get_fc_layer(self):
        return self.fc


def _rand_batch(b=4, c=3, h=4, w=4, seed=0, eps=1e-3):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand((b, c, h, w), generator=g) + eps
    return x.float()


def test_ash_helpers_and_nets_cpu_exhaustive():
    device = torch.device("cpu")
    x = _rand_batch().to(device)

    ash_module = importlib.import_module(ASHNet.__module__)
    ash_b = getattr(ash_module, "ash_b", None)
    ash_p = getattr(ash_module, "ash_p", None)
    ash_s = getattr(ash_module, "ash_s", None)
    ash_rand = getattr(ash_module, "ash_rand", None)

    assert callable(ash_b), "ash_b not found next to ASHNet"
    assert callable(ash_p), "ash_p not found next to ASHNet"
    assert callable(ash_s), "ash_s not found next to ASHNet"
    assert callable(ash_rand), "ash_rand not found next to ASHNet"

    for pct in (0, 50):
        t = x.clone()
        outs = (
            ash_b(t.clone(), percentile=pct),
            ash_p(t.clone(), percentile=pct),
            ash_s(t.clone(), percentile=pct),
            ash_rand(t.clone(), percentile=pct, r1=0.0, r2=1.0),
        )
        for out in outs:
            assert out.shape == t.shape
            assert torch.isfinite(out).all(), f"Non-finite values at percentile {pct}"

    pct = 100
    t = x.clone()
    outs_100 = (
        ash_b(t.clone(), percentile=pct),
        ash_p(t.clone(), percentile=pct),
        ash_s(t.clone(), percentile=pct),
        ash_rand(t.clone(), percentile=pct, r1=0.0, r2=1.0),
    )
    for out in outs_100:
        assert out.shape == t.shape

    backbone = _TinyBackbone(in_ch=3, num_classes=3).to(device)

    ash_net = ASHNet(backbone)
    y_ash = ash_net.forward_threshold(x, percentile=70)
    assert y_ash.shape == (x.size(0), 3)
    assert torch.isfinite(y_ash).all()

    react_net = ReactNet(backbone)
    y_react = react_net.forward_threshold(x, threshold=1.0)
    assert y_react.shape == (x.size(0), 3)
    assert torch.isfinite(y_react).all()

    scale_net = ScaleNet(backbone)
    y_scale = scale_net.forward_threshold(x, percentile=65)
    assert y_scale.shape == (x.size(0), 3)
    assert torch.isfinite(y_scale).all()

    w, b = scale_net.get_fc()
    assert isinstance(w, np.ndarray)
    assert isinstance(b, np.ndarray)
    assert w.shape == (3, backbone.feature_size)
    assert b.shape == (3,)
