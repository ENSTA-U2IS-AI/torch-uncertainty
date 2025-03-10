import torch
from torch import nn

from torch_uncertainty.models.wrappers.batch_ensemble import BatchEnsemble


# Define a simple model for testing wrapper functionality (disregarding the actual BatchEnsemble architecture)
class _DummyModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.r_group = nn.Parameter(torch.randn(in_features))
        self.s_group = nn.Parameter(torch.randn(out_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        return self.fc(x)


class TestBatchEnsembleModel:
    def test_forward_pass(self):
        in_features = 10
        out_features = 5
        num_estimators = 3
        model = _DummyModel(in_features, out_features)
        wrapped_model = BatchEnsemble(model, num_estimators)

        # Test forward pass
        x = torch.randn(2, in_features)  # Batch size of 2
        logits = wrapped_model(x)
        assert logits.shape == (2 * num_estimators, out_features)
