import torch

from tests._dummies.model import dummy_model
from torch_uncertainty.models import CheckpointEnsemble


class TestCheckpointEnsemble:
    """Testing the CheckpointEnsemble class."""

    def test_training(self):
        ens = CheckpointEnsemble(dummy_model(1, 10))
        ens.eval()
        ens(torch.randn(1, 1))

        ens.train()
        ens(torch.randn(1, 1))
        ens.update_wrapper(0)
        ens.eval()
        ens(torch.randn(1, 1))

        ens = CheckpointEnsemble(dummy_model(1, 10), use_final_checkpoint=False)
        ens.train()
        ens(torch.randn(1, 1))
        ens.update_wrapper(0)
        ens.eval()
        ens(torch.randn(1, 1))
