import copy

import torch
from torch import nn


class CheckpointEnsemble(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        save_schedule: list[int] | None = None,
        use_final_checkpoint: bool = True,
    ) -> None:
        """Ensemble of models at different points in the training trajectory.

        Args:
            model (nn.Module): The model to train and ensemble.
            save_schedule (list[int]): The epochs at which to save the model.
                If save schedule is None, save the model at every epoch.
                Defaults to None.
            use_final_checkpoint (bool, optional): Whether to use the final
                model as a checkpoint. Defaults to True.

        Reference:
            Checkpoint Ensembles: Ensemble Methods from a Single Training Process.
            Hugh Chen, Scott Lundberg, Su-In Lee. In ArXiv 2018.
        """
        super().__init__()
        self.core_model = model
        self.save_schedule = save_schedule
        self.use_final_checkpoint = use_final_checkpoint
        self.num_estimators = int(use_final_checkpoint)
        self.saved_models = []
        self.num_estimators = 1

    @torch.no_grad()
    def update_wrapper(self, epoch: int) -> None:
        """Save the model at the given epoch if included in the schedule.

        Args:
            epoch (int): The current epoch.
        """
        if self.save_schedule is None or epoch in self.save_schedule:
            self.saved_models.append(copy.deepcopy(self.core_model))
            self.num_estimators += 1

    def eval_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for evaluation.

        If the model is in evaluation mode, this method will return the
        ensemble prediction. Otherwise, it will return the prediction of the
        current model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The model or ensemble output.
        """
        if not len(self.saved_models):
            return self.core_model.forward(x)
        preds = torch.cat(
            [model.forward(x) for model in self.saved_models], dim=0
        )
        if self.use_final_checkpoint:
            model_forward = self.core_model.forward(x)
            preds = torch.cat([model_forward, preds], dim=0)
        return preds

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.core_model.forward(x)
        return self.eval_forward(x)
