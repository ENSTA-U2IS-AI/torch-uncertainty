import copy

import torch
from torch import nn


class TrajectoryModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        save_schedule: list[int] | None = None,
    ) -> None:
        """Ensemble of models at different points in the training trajectory.

        Args:
            model (nn.Module): The model to train and ensemble.
            save_schedule (list[int]): The epochs at which to save the model.
                If save schedule is None, save the model at every epoch.
                Defaults to None.
        """
        super().__init__()
        self.model = model
        self.save_schedule = save_schedule

        self.saved_models = []

    @torch.no_grad()
    def save_model(self, epoch: int) -> None:
        """Save the model at the given epoch if included in the schedule.

        Args:
            epoch (int): The current epoch.
        """
        if self.save_schedule is None or epoch in self.save_schedule:
            self.saved_models.append(copy.deepcopy(self.model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement this method.")


class TrajectoryEnsemble(TrajectoryModel):
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
        """
        super().__init__(model, save_schedule)
        self.use_final_checkpoint = use_final_checkpoint
        self.num_estimators = int(use_final_checkpoint)

    @torch.no_grad()
    def save_model(self, epoch: int) -> None:
        """Save the model at the given epoch if included in the schedule.

        Args:
            epoch (int): The current epoch.
        """
        if self.save_schedule is None or epoch in self.save_schedule:
            self.saved_models.append(copy.deepcopy(self.model))
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
            return self.model.forward(x)
        preds = torch.cat(
            [model.forward(x) for model in self.saved_models], dim=0
        )
        if self.use_final_checkpoint:
            model_forward = self.model.forward(x)
            preds = torch.cat([model_forward, preds], dim=0)
        return preds

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.model.forward(x)
        return self.eval_forward(x)


def trajectory_ensemble(
    model: nn.Module,
    save_schedule: list[int],
) -> TrajectoryEnsemble:
    return TrajectoryEnsemble(model, save_schedule)
