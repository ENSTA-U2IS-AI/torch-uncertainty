import copy

import torch
from torch import nn


class TrajectoryModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
    ) -> None:
        """Updated model at different points in the training trajectory.

        Args:
            model (nn.Module): The inner model.
        """
        super().__init__()
        self.model = model
        self.num_estimators = 1

    def update_model(self, epoch: int) -> None:
        """Update the model.

        Args:
            epoch (int): The current epoch.
        """
        raise NotImplementedError("Subclasses must implement this method.")

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
        super().__init__(model)
        self.save_schedule = save_schedule
        self.use_final_checkpoint = use_final_checkpoint
        self.num_estimators = int(use_final_checkpoint)
        self.saved_models = []

    @torch.no_grad()
    def update_model(self, epoch: int) -> None:
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


class EMA(TrajectoryModel):
    def __init__(
        self,
        model: nn.Module,
        momentum: float,
    ) -> None:
        """Exponential moving average model.

        Args:
            model (nn.Module): The model to train and ensemble.
            momentum (float): The momentum of the moving average.
        """
        super().__init__(model)
        self.ema_model = None
        self.momentum = momentum
        self.remainder = 1 - momentum

    def update_model(self, epoch: int) -> None:
        """Update the EMA model.

        Args:
            epoch (int): The current epoch. For API consistency.
        """
        if self.ema_model is None:
            self.ema_model = copy.deepcopy(self.model)
        else:
            for ema_param, param in zip(
                self.ema_model.parameters(),
                self.model.parameters(),
                strict=False,
            ):
                ema_param.data = (
                    ema_param.data * self.momentum + param.data * self.remainder
                )

    def eval_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ema_model is None:
            return self.model.forward(x)
        return self.ema_model.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.model.forward(x)
        return self.eval_forward(x)


class SWA(TrajectoryModel):
    def __init__(
        self,
        model: nn.Module,
        cycle_start: int,
        cycle_length: int,
    ) -> None:
        super().__init__(model)
        self.cycle_start = cycle_start
        self.cycle_length = cycle_length
        self.num_averaged = 0
        self.swa_model = None
        self.need_bn_update = False

    @torch.no_grad()
    def update_model(self, epoch: int) -> None:
        if (
            epoch >= self.cycle_start
            and (epoch - self.cycle_start) % self.cycle_length == 0
        ):
            if self.swa_model is None:
                self.swa_model = copy.deepcopy(self.model)
                self.num_averaged = 1
            else:
                for swa_param, param in zip(
                    self.swa_model.parameters(),
                    self.model.parameters(),
                    strict=False,
                ):
                    swa_param.data += (param.data - swa_param.data) / (
                        self.num_averaged + 1
                    )
            self.num_averaged += 1
            self.need_bn_update = True

    def eval_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.swa_model is None:
            return self.model.forward(x)
        return self.swa_model.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.model.forward(x)
        return self.eval_forward(x)
