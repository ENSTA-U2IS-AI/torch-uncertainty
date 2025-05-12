import copy

import torch
from torch import Tensor, nn


class CheckpointEnsemble(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        save_schedule: list[int] | None = None,
        use_final_model: bool = True,
        store_on_cpu: bool = False,
    ) -> None:
        """Ensemble of models at different points in the training trajectory.

        Args:
            model (nn.Module): The model to train and ensemble.
            save_schedule (list[int] | None): The epochs at which to save the model. If save schedule is
                ``None``, save the model at every epoch. Defaults to ``None``.
            use_final_model (bool): Whether to use the final model as a checkpoint. Defaults to ``True``.
            store_on_cpu (bool): Whether to put the models on the CPU when unused. Defaults to ``False``.

        Reference:
            Checkpoint Ensembles: Ensemble Methods from a Single Training Process.
            Hugh Chen, Scott Lundberg, Su-In Lee. In ArXiv 2018.
        """
        super().__init__()
        self.core_model = model
        self.save_schedule = save_schedule
        self.use_final_model = use_final_model
        self.store_on_cpu = store_on_cpu
        self.num_estimators = int(use_final_model)
        self.saved_models = nn.ModuleList()

    @torch.no_grad()
    def update_wrapper(self, epoch: int) -> None:
        """Save the model at the given epoch if included in the schedule.

        Args:
            epoch (int): The current epoch.
        """
        if self.save_schedule is None or epoch in self.save_schedule:
            self.saved_models.append(
                copy.deepcopy(self.core_model)
                if not self.store_on_cpu
                else copy.deepcopy(self.core_model).cpu()
            )
            self.num_estimators += 1

    def eval_forward(self, x: Tensor) -> Tensor:
        """Forward pass for evaluation.

        If the model is in evaluation mode, this method will return the
        ensemble prediction. Otherwise, it will return the prediction of the
        current model.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The model or ensemble output.
        """
        preds: list[Tensor] = []
        if not len(self.saved_models):
            return self.core_model.forward(x)
        if self.store_on_cpu:
            for model in self.saved_models:
                model.to(x.device)
                preds.append(model.forward(x))
                model.cpu()
        else:
            preds = [model.forward(x) for model in self.saved_models]
        if self.use_final_model:
            preds.append(self.core_model.forward(x))
        return torch.cat(preds, dim=0)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            return self.core_model.forward(x)
        return self.eval_forward(x)

    def to(self, *args, **kwargs):
        device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)[:3]

        if self.store_on_cpu:
            device = torch.device("cpu")
        return super().to(device=device, dtype=dtype, non_blocking=non_blocking)
