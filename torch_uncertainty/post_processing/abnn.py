import copy

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from torch_uncertainty.layers.bayesian.abnn import BatchNormAdapter2d
from torch_uncertainty.models import deep_ensembles
from torch_uncertainty.optim_recipes import optim_abnn
from torch_uncertainty.routines import ClassificationRoutine
from torch_uncertainty.utils import TUTrainer

from .abstract import PostProcessing


class ABNN(PostProcessing):
    def __init__(
        self,
        num_classes: int,
        random_prior: float,
        alpha: float,
        num_models: int,
        num_samples: int,
        base_lr: float,
        device: torch.device | str,
        max_epochs: int = 5,
        use_original_model: bool = True,
        batch_size: int = 128,
        precision: str = "32",
        model: nn.Module | None = None,
    ):
        """ABNN post-processing.

        Args:
            num_classes (int): Number of classes of the inner model.
            random_prior (float): Random prior specializing estimators on
                certain classes.
            alpha (float): Alpha value for ABNN to control the diversity of
                the predictions.
            num_models (int): Number of stochastic models.
            num_samples (int): Number of samples per model.
            base_lr (float): Base learning rate.
            device (torch.device): Device to use.
            max_epochs (int, optional): Number of training epochs. Defaults
                to 5.
            use_original_model (bool, optional): Use original model during
                evaluation. Defaults to True.
            batch_size (int, optional): Batch size for the training of ABNN.
                Defaults to 128.
            precision (str, optional): Machine precision for training & eval.
                Defaults to "32".
            model (nn.Module | None, optional): Model to use. Defaults to None.

        Reference:

        """
        super().__init__(model)
        _abnn_checks(
            num_classes=num_classes,
            random_prior=random_prior,
            alpha=alpha,
            max_epochs=max_epochs,
            num_models=num_models,
            num_samples=num_samples,
            base_lr=base_lr,
            batch_size=batch_size,
        )
        self.num_classes = num_classes
        self.alpha = alpha
        self.base_lr = base_lr
        self.num_models = num_models
        self.num_samples = num_samples
        self.total_models = num_models + int(use_original_model)
        self.use_original_model = use_original_model
        self.max_epochs = max_epochs

        self.batch_size = batch_size
        self.precision = precision
        self.device = device

        self.final_model = None

        # Build random prior
        num_rp_classes = int(num_classes**0.5)
        self.weights = []
        for _ in range(num_models):
            weight = torch.ones([num_classes])
            weight[torch.randperm(num_classes)[:num_rp_classes]] += (
                random_prior - 1
            )
            self.weights.append(weight)

    def fit(self, dataset: Dataset) -> None:
        if self.model is None:
            raise ValueError("Model must be set before fitting.")
        dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        source_model = copy.deepcopy(self.model)
        _replace_bn_layers(source_model, self.alpha)

        models = [copy.deepcopy(source_model) for _ in range(self.num_models)]

        baselines = [
            ClassificationRoutine(
                num_classes=self.num_classes,
                model=mod,
                loss=nn.CrossEntropyLoss(
                    weight=self.weights[i].to(device=self.device)
                ),
                optim_recipe=optim_abnn(mod, lr=self.base_lr),
                eval_ood=True,
            )
            for i, mod in enumerate(models)
        ]

        for baseline in baselines:
            trainer = TUTrainer(
                max_epochs=self.max_epochs,
                accelerator=self.device,
                enable_progress_bar=False,
                precision=self.precision,
                enable_checkpointing=False,
                logger=None,
                enable_model_summary=False,
            )
            trainer.fit(model=baseline, train_dataloaders=dl)

        final_models = (
            [copy.deepcopy(source_model) for _ in range(self.num_samples)]
            if self.use_original_model
            else []
        )
        for baseline in baselines:
            model = copy.deepcopy(source_model)
            model.load_state_dict(baseline.model.state_dict())
            final_models.extend(
                [copy.deepcopy(model) for _ in range(self.num_samples)]
            )

        self.final_model = deep_ensembles(final_models)

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        if self.final_model is not None:
            return self.final_model(x)
        if self.model is not None:
            return self.model(x)
        raise ValueError("Model must be set before calling forward.")


def _abnn_checks(
    num_classes,
    random_prior,
    alpha,
    max_epochs,
    num_models,
    num_samples,
    base_lr,
    batch_size,
) -> None:
    if random_prior < 0:
        raise ValueError(
            f"random_prior must be greater than 0. Got {random_prior}."
        )
    if batch_size < 1:
        raise ValueError(
            f"batch_size must be greater than 0. Got {batch_size}."
        )
    if max_epochs < 1:
        raise ValueError(f"epoch must be greater than 0. Got {max_epochs}.")
    if num_models < 1:
        raise ValueError(
            f"num_models must be greater than 0. Got {num_models}."
        )
    if num_samples < 1:
        raise ValueError(
            f"num_samples must be greater than 0. Got {num_samples}."
        )
    if alpha < 0:
        raise ValueError(f"alpha must be greater than 0. Got {alpha}.")
    if base_lr < 0:
        raise ValueError(f"base_lr must be greater than 0. Got {base_lr}.")
    if num_classes < 1:
        raise ValueError(
            f"num_classes must be greater than 0. Got {num_classes}."
        )


def _replace_bn_layers(model: nn.Module, alpha: float) -> None:
    """Recursively replace batch normalization layers with ABNN layers.

    Args:
        model (nn.Module): Model to replace batch normalization layers.
        alpha (float): Alpha value for ABNN.
    """
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            _replace_bn_layers(module, alpha)
        if isinstance(module, nn.BatchNorm2d) and module.track_running_stats:
            num_channels = module.num_features
            new_module = BatchNormAdapter2d(num_channels, alpha=alpha)
            new_module.running_mean = module.running_mean
            new_module.running_var = module.running_var
            new_module.num_batches_tracked = module.num_batches_tracked

            new_module.weight.data = module.weight.data
            new_module.bias.data = module.bias.data
            setattr(model, name, new_module)
        else:
            _replace_bn_layers(module, alpha)
