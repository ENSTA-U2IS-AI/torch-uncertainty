from importlib import util
from typing import Literal

from torch import Tensor, nn
from torch.utils.data import DataLoader

from .abstract import PostProcessing

if util.find_spec("laplace"):
    from laplace import Laplace

    laplace_installed = True
else:  # coverage: ignore
    laplace_installed = False


class LaplaceApprox(PostProcessing):
    def __init__(
        self,
        task: Literal["classification", "regression"],
        model: nn.Module | None = None,
        weight_subset: str = "last_layer",
        hessian_struct: str = "kron",
        pred_type: Literal["glm", "nn"] = "glm",
        link_approx: Literal["mc", "probit", "bridge", "bridge_norm"] = "probit",
        optimize_prior_precision: bool = True,
    ) -> None:
        """Laplace approximation for uncertainty estimation.

        This class is a wrapper of Laplace classes from the laplace-torch library.

        Args:
            task (Literal[``"classification"``, ``"regression"``]): task type.
            model (nn.Module): model to be converted.
            weight_subset (str): subset of weights to be considered. Defaults to
                "last_layer".
            hessian_struct (str): structure of the Hessian matrix. Defaults to
                "kron".
            pred_type (Literal["glm", "nn"], optional): type of posterior predictive,
                See the Laplace library for more details. Defaults to "glm".
            link_approx (Literal["mc", "probit", "bridge", "bridge_norm"], optional):
                how to approximate the classification link function for the `'glm'`.
                See the Laplace library for more details. Defaults to "probit".
            optimize_prior_precision (bool, optional): whether to optimize the prior
                precision. Defaults to True.

        References:
            [1] `Daxberger et al. Laplace Redux - Effortless Bayesian Deep Learning. In NeurIPS 2021
            <https://arxiv.org/abs/2106.14806>`_.
        """
        super().__init__()
        if not laplace_installed:
            raise ImportError(
                "The laplace-torch library is not installed. Please install "
                "torch_uncertainty with the all option: pip install -U torch_uncertainty[all]"
            )

        self.pred_type = pred_type
        self.link_approx = link_approx
        self.task = task
        self.weight_subset = weight_subset
        self.hessian_struct = hessian_struct
        self.optimize_prior_precision = optimize_prior_precision

        if model is not None:
            self.set_model(model)

    def set_model(self, model: nn.Module) -> None:
        super().set_model(model)
        self.la = Laplace(
            model=model,
            likelihood=self.task,
            subset_of_weights=self.weight_subset,
            hessian_structure=self.hessian_struct,
        )

    def fit(self, dataloader: DataLoader) -> None:
        self.la.fit(train_loader=dataloader)
        if self.optimize_prior_precision:
            self.la.optimize_prior_precision(method="marglik")

    def forward(
        self,
        inputs: Tensor,
    ) -> Tensor:
        return self.la(inputs, pred_type=self.pred_type, link_approx=self.link_approx)
