from importlib import util
from typing import Literal

from torch import Tensor, nn
from torch.utils.data import Dataset

if util.find_spec("laplace"):
    from laplace import Laplace

    laplace_installed = True


class Laplace(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        task: Literal["classification", "regression"],
        subset_of_weights="last_layer",
        hessian_structure="kron",
        pred_type: Literal["glm", "nn"] = "glm",
        link_approx: Literal[
            "mc", "probit", "bridge", "bridge_norm"
        ] = "probit",
    ) -> None:
        """Laplace approximation for uncertainty estimation.

        This class is a wrapper of Laplace classes from the laplace-torch library.

        Args:
            model (nn.Module): model to be converted.
            task (Literal["classification", "regression"]): task type.
            subset_of_weights (str): subset of weights to be considered. Defaults to
                "last_layer".
            hessian_structure (str): structure of the Hessian matrix. Defaults to
                "kron".
            pred_type (Literal["glm", "nn"], optional): type of posterior predictive,
                See the Laplace library for more details. Defaults to "glm".
            link_approx (Literal["mc", "probit", "bridge", "bridge_norm"], optional):
                how to approximate the classification link function for the `'glm'`.
                See the Laplace library for more details. Defaults to "probit".

        Reference:
            Daxberger et al. Laplace Redux - Effortless Bayesian Deep Learning. In NeurIPS 2021.
        """
        super().__init__()
        if not laplace_installed:
            raise ImportError(
                "The laplace-torch library is not installed. Please install it via `pip install laplace-torch`."
            )
        self.la = Laplace(
            model=model,
            task=task,
            subset_of_weights=subset_of_weights,
            hessian_structure=hessian_structure,
        )
        self.pred_type = pred_type
        self.link_approx = link_approx

    def fit(self, dataset: Dataset) -> None:
        self.la.fit(dataset=dataset)

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        return self.la(
            x, pred_type=self.pred_type, link_approx=self.link_approx
        )
