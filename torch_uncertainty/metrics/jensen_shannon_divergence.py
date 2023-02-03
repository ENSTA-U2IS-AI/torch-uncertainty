import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat


class JensenShannonDivergence(Metric):
    full_state_update: bool = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.add_state("probs", [], dist_reduce_fx="cat")
        self.kl = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

        rank_zero_warn(
            "Metric `JensenShannonDivergence` will save all "
            "predictions in buffer. For large datasets this may lead to large "
            "memory footprint."
        )

    def update(self, probs: Tensor) -> None:  # type: ignore
        # store data as (example, estimator, class)
        self.probs.append(probs.transpose(0, 1))

    def compute(self) -> Tensor:
        probs = dim_zero_cat(self.probs)
        mean_proba = probs.mean(1, keepdim=True).repeat(1, probs.shape[1], 1)

        return (
            F.kl_div(
                mean_proba.log(),
                probs.log(),
                log_target=True,
                reduction="batchmean",
            )
            / probs.shape[1]
        )
