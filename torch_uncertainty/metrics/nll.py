# fmt: on
import torch
import torch.nn.functional as F
from torchmetrics import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat


# fmt:off
class NegativeLogLikelihood(Metric):
    full_state_update: bool = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.add_state("log_probs", [], dist_reduce_fx="cat")
        self.add_state("targets", [], dist_reduce_fx="cat")

        rank_zero_warn(
            "Metric `NLLMetric` will save all targets and predictions"
            " in buffer. For large datasets this may lead to large memory"
            " footprint."
        )

    def update(self, probs: torch.Tensor, target: torch.Tensor) -> None:
        self.log_probs.append(torch.log(probs))
        self.targets.append(target)

    def compute(self) -> torch.Tensor:
        log_probs = dim_zero_cat(self.log_probs)
        targets = dim_zero_cat(self.targets)
        return F.nll_loss(log_probs, targets)


# class NegativeLogLikelihood(Metric):
#     is_differentiable: bool = False
#     higher_is_better: Optional[bool] = False
#     full_state_update: bool = False

#     def __init__(
#         self,
#         reduction: Literal["mean", "sum", "none", None] = "mean",
#         **kwargs: Any,
#     ) -> None:
#         super().__init__(**kwargs)

#         allowed_reduction = ("sum", "mean", "none", None)
#         if reduction not in allowed_reduction:
#             raise ValueError(
#                 "Expected argument `reduction` to be one of ",
#                 f"{allowed_reduction} but got {reduction}",
#             )

#         self.reduction = reduction

#         if self.reduction in ["mean", "sum"]:
#             self.add_state(
#                 "values",
#                 default=torch.tensor(0.0),
#                 dist_reduce_fx="sum",
#             )
#         else:
#             self.add_state("values", default=[], dist_reduce_fx="cat")
#         self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

#     def update(self, probs: torch.Tensor, target: torch.Tensor) -> None:
#         if self.reduction is None or self.reduction == "none":
#             self.values.append(F.nll_loss(probs, target, reduction="none"))
#         else:
#             self.values += F.nll_loss(probs, target, reduction="none").sum(
#                 dim=-1
#             )
#             self.total += target.size(0)

#     def compute(self) -> torch.Tensor:
#         """Computes NLL based on inputs passed in to ``update`` previously."""
#         values = dim_zero_cat(self.values)
#         # print(values.size())
#         if self.reduction == "sum":
#             return values.sum(dim=-1)
#         elif self.reduction == "mean":
#             return values.sum(dim=-1) / self.total
#         elif self.reduction is None or self.reduction == "none":
#             return values
#         else:
#             raise ValueError(
#                 "Expected argument `reduction` to be one of ",
#                 "['mean','sum','none',None] but got {reduction}",
#             )
