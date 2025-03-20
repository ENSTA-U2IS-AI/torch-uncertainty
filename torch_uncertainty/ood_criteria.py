from abc import ABC, abstractmethod
from enum import Enum

import torch
from torch import Tensor, nn

from torch_uncertainty.metrics import MutualInformation, VariationRatio


class OODCriterionInputType(Enum):
    LOGIT = 1
    PROB = 2
    ESTIMATOR_PROB = 3


class TUOODCriterion(ABC, nn.Module):
    input_type: OODCriterionInputType
    ensemble_only = False

    @abstractmethod
    def forward(self, inputs: Tensor) -> Tensor:
        pass


class LogitCriterion(TUOODCriterion):
    input_type = OODCriterionInputType.LOGIT

    def forward(self, inputs: Tensor) -> Tensor:
        return -inputs.mean(dim=1).max(dim=-1).values


class EnergyCriterion(TUOODCriterion):
    input_type = OODCriterionInputType.LOGIT

    def forward(self, inputs: Tensor) -> Tensor:
        return -inputs.mean(dim=1).logsumexp(dim=-1)


class MaxSoftmaxProbabilityCriterion(TUOODCriterion):
    input_type = OODCriterionInputType.PROB

    def forward(self, inputs: Tensor) -> Tensor:
        return -inputs.max(-1)[0]


class EntropyCriterion(TUOODCriterion):
    input_type = OODCriterionInputType.ESTIMATOR_PROB

    def forward(self, inputs: Tensor) -> Tensor:
        return torch.special.entr(inputs).sum(dim=-1).mean(dim=1)


class MutualInformationCriterion(TUOODCriterion):
    ensemble_only = True
    input_type = OODCriterionInputType.ESTIMATOR_PROB

    def __init__(self) -> None:
        super().__init__()
        self.mi_metric = MutualInformation(reduction="none")

    def forward(self, inputs: Tensor) -> Tensor:
        return self.mi_metric(inputs)


class VariationRatioCriterion(TUOODCriterion):
    ensemble_only = True
    input_type = OODCriterionInputType.ESTIMATOR_PROB

    def __init__(self) -> None:
        super().__init__()
        self.vr_metric = VariationRatio(reduction="none", probabilistic=False)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.vr_metric(inputs.transpose(0, 1))
