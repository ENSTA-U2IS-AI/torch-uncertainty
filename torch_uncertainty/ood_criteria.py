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
    def forward(self, inputs: Tensor) -> Tensor:  # coverage: ignore
        pass


class MaxLogitCriterion(TUOODCriterion):
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


def get_ood_criterion(ood_criterion):
    if isinstance(ood_criterion, str):
        if ood_criterion == "logit":
            return MaxLogitCriterion()
        if ood_criterion == "energy":
            return EnergyCriterion()
        if ood_criterion == "msp":
            return MaxSoftmaxProbabilityCriterion()
        if ood_criterion == "entropy":
            return EntropyCriterion()
        if ood_criterion == "mutual_information":
            return MutualInformationCriterion()
        if ood_criterion == "variation_ratio":
            return VariationRatioCriterion()
        raise ValueError(
            "The OOD criterion must be one of 'msp', 'logit', 'energy', 'entropy',"
            f" 'mutual_information' or 'variation_ratio'. Got {ood_criterion}."
        )
    if isinstance(ood_criterion, type) and issubclass(ood_criterion, TUOODCriterion):
        return ood_criterion()
    raise ValueError(
        f"The OOD criterion should be a string or a subclass of TUOODCriterion. Got {type(ood_criterion)}."
    )
