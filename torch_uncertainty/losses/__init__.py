# ruff: noqa: F401
from .bayesian import ELBOLoss, KLDiv
from .classification import (
    ConfidencePenaltyLoss,
    ConflictualLoss,
    DECLoss,
    FocalLoss,
)
from .regression import BetaNLL, DERLoss, DistributionNLLLoss
