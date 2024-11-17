# ruff: noqa: F401
from .bayesian import ELBOLoss, KLDiv
from .classification import (
    BCEWithLogitsLSLoss,
    ConfidencePenaltyLoss,
    ConflictualLoss,
    DECLoss,
    FocalLoss,
)
from .regression import BetaNLL, DERLoss, DistributionNLLLoss
