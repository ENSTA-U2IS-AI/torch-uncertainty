# ruff: noqa: F401
from .bayes_conv import BayesConv1d, BayesConv2d, BayesConv3d
from .bayes_linear import BayesLinear
from .lpbnn import LPBNNConv2d, LPBNNLinear

bayesian_modules = (
    BayesConv1d,
    BayesConv2d,
    BayesConv3d,
    BayesLinear,
    LPBNNLinear,
    LPBNNConv2d,
)
