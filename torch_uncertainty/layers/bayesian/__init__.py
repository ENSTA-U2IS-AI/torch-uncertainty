# ruff: noqa: F401
from .bayes_conv import BayesConv1d, BayesConv2d, BayesConv3d
from .bayes_linear import BayesLinear

bayesian_modules = (BayesConv1d, BayesConv2d, BayesConv3d, BayesLinear)
