# ruff: noqa: F401
from .checkpoint_ensemble import (
    CheckpointEnsemble,
)
from .deep_ensembles import deep_ensembles
from .ema import EMA
from .mc_dropout import MCDropout, mc_dropout
from .swa import SWA

STEP_UPDATE_MODEL = (EMA,)
EPOCH_UPDATE_MODEL = (SWA, CheckpointEnsemble)