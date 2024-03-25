from lightning.pytorch import Trainer
from lightning.pytorch.trainer.states import (
    RunningStage,
    TrainerFn,
)

from torch_uncertainty.utils.evaluation_loop import TUEvaluationLoop


class TUTrainer(Trainer):
    def __init__(self, inference_mode: bool = True, **kwargs):
        super().__init__(inference_mode=inference_mode, **kwargs)

        self.test_loop = TUEvaluationLoop(
            self,
            TrainerFn.TESTING,
            RunningStage.TESTING,
            inference_mode=inference_mode,
        )
