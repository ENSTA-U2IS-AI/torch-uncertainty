# fmt: off
from pathlib import Path

import torch.nn as nn

from torch_uncertainty import cls_main, init_args
from torch_uncertainty.baselines.regression.mlp import MLP
from torch_uncertainty.datamodules.uci_regression import UCIDataModule
from torch_uncertainty.optimization_procedures import optim_regression

# fmt: on
if __name__ == "__main__":
    root = Path(__file__).parent.absolute().parents[2]

    args = init_args(MLP, UCIDataModule)

    net_name = "mlp-10neurons-2layers-kin8nm"

    # datamodule
    args.root = str(root / "data")
    dm = UCIDataModule(dataset_name="kin8nm", **vars(args))

    # model
    model = MLP(
        num_outputs=2,
        in_features=8,
        loss=nn.GaussianNLLLoss,
        optimization_procedure=optim_regression,
        **vars(args),
    )

    cls_main(model, dm, root, net_name, "regression", args)
