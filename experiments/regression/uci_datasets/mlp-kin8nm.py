from pathlib import Path

from torch import nn, optim

from torch_uncertainty import cli_main, init_args
from torch_uncertainty.baselines.regression.mlp import MLP
from torch_uncertainty.datamodules import UCIDataModule


def optim_regression(
    model: nn.Module,
    learning_rate: float = 5e-3,
) -> dict:
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0,
    )
    return {
        "optimizer": optimizer,
    }


if __name__ == "__main__":
    args = init_args(MLP, UCIDataModule)
    if args.root == "./data/":
        root = Path(__file__).parent.absolute().parents[2]
    else:
        root = Path(args.root)

    net_name = "mlp-kin8nm"

    # datamodule
    args.root = str(root / "data")
    dm = UCIDataModule(dataset_name="kin8nm", **vars(args))

    # model
    model = MLP(
        num_outputs=2,
        in_features=8,
        hidden_dims=[100],
        loss=nn.GaussianNLLLoss,
        optimization_procedure=optim_regression,
        dist_estimation=2,
        **vars(args),
    )

    cli_main(model, dm, root, net_name, args)
