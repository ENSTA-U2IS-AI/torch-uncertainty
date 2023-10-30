from pathlib import Path

from torch_uncertainty import cli_main, init_args
from torch_uncertainty.baselines import DeepEnsembles
from torch_uncertainty.datamodules import UCIDataModule

if __name__ == "__main__":
    args = init_args(DeepEnsembles, UCIDataModule)
    if args.root == "./data/":
        root = Path(__file__).parent.absolute().parents[2]
    else:
        root = Path(args.root)

    net_name = f"de-{args.backbone}-kin8nm"

    # datamodule
    args.root = str(root / "data")
    dm = UCIDataModule(dataset_name="kin8nm", **vars(args))

    # model
    args.task = "regression"
    model = DeepEnsembles(
        **vars(args),
    )

    args.test = -1

    cli_main(model, dm, root, net_name, args)
