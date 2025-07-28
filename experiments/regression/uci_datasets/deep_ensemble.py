# NOTE: This experiment has been temporarily disabled because it relied on the
# DeepEnsemblesBaseline which was part of the removed baselines abstraction layer.
# To re-enable, refactor to use RegressionRoutine directly with deep_ensembles
# from torch_uncertainty.models

# from pathlib import Path
# 
# from torch_uncertainty import cli_main, init_args
# from torch_uncertainty.baselines import DeepEnsemblesBaseline
# from torch_uncertainty.datamodules import UCIRegressionDataModule

if __name__ == "__main__":
    raise NotImplementedError(
        "This experiment needs to be refactored after baseline removal. "
        "Use RegressionRoutine with deep_ensembles directly."
    )

if __name__ == "__main__":
    args = init_args(DeepEnsemblesBaseline, UCIRegressionDataModule)
    if args.root == "./data/":
        root = Path(__file__).parent.absolute().parents[2]
    else:
        root = Path(args.root)

    net_name = f"de-{args.backbone}-kin8nm"

    # datamodule
    args.root = str(root / "data")
    dm = UCIRegressionDataModule(dataset_name="kin8nm", **vars(args))

    # model
    args.task = "regression"
    model = DeepEnsemblesBaseline(
        **vars(args),
    )

    args.test = -1

    cli_main(model, dm, root, net_name, args)
