import torch
from lightning.pytorch.cli import LightningArgumentParser
from torch import nn

from torch_uncertainty import TULightningCLI
from torch_uncertainty.datamodules import UCIRegressionDataModule
from torch_uncertainty.models.mlp import mlp
from torch_uncertainty.routines.regression import RegressionRoutine


class MLPModel(RegressionRoutine):
    def __init__(
        self,
        output_dim: int,
        loss: nn.Module,
        in_features: int,
        hidden_dims: list[int],
        dropout_rate: float = 0.0,
        optim_recipe: dict | None = None,
        **kwargs,
    ) -> None:
        """MLP model for regression."""
        model = mlp(
            in_features=in_features,
            num_outputs=output_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
        )
        super().__init__(
            output_dim=output_dim,
            model=model,
            loss=loss,
            optim_recipe=optim_recipe,
            **kwargs,
        )
        self.save_hyperparameters(ignore=["loss"])


class MLPCLI(TULightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_optimizer_args(torch.optim.Adam)


def cli_main() -> MLPCLI:
    return MLPCLI(MLPModel, UCIRegressionDataModule)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    cli = cli_main()
    if (
        (not cli.trainer.fast_dev_run)
        and cli.subcommand == "fit"
        and cli._get(cli.config, "eval_after_fit")
    ):
        cli.trainer.test(datamodule=cli.datamodule, ckpt_path="best")
