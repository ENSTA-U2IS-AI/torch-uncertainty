from typing import Any

from torch import nn


class Identity(nn.Module):
    # ruff: noqa: ARG002
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def forward(self, *args) -> Any:
        return args
