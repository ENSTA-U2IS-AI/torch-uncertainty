from einops import rearrange
from torch import Tensor, nn


class ChannelBack(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return rearrange(x, "b c h w -> b h w c")


class ChannelFront(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return rearrange(x, "b h w c -> b c h w")
