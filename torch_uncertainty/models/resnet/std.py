# fmt: off
from typing import List, Type, Union

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, Linear


# fmt: on
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        groups: int = 1,
    ):
        super(BasicBlock, self).__init__()

        # No subgroups for the first layer
        self.conv1 = Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    groups=groups,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, groups=1):
        super(Bottleneck, self).__init__()

        # No subgroups for the first layer
        self.conv1 = Conv2d(
            in_planes,
            planes,
            kernel_size=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = Conv2d(
            planes,
            self.expansion * planes,
            kernel_size=1,
            groups=groups,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    groups=groups,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class _ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        num_blocks: List[int],
        in_channels: int,
        num_classes: int,
        groups: int,
        # dataset: str = "cifar",
    ) -> None:
        super().__init__()
        # assert dataset in [
        #     "cifar",
        #     "mnist",
        #     "tinyimagenet",
        #     "imagenet",
        # ], "The dataset is not taken in charge by this implementation."
        # self.dataset = dataset
        self.in_planes = 64
        block_planes = self.in_planes

        # No subgroups in the first layer
        # if self.dataset == "imagenet":
        #     self.conv1 = Conv2d(
        #         3 * self.num_estimators,
        #         block_planes,
        #         kernel_size=7,
        #         stride=2,
        #         padding=3,
        #         groups=1,
        #         num_estimators=num_estimators,
        #         bias=False,
        #     )
        # elif self.dataset == "mnist":
        #     self.conv1 = Conv2d(
        #         1 * self.num_estimators,
        #         block_planes,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1,
        #         groups=1,
        #         num_estimators=num_estimators,
        #         bias=False,
        #     )
        # else:
        self.conv1 = Conv2d(
            in_channels,
            block_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(block_planes)

        self.optional_pool = nn.Identity()

        self.layer1 = self._make_layer(
            block,
            block_planes,
            num_blocks[0],
            stride=1,
            groups=groups,
        )
        self.layer2 = self._make_layer(
            block,
            block_planes * 2,
            num_blocks[1],
            stride=2,
            groups=groups,
        )
        self.layer3 = self._make_layer(
            block,
            block_planes * 4,
            num_blocks[2],
            stride=2,
            groups=groups,
        )
        self.layer4 = self._make_layer(
            block,
            block_planes * 8,
            num_blocks[3],
            stride=2,
            groups=groups,
        )

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten(1)

        self.linear = Linear(
            block_planes * 8 * block.expansion,
            num_classes,
        )

    def _make_layer(
        self,
        block: nn.Module,
        planes: int,
        num_blocks: int,
        stride: int,
        groups: int,
    ):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    in_planes=self.in_planes,
                    planes=planes,
                    stride=stride,
                    groups=groups,
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.optional_pool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
        out = self.flatten(out)
        out = self.linear(out)
        return out


def ResNet18(
    in_channels: int,
    num_classes: int,
    groups: int,
) -> _ResNet:
    return _ResNet(
        block=BasicBlock,
        num_blocks=[2, 2, 2, 2],
        in_channels=in_channels,
        num_classes=num_classes,
        groups=groups,
    )


def ResNet34(
    in_channels: int,
    num_classes: int,
    groups: int,
) -> _ResNet:
    return _ResNet(
        block=BasicBlock,
        num_blocks=[3, 4, 6, 3],
        in_channels=in_channels,
        groups=groups,
        num_classes=num_classes,
    )


def ResNet50(
    in_channels: int,
    num_classes: int,
    groups: int,
) -> _ResNet:
    return _ResNet(
        block=Bottleneck,
        num_blocks=[3, 4, 6, 3],
        in_channels=in_channels,
        groups=groups,
        num_classes=num_classes,
    )


def ResNet101(
    in_channels: int,
    num_classes: int,
    groups: int,
) -> _ResNet:
    return _ResNet(
        block=Bottleneck,
        num_blocks=[3, 4, 23, 3],
        in_channels=in_channels,
        groups=groups,
        num_classes=num_classes,
    )


def ResNet152(
    in_channels: int,
    num_classes: int,
    groups: int,
) -> _ResNet:
    return _ResNet(
        block=Bottleneck,
        num_blocks=[3, 8, 36, 3],
        in_channels=in_channels,
        groups=groups,
        num_classes=num_classes,
    )
