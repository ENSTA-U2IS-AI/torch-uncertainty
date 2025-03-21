"""
Packed ensembles Segmentation Tutorial using Muad Dataset
==========================================================

This tutorial demonstrates how to train a segmentation model on the MUAD dataset using TorchUncertainty.
MUAD is a synthetic dataset designed for evaluating autonomous driving under diverse uncertainties.
It includes **10,413 images** across training, validation, and test sets, featuring adverse weather,
lighting conditions, and out-of-distribution (OOD) objects. The dataset supports tasks like semantic segmentation,
depth estimation, and object detection.

For details and access, visit the `MUAD Website <https://muad-dataset.github.io/>`_.

"""
# %% 
# 1. Load Muad dataset using Torch Uncertainty
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# %% 
# Let's start by defining the training parameters.

batch_size = 10
learning_rate =1e-3
weight_decay=2e-4
lr_decay_epochs=20
lr_decay=0.1
nb_epochs=50

# %% 
# In this Tutorial we are using the small version a bigger version can be specified with keyword "full" instead of small.


import torch
from einops import rearrange
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F
import matplotlib.pyplot as plt

from torch_uncertainty.datasets import MUAD

train_transform = v2.Compose(
    [
        v2.Resize(size=(256, 512), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(
            dtype={
                tv_tensors.Image: torch.float32,
                tv_tensors.Mask: torch.int64,
                "others": None,
            },
            scale=True,
        ),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transform = v2.Compose(
    [
        v2.Resize(size=(256, 512), antialias=True),
        v2.ToDtype(
            dtype={
                tv_tensors.Image: torch.float32,
                tv_tensors.Mask: torch.int64,
                "others": None,
            },
            scale=True,
        ),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

train_set = MUAD(root="./data", target_type="semantic", version="small", split="train" , transforms=train_transform, download=True)
val_set = MUAD(root="./data", target_type="semantic", version="small", split="val" , transforms=val_transform, download=True)
test_set = MUAD(root="./data", target_type="semantic", version="small", split="test" , transforms=val_transform, download=True)

# %% 
# Visualize a validation input sample (and RGB image)

# Undo normalization on the image and convert to uint8.
sample = train_set[0]
img, tgt = sample
mean = torch.tensor([0.485, 0.456, 0.406], device=img.device)
std = torch.tensor([0.229, 0.224, 0.225], device=img.device)
img = img * std[:, None, None] + mean[:, None, None]
img = F.to_dtype(img, torch.uint8, scale=True)
img_pil = F.to_pil_image(img)

plt.figure(figsize=(6,6))
plt.imshow(img_pil)
plt.axis("off") 
plt.show()

# %% 
# Visualize the same image above but segmented.

from torchvision.utils import draw_segmentation_masks

tmp_tgt = tgt.masked_fill(tgt == 255, 21)
tgt_masks = tmp_tgt == torch.arange(22, device=tgt.device)[:, None, None]
img_segmented = draw_segmentation_masks(img, tgt_masks, alpha=1, colors=val_set.color_palette)
img_pil = F.to_pil_image(img)

plt.figure(figsize=(6,6))
plt.imshow(img_pil)
plt.axis("off") 
plt.show()

# %% 
# Below is the complete list of classes in MUAD, presented as:
# 
# 1.   Class Name
# 2.   Train ID
# 3.   Segmentation Color in RGB format [R,G, B].

for muad_class in train_set.classes:
    class_name = muad_class.name
    train_id = muad_class.id
    color = muad_class.color
    print(f"Class: {class_name}, Train ID: {train_id}, Color: {color}")

# %% 
# Let's now calculate each class weight

import numpy as np
import torch
from torch.utils.data import DataLoader

train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4)

val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4)

test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4)


def enet_weighing(dataloader, num_classes, c=1.02):
    """Computes class weights as described in the ENet paper.

        w_class = 1 / (ln(c + p_class)),

    where c is usually 1.02 and p_class is the propensity score of that
    class:

        propensity_score = freq_class / total_pixels.

    References:
        https://arxiv.org/abs/1606.02147

    Args:
        dataloader (``data.Dataloader``): A data loader to iterate over the
            dataset.
        num_classes (``int``): The number of classes.
        c (``int``, optional): AN additional hyper-parameter which restricts
            the interval of values for the weights. Default: 1.02.

    """
    class_count = 0
    total = 0
    for _, label in dataloader:
      label = label.cpu().numpy()
      # Flatten label
      flat_label = label.flatten()
      flat_label = flat_label[flat_label != 255]

      # Sum up the number of pixels of each class and the total pixel
      # counts for each label
      class_count += np.bincount(flat_label, minlength=num_classes)
      total += flat_label.size

    # Compute propensity score and then the weights for each class
    propensity_score = class_count / total
    return 1 / (np.log(c + propensity_score))

print("\nComputing class weights...")
print("(this can take a while depending on the dataset size)")
class_weights = enet_weighing(train_loader, 19)
class_weights = torch.from_numpy(class_weights).float()
print("Class weights:", class_weights)

# %% 
# 2. Building the DNN
# ~~~~~~~~~~~~~~~~~~~

# %%
from torch import nn


class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.mpconv(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        self.bilinear = bilinear

        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = F.resize(x1, size=[2*x1.size()[2],2*x1.size()[3]],
                          interpolation=v2.InterpolationMode.BILINEAR)
        else:
            x1 = self.up(x1)

        # input is CHW
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)

#please note that we have added dropout layer to be abble to use MC dropout

class UNet(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.inc = InConv(3, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 256)
        self.up1 = Up(512, 128)
        self.up2 = Up(256, 64)
        self.up3 = Up(128, 32)
        self.up4 = Up(64, 32)
        self.dropout = nn.Dropout2d(0.1)
        self.outc = OutConv(32, classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.dropout(x)
        x = self.up2(x, x3)
        x = self.dropout(x)
        x = self.up3(x, x2)
        x = self.dropout(x)
        x = self.up4(x, x1)
        x = self.dropout(x)
        return self.outc(x)










# %% 
# 3. Train a Packed-Ensembles
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Using Torch Uncertainty we will easily train a `Packed-Ensembles <https://arxiv.org/pdf/2210.09184>`_ on our segmentation task.
# 

# %%
from torch import Tensor, nn
from torch.nn.common_types import _size_2_t

from torch_uncertainty.layers.packed import check_packed_parameters_consistency


class PackedConvTranspose2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        alpha: int,
        num_estimators: int,
        gamma: int = 1,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        output_padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        minimum_channels_per_group: int = 64,
        bias: bool = True,
        first: bool = False,
        last: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        r"""Packed-Ensembles-style ConvTranspose2d layer with debug flags.

        Args:
            in_channels (int): Number of channels in the input.
            out_channels (int): Number of channels produced by the transposed convolution.
            kernel_size (int or tuple): Size of the convolving kernel.
            alpha (int): The channel multiplier for the layer.
            num_estimators (int): Number of estimators in the ensemble.
            gamma (int, optional): Defaults to ``1``.
            stride (int or tuple, optional): Stride of the convolution. Defaults to ``1``.
            padding (int or tuple, optional): Zero-padding added to both sides of the input. Defaults to ``0``.
            output_padding (int or tuple, optional): Additional size added to one side of the output shape. Defaults to ``0``.
            dilation (int or tuple, optional): Spacing between kernel elements. Defaults to ``1``.
            groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to ``1``.
            minimum_channels_per_group (int, optional): Smallest possible number of channels per group.
            bias (bool, optional): If ``True``, adds a learnable bias to the output. Defaults to ``True``.
            first (bool, optional): Whether this is the first layer of the network. Defaults to ``False``.
            last (bool, optional): Whether this is the last layer of the network. Defaults to ``False``.
            device (torch.device, optional): The device to use for the layer's parameters. Defaults to ``None``.
            dtype (torch.dtype, optional): The dtype to use for the layer's parameters. Defaults to ``None``.
        """
        check_packed_parameters_consistency(alpha, gamma, num_estimators)
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.num_estimators = num_estimators
        self.first = first
        self.last = last

        # Define the number of channels for the underlying convolution
        self.extended_in_channels = int(in_channels * (1 if first else alpha))
        self.extended_out_channels = int(out_channels * (num_estimators if last else alpha))

        # Define the number of groups of the underlying convolution
        self.actual_groups = 1 if first else gamma * groups * num_estimators

        while (
            self.extended_in_channels % self.actual_groups != 0
            or self.extended_in_channels // self.actual_groups < minimum_channels_per_group
        ) and self.actual_groups // (groups * num_estimators) > 1:
            gamma -= 1
            self.actual_groups = gamma * groups * num_estimators

        # Fix dimensions to be divisible by groups
        if self.extended_in_channels % self.actual_groups:
            self.extended_in_channels += (
                num_estimators - self.extended_in_channels % self.actual_groups
            )
        if self.extended_out_channels % self.actual_groups:
            self.extended_out_channels += (
                num_estimators - self.extended_out_channels % self.actual_groups
            )

        # Initialize the transposed convolutional layer
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels=self.extended_in_channels,
            out_channels=self.extended_out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=self.actual_groups,
            bias=bias,
            **factory_kwargs,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv_transpose(inputs)

    @property
    def weight(self) -> Tensor:
        r"""The weight of the underlying transposed convolutional layer."""
        return self.conv_transpose.weight

    @property
    def bias(self) -> Tensor | None:
        r"""The bias of the underlying transposed convolutional layer."""
        return self.conv_transpose.bias

# %%
# defining the Packed-Ensembles UNet model


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_uncertainty.layers.packed import PackedConv2d


class PackedDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, alpha, num_estimators, gamma, first=False, last=False):
        super().__init__()
        self.conv = nn.Sequential(
            PackedConv2d(
                in_ch, out_ch, 3, alpha=alpha, num_estimators=num_estimators, gamma=gamma, padding=1, first=first
            ),
            nn.BatchNorm2d(out_ch * (num_estimators if last else alpha)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class PackedInconv(nn.Module):
    def __init__(self, in_ch, out_ch, alpha, num_estimators, gamma):
        super().__init__()
        self.conv = PackedDoubleConv(in_ch, out_ch, alpha, num_estimators, gamma, first=True)

    def forward(self, x):
        return self.conv(x)


class PackedDown(nn.Module):
    def __init__(self, in_ch, out_ch, alpha, num_estimators, gamma):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            PackedDoubleConv(in_ch, out_ch, alpha, num_estimators, gamma)
        )

    def forward(self, x):
        return self.mpconv(x)


class PackedUp(nn.Module):
    def __init__(self, in_ch, out_ch, alpha, num_estimators, gamma):
        super().__init__()
        self.up = PackedConvTranspose2d(
            in_ch//2, in_ch // 2, kernel_size=2, stride=2, alpha=alpha, num_estimators=num_estimators, gamma=gamma
        )
        self.conv = PackedDoubleConv(in_ch, out_ch, alpha, num_estimators, gamma)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class PackedOutconv(nn.Module):
    def __init__(self, in_ch, out_ch, alpha, num_estimators, gamma):
        super().__init__()
        self.conv = PackedConv2d(
            in_ch, out_ch, kernel_size=1, alpha=alpha, num_estimators=num_estimators, gamma=gamma, last=True
        )

    def forward(self, x):
        return self.conv(x)


class PackedUNet(nn.Module):
    def __init__(self, classes, alpha=1, num_estimators=1, gamma=1):
        super().__init__()
        self.alpha = alpha
        self.num_estimators = num_estimators
        self.gamma = gamma

        # Downsampling
        self.inc = PackedInconv(3, 32, alpha, num_estimators, gamma)
        self.down1 = PackedDown(32, 64, alpha, num_estimators, gamma)
        self.down2 = PackedDown(64, 128, alpha, num_estimators, gamma)
        self.down3 = PackedDown(128, 256, alpha, num_estimators, gamma)
        self.down4 = PackedDown(256, 256, alpha, num_estimators, gamma)

        # Upsampling
        self.up1 = PackedUp(512, 128, alpha, num_estimators, gamma)
        self.up2 = PackedUp(256, 64, alpha, num_estimators, gamma)
        self.up3 = PackedUp(128, 32, alpha, num_estimators, gamma)
        self.up4 = PackedUp(64, 32, alpha, num_estimators, gamma)

        # Dropout
        self.dropout = nn.Dropout2d(0.1)

        # Final output
        self.outc = PackedOutconv(32, classes, alpha, num_estimators, gamma)

    def forward(self, x):
        # Downsampling
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # Upsampling
        x = self.up1(x5, x4)
        x = self.dropout(x)
        x = self.up2(x, x3)
        x = self.dropout(x)
        x = self.up3(x, x2)
        x = self.dropout(x)
        x = self.up4(x, x1)
        x = self.dropout(x)

        # Final output
        x = self.outc(x)
        return rearrange(x, "b (m c) h w -> (m b) c h w", m=self.num_estimators)

# %% 
# Train on 1 epoch for demonstration purposes

from torch_uncertainty import TUTrainer
from torch_uncertainty.routines import SegmentationRoutine
from torch import optim
from torch.optim import lr_scheduler
from torch_uncertainty.transforms import RepeatTarget


num_estimators = 4
alpha = 2
gamma = 1

packed_model = PackedUNet(
    classes=19,
	alpha=alpha,
	num_estimators=num_estimators,
	gamma=gamma,
)

# We build the optimizer
optimizer = optim.Adam(
	packed_model.parameters(),
	lr=learning_rate*num_estimators,
	weight_decay=weight_decay
)

# Learning rate decay scheduler
lr_updater = lr_scheduler.StepLR(
    optimizer, lr_decay_epochs, lr_decay
)

trainer = TUTrainer(accelerator="gpu", devices=1, max_epochs=1, precision=16, logger=False)

packed_routine = SegmentationRoutine(
    model=packed_model,
	num_classes=19,
	loss=torch.nn.CrossEntropyLoss(weight=class_weights),
	format_batch_fn=RepeatTarget(num_estimators),  # Repeat the target 4 times for the ensemble
	optim_recipe={"optimizer": optimizer, "lr_scheduler": lr_updater},
)

# %%
#trainer.fit(packed_routine, train_loader, val_loader)

# %%
#results = trainer.test(packed_routine, test_loader)

# %% 
# Load a pre-trained ensembles from huggingface to continue the tutorial

from huggingface_hub import hf_hub_download

# Download the model
model_path = hf_hub_download(repo_id="torch-uncertainty/muad_tutorials", filename="packed_unet_tuto.pth")

model = packed_model = PackedUNet(
    classes=19,
	alpha=alpha,
	num_estimators=num_estimators,
	gamma=gamma,
)
model.load_state_dict(torch.load(model_path))
model = model.to('cpu')

# %% 
# 4. Uncertainty evaluations with MCP
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Here we will just use as confidence score the Maximum class probability (MCP)
# 

import matplotlib.pyplot as plt
from torchvision.transforms.v2 import functional as F

sample_idx = 0
img, target = test_set[sample_idx]

batch_img = img.unsqueeze(0)
batch_target = target.unsqueeze(0)
model.eval()
with torch.no_grad():
	# Forward propagation
	outputs = model(batch_img)
	outputs_proba = outputs.softmax(dim=1)
	outputs_proba = outputs_proba.mean(dim=0)

	# remove the batch dimension
	outputs_proba = outputs_proba.squeeze(0)
	confidence, pred = outputs_proba.max(0)

# Undo normalization on the image and convert to uint8.
mean = torch.tensor([0.485, 0.456, 0.406], device=img.device)
std = torch.tensor([0.229, 0.224, 0.225], device=img.device)
img = img * std[:, None, None] + mean[:, None, None]
img = F.to_dtype(img, torch.uint8, scale=True)

tmp_target = target.masked_fill(target == 255, 21)
target_masks = tmp_target == torch.arange(22, device=target.device)[:, None, None]
img_segmented = draw_segmentation_masks(img, target_masks, alpha=1, colors=test_set.color_palette)

pred_masks = pred == torch.arange(22, device=pred.device)[:, None, None]

pred_img = draw_segmentation_masks(img, pred_masks, alpha=1, colors=test_set.color_palette)

if confidence.ndim == 2:
    confidence = confidence.unsqueeze(0)

img = F.to_pil_image(F.resize(img, 1024))
img_segmented = F.to_pil_image(F.resize(img_segmented, 1024))
pred_img = F.to_pil_image(F.resize(pred_img, 1024))
confidence_img = F.to_pil_image(F.resize(confidence, 1024))

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(30, 15))
ax1.imshow(img)
ax2.imshow(img_segmented)
ax3.imshow(pred_img)
ax4.imshow(confidence_img)
plt.show()

# %% 
# Now let's load the OOD test set

# %%
test_ood_set = MUAD(root="./data", target_type="semantic", version="small", split="ood" , transforms=val_transform, download=True)


# %%
# Testing on the ood set

sample_idx = 0
img, target = test_ood_set[sample_idx]

batch_img = img.unsqueeze(0)
batch_target = target.unsqueeze(0)
model.eval()
with torch.no_grad():
	# Forward propagation
	outputs = model(batch_img)
	outputs_proba = outputs.softmax(dim=1)
	outputs_proba = outputs_proba.mean(dim=0)
	
	# remove the batch dimension
	outputs_proba = outputs_proba.squeeze(0)
	confidence, pred = outputs_proba.max(0)

# Undo normalization on the image and convert to uint8.
mean = torch.tensor([0.485, 0.456, 0.406], device=img.device)
std = torch.tensor([0.229, 0.224, 0.225], device=img.device)
img = img * std[:, None, None] + mean[:, None, None]
img = F.to_dtype(img, torch.uint8, scale=True)

tmp_target = target.masked_fill(target == 255, 21)
target_masks = tmp_target == torch.arange(22, device=target.device)[:, None, None]
img_segmented = draw_segmentation_masks(img, target_masks, alpha=1, colors=test_set.color_palette)

pred_masks = pred == torch.arange(22, device=pred.device)[:, None, None]

pred_img = draw_segmentation_masks(img, pred_masks, alpha=1, colors=test_set.color_palette)

if confidence.ndim == 2:
    confidence = confidence.unsqueeze(0)

img = F.to_pil_image(F.resize(img, 1024))
img_segmented = F.to_pil_image(F.resize(img_segmented, 1024))
pred_img = F.to_pil_image(F.resize(pred_img, 1024))
confidence_img = F.to_pil_image(F.resize(confidence, 1024))

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(30, 15))
ax1.imshow(img_pil)
ax2.imshow(img_segmented)
ax3.imshow(pred_img)
ax4.imshow(confidence_img)
plt.show()
