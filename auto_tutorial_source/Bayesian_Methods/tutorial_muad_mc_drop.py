"""
Monte Carlo Dropout for Semantic Segmentation on MUAD
=====================================================

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
learning_rate = 1e-3
weight_decay = 2e-4
lr_decay_epochs = 20
lr_decay = 0.1
nb_epochs = 50

# %%
# In this Tutorial we are using the small version a bigger version can be specified with keyword "full" instead of small.


import matplotlib.pyplot as plt
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F

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

train_set = MUAD(
    root="./data",
    target_type="semantic",
    version="small",
    split="train",
    transforms=train_transform,
    download=True,
)
val_set = MUAD(
    root="./data",
    target_type="semantic",
    version="small",
    split="val",
    transforms=val_transform,
    download=True,
)
test_set = MUAD(
    root="./data",
    target_type="semantic",
    version="small",
    split="test",
    transforms=val_transform,
    download=True,
)

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

plt.figure(figsize=(6, 6))
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

plt.figure(figsize=(6, 6))
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

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)

val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)

test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)


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
            nn.ReLU(inplace=True),
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
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

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
            x1 = F.resize(
                x1,
                size=[2 * x1.size()[2], 2 * x1.size()[3]],
                interpolation=v2.InterpolationMode.BILINEAR,
            )
        else:
            x1 = self.up(x1)

        # input is CHW
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

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


# please note that we have added dropout layer to be abble to use MC dropout


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
# 3. Training Process
# ~~~~~~~~~~~~~~~~~~~

# %%
# Below we will train only for 1 epoch for demonstartion a good range for this setup is generally 50 epochs

num_classes = 19
# Intialize UNet
model = UNet(num_classes)

from torch import optim
from torch.optim import lr_scheduler

from torch_uncertainty import TUTrainer
from torch_uncertainty.routines import SegmentationRoutine

# We build the optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Learning rate decay scheduler
lr_updater = lr_scheduler.StepLR(optimizer, lr_decay_epochs, lr_decay)

# Initialize the trainer
trainer = TUTrainer(accelerator="gpu", devices=1, max_epochs=1, logger=False)
ens_routine = SegmentationRoutine(
    model=model,
    num_classes=19,
    loss=torch.nn.CrossEntropyLoss(weight=class_weights),
    optim_recipe={"optimizer": optimizer, "lr_scheduler": lr_updater},
)

# %%
# trainer.fit(ens_routine, train_loader, val_loader)


# %%
# 4. Evaluation
# ~~~~~~~~~~~~~

# %%
# results = trainer.test(ens_routine, test_loader)

# %%
# Let's now load a fully trained model to continue this tutorial

from huggingface_hub import hf_hub_download

# Download the model
model_path = hf_hub_download(repo_id="torch-uncertainty/muad_tutorials", filename="tuto_muad.pth")

model = UNet(num_classes)
model.load_state_dict(torch.load(model_path))
model = model.to("cpu")

# %%
# 5. Uncertainty evaluations with MCP
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Here we will just use as confidence score the Maximum class probability (MCP)


# %%
sample_idx = 0
img, target = test_set[sample_idx]

batch_img = img.unsqueeze(0)
batch_target = target.unsqueeze(0)
model.eval()
with torch.no_grad():
    # Forward propagation
    outputs = model(batch_img)
    outputs_proba = outputs.softmax(dim=1)
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
test_ood_set = MUAD(
    root="./data",
    target_type="semantic",
    version="small",
    split="ood",
    transforms=val_transform,
    download=True,
)
test_ood_set

# %%
sample_idx = 0
img, target = test_ood_set[sample_idx]

batch_img = img.unsqueeze(0)
batch_target = target.unsqueeze(0)
model.eval()
with torch.no_grad():
    # Forward propagation
    outputs = model(batch_img)
    outputs_proba = outputs.softmax(dim=1)
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

# %%
# 6. Uncertainty evaluations with MC Dropout
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Let us use now **MC dropout** via TorchUncertainty. This technique decribed in this `paper <https://arxiv.org/abs/1506.02142/>`_ allow us to have a better confindence score by using the dropout during test time.


# %%
from torch_uncertainty.models.wrappers.mc_dropout import mc_dropout

# Wrap model with MCDropout from torch_uncertainty

num_estimators = 20
num_classes = 19

mc_model = mc_dropout(model, num_estimators=num_estimators, last_layer=False, on_batch=False)

mc_model.eval()  # to enable MCdropout at inferene and get many predictions

sample_idx = 0
img, target = test_ood_set[sample_idx]


batch_img = img.unsqueeze(0)
batch_target = target.unsqueeze(0)

# %%
# Perform stochastic forward passes

with torch.no_grad():
    outputs = mc_model(batch_img)
    y_hat = outputs.softmax(dim=1)

    mean_y_hat = torch.mean(y_hat, dim=0)

    var_y_hat = torch.sqrt(torch.mean((y_hat - mean_y_hat) ** 2, dim=0))

    # Prepare for visualization
    img_id = 0
    aggregated_uncertainty = var_y_hat.mean(dim=0)

    rescaled_uncertainty = aggregated_uncertainty / aggregated_uncertainty.max()
    inverted_uncertainty = 1 - rescaled_uncertainty

    pred = torch.argmax(mean_y_hat, dim=0)  # Shape: [H, W]

    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device)
    std = torch.tensor([0.229, 0.224, 0.225], device=img.device)
    img = img * std[:, None, None] + mean[:, None, None]
    img = F.to_dtype(img, torch.uint8, scale=True)

    tmp_target = target.masked_fill(target == 255, 21)
    target_masks = tmp_target == torch.arange(22, device=target.device)[:, None, None]
    img_segmented = draw_segmentation_masks(
        img, target_masks, alpha=1, colors=test_set.color_palette
    )

    pred_masks = pred == torch.arange(22, device=pred.device)[:, None, None]

    pred_img = draw_segmentation_masks(img, pred_masks, alpha=1, colors=test_set.color_palette)

    img_pil = F.to_pil_image(img)
    img_segmented = F.to_pil_image(img_segmented)
    uncertainty_img = F.to_pil_image(inverted_uncertainty)
    pred_img = F.to_pil_image(pred_img)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(30, 15))
    ax1.imshow(img_pil)
    ax2.imshow(img_segmented)
    ax3.imshow(pred_img)
    ax4.imshow(uncertainty_img)
    plt.show()
