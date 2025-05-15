# ruff: noqa: E402, E703, D212, D415
"""
Conformal Prediction on CIFAR-10 with TorchUncertainty.
======================================================

*This notebook follows the TorchUncertainty tutorial style and demonstrates how to calibrate a pretrained ResNet model using Conformal Prediction on CIFAR-10.*

We evaluate the model's performance both before and after applying different conformal predictors (THR, APS, RAPS), and visualize how conformal prediction modifies the prediction sets.

We use the pretrained ResNet models provided on Hugging Face.

Throughout this tutorial, we rely on the `TorchUncertainty <https://torch-uncertainty.github.io/>`_ library which simplifies training, calibration, and evaluation of uncertainty-aware models in PyTorch.
"""

# %%
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms.v2 as v2
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader, random_split

from torch_uncertainty import TUTrainer
from torch_uncertainty.models.classification.resnet import resnet
from torch_uncertainty.post_processing import ConformalClsAPS, ConformalClsRAPS, ConformalClsTHR
from torch_uncertainty.routines import ClassificationRoutine

# %%
# 1. Load pretrained model from Hugging Face repository
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We use a ResNet18 model trained on CIFAR-10, provided by the TorchUncertainty team
ckpt_path = hf_hub_download(repo_id="torch-uncertainty/resnet18_c10", filename="resnet18_c10.ckpt")
model = resnet(in_channels=3, num_classes=10, arch=18, conv_bias=False, style="cifar")
ckpt = torch.load(ckpt_path, weights_only=True)
model.load_state_dict(ckpt)
model = model.cuda().eval()

# %%
# 2. Load CIFAR-10 dataset & define dataloaders
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
batch_size = 128
transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ]
)
# For calibration and Conformal it is important to have a validation set that we calibration_set
# Without these two sets you might have an unfair behaviour.
# To build these set we have decided to split the test set into 2.
train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_full = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
test_set, calibration_set = random_split(test_full, [8000, 2000])

# We also load SVHN as an out-of-distribution (OOD) dataset
# Please note that this not necessary useful for conformal, but since it is easy with TU
# why not seeing how conformal behave on OOD
ood_data = datasets.SVHN(root="./data", split="test", download=True, transform=transform)

train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=10)
test_dl = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=10)
calibration_dl = DataLoader(calibration_set, batch_size=batch_size, shuffle=False, num_workers=10)
ood_dl = DataLoader(ood_data, batch_size=batch_size, shuffle=False, num_workers=10)


# %%
# 3. Define training configuration (optimizer and scheduler)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def optim_recipe(model, lr_mult: float = 1.0):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05 * lr_mult)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    return {"optimizer": optimizer, "scheduler": scheduler}


trainer = TUTrainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", max_epochs=5)

# %%
# 4. Evaluate pretrained model before conformal calibration
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In the following, we will just assess a classical model

print("\n[Phase 1]: Evaluation of pretrained model")
routine = ClassificationRoutine(
    num_classes=10,
    model=model,
    log_plots=True,
    loss=nn.CrossEntropyLoss(),
    optim_recipe=optim_recipe(model),
    eval_ood=True,
    is_conformal=False,
)
perf = trainer.test(routine, dataloaders=[test_dl, ood_dl])

# %%
# 5. Calibrate with ConformalClsTHR
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\n[Phase 2]: ConformalClsTHR calibration")
conformal_model = ConformalClsTHR(model, device="cuda", alpha=0.01)
conformal_model.cuda()
conformal_model.fit(dataloader=calibration_dl)
print(f"Q-hat (THR): {conformal_model.q_hat:.4f}")

routine_thr = ClassificationRoutine(
    num_classes=10,
    model=conformal_model,
    log_plots=True,
    loss=nn.CrossEntropyLoss(),
    optim_recipe=optim_recipe(model),
    eval_ood=True,
    is_conformal=True,
)
perf_thr = trainer.test(routine_thr, dataloaders=[test_dl, ood_dl])

# %%
# 6. Visualization of ConformalClsTHR prediction sets
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
inputs, labels = next(iter(test_dl))

conformal_model.cuda()
prediction_sets, confidence_scores = conformal_model.conformal_visu(inputs.cuda())

classes = test_full.classes


def visualize_prediction_sets(
    inputs, labels, prediction_sets, confidence_scores, classes, num_examples=5
):
    fig, axs = plt.subplots(2, num_examples, figsize=(15, 5))
    for i in range(num_examples):
        ax = axs[0, i]
        img = inputs[i].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5  # unnormalize
        ax.imshow(img)
        ax.set_title(f"True: {classes[labels[i]]}")
        ax.axis("off")

        ax = axs[1, i]
        pred_set = prediction_sets[i]
        conf_scores = confidence_scores[i]
        for j in range(len(classes)):
            color = "green" if pred_set[j] else "red"
            ax.barh(classes[j], conf_scores[j], color=color)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Confidence Score")
    plt.tight_layout()
    plt.show()


visualize_prediction_sets(
    inputs, labels, prediction_sets[:5].cpu().numpy(), confidence_scores[:5].cpu().numpy(), classes
)

# %%
# 7. Calibrate with ConformalClsAPS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\n[Phase 3]: ConformalClsAPS calibration")
conformal_model = ConformalClsAPS(model, device="cuda", alpha=0.01)

conformal_model.fit(dataloader=calibration_dl)
print(f"Q-hat (APS): {conformal_model.q_hat:.4f}")

routine_aps = ClassificationRoutine(
    num_classes=10,
    model=conformal_model,
    loss=nn.CrossEntropyLoss(),
    optim_recipe=optim_recipe(model),
    eval_ood=True,
    is_conformal=True,
)
perf_aps = trainer.test(routine_aps, dataloaders=[test_dl, ood_dl])
conformal_model.cuda()
prediction_sets, confidence_scores = conformal_model.conformal_visu(inputs.cuda())
visualize_prediction_sets(
    inputs, labels, prediction_sets[:5].cpu().numpy(), confidence_scores[:5].cpu().numpy(), classes
)

# %%
# 8. Calibrate with ConformalClsRAPS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\n[Phase 4]: ConformalClsRAPS calibration")
conformal_model = ConformalClsRAPS(model, device="cuda", alpha=0.01)

conformal_model.fit(dataloader=calibration_dl)
print(f"Q-hat (RAPS): {conformal_model.q_hat:.4f}")

routine_raps = ClassificationRoutine(
    num_classes=10,
    model=conformal_model,
    loss=nn.CrossEntropyLoss(),
    optim_recipe=optim_recipe(model),
    eval_ood=True,
    is_conformal=True,
)
perf_raps = trainer.test(routine_raps, dataloaders=[test_dl, ood_dl])
conformal_model.cuda()
prediction_sets, confidence_scores = conformal_model.conformal_visu(inputs.cuda())
visualize_prediction_sets(
    inputs, labels, prediction_sets[:5].cpu().numpy(), confidence_scores[:5].cpu().numpy(), classes
)

# %%
# Summary
# ~~~~~~~
# In this tutorial, we explored how to apply conformal prediction to a pretrained ResNet on CIFAR-10.
# We evaluated three methods: Thresholding (THR), Adaptive Prediction Sets (APS), and Regularized APS (RAPS).
# For each, we calibrated on a validation set, evaluated OOD performance, and visualized prediction sets.

# You can explore further by adjusting `alpha`, changing the model, or testing on other datasets.
