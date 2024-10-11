import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F
from torch import Tensor


def show(prediction: Tensor, target: Tensor):
    imgs = [prediction, target]
    fig, axs = plt.subplots(ncols=len(imgs), figsize=(12, 6))
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[i].imshow(np.asarray(img))
        axs[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    axs[0].set(title="Prediction")
    axs[1].set(title="Ground Truth")

    return fig
