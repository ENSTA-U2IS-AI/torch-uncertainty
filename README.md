<div align="center">

![TorchUncertaintyLogo](https://github.com/ENSTA-U2IS-AI/torch-uncertainty/blob/main/docs/source/_static/images/torch_uncertainty.png)

[![pypi](https://img.shields.io/pypi/v/torch_uncertainty.svg)](https://pypi.python.org/pypi/torch_uncertainty)
[![tests](https://github.com/ENSTA-U2IS-AI/torch-uncertainty/actions/workflows/run-tests.yml/badge.svg?branch=main&event=push)](https://github.com/ENSTA-U2IS-AI/torch-uncertainty/actions/workflows/run-tests.yml)
[![Docs](https://github.com/ENSTA-U2IS-AI/torch-uncertainty/actions/workflows/build-docs.yml/badge.svg)](https://torch-uncertainty.github.io/)
[![PRWelcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/ENSTA-U2IS-AI/torch-uncertainty/pulls)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code Coverage](https://codecov.io/github/ENSTA-U2IS-AI/torch-uncertainty/coverage.svg?branch=master)](https://codecov.io/gh/ENSTA-U2IS-AI/torch-uncertainty)
[![Downloads](https://static.pepy.tech/badge/torch-uncertainty)](https://pepy.tech/project/torch-uncertainty)
[![Discord Badge](https://dcbadge.vercel.app/api/server/HMCawt5MJu?compact=true&style=flat)](https://discord.gg/HMCawt5MJu)
</div>

_TorchUncertainty_ is a package designed to help you leverage [uncertainty quantification techniques](https://github.com/ENSTA-U2IS-AI/awesome-uncertainty-deeplearning) and make your deep neural networks more reliable. It aims at being collaborative and including as many methods as possible, so reach out to add yours!

:construction: _TorchUncertainty_ is in early development :construction: - expect changes, but reach out and contribute if you are interested in the project! **Please raise an issue if you have any bugs or difficulties and join the [discord server](https://discord.gg/HMCawt5MJu).**

:books: Our webpage and documentation is available here: [torch-uncertainty.github.io](https://torch-uncertainty.github.io). :books:

TorchUncertainty contains the *official implementations* of multiple papers from *major machine-learning and computer vision conferences* and was/will be featured in tutorials at **[WACV](https://wacv2024.thecvf.com/) 2024**, **[HAICON](https://haicon24.de/) 2024** and **[ECCV](https://eccv.ecva.net/) 2024**.

---

This package provides a multi-level API, including:

- easy-to-use :zap: lightning **uncertainty-aware** training & evaluation routines for **4 tasks**: classification, probabilistic and pointwise regression, and segmentation.
- ready-to-train baselines on research datasets, such as ImageNet and CIFAR
- [pretrained weights](https://huggingface.co/torch-uncertainty) for these baselines on ImageNet and CIFAR ( :construction: work in progress :construction: ).
- **layers**, **models**, **metrics**, & **losses** available for use in your networks
- scikit-learn style post-processing methods such as Temperature Scaling.

Have a look at the [Reference page](https://torch-uncertainty.github.io/references.html) or the [API reference](https://torch-uncertainty.github.io/api.html) for a more exhaustive list of the implemented methods, datasets, metrics, etc.

## :gear: Installation

TorchUncertainty requires Python 3.10 or greater. Install the desired PyTorch version in your environment.
Then, install the package from PyPI:

```sh
pip install torch-uncertainty
```

The installation procedure for contributors is different: have a look at the [contribution page](https://torch-uncertainty.github.io/contributing.html).

## :racehorse: Quickstart

We make a quickstart available at [torch-uncertainty.github.io/quickstart](https://torch-uncertainty.github.io/quickstart.html).

## :books: Implemented methods

TorchUncertainty currently supports **classification**, **probabilistic** and pointwise **regression**, **segmentation** and **pixelwise regression** (such as monocular depth estimation). It includes the official codes of the following papers:

- *LP-BNN: Encoding the latent posterior of Bayesian Neural Networks for uncertainty quantification* - [IEEE TPAMI](https://arxiv.org/abs/2012.02818)
- *Packed-Ensembles for Efficient Uncertainty Estimation* - [ICLR 2023](https://arxiv.org/abs/2210.09184) - [Tutorial](https://torch-uncertainty.github.io/auto_tutorials/tutorial_pe_cifar10.html)
- *MUAD: Multiple Uncertainties for Autonomous Driving, a benchmark for multiple uncertainty types and tasks* - [BMVC 2022](https://arxiv.org/abs/2203.01437)

We also provide the following methods:

### Baselines

To date, the following deep learning baselines have been implemented. **Click** :inbox_tray: **on the methods for tutorials**:

- [Deep Ensembles](https://torch-uncertainty.github.io/auto_tutorials/tutorial_from_de_to_pe.html), BatchEnsemble, Masksembles, & MIMO
- [MC-Dropout](https://torch-uncertainty.github.io/auto_tutorials/tutorial_mc_dropout.html)
- [Packed-Ensembles](https://torch-uncertainty.github.io/auto_tutorials/tutorial_from_de_to_pe.html) (see [Blog post](https://medium.com/@adrien.lafage/make-your-neural-networks-more-reliable-with-packed-ensembles-7ad0b737a873))
- [Variational Bayesian Neural Networks](https://torch-uncertainty.github.io/auto_tutorials/tutorial_bayesian.html)
- Checkpoint Ensembles & Snapshot Ensembles
- Stochastic Weight Averaging & Stochastic Weight Averaging Gaussian
- Regression with Beta Gaussian NLL Loss
- [Deep Evidential Classification](https://torch-uncertainty.github.io/auto_tutorials/tutorial_evidential_classification.html) & [Regression](https://torch-uncertainty.github.io/auto_tutorials/tutorial_der_cubic.html)

### Augmentation methods

The following data augmentation methods have been implemented:

- Mixup, MixupIO, RegMixup, WarpingMixup

### Post-processing methods

To date, the following post-processing methods have been implemented:

- [Temperature](https://torch-uncertainty.github.io/auto_tutorials/tutorial_scaler.html), Vector, & Matrix scaling
- [Monte Carlo Batch Normalization](https://torch-uncertainty.github.io/auto_tutorials/tutorial_mc_batch_norm.html)
- Laplace approximation using the [Laplace library](https://github.com/aleximmer/Laplace)

## Tutorials

Check out our tutorials at [torch-uncertainty.github.io/auto_tutorials](https://torch-uncertainty.github.io/auto_tutorials/index.html).

## :telescope: Projects using TorchUncertainty

The following projects use TorchUncertainty:

- *A Symmetry-Aware Exploration of Bayesian Neural Network Posteriors* - [ICLR 2024](https://arxiv.org/abs/2310.08287)

**If you are using TorchUncertainty in your project, please let us know, we will add your project to this list!**
