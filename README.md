<div align="center">

![Torch Uncertainty Logo](https://github.com/ENSTA-U2IS/torch-uncertainty/blob/main/docs/source/_static/images/torch_uncertainty.png)

[![pypi](https://img.shields.io/pypi/v/torch_uncertainty.svg)](https://pypi.python.org/pypi/torch_uncertainty)
[![tests](https://github.com/ENSTA-U2IS/torch-uncertainty/actions/workflows/run-tests.yml/badge.svg?branch=main&event=push)](https://github.com/ENSTA-U2IS/torch-uncertainty/actions/workflows/run-tests.yml)
[![Docs](https://github.com/ENSTA-U2IS/torch-uncertainty/actions/workflows/build-docs.yml/badge.svg)](https://torch-uncertainty.github.io/)
[![Code Coverage](https://codecov.io/github/ENSTA-U2IS/torch-uncertainty/coverage.svg?branch=master)](https://codecov.io/gh/ENSTA-U2IS/torch-uncertainty)
[![Code style: black](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/psf/black)
</div>

_TorchUncertainty_ is a package designed to help you leverage uncertainty quantification techniques and make your deep neural networks more reliable. It aims at being collaborative and including as many methods as possible, so reach out to add yours!

:construction: _TorchUncertainty_ is in early development :construction: - expect massive changes, but reach out and contribute if you are interested in the project! **Please raise an issue if you have any bugs or difficulties.**

---

This package provides a multi-level API, including:

- ready-to-train baselines on research datasets, such as ImageNet and CIFAR
- deep learning baselines available for training on your datasets
- [pretrained weights](https://huggingface.co/torch-uncertainty) for these baselines on ImageNet and CIFAR (work in progress 🚧).
- layers available for use in your networks
- scikit-learn style post-processing methods such as Temperature Scaling

See the [Reference page](https://torch-uncertainty.github.io/references.html) or the [API reference](https://torch-uncertainty.github.io/api.html) for a more exhaustive list of the implemented methods, datasets, metrics, etc.

## Installation

The package can be installed from PyPI:

```sh
pip install torch-uncertainty
```

Then, install the desired PyTorch version in your environment.

If you aim to contribute (thank you!), have a look at the [contribution page](https://torch-uncertainty.github.io/contributing.html).

## Getting Started and Documentation

Please find the documentation at [torch-uncertainty.github.io](https://torch-uncertainty.github.io).

A quickstart is available at [torch-uncertainty.github.io/quickstart](https://torch-uncertainty.github.io/quickstart.html).

## Implemented methods

### Baselines

To date, the following deep learning baselines have been implemented:

- Deep Ensembles
- BatchEnsemble
- Masksembles
- MIMO
- Packed-Ensembles (see [blog post](https://medium.com/@adrien.lafage/make-your-neural-networks-more-reliable-with-packed-ensembles-7ad0b737a873))
- Bayesian Neural Networks

### Post-processing methods

To date, the following post-processing methods have been implemented:

- Temperature, Vector, & Matrix scaling

## Tutorials

We provide the following tutorials in our documentation:

- [From a Vanilla Classifier to a Packed-Ensemble](https://torch-uncertainty.github.io/auto_tutorials/tutorial_pe_cifar10.html)
- [Training a Bayesian Neural Network in 3 minutes](https://torch-uncertainty.github.io/auto_tutorials/tutorial_bayesian.html)
- [Improve Top-label Calibration with Temperature Scaling](https://torch-uncertainty.github.io/auto_tutorials/tutorial_scaler.html)
  
## Awesome Uncertainty repositories

You may find a lot of papers about modern uncertainty estimation techniques on the [Awesome Uncertainty in Deep Learning](https://github.com/ENSTA-U2IS/awesome-uncertainty-deeplearning).

## Other References

This package also contains the official implementation of Packed-Ensembles.

If you find the corresponding models interesting, please consider citing our [paper](https://arxiv.org/abs/2210.09184):

```text
@inproceedings{laurent2023packed,
    title={Packed-Ensembles for Efficient Uncertainty Estimation},
    author={Laurent, Olivier and Lafage, Adrien and Tartaglione, Enzo and Daniel, Geoffrey and Martinez, Jean-Marc and Bursuc, Andrei and Franchi, Gianni},
    booktitle={ICLR},
    year={2023}
}
```
