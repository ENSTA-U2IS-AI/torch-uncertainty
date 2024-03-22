# CIFAR10 - Benchmark

This folder contains the code to train models on the CIFAR10 dataset. The task is to classify images into $10$ classes.

## ResNet-backbone models

`torch-uncertainty` leverages [LightningCLI](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.cli.LightningCLI.html#lightning.pytorch.cli.LightningCLI) the configurable command line tool for pytorch-lightning. To ease the train of models, we provide a set of predefined configurations for the CIFAR10 dataset (corresponding to the experiments reported in [Packed-Ensembles for Efficient Uncertainty Estimation](https://arxiv.org/abs/2210.09184)). The configurations are located in the `configs` folder.

*Examples:*

* Training a standard ResNet18 model as in [Packed-Ensembles for Efficient Uncertainty Estimation](https://arxiv.org/abs/2210.09184):

```bash
python resnet.py fit --config configs/resnet18/standard.yaml
```

* Training Packed-Ensembles ResNet50 model as in [Packed-Ensembles for Efficient Uncertainty Estimation](https://arxiv.org/abs/2210.09184):

```bash
python resnet.py fit --config configs/resnet50/packed.yaml
```


**Note:** In addition we provide a default resnet config file (`configs/resnet.yaml`) to enable the training of any ResNet model. Here a basic example to train a MIMO ResNet101 model with $4$ estimators and $\rho=1.0$:

```bash
python resnet.py fit --config configs/resnet.yaml --model.arch 101 --model.version mimo --model.num_estimators 4 --model.rho 1.0
```

## Available configurations:

### ResNet

||ResNet18|ResNet34|ResNet50|ResNet101|ResNet152|
|---|---|---|---|---|---|
|Standard|✅|✅|✅|✅|✅|
|Packed-Ensembles|✅|✅|✅|✅|✅|
|BatchEnsemble|✅|✅|✅|✅|✅|
|Masked-Ensembles|✅|✅|✅|✅|✅|
|MIMO|✅|✅|✅|✅|✅|
|MC Dropout|✅|✅|✅|✅|✅|


### WideResNet

||WideResNet28-10|
|---|---|
|Standard|✅|
|Packed-Ensembles|✅|
|BatchEnsemble|✅|
|Masked-Ensembles|✅|
|MIMO|✅|
|MC Dropout|✅|

### VGG

||VGG11|VGG13|VGG16|VGG19|
|---|---|---|---|---|
|Standard|✅|✅|✅|✅|
|Packed-Ensembles|✅|✅|✅|✅|
|BatchEnsemble|||||
|Masked-Ensembles|||||
|MIMO|||||
|MC Dropout|✅|✅|✅|✅|
