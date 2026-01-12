# Benchmark

Below are the results for InceptionTime model variants on UCR/UEA datasets. The metrics include accuracy Acc, expected calibration error ECE, false positive rate at 95% recall **FPR**$_{95}$, and the number of floating point operations when executing the forward on a single batch of size $16$ **FLOPS** in Giga.

The (FPR$_{95}$) measures how well the model is separating the in-distribution and out-of-distribution data, with lower values indicating better performance. The FLOPS metric indicates the computational cost of the model.

## Launch experiments

```bash
cd experiments/classification/ucr-uea
python main.py fit --config configs/{dataset_name}/inception-time/{method}.yaml
```

`dataset_name` can be one of the following: `adiac`, `beef`, `crickety`, `cricketz`, `inline-skate`, `lightning7`, `olive-oil`, and `two-patterns`.

`method` can be one of the following: `standard`, `deep_ensembles`, `mc_dropout`, `mimo`, `batch_ensemble`, `packed_ensembles`, and `bayesian`.


## Results

### Adiac

| Method | **Acc (%)** | **ECE** | **FPR**$_{95}$ | **FLOPS (G)** |
|--------|-------------|---------|----------------|---------------|
| Baseline | 76.34 | 5.84 | 34.39 | 2.17 |
| BNN | 78.01 | 6.94 | 29.82 | 34.81 |
| MC Dropout | 100.00 | 6.03 | 53.51 | 15.82 |
| MIMO ($\rho=0.5$) | 72.12 | 12.09 | 39.05 | 2.22 |
| BatchEnsemble | 73.27 | 11.99 | 47.49 | 8.70 |
| Packed-Ensembles ($\alpha=2$) | 75.81 | 9.93 | 36.41 | 2.19 |
| Deep Ensembles | 78.28 | 6.71 | 41.34 | 8.70 |

### Beef

| Method | **Acc (%)** | **ECE** | **FPR**$_{95}$ | **FLOPS (G)** |
|--------|-------------|---------|----------------|---------------|
| Baseline | 75.00 | 21.08 | 70.83 | 5.81 |
| BNN | 77.78 | 20.86 | 70.83 | 92.96 |
| MC Dropout | 70.83 | 18.70 | 63.89 | 58.10 |
| MIMO ($\rho=0.5$) | 65.28 | 18.64 | 77.78 | 5.91 |
| BatchEnsemble | 65.28 | 15.80 | 76.39 | 23.24 |
| Packed-Ensembles ($\alpha=2$) | 73.61 | 17.73 | 70.84 | 5.84 |
| Deep Ensembles | 66.67 | 16.80 | 81.94 | 23.24 |

### CricketY

| Method | **Acc (%)** | **ECE** | **FPR**$_{95}$ | **FLOPS (G)** |
|--------|-------------|---------|----------------|---------------|
| Baseline | 87.28 | 4.15 | 83.10 | 3.71 |
| BNN | 87.00 | 7.15 | 81.62 | 59.34 |
| MC Dropout | 87.56 | 5.50 | 85.61 | 37.09 |
| MIMO ($\rho=0.5$) | 84.96 | 6.14 | 97.21 | 3.78 |
| BatchEnsemble | 85.79 | 7.70 | 69.73 | 14.83 |
| Packed-Ensembles ($\alpha=2$) | 87.00 | 9.30 | 77.53 | 3.73 |
| Deep Ensembles | 85.24 | 3.82 | 80.22 | 14.83 |

### CricketZ

| Method | **Acc (%)** | **ECE** | **FPR**$_{95}$ | **FLOPS (G)** |
|--------|-------------|---------|----------------|---------------|
| Baseline | 88.18 | 3.64 | 56.51 | 3.71 |
| BNN | 88.09 | 5.65 | 60.39 | 59.34 |
| MC Dropout | 88.37 | 7.80 | 53.19 | 37.09 |
| MIMO ($\rho=0.5$) | 86.61 | 8.14 | 64.82 | 3.78 |
| BatchEnsemble | 87.17 | 9.21 | 49.58 | 14.83 |
| Packed-Ensembles ($\alpha=2$) | 88.55 | 10.62 | 49.49 | 3.73 |
| Deep Ensembles | 87.26 | 7.95 | 54.20 | 14.83 |

### InlineSkate

| Method | **Acc (%)** | **ECE** | **FPR**$_{95}$ | **FLOPS (G)** |
|--------|-------------|---------|----------------|---------------|
| Baseline | 40.41 | 4.95 | 83.44 | 23.26 |
| BNN | 40.88 | 6.39 | 84.63 | 372.24 |
| MC Dropout | 26.18 | 6.66 | 76.62 | 232.65 |
| MIMO ($\rho=0.5$) | 26.18 | 3.68 | 89.18 | 23.68 |
| BatchEnsemble | 41.82 | 6.91 | 84.70 | 93.06 |
| Packed-Ensembles ($\alpha=2$) | 43.02 | 7.57 | 79.16 | 23.40 |
| Deep Ensembles | 40.28 | 4.66 | 87.04 | 93.06 |

### Lightning7

| Method | **Acc (%)** | **ECE** | **FPR**$_{95}$ | **FLOPS (G)** |
|--------|-------------|---------|----------------|---------------|
| Baseline | 86.89 | 20.76 | 87.43 | 3.94 |
| BNN | 83.06 | 16.55 | 89.07 | 63.09 |
| MC Dropout | 85.79 | 20.44 | 80.33 | 39.43 |
| MIMO ($\rho=0.5$) | 83.61 | 17.35 | 86.88 | 4.01 |
| BatchEnsemble | 84.70 | 18.65 | 97.81 | 15.77 |
| Packed-Ensembles ($\alpha=2$) | 85.25 | 23.64 | 80.87 | 3.97 |
| Deep Ensembles | 85.80 | 14.57 | 92.35 | 15.77 |

### Olive Oil

| Method | **Acc (%)** | **ECE** | **FPR**$_{95}$ | **FLOPS (G)** |
|--------|-------------|---------|----------------|---------------|
| Baseline | 77.78 | 29.00 | 83.33 | 7.05 |
| BNN | 85.18 | 22.63 | 81.48 | 112.74 |
| MC Dropout | 81.48 | 18.82 | 88.89 | 70.46 |
| MIMO ($\rho=0.5$) | 74.07 | 25.06 | 88.89 | 7.17 |
| BatchEnsemble | 70.37 | 18.61 | 88.89 | 28.18 |
| Packed-Ensembles ($\alpha=2$) | 79.63 | 25.46 | 88.89 | 7.09 |
| Deep Ensembles | 72.22 | 25.35 | 88.89 | 28.18 |

### Two Patterns

| Method | **Acc (%)** | **ECE** | **FPR**$_{95}$ | **FLOPS (G)** |
|--------|-------------|---------|----------------|---------------|
| Baseline | 100.00 | 0.41 | 51.84 | 1.58 |
| BNN | 100.00 | 0.25 | 63.23 | 25.32 |
| MC Dropout | 100.00 | 0.16 | 70.69 | 15.82 |
| MIMO ($\rho=0.5$) | 100.00 | 1.26 | 9.72 | 1.61 |
| BatchEnsemble | 100.00 | 2.15 | 78.05 | 6.33 |
| Packed-Ensembles ($\alpha=2$) | 100.00 | 0.14 | 40.54 | 1.59 |
| Deep Ensembles | 100.00 | 2.15 | 78.05 | 6.33 |
