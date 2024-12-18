# UCI Regression - Benchmark


| Dataset | Number of Instances | Number of Features |
| --- | --- | --- |
| Boston Housing | 506 | 13 |
| Concrete Compression Strength | 1030 | 8 |
| Energy Efficiency | 768 | 8 |
| Kin8nm | 8192 | 8 |
| Naval Propulsion | 11,934 | 16 |
| Combined Cycle Power Plant | 9568 | 4 |
| Protein Structure | 45730 | 9 |
| Wine Quality (Red) | 1599 | 11 |
| Yacht Hydrodynamics | 308 | 6 |


> [!WARNING]
> Some datasets require installing additional packages.


This folder contains the code to train models on the UCI regression datasets. The task is to predict (a) continuous target variable(s).

**General command to train a model:**

```bash
python mlp.py fit --config configs/{dataset}/{network}/{dist_family}.yaml
```

*Example:*

```bash
python mlp.py fit --config configs/kinn8nm/mlp/laplace.yaml
```
