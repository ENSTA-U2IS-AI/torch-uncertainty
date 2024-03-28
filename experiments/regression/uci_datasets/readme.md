# UCI Regression - Benchmark

This folder contains the code to train models on the UCI regression datasets. The task is to predict (a) continuous target variable(s).

Three experiments are provided:

```bash
python mlp.py fit --config configs/pw_mlp_kin8nm.yaml
```

```bash
python mlp.py fit --config configs/gaussian_mlp_kin8nm.yaml
```

```bash
python mlp.py fit --config configs/laplace_mlp_kin8nm.yaml
```
