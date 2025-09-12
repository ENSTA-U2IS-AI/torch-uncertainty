import logging
import types
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from tests._dummies import (
    DummyClassificationBaseline,
    DummyClassificationDataModule,
    dummy_model,
    dummy_ood_model,
)
from torch_uncertainty import TUTrainer
from torch_uncertainty.losses import DECLoss, ELBOLoss
from torch_uncertainty.ood.ood_criteria import (
    EntropyCriterion,
    MaxSoftmaxCriterion,
)
from torch_uncertainty.post_processing import ConformalClsTHR
from torch_uncertainty.routines import ClassificationRoutine


class TestClassification:
    """Testing the classification routine."""

    def test_one_estimator_binary(self) -> None:
        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True)

        dm = DummyClassificationDataModule(
            root=Path(),
            batch_size=16,
            num_classes=1,
            num_images=100,
        )
        model = DummyClassificationBaseline(
            in_channels=dm.num_channels,
            num_classes=dm.num_classes,
            loss=nn.BCEWithLogitsLoss(),
            baseline_type="single",
            ema=True,
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_two_estimators_binary(self) -> None:
        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True)

        dm = DummyClassificationDataModule(
            root=Path(),
            batch_size=16,
            num_classes=1,
            num_images=100,
        )
        model = DummyClassificationBaseline(
            in_channels=dm.num_channels,
            num_classes=dm.num_classes,
            loss=nn.BCEWithLogitsLoss(),
            baseline_type="single",
            ood_criterion="logit",
            swa=True,
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_one_estimator_two_classes(self) -> None:
        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True)

        dm = DummyClassificationDataModule(
            root=Path(),
            batch_size=16,
            num_classes=2,
            num_images=100,
            eval_ood=True,
            eval_shift=True,
        )
        model = DummyClassificationBaseline(
            num_classes=dm.num_classes,
            in_channels=dm.num_channels,
            loss=nn.CrossEntropyLoss(),
            baseline_type="single",
            ood_criterion=EntropyCriterion,
            eval_ood=True,
            eval_shift=True,
            no_mixup_params=True,
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_one_estimator_two_classes_timm(self) -> None:
        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True)

        dm = DummyClassificationDataModule(
            root=Path(),
            batch_size=16,
            num_classes=2,
            num_images=100,
            eval_ood=True,
        )
        model = DummyClassificationBaseline(
            num_classes=dm.num_classes,
            in_channels=dm.num_channels,
            loss=nn.CrossEntropyLoss(),
            baseline_type="single",
            ood_criterion="entropy",
            eval_ood=True,
            mixtype="timm",
            mixup_alpha=1.0,
            cutmix_alpha=0.5,
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_one_estimator_two_classes_mixup(self) -> None:
        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True)

        dm = DummyClassificationDataModule(
            root=Path(),
            batch_size=16,
            num_classes=2,
            num_images=100,
            eval_ood=True,
        )
        model = DummyClassificationBaseline(
            num_classes=dm.num_classes,
            in_channels=dm.num_channels,
            loss=nn.CrossEntropyLoss(),
            baseline_type="single",
            ood_criterion="entropy",
            eval_ood=True,
            mixtype="mixup",
            mixup_alpha=1.0,
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_one_estimator_two_classes_mixup_io(self) -> None:
        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True)

        dm = DummyClassificationDataModule(
            root=Path(),
            batch_size=16,
            num_classes=2,
            num_images=100,
            eval_ood=True,
        )
        model = DummyClassificationBaseline(
            num_classes=dm.num_classes,
            in_channels=dm.num_channels,
            loss=nn.CrossEntropyLoss(),
            baseline_type="single",
            ood_criterion="entropy",
            eval_ood=True,
            mixtype="mixup_io",
            mixup_alpha=1.0,
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_one_estimator_two_classes_regmixup(self) -> None:
        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True)

        dm = DummyClassificationDataModule(
            root=Path(),
            batch_size=16,
            num_classes=2,
            num_images=100,
            eval_ood=True,
        )
        model = DummyClassificationBaseline(
            num_classes=dm.num_classes,
            in_channels=dm.num_channels,
            loss=nn.CrossEntropyLoss(),
            baseline_type="single",
            ood_criterion="entropy",
            eval_ood=True,
            mixtype="regmixup",
            mixup_alpha=1.0,
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_one_estimator_two_classes_kernel_warping_emb(self) -> None:
        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True)

        dm = DummyClassificationDataModule(
            root=Path(),
            batch_size=16,
            num_classes=2,
            num_images=100,
            eval_ood=True,
        )
        model = DummyClassificationBaseline(
            num_classes=dm.num_classes,
            in_channels=dm.num_channels,
            loss=nn.CrossEntropyLoss(),
            baseline_type="single",
            ood_criterion="entropy",
            eval_ood=True,
            mixtype="kernel_warping",
            mixup_alpha=0.5,
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_one_estimator_two_classes_kernel_warping_inp(self) -> None:
        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True)

        dm = DummyClassificationDataModule(
            root=Path(),
            batch_size=16,
            num_classes=2,
            num_images=100,
            eval_ood=True,
        )
        model = DummyClassificationBaseline(
            num_classes=dm.num_classes,
            in_channels=dm.num_channels,
            loss=nn.CrossEntropyLoss(),
            baseline_type="single",
            ood_criterion="entropy",
            eval_ood=True,
            mixtype="kernel_warping",
            dist_sim="inp",
            mixup_alpha=0.5,
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_one_estimator_two_classes_calibrated_with_ood(self) -> None:
        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True, logger=True)

        dm = DummyClassificationDataModule(
            root=Path(),
            batch_size=19,  # lower than 19 it doesn't work :'(
            num_classes=2,
            num_images=100,
            eval_ood=True,
        )
        model = DummyClassificationBaseline(
            num_classes=dm.num_classes,
            in_channels=dm.num_channels,
            loss=nn.CrossEntropyLoss(),
            baseline_type="single",
            ood_criterion="energy",
            eval_ood=True,
            eval_grouping_loss=True,
            calibrate=True,
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_two_estimators_two_classes_mi(self) -> None:
        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True)

        dm = DummyClassificationDataModule(
            root=Path(),
            batch_size=16,
            num_classes=2,
            num_images=100,
            eval_ood=True,
            eval_shift=True,
        )
        model = DummyClassificationBaseline(
            num_classes=dm.num_classes,
            in_channels=dm.num_channels,
            loss=DECLoss(1, 1e-2),
            baseline_type="ensemble",
            ood_criterion="mutual_information",
            eval_ood=True,
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_two_estimator_two_classes_elbo_vr_logs(self) -> None:
        trainer = TUTrainer(
            accelerator="cpu",
            max_epochs=1,
            limit_train_batches=1,
            limit_val_batches=1,
            limit_test_batches=1,
            enable_checkpointing=False,
        )

        dm = DummyClassificationDataModule(
            root=Path(),
            batch_size=16,
            num_classes=2,
            num_images=100,
            eval_ood=True,
        )
        model = DummyClassificationBaseline(
            num_classes=dm.num_classes,
            in_channels=dm.num_channels,
            loss=ELBOLoss(None, nn.CrossEntropyLoss(), kl_weight=1.0, num_samples=4),
            baseline_type="ensemble",
            ood_criterion="variation_ratio",
            eval_ood=True,
            save_in_csv=True,
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_one_estimator_conformal(self) -> None:
        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True)

        dm = DummyClassificationDataModule(
            root=Path(),
            batch_size=16,
            num_classes=3,
            num_images=100,
            eval_ood=True,
        )

        model = dummy_model(
            in_channels=dm.num_channels,
            num_classes=dm.num_classes,
        )
        routine = ClassificationRoutine(
            model=model,
            loss=None,
            num_classes=3,
            post_processing=ConformalClsTHR(alpha=0.1),
            eval_ood=True,
        )
        trainer.test(routine, dm)

        model = ConformalClsTHR(
            alpha=0.1,
            model=dummy_model(
                in_channels=dm.num_channels,
                num_classes=dm.num_classes,
            ),
        )
        model.fit(dm.postprocess_dataloader())

        routine = ClassificationRoutine(
            model=model,
            loss=None,
            num_classes=3,
            post_processing=None,
        )
        trainer.test(routine, dm)

    OOD_CRITS = [
        "scale",
        "ash",
        "react",
        "adascale_a",
        "vim",
        "odin",
        "knn",
        "gen",
        "nnguide",
    ]

    @pytest.mark.parametrize("crit", OOD_CRITS)
    def test_all_other_ood_criteria_with_dummy_ood_model(self, crit, monkeypatch):
        device = torch.device("cpu")

        monkeypatch.setattr(
            torch.Tensor,
            "cuda",
            lambda self, *a, **k: self.to(device),  # noqa: ARG005
            raising=False,
        )
        monkeypatch.setattr(nn.Module, "cuda", lambda self, *a, **k: self.to(device), raising=False)  # noqa: ARG005

        trainer = TUTrainer(
            accelerator="cpu",
            devices=1,
            inference_mode=False,
            num_sanity_val_steps=0,
            max_epochs=1,
            limit_train_batches=1,
            limit_val_batches=1,
            limit_test_batches=1,
            enable_checkpointing=False,
            logger=False,
        )

        class _TensorDS(Dataset):
            def __init__(self, x, y, name="tensor_ds"):
                self.x, self.y, self.dataset_name = x, y, name

            def __len__(self):
                return self.x.shape[0]

            def __getitem__(self, i):
                return self.x[i], self.y[i]

        def _mk_split(n, c, h, w, num_classes, shift=0.0, seed=0):
            g = torch.Generator().manual_seed(seed)
            x = (torch.rand((n, c, h, w), generator=g) + shift).clamp(0, 1)
            y = (torch.arange(n) % num_classes).long()
            return x.float(), y

        dm = DummyClassificationDataModule(
            root=Path(),
            batch_size=8,
            num_classes=3,
            num_images=64,
            num_workers=0,
            eval_ood=True,
            persistent_workers=False,
        )

        def patched_setup(self, stage=None):
            self.num_channels = 3
            h = w = self.image_size
            n = self.num_images

            if stage in (None, "fit"):
                x_tr, y_tr = _mk_split(n, 3, h, w, self.num_classes, 0.0, 123)
                x_va, y_va = _mk_split(n, 3, h, w, self.num_classes, 0.0, 456)
                self.train = _TensorDS(x_tr, y_tr, "train")
                self.val = _TensorDS(x_va, y_va, "val")

            if stage in (None, "test"):
                x_te, y_te = _mk_split(n, 3, h, w, self.num_classes, 0.0, 789)
                self.test = _TensorDS(x_te, y_te, "test")
                if self.eval_ood:
                    x_vo, y_vo = _mk_split(n, 3, h, w, self.num_classes, 0.10, 321)
                    self.val_ood = _TensorDS(x_vo, y_vo, "val_ood")
                    x_near, y_near = _mk_split(n, 3, h, w, self.num_classes, 0.15, 654)
                    self.near_oods = [_TensorDS(x_near, y_near, "near")]
                    x_far, y_far = _mk_split(n, 3, h, w, self.num_classes, 0.35, 987)
                    self.far_oods = [_TensorDS(x_far, y_far, "far")]

            if self.eval_shift:
                x_sh, y_sh = _mk_split(n, 3, h, w, self.num_classes, 0.20, 111)
                self.shift = _TensorDS(x_sh, y_sh, "shift")
                self.shift_severity = 1

        monkeypatch.setattr(dm, "setup", patched_setup.__get__(dm, type(dm)), raising=True)

        model = dummy_ood_model(in_channels=3, feat_dim=4096, num_classes=dm.num_classes).to(device)

        routine = ClassificationRoutine(
            model=model,
            loss=None,
            num_classes=dm.num_classes,
            eval_ood=True,
            ood_criterion=crit,
            log_plots=False,
        )

        c = routine.ood_criterion
        if hasattr(c, "args_dict"):
            if crit in ("scale", "ash", "react"):
                c.args_dict = {"percentile": [70]}
            elif crit == "adascale_a":
                c.args_dict = {
                    "percentile": [(40, 60)],
                    "k1": [1],
                    "k2": [1],
                    "lmbda": [0.1],
                    "o": [0.05],
                }
            elif crit == "vim":
                safe_dim = min(64, getattr(model, "feature_size", 256) - 1)
                c.args_dict = {"dim": [safe_dim]}
                c.dim = safe_dim
            elif crit == "odin":
                c.args_dict = {"temperature": [1.0], "noise": [0.0014]}
            elif crit == "knn":
                c.args_dict = {"K": [5]}
            elif crit == "gen":
                # ensure m ≤ num_classes to avoid degenerate slices
                c.gamma = getattr(c, "gamma", 0.1)
                c.m = min(getattr(c, "m", 10), dm.num_classes)
                c.args_dict = {"gamma": [c.gamma], "m": [c.m]}
            elif crit == "nnguide":
                c.args_dict = {"K": [5], "alpha": [0.5]}
            c.hyperparam_search_done = False

        trainer.test(routine, dm)

        if hasattr(c, "args_dict"):
            assert getattr(c, "hyperparam_search_done", False), (
                f"Hyperparam search did not complete for '{crit}'."
            )
        for needs_setup in {"react", "adascale_a", "vim", "knn", "nnguide"}:
            if crit == needs_setup:
                assert getattr(c, "setup_flag", False), f"Setup not executed for '{crit}'."

    def test_setup_logs_when_no_train_loader(self, caplog, monkeypatch):
        dm = DummyClassificationDataModule(
            root=Path(),
            batch_size=4,
            num_classes=3,
            num_images=16,
            eval_ood=True,
        )

        def _raise_train_loader(*_a, **_k):
            raise RuntimeError("no train loader")

        monkeypatch.setattr(
            ClassificationRoutine, "_hyperparam_search_ood", lambda _self: None, raising=True
        )
        monkeypatch.setattr(dm, "train_dataloader", _raise_train_loader, raising=True)

        model = dummy_ood_model(in_channels=3, feat_dim=64, num_classes=3)
        routine = ClassificationRoutine(
            model=model,
            loss=None,
            num_classes=3,
            eval_ood=True,
        )
        routine.ood_criterion = MaxSoftmaxCriterion()  # no setup() side-effects

        routine.trainer = types.SimpleNamespace(datamodule=dm)

        with caplog.at_level(logging.INFO):
            routine.setup("test")
        assert any("No train loader detected" in r.message for r in caplog.records)

    def test_create_near_far_metric_dicts_non_ensemble(self):
        model = dummy_ood_model(in_channels=3, feat_dim=64, num_classes=3)
        routine = ClassificationRoutine(
            model=model, loss=None, num_classes=3, eval_ood=True, is_ensemble=False
        )
        routine.ood_criterion = MaxSoftmaxCriterion()

        x = torch.rand(4, 3, 8, 8)
        y = torch.tensor([0, 1, 2, 0])

        class _DS:
            def __init__(self, name):
                self.dataset_name = name

        routine.trainer = types.SimpleNamespace(
            datamodule=types.SimpleNamespace(
                get_indices=lambda: {"val_ood": 9, "near_oods": [2], "far_oods": [3], "shift": []},
                near_oods=[_DS("nearX")],
                far_oods=[_DS("farY")],
            )
        )

        routine.test_step((x, y), batch_idx=0, dataloader_idx=2)  # near
        assert "nearX" in routine.test_ood_metrics_near

        routine.test_step((x, y), batch_idx=0, dataloader_idx=3)  # far
        assert "farY" in routine.test_ood_metrics_far

    def test_create_near_far_metric_dicts_ensemble_and_aggregator(self):
        model = dummy_ood_model(in_channels=3, feat_dim=64, num_classes=3)
        routine = ClassificationRoutine(
            model=model, loss=None, num_classes=3, eval_ood=True, is_ensemble=True
        )
        routine.ood_criterion = MaxSoftmaxCriterion()

        x = torch.rand(4, 3, 8, 8)
        y = torch.tensor([0, 1, 2, 0])

        class _DS:
            def __init__(self, name):
                self.dataset_name = name

        routine.trainer = types.SimpleNamespace(
            datamodule=types.SimpleNamespace(
                get_indices=lambda: {
                    "val_ood": 9,
                    "near_oods": [5],
                    "far_oods": [6],
                    "shift": [7],
                },
                near_oods=[_DS("n1")],
                far_oods=[_DS("f1")],
            )
        )

        routine.test_step((x, y), batch_idx=0, dataloader_idx=1)  # aggregator
        assert "n1" in routine.test_ood_ens_metrics_near
        assert "f1" in routine.test_ood_ens_metrics_far

        routine.test_step((x, y), batch_idx=0, dataloader_idx=5)  # near
        routine.test_step((x, y), batch_idx=0, dataloader_idx=6)  # far
        assert "n1" in routine.test_ood_ens_metrics_near
        assert "f1" in routine.test_ood_ens_metrics_far

    def test_skip_when_val_ood_loader(self):
        model = dummy_ood_model(in_channels=3, feat_dim=64, num_classes=3)
        routine = ClassificationRoutine(model=model, loss=None, num_classes=3, eval_ood=True)
        routine.ood_criterion = MaxSoftmaxCriterion()

        routine.trainer = types.SimpleNamespace(
            datamodule=types.SimpleNamespace(
                get_indices=lambda: {"val_ood": 4, "near_oods": [], "far_oods": [], "shift": []}
            )
        )
        x = torch.rand(2, 3, 8, 8)
        y = torch.tensor([0, 1])
        routine.test_step((x, y), batch_idx=0, dataloader_idx=4)

    def test_init_metrics_creates_shift_ens_metrics_when_ensemble_and_eval_shift(self):
        model = dummy_ood_model(in_channels=3, feat_dim=64, num_classes=3)
        routine = ClassificationRoutine(
            model=model, loss=None, num_classes=3, eval_shift=True, is_ensemble=True
        )
        assert hasattr(routine, "test_shift_ens_metrics")

    def test_shift_ens_update_path(self):
        model = dummy_ood_model(in_channels=3, feat_dim=64, num_classes=3)
        routine = ClassificationRoutine(
            model=model, loss=None, num_classes=3, eval_shift=True, is_ensemble=True
        )
        routine.ood_criterion = MaxSoftmaxCriterion()

        x = torch.rand(4, 3, 8, 8)
        y = torch.tensor([0, 1, 2, 0])

        routine.trainer = types.SimpleNamespace(
            datamodule=types.SimpleNamespace(
                get_indices=lambda: {"val_ood": 99, "near_oods": [], "far_oods": [], "shift": [7]}
            )
        )
        routine.test_step((x, y), batch_idx=0, dataloader_idx=7)
