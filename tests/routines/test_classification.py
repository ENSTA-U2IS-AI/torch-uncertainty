import importlib.util
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
        if not torch.cuda.is_available():
            pytest.skip(f"{crit} requires CUDA.")

        if crit == "knn" and importlib.util.find_spec("faiss") is None:
            pytest.skip("faiss not available for KNN criterion.")
        if crit == "adascale_a" and importlib.util.find_spec("statsmodels") is None:
            pytest.skip("statsmodels not available for AdaScale criterion.")

        trainer = TUTrainer(
            accelerator="gpu",
            devices=1,
            max_epochs=1,
            limit_train_batches=1,
            limit_val_batches=1,
            limit_test_batches=1,
            enable_checkpointing=False,
            logger=False,
        )

        dm = DummyClassificationDataModule(
            root=Path(),
            batch_size=8,
            num_classes=3,
            num_images=64,
            eval_ood=True,
        )

        # ---- monkeypatch dataset----
        class _TensorDS(Dataset):
            def __init__(self, x, y, name="tensor_ds"):
                self.x, self.y = x, y
                self.dataset_name = name

            def __len__(self):
                return self.x.shape[0]

            def __getitem__(self, i):
                return self.x[i], self.y[i]

        def _mk_split(n, c, h, w, num_classes, shift=0.0, seed=0):
            g = torch.Generator().manual_seed(seed)
            x = torch.rand((n, c, h, w), generator=g) + shift
            x = x.clamp(0, 1)
            y = torch.arange(n) % num_classes
            return x.float(), y.long()

        def patched_setup(self, stage=None):
            self.num_channels = 3
            h = w = self.image_size
            n = self.num_images

            if stage in (None, "fit"):
                x_tr, y_tr = _mk_split(n, 3, h, w, self.num_classes, shift=0.0, seed=123)
                x_va, y_va = _mk_split(n, 3, h, w, self.num_classes, shift=0.0, seed=456)
                self.train = _TensorDS(x_tr, y_tr, "train")
                self.val = _TensorDS(x_va, y_va, "val")

            if stage in (None, "test"):
                x_te, y_te = _mk_split(n, 3, h, w, self.num_classes, shift=0.0, seed=789)
                self.test = _TensorDS(x_te, y_te, "test")
                if self.eval_ood:
                    x_vo, y_vo = _mk_split(n, 3, h, w, self.num_classes, shift=0.10, seed=321)
                    self.val_ood = _TensorDS(x_vo, y_vo, "val_ood")
                    x_near, y_near = _mk_split(n, 3, h, w, self.num_classes, shift=0.15, seed=654)
                    self.near_oods = [_TensorDS(x_near, y_near, "near")]
                    x_far, y_far = _mk_split(n, 3, h, w, self.num_classes, shift=0.35, seed=987)
                    self.far_oods = [_TensorDS(x_far, y_far, "far")]

            if self.eval_shift:
                x_sh, y_sh = _mk_split(n, 3, h, w, self.num_classes, shift=0.20, seed=111)
                self.shift = _TensorDS(x_sh, y_sh, "shift")
                self.shift_severity = 1

        monkeypatch.setattr(dm, "setup", patched_setup.__get__(dm, type(dm)), raising=True)
        # ------------------------------------------------------------------------------

        model = dummy_ood_model(in_channels=3, feat_dim=4096, num_classes=dm.num_classes).to("cuda")

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
