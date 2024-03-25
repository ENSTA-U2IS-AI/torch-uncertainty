from pathlib import Path

import pytest
from torch import nn

from tests._dummies import (
    DummyClassificationBaseline,
    DummyClassificationDataModule,
    dummy_model,
)
from torch_uncertainty.losses import DECLoss, ELBOLoss
from torch_uncertainty.optim_recipes import optim_cifar10_resnet18
from torch_uncertainty.routines import ClassificationRoutine
from torch_uncertainty.utils import TUTrainer


class TestClassification:
    """Testing the classification routine."""

    def test_one_estimator_binary(self):
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
            optim_recipe=optim_cifar10_resnet18,
            baseline_type="single",
            ood_criterion="msp",
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_two_estimators_binary(self):
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
            optim_recipe=optim_cifar10_resnet18,
            baseline_type="single",
            ood_criterion="logit",
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_one_estimator_two_classes(self):
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
            optim_recipe=optim_cifar10_resnet18,
            baseline_type="single",
            ood_criterion="entropy",
            eval_ood=True,
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_one_estimator_two_classes_timm(self):
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
            optim_recipe=optim_cifar10_resnet18,
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

    def test_one_estimator_two_classes_mixup(self):
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
            optim_recipe=optim_cifar10_resnet18,
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

    def test_one_estimator_two_classes_mixup_io(self):
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
            optim_recipe=optim_cifar10_resnet18,
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

    def test_one_estimator_two_classes_regmixup(self):
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
            optim_recipe=optim_cifar10_resnet18,
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

    def test_one_estimator_two_classes_kernel_warping_emb(self):
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
            optim_recipe=optim_cifar10_resnet18,
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

    def test_one_estimator_two_classes_kernel_warping_inp(self):
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
            optim_recipe=optim_cifar10_resnet18,
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

    def test_one_estimator_two_classes_calibrated_with_ood(self):
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
            optim_recipe=optim_cifar10_resnet18,
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

    def test_two_estimators_two_classes_mi(self):
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
            loss=DECLoss(1, 1e-2),
            optim_recipe=optim_cifar10_resnet18,
            baseline_type="ensemble",
            ood_criterion="mi",
            eval_ood=True,
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_two_estimator_two_classes_elbo_vr_logs(self):
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
            loss=ELBOLoss(
                None, nn.CrossEntropyLoss(), kl_weight=1.0, num_samples=4
            ),
            optim_recipe=optim_cifar10_resnet18,
            baseline_type="ensemble",
            ood_criterion="vr",
            eval_ood=True,
            save_in_csv=True,
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_classification_failures(self):
        # num_estimators
        with pytest.raises(ValueError):
            ClassificationRoutine(
                num_classes=10, model=nn.Module(), loss=None, num_estimators=-1
            )
        # num_classes
        with pytest.raises(ValueError):
            ClassificationRoutine(num_classes=0, model=nn.Module(), loss=None)
        # single & MI
        with pytest.raises(ValueError):
            ClassificationRoutine(
                num_classes=10,
                model=nn.Module(),
                loss=None,
                num_estimators=1,
                ood_criterion="mi",
            )
        with pytest.raises(ValueError):
            ClassificationRoutine(
                num_classes=10,
                model=nn.Module(),
                loss=None,
                ood_criterion="other",
            )

        with pytest.raises(ValueError):
            ClassificationRoutine(
                num_classes=10, model=nn.Module(), loss=None, cutmix_alpha=-1
            )

        with pytest.raises(ValueError):
            ClassificationRoutine(
                num_classes=10,
                model=nn.Module(),
                loss=None,
                eval_grouping_loss=True,
            )

        with pytest.raises(NotImplementedError):
            ClassificationRoutine(
                num_classes=10,
                model=nn.Module(),
                loss=None,
                num_estimators=2,
                eval_grouping_loss=True,
            )

        model = dummy_model(1, 1, 0, with_feats=False, with_linear=True)
        with pytest.raises(ValueError):
            ClassificationRoutine(
                num_classes=10, model=model, loss=None, eval_grouping_loss=True
            )

        model = dummy_model(1, 1, 0, with_feats=True, with_linear=False)
        with pytest.raises(ValueError):
            ClassificationRoutine(
                num_classes=10, model=model, loss=None, eval_grouping_loss=True
            )
