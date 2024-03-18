from pathlib import Path

import pytest
from lightning import Trainer
from torch import nn

from tests._dummies import (
    DummyClassificationBaseline,
    DummyClassificationDataModule,
    dummy_model,
)
from torch_uncertainty.optimization_procedures import optim_cifar10_resnet18
from torch_uncertainty.routines import ClassificationRoutine


class TestClassificationSingle:
    """Testing the classification routine with a single model."""

    def test_one_estimator_binary(self):
        trainer = Trainer(accelerator="cpu", fast_dev_run=True)

        dm = DummyClassificationDataModule(
            root=Path(),
            batch_size=16,
            num_classes=1,
            num_images=100,
        )
        model = DummyClassificationBaseline(
            in_channels=dm.num_channels,
            num_classes=dm.num_classes,
            loss=nn.BCEWithLogitsLoss,
            optimization_procedure=optim_cifar10_resnet18,
            ensemble=False,
            ood_criterion="msp",
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_two_estimators_binary(self):
        trainer = Trainer(accelerator="cpu", fast_dev_run=True)

        dm = DummyClassificationDataModule(
            root=Path(),
            batch_size=16,
            num_classes=1,
            num_images=100,
        )
        model = DummyClassificationBaseline(
            in_channels=dm.num_channels,
            num_classes=dm.num_classes,
            loss=nn.BCEWithLogitsLoss,
            optimization_procedure=optim_cifar10_resnet18,
            ensemble=True,
            ood_criterion="logit",
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_one_estimator_two_classes(self):
        trainer = Trainer(accelerator="cpu", fast_dev_run=True)

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
            loss=nn.CrossEntropyLoss,
            optimization_procedure=optim_cifar10_resnet18,
            ensemble=False,
            ood_criterion="entropy",
            eval_ood=True,
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_two_estimators_two_classes(self):
        trainer = Trainer(accelerator="cpu", fast_dev_run=True)

        dm = DummyClassificationDataModule(
            root=Path(),
            batch_size=16,
            num_classes=2,
            num_images=100,
        )
        model = DummyClassificationBaseline(
            num_classes=dm.num_classes,
            in_channels=dm.num_channels,
            loss=nn.CrossEntropyLoss,
            optimization_procedure=optim_cifar10_resnet18,
            ensemble=True,
            ood_criterion="energy",
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_classification_failures(self):
        # num_estimators
        with pytest.raises(ValueError):
            ClassificationRoutine(10, nn.Module(), None, num_estimators=-1)
        # single & MI
        with pytest.raises(ValueError):
            ClassificationRoutine(
                10, nn.Module(), None, num_estimators=1, ood_criterion="mi"
            )

        with pytest.raises(ValueError):
            ClassificationRoutine(10, nn.Module(), None, ood_criterion="other")

        with pytest.raises(ValueError):
            ClassificationRoutine(10, nn.Module(), None, cutmix_alpha=-1)

        with pytest.raises(ValueError):
            ClassificationRoutine(
                10, nn.Module(), None, eval_grouping_loss=True
            )

        with pytest.raises(NotImplementedError):
            ClassificationRoutine(
                10, nn.Module(), None, 2, eval_grouping_loss=True
            )

        model = dummy_model(1, 1, 1, 0, with_feats=False, with_linear=True)
        with pytest.raises(ValueError):
            ClassificationRoutine(10, model, None, eval_grouping_loss=True)

        model = dummy_model(1, 1, 1, 0, with_feats=True, with_linear=False)
        with pytest.raises(ValueError):
            ClassificationRoutine(10, model, None, eval_grouping_loss=True)


# from functools import partial
# from pathlib import Path

# import pytest
# from cli_test_helpers import ArgvContext
# from torch import nn

# from tests._dummies import (
#     DummyClassificationBaseline,
#     DummyClassificationDataModule,
#     DummyClassificationDataset,
#     dummy_model,
# )
# from torch_uncertainty import cli_main, init_args
# from torch_uncertainty.losses import DECLoss, ELBOLoss

# with ArgvContext(
#     "file.py",
#     "--eval-ood",
#     "--entropy",
#     "--cutmix_alpha",
#     "0.5",
#     "--mixtype",
#     "timm",
# ):
#     args = init_args(
#         DummyClassificationBaseline, DummyClassificationDataModule
#     )

#             args.root = str(root / "data")
#             dm = DummyClassificationDataModule(**vars(args))

#             model = DummyClassificationBaseline(
#                 num_classes=dm.num_classes,
#                 in_channels=dm.num_channels,
#                 loss=DECLoss,
#                 optimization_procedure=optim_cifar10_resnet18,
#                 ensemble=False,
#                 **vars(args),
#             )
#             with pytest.raises(NotImplementedError):
#                 cli_main(model, dm, root, "logs/dummy", args)

#     def test_cli_main_dummy_mixup_ts_cv(self):
#         root = Path(__file__).parent.absolute().parents[0]
#         with ArgvContext(
#             "file.py",
#             "--mixtype",
#             "kernel_warping",
#             "--mixup_alpha",
#             "1.",
#             "--dist_sim",
#             "inp",
#             "--val_temp_scaling",
#             "--use_cv",
#         ):
#             args = init_args(DummyClassificationBaseline, DummyClassificationDataModule)

# args.root = str(root / "data")
# dm = DummyClassificationDataModule(num_classes=10, **vars(args))
# dm.dataset = (
#     lambda root,
#     num_channels,
#     num_classes,
#     image_size,
#     transform,
#     num_images: DummyClassificationDataset(
#         root,
#         num_channels=num_channels,
#         num_classes=num_classes,
#         image_size=image_size,
#         transform=transform,
#         num_images=20,
#     )
# )

#             list_dm = dm.make_cross_val_splits(2, 1)
#             list_model = [
#                 DummyClassificationBaseline(
#                     num_classes=list_dm[i].dm.num_classes,
#                     in_channels=list_dm[i].dm.num_channels,
#                     loss=nn.CrossEntropyLoss,
#                     optimization_procedure=optim_cifar10_resnet18,
#                     ensemble=False,
#                     calibration_set=dm.get_val_set,
#                     **vars(args),
#                 )
#                 for i in range(len(list_dm))
#             ]

#             cli_main(list_model, list_dm, root, "logs/dummy", args)

#         with ArgvContext(
#             "file.py",
#             "--mixtype",
#             "kernel_warping",
#             "--mixup_alpha",
#             "1.",
#             "--dist_sim",
#             "emb",
#             "--val_temp_scaling",
#             "--use_cv",
#         ):
#             args = init_args(DummyClassificationBaseline, DummyClassificationDataModule)

# args.root = str(root / "data")
# dm = DummyClassificationDataModule(num_classes=10, **vars(args))
# dm.dataset = (
#     lambda root,
#     num_channels,
#     num_classes,
#     image_size,
#     transform,
#     num_images: DummyClassificationDataset(
#         root,
#         num_channels=num_channels,
#         num_classes=num_classes,
#         image_size=image_size,
#         transform=transform,
#         num_images=20,
#     )
# )

#             list_dm = dm.make_cross_val_splits(2, 1)
#             list_model = []
#             for i in range(len(list_dm)):
#                 list_model.append(
#                     DummyClassificationBaseline(
#                         num_classes=list_dm[i].dm.num_classes,
#                         in_channels=list_dm[i].dm.num_channels,
#                         loss=nn.CrossEntropyLoss,
#                         optimization_procedure=optim_cifar10_resnet18,
#                         ensemble=False,
#                         calibration_set=dm.get_val_set,
#                         **vars(args),
#                     )
#                 )

#             cli_main(list_model, list_dm, root, "logs/dummy", args)
#         with ArgvContext("file.py", "--mutual_information"):
#             args = init_args(DummyClassificationBaseline, DummyClassificationDataModule)

#             # datamodule
#             args.root = str(root / "data")
#             dm = DummyClassificationDataModule(num_classes=1, **vars(args))

#             model = DummyClassificationBaseline(
#                 num_classes=dm.num_classes,
#                 in_channels=dm.num_channels,
#                 loss=nn.BCEWithLogitsLoss,
#                 optimization_procedure=optim_cifar10_resnet18,
#                 ensemble=True,
#                 **vars(args),
#             )

#             cli_main(model, dm, root, "logs/dummy", args)


# with ArgvContext("file.py", "--eval-ood", "--variation_ratio"):
#     args = init_args(
#         DummyClassificationBaseline, DummyClassificationDataModule
#     )

#             # datamodule
#             args.root = str(root / "data")
#             dm = DummyClassificationDataModule(**vars(args))

#             model = DummyClassificationBaseline(
#                 num_classes=dm.num_classes,
#                 in_channels=dm.num_channels,
#                 loss=nn.CrossEntropyLoss,
#                 optimization_procedure=optim_cifar10_resnet18,
#                 ensemble=True,
#                 **vars(args),
#             )

#             cli_main(model, dm, root, "logs/dummy", args)

#     def test_classification_failures(self):
#         with pytest.raises(ValueError):
#             ClassificationRoutine(
#                 10,
#                 nn.Module(),
#                 None,
#                 None,
#                 2,
#                 use_entropy=True,
#                 use_variation_ratio=True,
#             )
