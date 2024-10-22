from collections import OrderedDict

from lightning.pytorch.loops.evaluation_loop import _EvaluationLoop
from lightning.pytorch.trainer.connectors.logger_connector.result import (
    _OUT_DICT,
)
from rich import get_console
from rich.console import Group
from rich.table import Table


class TUEvaluationLoop(_EvaluationLoop):
    @staticmethod
    def _print_results(results: list[_OUT_DICT], stage: str) -> None:
        # test/cls: Classification Metrics
        # test/cal: Calibration Metrics
        # ood: OOD Detection Metrics
        # shift: Distribution shift Metrics
        # test/sc: Selective Classification Metrics
        # test/post: Post-Processing Metrics
        # test/seg: Segmentation Metrics

        # In percentage
        percentage_metrics = [
            "Acc",
            "AUPR",
            "AUROC",
            "FPR95",
            "Cov@5Risk",
            "Risk@80Cov",
            "pixAcc",
            "mIoU",
            "AURC",
            "AUGRC",
            "mAcc",
        ]

        metrics = {}
        for result in results:
            for key, value in result.items():
                if key.startswith("test/cls"):
                    if "cls" not in metrics:
                        metrics["cls"] = {}
                    metric_name = key.split("/")[-1]
                    metrics["cls"].update({metric_name: value})
                elif key.startswith("test/cal"):
                    if "cal" not in metrics:
                        metrics["cal"] = {}
                    metric_name = key.split("/")[-1]
                    metrics["cal"].update({metric_name: value})
                elif key.startswith("ood"):
                    if "ood" not in metrics:
                        metrics["ood"] = {}
                    metric_name = key.split("/")[-1]
                    metrics["ood"].update({metric_name: value})
                elif key.startswith("shift"):
                    if "shift" not in metrics:
                        metrics["shift"] = {}
                    metric_name = key.split("/")[-1]
                    metrics["shift"].update({metric_name: value})
                elif key.startswith("test/sc"):
                    if "sc" not in metrics:
                        metrics["sc"] = {}
                    metric_name = key.split("/")[-1]
                    metrics["sc"].update({metric_name: value})
                elif key.startswith("test/post"):
                    if "post" not in metrics:
                        metrics["post"] = {}
                    metric_name = key.split("/")[-1]
                    metrics["post"].update({metric_name: value})
                elif key.startswith("test/seg"):
                    if "seg" not in metrics:
                        metrics["seg"] = {}
                    metric_name = key.split("/")[-1]
                    metrics["seg"].update({metric_name: value})
                elif key.startswith("test/reg"):
                    if "reg" not in metrics:
                        metrics["reg"] = {}
                    metric_name = key.split("/")[-1]
                    metrics["reg"].update({metric_name: value})

        tables = []

        first_col_name = f"{stage.capitalize()} metric"

        if "cls" in metrics:
            table = Table()
            table.add_column(
                first_col_name, justify="center", style="cyan", width=12
            )
            table.add_column(
                "Classification", justify="center", style="magenta", width=25
            )
            cls_metrics = OrderedDict(sorted(metrics["cls"].items()))
            for metric, value in cls_metrics.items():
                if metric in percentage_metrics:
                    value = value * 100
                    table.add_row(metric, f"{value.item():.2f}%")
                else:
                    table.add_row(metric, f"{value.item():.5f}")
            tables.append(table)

        if "seg" in metrics:
            table = Table()
            table.add_column(
                first_col_name, justify="center", style="cyan", width=12
            )
            table.add_column(
                "Segmentation", justify="center", style="magenta", width=25
            )
            seg_metrics = OrderedDict(sorted(metrics["seg"].items()))
            for metric, value in seg_metrics.items():
                if metric in percentage_metrics:
                    value = value * 100
                    table.add_row(metric, f"{value.item():.2f}%")
                else:
                    table.add_row(metric, f"{value.item():.5f}")
            tables.append(table)

        if "reg" in metrics:
            table = Table()
            table.add_column(
                first_col_name, justify="center", style="cyan", width=12
            )
            table.add_column(
                "Regression", justify="center", style="magenta", width=25
            )
            reg_metrics = OrderedDict(sorted(metrics["reg"].items()))
            for metric, value in reg_metrics.items():
                if metric in percentage_metrics:  # coverage: ignore
                    value = value * 100
                    table.add_row(metric, f"{value.item():.2f}%")
                else:
                    table.add_row(metric, f"{value.item():.5f}")
            tables.append(table)

        if "cal" in metrics:
            table = Table()
            table.add_column(
                first_col_name, justify="center", style="cyan", width=12
            )
            table.add_column(
                "Calibration", justify="center", style="magenta", width=25
            )
            cal_metrics = OrderedDict(sorted(metrics["cal"].items()))
            for metric, value in cal_metrics.items():
                if metric in percentage_metrics:
                    value = value * 100
                    table.add_row(metric, f"{value.item():.2f}%")
                else:
                    table.add_row(metric, f"{value.item():.5f}")
            tables.append(table)

        if "ood" in metrics:
            table = Table()
            table.add_column(
                first_col_name, justify="center", style="cyan", width=12
            )
            table.add_column(
                "OOD Detection", justify="center", style="magenta", width=25
            )
            ood_metrics = OrderedDict(sorted(metrics["ood"].items()))
            for metric, value in ood_metrics.items():
                if metric in percentage_metrics:
                    value = value * 100
                    table.add_row(metric, f"{value.item():.2f}%")
                else:
                    table.add_row(metric, f"{value.item():.5f}")
            tables.append(table)

        if "sc" in metrics:
            table = Table()
            table.add_column(
                first_col_name, justify="center", style="cyan", width=12
            )
            table.add_column(
                "Selective Classification",
                justify="center",
                style="magenta",
                width=25,
            )
            sc_metrics = OrderedDict(sorted(metrics["sc"].items()))
            for metric, value in sc_metrics.items():
                if metric in percentage_metrics:
                    value = value * 100
                    table.add_row(metric, f"{value.item():.2f}%")
                else:
                    table.add_row(metric, f"{value.item():.5f}")
            tables.append(table)

        if "post" in metrics:
            table = Table()
            table.add_column(
                first_col_name, justify="center", style="cyan", width=12
            )
            table.add_column(
                "Post-Processing", justify="center", style="magenta", width=25
            )
            post_metrics = OrderedDict(sorted(metrics["post"].items()))
            for metric, value in post_metrics.items():
                if metric in percentage_metrics:
                    value = value * 100
                    table.add_row(metric, f"{value.item():.2f}%")
                else:
                    table.add_row(metric, f"{value.item():.5f}")
            tables.append(table)

        if "shift" in metrics:
            table = Table()
            table.add_column(
                first_col_name, justify="center", style="cyan", width=12
            )
            shift_severity = int(metrics["shift"]["shift_severity"])
            table.add_column(
                f"Distribution Shift lvl{shift_severity}",
                justify="center",
                style="magenta",
                width=25,
            )
            shift_metrics = OrderedDict(sorted(metrics["shift"].items()))
            for metric, value in shift_metrics.items():
                if metric == "shift_severity":
                    continue
                if metric in percentage_metrics:
                    value = value * 100
                    table.add_row(metric, f"{value.item():.2f}%")
                else:
                    table.add_row(metric, f"{value.item():.5f}")
            tables.append(table)

        console = get_console()
        group = Group(*tables)
        console.print(group)
