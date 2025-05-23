from collections import OrderedDict, defaultdict

from lightning.pytorch.loops.evaluation_loop import _EvaluationLoop
from lightning.pytorch.trainer.connectors.logger_connector.result import (
    _OUT_DICT,
)
from rich import get_console
from rich.box import HEAVY_EDGE
from rich.console import Group
from rich.table import Table
from torch import Tensor

PERCENTAGE_METRICS = [
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


def _add_row(table: Table, metric_name: str, value: Tensor) -> None:
    if metric_name in PERCENTAGE_METRICS:
        value = value * 100
        table.add_row(metric_name, f"{value.item():.3f}%")
    else:
        table.add_row(metric_name, f"{value.item():.5f}")


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
                    # Initialize the ood dict if it isnt already.
                    if "ood" not in metrics:
                        metrics["ood"] = {
                            "individual": {"near": {}, "far": {}},
                            "NearOOD": {"auroc": [], "fpr95": [], "aupr": []},
                            "FarOOD": {"auroc": [], "fpr95": [], "aupr": []},
                        }
                    # Here, we expect keys of the form: "ood_{group}_{datasetName}_{metric}"
                    parts = key.split("_")
                    group = parts[1].lower()  # either "near" or "far"
                    dataset_name = parts[2]
                    metric_name = parts[-1].lower()  # e.g. "auroc", "fpr95", "aupr"

                    # Store individual dataset results grouped by near or far.
                    if dataset_name not in metrics["ood"]["individual"][group]:
                        metrics["ood"]["individual"][group][dataset_name] = {}
                    metrics["ood"]["individual"][group][dataset_name][metric_name] = value

                    # Also, add this value to the corresponding average accumulator.
                    if group == "near":
                        metrics["ood"]["NearOOD"][metric_name].append(value)
                    elif group == "far":
                        metrics["ood"]["FarOOD"][metric_name].append(value)

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
            table.add_column(first_col_name, justify="center", style="cyan", width=12)
            table.add_column("Classification", justify="center", style="magenta", width=25)
            cls_metrics = OrderedDict(sorted(metrics["cls"].items()))
            for metric_name, value in cls_metrics.items():
                _add_row(table, metric_name, value)
            tables.append(table)

        if "seg" in metrics:
            table = Table()
            table.add_column(first_col_name, justify="center", style="cyan", width=12)
            table.add_column("Segmentation", justify="center", style="magenta", width=25)
            seg_metrics = OrderedDict(sorted(metrics["seg"].items()))
            for metric_name, value in seg_metrics.items():
                _add_row(table, metric_name, value)
            tables.append(table)

        if "reg" in metrics:
            table = Table()
            table.add_column(first_col_name, justify="center", style="cyan", width=12)
            table.add_column("Regression", justify="center", style="magenta", width=25)
            reg_metrics = OrderedDict(sorted(metrics["reg"].items()))
            for metric_name, value in reg_metrics.items():
                _add_row(table, metric_name, value)
            tables.append(table)

        if "cal" in metrics:
            table = Table()
            table.add_column(first_col_name, justify="center", style="cyan", width=12)
            table.add_column("Calibration", justify="center", style="magenta", width=25)
            cal_metrics = OrderedDict(sorted(metrics["cal"].items()))
            for metric_name, value in cal_metrics.items():
                _add_row(table, metric_name, value)
            tables.append(table)

        if "ood" in metrics:
            final_ood_results = defaultdict(lambda: {"auroc": None, "fpr95": None, "aupr": None})

            for key, val in metrics["ood"].items():
                parts = key.split("_")

                if len(parts) == 2:
                    dataset_name, metric_postfix = parts[0], parts[1].lower()

                    if metric_postfix in ["auroc", "fpr95", "aupr"]:
                        final_ood_results[dataset_name][metric_postfix] = val

            for key in ["NearOOD", "FarOOD"]:
                if key in metrics["ood"]:
                    for key2, val2 in metrics["ood"][key].items():
                        final_ood_results[key][key2] = val2

            table = Table(
                title="[bold]OOD Results[/bold]",
                box=HEAVY_EDGE,
                show_header=True,
                show_lines=False,
            )
            table.add_column("Dataset", justify="center", style="cyan", width=16)
            table.add_column("AUROC", justify="center", style="magenta", width=12)
            table.add_column("FPR95", justify="center", style="magenta", width=12)
            table.add_column("AUPR", justify="center", style="magenta", width=12)

            def format_val(val):
                if val is None:
                    return "N/A"
                # If we have a list, compute the average.
                if isinstance(val, list) and len(val) > 0:
                    val = sum(val) / len(val)
                return f"{val.item() * 100:.3f}%" if val is not None else "N/A"

            # First output the Near OOD individual rows.
            for dataset_name, m_dict in metrics["ood"]["individual"]["near"].items():
                row_auroc = format_val(m_dict.get("auroc"))
                row_fpr95 = format_val(m_dict.get("fpr95"))
                row_aupr = format_val(m_dict.get("aupr"))
                table.add_row(f"{dataset_name}", row_auroc, row_fpr95, row_aupr)

            # Then add the NearOOD average row.
            near_avg = metrics["ood"]["NearOOD"]
            table.add_row(
                "NearOOD Average",
                format_val(near_avg.get("auroc")),
                format_val(near_avg.get("fpr95")),
                format_val(near_avg.get("aupr")),
            )

            # Next output the Far OOD individual rows.
            for dataset_name, m_dict in metrics["ood"]["individual"]["far"].items():
                row_auroc = format_val(m_dict.get("auroc"))
                row_fpr95 = format_val(m_dict.get("fpr95"))
                row_aupr = format_val(m_dict.get("aupr"))
                table.add_row(f"{dataset_name}", row_auroc, row_fpr95, row_aupr)

            # And add the FarOOD average row.
            far_avg = metrics["ood"]["FarOOD"]
            table.add_row(
                "FarOOD Average",
                format_val(far_avg.get("auroc")),
                format_val(far_avg.get("fpr95")),
                format_val(far_avg.get("aupr")),
            )

            tables.append(table)

        if "sc" in metrics:
            table = Table()
            table.add_column(first_col_name, justify="center", style="cyan", width=12)
            table.add_column(
                "Selective Classification",
                justify="center",
                style="magenta",
                width=25,
            )
            sc_metrics = OrderedDict(sorted(metrics["sc"].items()))
            for metric_name, value in sc_metrics.items():
                _add_row(table, metric_name, value)
            tables.append(table)

        if "post" in metrics:
            table = Table()
            table.add_column(first_col_name, justify="center", style="cyan", width=12)
            table.add_column("Post-Processing", justify="center", style="magenta", width=25)
            post_metrics = OrderedDict(sorted(metrics["post"].items()))
            for metric_name, value in post_metrics.items():
                _add_row(table, metric_name, value)
            tables.append(table)

        if "shift" in metrics:
            table = Table()
            table.add_column(first_col_name, justify="center", style="cyan", width=12)
            shift_severity = int(metrics["shift"]["severity"])
            table.add_column(
                f"Distribution Shift lvl{shift_severity}",
                justify="center",
                style="magenta",
                width=25,
            )
            shift_metrics = OrderedDict(sorted(metrics["shift"].items()))
            for metric_name, value in shift_metrics.items():
                if metric_name == "severity":
                    continue
                _add_row(table, metric_name, value)
            tables.append(table)

        console = get_console()
        group = Group(*tables)
        console.print(group)
