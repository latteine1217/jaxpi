"""
What:
    畫出 window-1 residual loss 對比圖，並輸出對齊後的 raw CSV 與摘要文字。
Why:
    現有 artifact 只覆蓋特定 run 配對；這支腳本把「no data vs sensor」比較流程變成可重跑工具，
    方便針對新 run（例如 3324）直接復用同一口徑。
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STEP_RE = re.compile(r"Time Window:\s*1/50 \| Step:\s*(\d+)/50000")
LOSS_RE = re.compile(r"^(rc_loss|ru_loss|rv_loss)\s+([0-9.eE+-]+)")
METRICS = ("rc_loss", "ru_loss", "rv_loss")


def load_run_from_csv(csv_path: Path, run_name: str) -> dict[str, list[tuple[int, float]]]:
    """
    What:
        從既有 raw CSV 讀取單一 run 的三條 residual 曲線。
    Why:
        舊分析流程已產生結構化 CSV，直接重用可避免再解析歷史 log。
    """

    series = {metric: [] for metric in METRICS}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if row["run"] != run_name:
                continue
            step = int(row["step"])
            for metric in METRICS:
                series[metric].append((step, float(row[metric])))
    return series


def load_run_from_stderr(stderr_path: Path) -> dict[str, list[tuple[int, float]]]:
    """
    What:
        從 train stderr 解析 window-1 的 residual loss 曲線。
    Why:
        新 run 常常只有訓練 log；把 log 解析成同一格式後，才能和歷史 CSV 對齊比較。
    """

    series = {metric: [] for metric in METRICS}
    current_step = None
    current_losses: dict[str, float] = {}

    for raw_line in stderr_path.read_text(errors="ignore").splitlines():
        step_match = STEP_RE.search(raw_line)
        if step_match:
            if current_step is not None and len(current_losses) == 3:
                for metric in METRICS:
                    series[metric].append((current_step, current_losses[metric]))
            current_step = int(step_match.group(1))
            current_losses = {}
            continue

        loss_match = LOSS_RE.match(raw_line.strip())
        if loss_match and current_step is not None:
            current_losses[loss_match.group(1)] = float(loss_match.group(2))

    if current_step is not None and len(current_losses) == 3:
        for metric in METRICS:
            series[metric].append((current_step, current_losses[metric]))

    return series


def trim_series_to_shared_steps(
    base_series: dict[str, list[tuple[int, float]]],
    compare_series: dict[str, list[tuple[int, float]]],
) -> tuple[dict[str, list[tuple[int, float]]], dict[str, list[tuple[int, float]]], int]:
    """
    What:
        將兩條 run 都裁到共同可比較的 step 範圍。
    Why:
        若一條 run 提前結束，直接拿各自尾段比較會混入非 shared horizon，
        對收斂速度與尾段 residual 的判讀不公平。
    """

    shared_max_step = min(
        max(step for step, _ in base_series[metric]) for metric in METRICS
    )
    shared_max_step = min(
        shared_max_step,
        min(max(step for step, _ in compare_series[metric]) for metric in METRICS),
    )

    def _trim(series: dict[str, list[tuple[int, float]]]) -> dict[str, list[tuple[int, float]]]:
        return {
            metric: [(step, value) for step, value in series[metric] if step <= shared_max_step]
            for metric in METRICS
        }

    return _trim(base_series), _trim(compare_series), shared_max_step


def save_combined_csv(
    output_path: Path,
    base_label: str,
    base_series: dict[str, list[tuple[int, float]]],
    compare_label: str,
    compare_series: dict[str, list[tuple[int, float]]],
) -> None:
    """
    What:
        輸出兩條 run 的對齊 raw CSV。
    Why:
        圖只是摘要，研究判讀仍需要可重算的中介表格。
    """

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "step", "rc_loss", "ru_loss", "rv_loss"])
        for label, series in ((base_label, base_series), (compare_label, compare_series)):
            by_metric = {metric: dict(points) for metric, points in series.items()}
            steps = sorted(set(by_metric["rc_loss"]) & set(by_metric["ru_loss"]) & set(by_metric["rv_loss"]))
            for step in steps:
                writer.writerow(
                    [label, step, by_metric["rc_loss"][step], by_metric["ru_loss"][step], by_metric["rv_loss"][step]]
                )


def build_summary(
    base_label: str,
    base_series: dict[str, list[tuple[int, float]]],
    compare_label: str,
    compare_series: dict[str, list[tuple[int, float]]],
    shared_max_step: int,
) -> str:
    """
    What:
        產生文字摘要，記錄 threshold crossing 與尾段差異。
    Why:
        讓圖表同時附帶可掃描的定量判讀，而不是只靠視覺印象。
    """

    thresholds = (1e-3, 1e-4, 5e-5, 1e-5)
    lines = [
        "=== Window 1 Loss Comparison ===",
        f"Scope: compare shared training residuals between {base_label} and {compare_label}.",
        f"Shared horizon: step <= {shared_max_step}.",
        "",
    ]

    for metric in METRICS:
        lines.append(f"--- {metric} ---")
        for label, series in ((base_label, base_series), (compare_label, compare_series)):
            values = [value for _, value in series[metric]]
            tail = values[-50:] if len(values) >= 50 else values
            lines.append(
                f"{label}: start={values[0]:.3e}, end={values[-1]:.3e}, "
                f"tail_mean(last50)={sum(tail)/len(tail):.3e}, tail_min(last50)={min(tail):.3e}"
            )
            for threshold in thresholds:
                crossing = next((step for step, value in series[metric] if value <= threshold), None)
                lines.append(
                    f"  first <= {threshold:.0e}: {'none' if crossing is None else f'step {crossing}'}"
                )
        base_end = base_series[metric][-1][1]
        compare_end = compare_series[metric][-1][1]
        lines.append(f"end_ratio(compare/base)={compare_end / base_end:.3f}")
        lines.append("")

    return "\n".join(lines)


def plot_losses(
    output_path: Path,
    base_label: str,
    base_series: dict[str, list[tuple[int, float]]],
    compare_label: str,
    compare_series: dict[str, list[tuple[int, float]]],
    shared_max_step: int,
) -> None:
    """
    What:
        畫三張 residual loss 曲線並標示 `5e-5` crossing。
    Why:
        直接把 no-data 與 sensor 的收斂差異可視化，避免只靠文字描述。
    """

    colors = {base_label: "#1d4ed8", compare_label: "#b91c1c"}
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)

    for ax, metric in zip(axes, METRICS):
        for label, series in ((base_label, base_series), (compare_label, compare_series)):
            xs = [step for step, _ in series[metric]]
            ys = [value for _, value in series[metric]]
            ax.plot(xs, ys, label=label, color=colors[label], linewidth=1.8)
            crossing = next(((step, value) for step, value in series[metric] if value <= 5e-5), None)
            if crossing is not None:
                ax.scatter([crossing[0]], [crossing[1]], color=colors[label], s=24, zorder=5)
                ax.annotate(
                    f"{label}@{crossing[0]}",
                    (crossing[0], crossing[1]),
                    textcoords="offset points",
                    xytext=(6, -10),
                    fontsize=8,
                    color=colors[label],
                )
        ax.set_yscale("log")
        ax.set_ylabel(metric)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="upper right", fontsize=9)
        ax.axhline(5e-5, color="#374151", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.text(shared_max_step + shared_max_step * 0.01, 5e-5, "5e-5", va="center", ha="left", fontsize=8, color="#374151")
        ax.set_xlim(0, shared_max_step)

    axes[-1].set_xlabel("Training Step")
    fig.suptitle("Window 1 Residual Loss Comparison: No Data vs Sensor", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot window-1 residual loss comparison.")
    parser.add_argument("--base-csv", required=True, help="Existing raw CSV path.")
    parser.add_argument("--base-run", required=True, help="Run label inside base CSV.")
    parser.add_argument("--compare-stderr", required=True, help="Train stderr path for compare run.")
    parser.add_argument("--compare-label", required=True, help="Display label for compare run.")
    parser.add_argument("--output-dir", required=True, help="Directory for plot/text/csv outputs.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_label = "No data (3137)" if args.base_run == "3137" else args.base_run
    base_series = load_run_from_csv(Path(args.base_csv), args.base_run)
    compare_series = load_run_from_stderr(Path(args.compare_stderr))
    base_series, compare_series, shared_max_step = trim_series_to_shared_steps(base_series, compare_series)

    plot_losses(
        output_dir / "window1_loss_compare.png",
        base_label,
        base_series,
        args.compare_label,
        compare_series,
        shared_max_step,
    )
    (output_dir / "window1_loss_compare.txt").write_text(
        build_summary(base_label, base_series, args.compare_label, compare_series, shared_max_step)
    )
    save_combined_csv(
        output_dir / "window1_loss_compare_raw.csv",
        args.base_run,
        base_series,
        args.compare_label,
        compare_series,
    )


if __name__ == "__main__":
    main()
