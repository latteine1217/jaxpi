"""
What:
    比較 window-1 的 residual loss 與 corrected field error 是否呈現一致趨勢。
Why:
    研究上需要回答「能不能用 loss 代替 error 判斷訓練狀態」；這支腳本把
    `3137`、`3155`、`3324` 放到同一張圖，避免再分散讀多份 artifact。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


LOSS_METRICS = ("rc_loss", "ru_loss", "rv_loss")
ERROR_METRICS = ("u_error", "v_error", "w_error")
RUN_ORDER = ("3137", "3155", "3324")
RUN_LABELS = {
    "3137": "3137 no data",
    "3155": "3155 sensor100",
    "3324": "3324 dw=23.1429",
}
RUN_COLORS = {
    "3137": "#1d4ed8",
    "3155": "#b91c1c",
    "3324": "#0f766e",
}


def load_loss_series(csv_path: Path) -> dict[str, dict[str, list[tuple[int, float]]]]:
    """
    What:
        讀取已整理好的 raw loss CSV。
    Why:
        直接重用既有 artifact，避免再次解析遠端 log。
    """

    rows: dict[str, dict[str, list[tuple[int, float]]]] = {
        run: {metric: [] for metric in LOSS_METRICS} for run in RUN_ORDER
    }
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            run = row["run"]
            if run not in rows:
                continue
            step = int(row["step"])
            for metric in LOSS_METRICS:
                rows[run][metric].append((step, float(row[metric])))
    return rows


def trim_to_shared_horizon(
    series_by_run: dict[str, dict[str, list[tuple[int, float]]]]
) -> tuple[dict[str, dict[str, list[tuple[int, float]]]], int]:
    """
    What:
        將所有 run 裁到共同 step 範圍。
    Why:
        `3324` 只跑到 50000 steps；若直接拿 `3137/3155` 的 100000-step 尾段比較，
        會把不同 horizon 混在一起。
    """

    shared_max_step = min(
        min(max(step for step, _ in series_by_run[run][metric]) for metric in LOSS_METRICS)
        for run in RUN_ORDER
    )
    trimmed = {
        run: {
            metric: [(step, value) for step, value in series_by_run[run][metric] if step <= shared_max_step]
            for metric in LOSS_METRICS
        }
        for run in RUN_ORDER
    }
    return trimmed, shared_max_step


def load_error_points(
    errors_3137: Path,
    errors_3155: Path,
    errors_3324: Path,
) -> dict[str, dict[str, float]]:
    """
    What:
        讀取三條 run 的 window-1 corrected field error 代表點。
    Why:
        這張圖要回答的是「loss 趨勢和 error 趨勢是否一致」，因此需要把 loss
        與最終可用的 field-quality 指標放在同一個框架中。
    """

    result: dict[str, dict[str, float]] = {}

    with errors_3137.open() as f:
        rows = list(csv.DictReader(f))
    row_3137 = max(rows, key=lambda row: int(row["checkpoint_step"]))
    result["3137"] = {
        "step": int(row_3137["checkpoint_step"]),
        **{metric: float(row_3137[metric]) for metric in ERROR_METRICS},
    }

    with errors_3155.open() as f:
        rows = list(csv.DictReader(f))
    row_3155 = next(row for row in rows if int(float(row["window"])) == 1)
    result["3155"] = {
        "step": int(float(row_3155["checkpoint_step"])),
        **{metric: float(row_3155[metric]) for metric in ERROR_METRICS},
    }

    with errors_3324.open() as f:
        rows = list(csv.DictReader(f))
    row_3324 = max(rows, key=lambda row: int(row["step"]))
    result["3324"] = {
        "step": int(row_3324["step"]),
        **{metric: float(row_3324[metric]) for metric in ERROR_METRICS},
    }

    return result


def find_crossing(series: list[tuple[int, float]], threshold: float) -> int | None:
    for step, value in series:
        if value <= threshold:
            return step
    return None


def build_summary(
    shared_loss: dict[str, dict[str, list[tuple[int, float]]]],
    shared_max_step: int,
    error_points: dict[str, dict[str, float]],
) -> str:
    """
    What:
        輸出可掃描的文字摘要。
    Why:
        圖可視化趨勢，但研究結論仍需要明確標記 crossing / tail / error 三類量。
    """

    thresholds = (1e-4, 5e-5, 1e-5)
    lines = [
        "=== Loss vs Error Alignment (Window 1) ===",
        f"Shared loss horizon: step <= {shared_max_step}",
        "Runs: 3137 no data, 3155 sensor100, 3324 dw=23.1429",
        "",
    ]

    for metric in LOSS_METRICS:
        lines.append(f"--- {metric} ---")
        for run in RUN_ORDER:
            values = [value for _, value in shared_loss[run][metric]]
            tail = values[-50:] if len(values) >= 50 else values
            lines.append(
                f"{RUN_LABELS[run]}: end={values[-1]:.3e}, tail_mean(last50)={sum(tail)/len(tail):.3e}"
            )
            for threshold in thresholds:
                crossing = find_crossing(shared_loss[run][metric], threshold)
                lines.append(
                    f"  first <= {threshold:.0e}: {'none' if crossing is None else f'step {crossing}'}"
                )
        lines.append("")

    lines.append("--- corrected field error ---")
    for run in RUN_ORDER:
        point = error_points[run]
        lines.append(
            f"{RUN_LABELS[run]} @ checkpoint_{point['step']}: "
            f"u={point['u_error']:.6e}, v={point['v_error']:.6e}, w={point['w_error']:.6e}"
        )

    lines.extend(
        [
            "",
            "Interpretation:",
            "- loss trend 可以反映訓練是否健康，但 run 之間的 loss crossing 順序不保證對應 field error 排名。",
            "- 3324 在部分 residual crossing 上不差，但 corrected field error 明顯差於 3137/3155。",
            "- 3155 和 3137 的 loss curve 幾乎重疊，但 shared-window mean error 並未因此出現整體改善。",
        ]
    )
    return "\n".join(lines) + "\n"


def add_text_panel(
    ax: plt.Axes,
    shared_loss: dict[str, dict[str, list[tuple[int, float]]]],
    error_points: dict[str, dict[str, float]],
) -> None:
    """
    What:
        在圖上補一個可直接讀的定量摘要框。
    Why:
        把結論寫進同一張圖，避免使用者還要再對照文字檔。
    """

    lines = ["Key Takeaways"]
    for metric in LOSS_METRICS:
        rcross = find_crossing(shared_loss["3137"][metric], 5e-5)
        pcross = find_crossing(shared_loss["3155"][metric], 5e-5)
        scross = find_crossing(shared_loss["3324"][metric], 5e-5)
        lines.append(
            f"{metric} <= 5e-5: 3137={rcross}, 3155={pcross}, 3324={scross}"
        )

    lines.extend(
        [
            "",
            "Field Error @ chosen ckpt",
            f"3137  w={error_points['3137']['w_error']:.2e}",
            f"3155  w={error_points['3155']['w_error']:.2e}",
            f"3324  w={error_points['3324']['w_error']:.2e}",
            "",
            "Loss rank != Error rank",
            "Use loss for health/progress.",
            "Use corrected error for selection.",
        ]
    )

    ax.axis("off")
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
    )


def plot(
    output_path: Path,
    shared_loss: dict[str, dict[str, list[tuple[int, float]]]],
    shared_max_step: int,
    error_points: dict[str, dict[str, float]],
) -> None:
    """
    What:
        產生 loss 與 error 對照圖。
    Why:
        用單一 artifact 回答「loss 趨勢是否等於 error 趨勢」這個研究問題。
    """

    fig = plt.figure(figsize=(18, 10))
    grid = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 0.95])
    loss_axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[0, 2])]

    for ax, metric in zip(loss_axes, LOSS_METRICS):
        for run in RUN_ORDER:
            xs = [step for step, _ in shared_loss[run][metric]]
            ys = [value for _, value in shared_loss[run][metric]]
            ax.plot(xs, ys, linewidth=2.0, color=RUN_COLORS[run], label=RUN_LABELS[run])
        ax.set_yscale("log")
        ax.set_xlim(0, shared_max_step)
        ax.set_xlabel("Training Step")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.grid(True, which="both", alpha=0.25)
        ax.axhline(5e-5, color="#475569", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.legend(fontsize=9)

    error_ax = fig.add_subplot(grid[1, 0:2])
    x = list(range(len(ERROR_METRICS)))
    width = 0.22
    offsets = {"3137": -width, "3155": 0.0, "3324": width}
    for run in RUN_ORDER:
        error_ax.bar(
            [idx + offsets[run] for idx in x],
            [error_points[run][metric] for metric in ERROR_METRICS],
            width=width,
            color=RUN_COLORS[run],
            label=f"{RUN_LABELS[run]} @ {error_points[run]['step']}",
        )
    error_ax.set_xticks(x)
    error_ax.set_xticklabels(["u error", "v error", "w error"])
    error_ax.set_yscale("log")
    error_ax.set_ylabel("Relative L2 error")
    error_ax.set_title("Corrected Field Error")
    error_ax.grid(True, which="both", axis="y", alpha=0.25)
    error_ax.legend(fontsize=9)

    text_ax = fig.add_subplot(grid[1, 2])
    add_text_panel(text_ax, shared_loss, error_points)

    fig.suptitle("Window 1 Loss vs Error Alignment")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot loss vs error alignment for window 1.")
    parser.add_argument("--loss-csv", required=True)
    parser.add_argument("--errors-3137", required=True)
    parser.add_argument("--errors-3155", required=True)
    parser.add_argument("--errors-3324", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_loss = load_loss_series(Path(args.loss_csv))
    shared_loss, shared_max_step = trim_to_shared_horizon(all_loss)
    error_points = load_error_points(
        Path(args.errors_3137),
        Path(args.errors_3155),
        Path(args.errors_3324),
    )

    plot(output_dir / "window1_loss_error_alignment.png", shared_loss, shared_max_step, error_points)
    (output_dir / "window1_loss_error_alignment.txt").write_text(
        build_summary(shared_loss, shared_max_step, error_points)
    )


if __name__ == "__main__":
    main()
