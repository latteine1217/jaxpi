"""
What:
    以 `3137/3155` 的 all-window direct CSV 重畫 per-window error trend。
Why:
    舊版 `with_vs_without_data_window_trends_to12` 混入了 `3145 window 1 ~1e-3`
    的歷史舊 artifact；現在 `3137/3155` 都已有 all-window direct reevaluation，
    這支腳本改為直接吃 CSV，避免任何硬編碼口徑漂移。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = ("u_error", "v_error", "w_error")


def load_rows(csv_path: Path) -> list[dict[str, float]]:
    rows = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "window": int(float(row["window"])),
                    "checkpoint_step": int(float(row["checkpoint_step"])),
                    "t_end": float(row["t_end"]),
                    "u_error": float(row["u_error"]),
                    "v_error": float(row["v_error"]),
                    "w_error": float(row["w_error"]),
                }
            )
    return sorted(rows, key=lambda row: row["window"])


def align_shared_windows(no_data_rows: list[dict[str, float]], with_data_rows: list[dict[str, float]]) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    shared = sorted({row["window"] for row in no_data_rows} & {row["window"] for row in with_data_rows})
    no_data = [row for row in no_data_rows if row["window"] in shared]
    with_data = [row for row in with_data_rows if row["window"] in shared]
    return no_data, with_data


def write_summary(output_path: Path, no_data_rows: list[dict[str, float]], with_data_rows: list[dict[str, float]]) -> None:
    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    u_no = mean([row["u_error"] for row in no_data_rows])
    v_no = mean([row["v_error"] for row in no_data_rows])
    w_no = mean([row["w_error"] for row in no_data_rows])
    u_yes = mean([row["u_error"] for row in with_data_rows])
    v_yes = mean([row["v_error"] for row in with_data_rows])
    w_yes = mean([row["w_error"] for row in with_data_rows])
    windows = [row["window"] for row in no_data_rows]

    lines = [
        "=== Compare 3155 (with data) vs 3137 (no data, all-window direct) ===",
        f"windows={windows[0]}..{windows[-1]}",
        f"u_error_mean_no_data={u_no:.6f}",
        f"u_error_mean_with_data={u_yes:.6f}",
        f"v_error_mean_no_data={v_no:.6f}",
        f"v_error_mean_with_data={v_yes:.6f}",
        f"w_error_mean_no_data={w_no:.6f}",
        f"w_error_mean_with_data={w_yes:.6f}",
        f"u_ratio_with_over_no={u_yes/u_no:.6f}",
        f"v_ratio_with_over_no={v_yes/v_no:.6f}",
        f"w_ratio_with_over_no={w_yes/w_no:.6f}",
        f"window{windows[0]}_no_data=u:{no_data_rows[0]['u_error']:.6f},v:{no_data_rows[0]['v_error']:.6f},w:{no_data_rows[0]['w_error']:.6f}",
        f"window{windows[0]}_with_data=u:{with_data_rows[0]['u_error']:.6f},v:{with_data_rows[0]['v_error']:.6f},w:{with_data_rows[0]['w_error']:.6f}",
        f"window{windows[-1]}_no_data=u:{no_data_rows[-1]['u_error']:.6f},v:{no_data_rows[-1]['v_error']:.6f},w:{no_data_rows[-1]['w_error']:.6f}",
        f"window{windows[-1]}_with_data=u:{with_data_rows[-1]['u_error']:.6f},v:{with_data_rows[-1]['v_error']:.6f},w:{with_data_rows[-1]['w_error']:.6f}",
    ]
    output_path.write_text("\n".join(lines) + "\n")


def plot(output_path: Path, no_data_rows: list[dict[str, float]], with_data_rows: list[dict[str, float]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)
    labels = {"u_error": "u Relative L2", "v_error": "v Relative L2", "w_error": "vorticity Relative L2"}
    colors = {"no_data": "#1d4ed8", "with_data": "#b91c1c"}
    windows = [row["window"] for row in no_data_rows]

    for ax, metric in zip(axes, METRICS):
        ax.plot(windows, [row[metric] for row in no_data_rows], marker="o", linewidth=2.0, color=colors["no_data"], label="3137 no data")
        ax.plot(windows, [row[metric] for row in with_data_rows], marker="o", linewidth=2.0, color=colors["with_data"], label="3155 with data")
        ax.set_yscale("log")
        ax.set_xlabel("Time Window")
        ax.set_ylabel("Relative L2 error")
        ax.set_title(labels[metric])
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=9)

    fig.suptitle(f"Window {windows[0]}-{windows[-1]} Direct Error Trends: 3137 vs 3155")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Redraw corrected 3137 vs 3155 error trends.")
    parser.add_argument("--no-data-csv", required=True)
    parser.add_argument("--with-data-csv", required=True)
    parser.add_argument("--output-plot", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--thesis-plot", default=None)
    args = parser.parse_args()

    output_plot = Path(args.output_plot)
    output_summary = Path(args.output_summary)
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)

    no_data_rows = load_rows(Path(args.no_data_csv))
    with_data_rows = load_rows(Path(args.with_data_csv))
    no_data_rows, with_data_rows = align_shared_windows(no_data_rows, with_data_rows)

    plot(output_plot, no_data_rows, with_data_rows)
    write_summary(output_summary, no_data_rows, with_data_rows)

    if args.thesis_plot:
        thesis_plot = Path(args.thesis_plot)
        thesis_plot.parent.mkdir(parents=True, exist_ok=True)
        plot(thesis_plot, no_data_rows, with_data_rows)


if __name__ == "__main__":
    main()
