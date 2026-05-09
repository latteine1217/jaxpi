"""
What:
    將 `window 1` 的關鍵 checkpoint error 點畫成同一張 `step vs error` 對照圖。
Why:
    釐清 `3324@50000`、`3329@50000/100000`、`3330@50000/100000` 在同一口徑下的
    field-error 位置，區分前期收斂速度與最終品質。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = ("u_error", "v_error", "w_error")
TITLES = {
    "u_error": "u Relative L2",
    "v_error": "v Relative L2",
    "w_error": "vorticity Relative L2",
}
STYLE = {
    "3324": {"label": "3324 dw=23.1429", "color": "#b91c1c", "marker": "D"},
    "3329": {"label": "3329 no data", "color": "#1d4ed8", "marker": "o"},
    "3330": {"label": "3330 sensor100", "color": "#047857", "marker": "s"},
}


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def collect_points(
    csv_50k: Path,
    csv_100k: Path,
    csv_3324: Path,
) -> dict[str, list[dict[str, float]]]:
    points = {"3324": [], "3329": [], "3330": []}

    for row in load_rows(csv_50k):
        if row["run"] == "no_data":
            points["3329"].append(_normalize_row(row))
        elif row["run"] == "sensor":
            points["3330"].append(_normalize_row(row))

    for row in load_rows(csv_100k):
        if row["run"] == "no_data":
            points["3329"].append(_normalize_row(row))
        elif row["run"] == "sensor":
            points["3330"].append(_normalize_row(row))

    for row in load_rows(csv_3324):
        if row["run"] == "3324_direct" and int(row["checkpoint_step"]) == 50000:
            points["3324"].append(_normalize_row(row))

    for key in points:
        points[key] = sorted(points[key], key=lambda row: row["step"])

    return points


def _normalize_row(row: dict[str, str]) -> dict[str, float]:
    return {
        "step": int(row["checkpoint_step"]),
        "u_error": float(row["u_error"]),
        "v_error": float(row["v_error"]),
        "w_error": float(row["w_error"]),
    }


def plot(points: dict[str, list[dict[str, float]]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)

    for ax, metric in zip(axes, METRICS):
        for run_id in ("3324", "3329", "3330"):
            rows = points[run_id]
            if not rows:
                continue
            style = STYLE[run_id]
            ax.plot(
                [row["step"] for row in rows],
                [row[metric] for row in rows],
                marker=style["marker"],
                markersize=7,
                linewidth=2.0,
                color=style["color"],
                label=style["label"],
            )
        ax.set_yscale("log")
        ax.set_xlabel("Checkpoint Step")
        ax.set_ylabel("Relative L2 error")
        ax.set_title(TITLES[metric])
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=9)

    fig.suptitle("Window 1 Step vs Error Comparison")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_summary(points: dict[str, list[dict[str, float]]]) -> str:
    lines = [
        "=== Window 1 Step vs Error Comparison ===",
        "",
        "Runs:",
        "- 3324: fixed data_weight=23.1429, direct eval at step 50000",
        "- 3329: no-data rerun, direct eval at steps 50000 and 100000",
        "- 3330: sensor100 rerun, direct eval at steps 50000 and 100000",
        "",
    ]

    for metric in METRICS:
        lines.append(f"--- {metric} ---")
        for run_id in ("3324", "3329", "3330"):
            for row in points[run_id]:
                lines.append(f"{run_id} step={row['step']}: {row[metric]:.6e}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot key window-1 checkpoint errors on one figure.")
    parser.add_argument("--csv-50k", required=True)
    parser.add_argument("--csv-100k", required=True)
    parser.add_argument("--csv-3324", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    points = collect_points(
        csv_50k=Path(args.csv_50k),
        csv_100k=Path(args.csv_100k),
        csv_3324=Path(args.csv_3324),
    )

    plot(points, output_dir / "window1_step_vs_error_points.png")
    (output_dir / "window1_step_vs_error_points.txt").write_text(build_summary(points))


if __name__ == "__main__":
    main()
