"""
What:
    畫出 window-1 corrected error 對比圖，直接比較 3137 與 3324。
Why:
    先前對話混入了 `3145 window 1 ~1e-3` 的歷史舊 artifact；這支腳本固定使用
    2026-04-21 重新核對後的 `3137` 真實 retained checkpoint 與 `3324` corrected eval，
    產出可重跑的比較圖與摘要。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = ("u_error", "v_error", "w_error")


def load_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def load_3137_rows(csv_path: Path) -> list[dict[str, float]]:
    rows = []
    for row in load_csv_rows(csv_path):
        rows.append(
            {
                "step": int(row["checkpoint_step"]),
                "u_error": float(row["u_error"]),
                "v_error": float(row["v_error"]),
                "w_error": float(row["w_error"]),
            }
        )
    return sorted(rows, key=lambda row: row["step"])


def load_3324_rows(csv_path: Path) -> list[dict[str, float]]:
    rows = []
    for row in load_csv_rows(csv_path):
        rows.append(
            {
                "step": int(row["step"]),
                "u_error": float(row["u_error"]),
                "v_error": float(row["v_error"]),
                "w_error": float(row["w_error"]),
            }
        )
    return sorted(rows, key=lambda row: row["step"])


def plot(output_path: Path, rows_3137: list[dict[str, float]], rows_3324: list[dict[str, float]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=False)
    labels = {"u_error": "u Relative L2", "v_error": "v Relative L2", "w_error": "vorticity Relative L2"}
    colors = {"3137": "#1d4ed8", "3324": "#b91c1c"}

    for ax, metric in zip(axes, METRICS):
        ax.plot(
            [row["step"] for row in rows_3137],
            [row[metric] for row in rows_3137],
            marker="o",
            linewidth=2.0,
            color=colors["3137"],
            label="3137 no data",
        )
        ax.plot(
            [row["step"] for row in rows_3324],
            [row[metric] for row in rows_3324],
            marker="o",
            linewidth=2.0,
            color=colors["3324"],
            label="3324 sensor dw=23.1429",
        )
        ax.set_yscale("log")
        ax.set_xlabel("Checkpoint Step")
        ax.set_ylabel("Relative L2 error")
        ax.set_title(labels[metric])
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=9)

    fig.suptitle("Window 1 Corrected Error Comparison: 3137 vs 3324")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_summary(rows_3137: list[dict[str, float]], rows_3324: list[dict[str, float]]) -> str:
    best_3137 = rows_3137[-1]
    best_3324 = min(rows_3324, key=lambda row: row["w_error"])
    lines = [
        "=== Window 1 Corrected Error Comparison ===",
        "Compared runs: 3137 no-data retained checkpoints vs 3324 sensor corrected eval.",
        "",
    ]

    for metric in METRICS:
        lines.append(f"--- {metric} ---")
        for label, rows in (("3137", rows_3137), ("3324", rows_3324)):
            for row in rows:
                lines.append(f"{label} step={row['step']}: {row[metric]:.6e}")
        ratio = best_3324[metric] / best_3137[metric]
        lines.append(
            f"best_3324_by_w / 3137@{best_3137['step']} = {ratio:.3f}"
        )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot window-1 corrected error comparison.")
    parser.add_argument("--errors-3137", required=True, help="CSV from 3137 partial sweep.")
    parser.add_argument("--errors-3324", required=True, help="CSV for 3324 corrected eval rows.")
    parser.add_argument("--output-dir", required=True, help="Directory for output artifacts.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_3137 = load_3137_rows(Path(args.errors_3137))
    rows_3324 = load_3324_rows(Path(args.errors_3324))

    plot(output_dir / "window1_error_compare_3137_vs_3324.png", rows_3137, rows_3324)
    (output_dir / "window1_error_compare_3137_vs_3324.txt").write_text(
        build_summary(rows_3137, rows_3324)
    )


if __name__ == "__main__":
    main()
