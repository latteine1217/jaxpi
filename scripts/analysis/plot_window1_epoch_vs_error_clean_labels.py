"""
What:
    重畫 window-1 的 epoch-vs-error 曲線，比較 no-data、sensor100 與
    data_weight=23.1429 三條路徑。
Why:
    使用研究語義上的方法名稱，而不是 job id；同時將 `3324` 的
    `10000~50000` direct evaluation 納入同一張圖，避免只看單點 `50000`
    造成收斂趨勢誤判。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = (
    ("u_error", "u Relative L2"),
    ("v_error", "v Relative L2"),
    ("w_error", "vorticity Relative L2"),
)

STYLE = {
    "no_data": {"label": "No data", "color": "#1d4ed8", "marker": "o"},
    "sensor100": {"label": "Sensor100", "color": "#c2410c", "marker": "s"},
    "dw_23_1429": {"label": "Sensor dw=23.1429", "color": "#7c3aed", "marker": "D"},
}


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def normalize_complete_rows(csv_path: Path) -> list[dict[str, float | str]]:
    rows = []
    for row in load_rows(csv_path):
        rows.append(
            {
                "run": row["run"],
                "checkpoint_step": int(row["checkpoint_step"]),
                "u_error": float(row["u_error"]),
                "v_error": float(row["v_error"]),
                "w_error": float(row["w_error"]),
            }
        )
    return rows


def normalize_dw_rows(csv_path: Path) -> list[dict[str, float | str]]:
    rows = []
    for row in load_rows(csv_path):
        rows.append(
            {
                "run": "dw_23_1429",
                "checkpoint_step": int(row["checkpoint_step"]),
                "u_error": float(row["u_error"]),
                "v_error": float(row["v_error"]),
                "w_error": float(row["w_error"]),
            }
        )
    return rows


def collect_rows(complete_csv: Path, dw_csv: Path) -> list[dict[str, float | str]]:
    rows = []
    for row in normalize_complete_rows(complete_csv):
        run = str(row["run"])
        if run == "sensor":
            row["run"] = "sensor100"
        rows.append(row)
    rows.extend(normalize_dw_rows(dw_csv))
    return sorted(rows, key=lambda row: (str(row["run"]), int(row["checkpoint_step"])))


def write_combined_csv(rows: list[dict[str, float | str]], output_path: Path) -> None:
    fieldnames = ["run", "checkpoint_step", "u_error", "v_error", "w_error"]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, float | str]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)
    reference_step = 50000

    for ax, (metric, title) in zip(axes, METRICS):
        for run_key in ("no_data", "sensor100", "dw_23_1429"):
            run_rows = [row for row in rows if row["run"] == run_key]
            if not run_rows:
                continue
            style = STYLE[run_key]
            ax.plot(
                [int(row["checkpoint_step"]) for row in run_rows],
                [float(row[metric]) for row in run_rows],
                marker=style["marker"],
                linewidth=2.0,
                markersize=6.5,
                color=style["color"],
                label=style["label"],
            )
        ax.axvline(
            reference_step,
            color="#6b7280",
            linestyle="--",
            linewidth=1.4,
            alpha=0.9,
        )
        ax.annotate(
            "50000",
            xy=(reference_step, 0.02),
            xycoords=("data", "axes fraction"),
            xytext=(4, 0),
            textcoords="offset points",
            rotation=90,
            va="bottom",
            ha="left",
            fontsize=8,
            color="#4b5563",
        )
        ax.set_title(title)
        ax.set_xlabel("Checkpoint Step")
        ax.set_ylabel("Relative L2 error")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle("Window 1 Epoch vs Error")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(rows: list[dict[str, float | str]], output_path: Path) -> None:
    lines = [
        "=== Window 1 Epoch vs Error ===",
        "",
        "Runs:",
        "- No data",
        "- Sensor100",
        "- Sensor dw=23.1429",
        "",
    ]
    for metric, _ in METRICS:
        lines.append(f"--- {metric} ---")
        for run_key in ("no_data", "sensor100", "dw_23_1429"):
            for row in [item for item in rows if item["run"] == run_key]:
                lines.append(
                    f"{STYLE[run_key]['label']} step={int(row['checkpoint_step'])}: "
                    f"{float(row[metric]):.6e}"
                )
        lines.append("")
    output_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot clean-label window-1 epoch-vs-error curves.")
    parser.add_argument("--complete-csv", required=True)
    parser.add_argument("--dw-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(Path(args.complete_csv), Path(args.dw_csv))
    write_combined_csv(rows, output_dir / "epoch_vs_error_clean_labels.csv")
    plot(rows, output_dir / "epoch_vs_error_clean_labels.png")
    write_summary(rows, output_dir / "epoch_vs_error_clean_labels.txt")


if __name__ == "__main__":
    main()
