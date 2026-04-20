#!/usr/bin/env python3
"""
What:
    從現有 markdown report 與 config 檔抽取 demo 頁面所需的結構化資料，
    輸出為 docs/demo-data.js。

Why:
    避免把實驗數值硬寫死在 HTML，讓 demo 頁面能跟著目前 repo 內的報告與設定同步更新。
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
EXPERIMENT_RECORD = ROOT / "EXPERIMENT_RECORD.md"
WINDOW_REPORT = ROOT / "examples/kolmogorov_flow/comparison/re1e6_n512_ds4_latest_per_window_report.md"
PIRATE_CONFIG = ROOT / "examples/kolmogorov_flow/configs/pirate.py"
SOAP_CONFIG = ROOT / "examples/kolmogorov_flow/configs/paper_repro_soap.py"
LEGACY_FULL_WINDOW_SUMMARY = ROOT / "eval_paper_repro_soap_0405" / "summary.txt"
LOCALTIME_FULL_WINDOW_SUMMARY = ROOT / "eval_paper_repro_soap_0405_localtime" / "summary.txt"
WINDOW12_DIAGNOSTICS_JSON = ROOT / "eval_paper_repro_soap_0405_localtime" / "window12_vorticity_diagnostics.json"
OUTPUT = DOCS_DIR / "demo-data.js"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def extract_block(text: str, heading_regex: str) -> str:
    match = re.search(heading_regex, text, flags=re.S | re.M)
    if not match:
        raise ValueError(f"Unable to find block: {heading_regex}")
    return match.group(1)


def parse_markdown_table(table_text: str) -> list[dict[str, Any]]:
    lines = [line.strip() for line in table_text.strip().splitlines() if line.strip()]
    if len(lines) < 2:
        return []

    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    rows: list[dict[str, Any]] = []
    for line in lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != len(header):
            continue
        row: dict[str, Any] = {}
        for key, raw in zip(header, cells):
            row[key] = coerce_value(raw)
        rows.append(row)
    return rows


def coerce_value(raw: str) -> Any:
    value = raw.strip()
    if value.startswith("`") and value.endswith("`"):
        value = value[1:-1]
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value) or re.fullmatch(r"-?\d+(?:\.\d+)?e[+-]?\d+", value, flags=re.I):
        return float(value)
    return value


def find_table_after(text: str, label: str) -> list[dict[str, Any]]:
    pattern = rf"{re.escape(label)}\n\n((?:\|.*\n)+)"
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"Unable to find table after label: {label}")
    return parse_markdown_table(match.group(1))


def find_loss_ranges(text: str) -> list[dict[str, str]]:
    losses = re.findall(r"- `([^`]+?) ≈ ([^`]+?)`", text)
    return [{"name": name, "range": value} for name, value in losses]


def find_job_progress(text: str) -> dict[str, str]:
    return {
        "time_window": must_match(text, r"- `Time Window ([^`]+)`"),
        "step": must_match(text, r"- `Step ([^`]+)`"),
        "checkpoint_path": must_match(text, r"- `/home[^`]+/checkpoint_([0-9]+)`"),
    }


def must_match(text: str, pattern: str) -> str:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"Unable to match pattern: {pattern}")
    return match.group(1)


def parse_eval_summary(path: Path) -> dict[str, Any]:
    text = read_text(path)
    row_pattern = re.compile(
        r"^\s*(\d+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$",
        flags=re.M,
    )
    rows = []
    for match in row_pattern.finditer(text):
        win, step, t_end, u_err, v_err, w_err = match.groups()
        rows.append(
            {
                "Window": int(win),
                "Checkpoint": int(step),
                "t_end": float(t_end),
                "u_err": float(u_err),
                "v_err": float(v_err),
                "w_err": float(w_err),
            }
        )

    mean_match = re.search(
        r"mean\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)",
        text,
        flags=re.M,
    )
    max_match = re.search(
        r"max\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)",
        text,
        flags=re.M,
    )
    if not mean_match or not max_match:
        raise ValueError(f"Unable to parse summary statistics from {path}")

    return {
        "rows": rows,
        "stats": {
            "u_err": {"Mean": float(mean_match.group(1)), "Max": float(max_match.group(1))},
            "v_err": {"Mean": float(mean_match.group(2)), "Max": float(max_match.group(2))},
            "w_err": {"Mean": float(mean_match.group(3)), "Max": float(max_match.group(3))},
        },
    }


def build_rerun_delta(legacy_rows: list[dict[str, Any]], local_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for legacy, local in zip(legacy_rows, local_rows):
        rows.append(
            {
                "Window": local["Window"],
                "legacy_w_err": legacy["w_err"],
                "localtime_w_err": local["w_err"],
                "drop_factor": legacy["w_err"] / local["w_err"] if local["w_err"] else None,
                "legacy_uv_mean": (legacy["u_err"] + legacy["v_err"]) / 2.0,
                "localtime_uv_mean": (local["u_err"] + local["v_err"]) / 2.0,
            }
        )
    return rows


def parse_config(path: Path) -> dict[str, Any]:
    text = read_text(path)

    def find(pattern: str, cast: Any = str) -> Any:
        match = re.search(pattern, text)
        if not match:
            raise ValueError(f"Unable to extract `{pattern}` from {path}")
        return cast(match.group(1))

    config = {
        "name": path.stem,
        "arch_name": find(r'arch\.arch_name = "([^"]+)"'),
        "num_layers": find(r"arch\.num_layers = (\d+)", int),
        "hidden_dim": find(r"arch\.hidden_dim = (\d+)", int),
        "out_dim": find(r"arch\.out_dim = (\d+)", int),
        "activation": find(r'arch\.activation = "([^"]+)"'),
        "embed_dim": find(r"embed_dim\": (\d+)", int),
        "optimizer": find(r'optim\.optimizer = "([^"]+)"'),
        "batch_size_per_device": find(r"training\.batch_size_per_device = (\d+)", int),
        "num_time_windows": find(r"training\.num_time_windows = (\d+)", int),
        "weighting_scheme": find(r'weighting\.scheme = "([^"]+)"'),
        "num_chunks": find(r"weighting\.num_chunks = (\d+)", int),
        "transfer_learning": find(r"config\.transfer_learning = (True|False)") == "True",
        "dataset_path": find(r'config\.dataset_path = "([^"]+)"'),
    }
    config["gating_layers"] = config["num_layers"] * 2
    return config


def build_data() -> dict[str, Any]:
    experiment_text = read_text(EXPERIMENT_RECORD)
    report_text = read_text(WINDOW_REPORT)

    active_update = extract_block(
        experiment_text,
        r"### 2026-04-01 更新\n\n(.*?)(?=\n## \[INDEX\] Critical Failures & Insights)",
    )
    verified_windows_block = extract_block(
        experiment_text,
        r"### \[2026-03-31\] `3137` \| `window 2` 與 `window 3` checkpoint 評估\n\n(.*?)(?=\n### )",
    )

    latest_final_step = find_table_after(active_update, "- `final_step` 評估：")
    latest_per_window = find_table_after(active_update, "- 各 `time_window` 最新 checkpoint 的 `final_step` 摘要：")
    verified_windows = find_table_after(verified_windows_block, "Evidence:")
    summary_statistics = find_table_after(report_text, "## Summary Statistics")
    report_results = find_table_after(report_text, "## Results")
    current_losses = find_loss_ranges(active_update)
    job_progress = find_job_progress(active_update)
    legacy_full_window = parse_eval_summary(LEGACY_FULL_WINDOW_SUMMARY)
    localtime_full_window = parse_eval_summary(LOCALTIME_FULL_WINDOW_SUMMARY)
    diagnostics = json.loads(read_text(WINDOW12_DIAGNOSTICS_JSON))

    pirate = parse_config(PIRATE_CONFIG)
    soap = parse_config(SOAP_CONFIG)

    latest_w = localtime_full_window["rows"][-1]["w_err"]
    summary_lookup = {
        "u_err": {
            "Metric": "u_err",
            "Mean": localtime_full_window["stats"]["u_err"]["Mean"],
            "Min": min(row["u_err"] for row in localtime_full_window["rows"]),
            "Max": localtime_full_window["stats"]["u_err"]["Max"],
        },
        "v_err": {
            "Metric": "v_err",
            "Mean": localtime_full_window["stats"]["v_err"]["Mean"],
            "Min": min(row["v_err"] for row in localtime_full_window["rows"]),
            "Max": localtime_full_window["stats"]["v_err"]["Max"],
        },
        "w_err": {
            "Metric": "w_err",
            "Mean": localtime_full_window["stats"]["w_err"]["Mean"],
            "Min": min(row["w_err"] for row in localtime_full_window["rows"]),
            "Max": localtime_full_window["stats"]["w_err"]["Max"],
        },
    }
    rerun_delta = build_rerun_delta(legacy_full_window["rows"], localtime_full_window["rows"])
    latest_rerun = localtime_full_window["rows"][-1]

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sources": {
            "experiment_record": str(EXPERIMENT_RECORD.relative_to(ROOT)),
            "window_report": str(WINDOW_REPORT.relative_to(ROOT)),
            "pirate_config": str(PIRATE_CONFIG.relative_to(ROOT)),
            "soap_config": str(SOAP_CONFIG.relative_to(ROOT)),
            "legacy_full_window_summary": str(LEGACY_FULL_WINDOW_SUMMARY.relative_to(ROOT)),
            "localtime_full_window_summary": str(LOCALTIME_FULL_WINDOW_SUMMARY.relative_to(ROOT)),
            "window12_diagnostics_json": str(WINDOW12_DIAGNOSTICS_JSON.relative_to(ROOT)),
        },
        "summary": {
            "key_message": "修正 eval time-axis 後，3137 的 u/v full-window 誤差維持低檔；真正仍需追蹤的是逐窗累積的 vorticity degradation。",
            "current_time_window": f"{latest_rerun['Window']}/{soap['num_time_windows']}",
            "current_step": f"{latest_rerun['Checkpoint']}/{100000}",
            "latest_checkpoint": str(latest_rerun["Checkpoint"]),
            "latest_w_err": latest_w,
            "max_verified_window": max(row["Window"] for row in localtime_full_window["rows"]),
            "u_err_mean": summary_lookup["u_err"]["Mean"],
            "v_err_mean": summary_lookup["v_err"]["Mean"],
            "w_err_mean": summary_lookup["w_err"]["Mean"],
            "legacy_w_err_mean": legacy_full_window["stats"]["w_err"]["Mean"],
            "window12_drop_factor": rerun_delta[-1]["drop_factor"],
        },
        "model": {
            "pirate": pirate,
            "soap": soap,
        },
        "reports": {
            "latest_final_step": latest_final_step,
            "latest_per_window": latest_per_window,
            "summary_statistics": list(summary_lookup.values()),
            "window_report_results": report_results,
            "verified_windows": verified_windows,
            "current_losses": current_losses,
            "full_window_legacy": legacy_full_window["rows"],
            "full_window_localtime": localtime_full_window["rows"],
            "full_window_rerun_delta": rerun_delta,
            "window12_diagnostics": diagnostics["summary"],
        },
    }


def main() -> None:
    data = build_data()
    OUTPUT.write_text(
        "window.DEMO_DATA = " + json.dumps(data, ensure_ascii=False, indent=2) + ";\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
