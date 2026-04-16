#!/usr/bin/env python3
"""
What:
    針對單一 sensor-constrained window checkpoint，比較 sensor 點、sensor 附近與全場誤差，
    並補充 final-step vorticity / spectrum / high-k drift 診斷。

Why:
    `full-window` 相對 L2 只能告訴我們整體失真有多大，卻無法回答：
    1. 模型是否至少在受監督的 sensor 點附近是對的；
    2. 失真是否主要來自 unsensed 區域；
    3. `w_err` 是否對應高波數渦度能量流失。
    這支腳本把這三件事放進同一份 artifact，讓 `window 2` 這類 failure
    可以被更直接地診斷。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.kolmogorov_flow import models
from examples.kolmogorov_flow.utils import get_dataset
from jaxpi.utils import restore_checkpoint


def load_config(config_path: Path):
    """What: 動態載入 config。 Why: 保持診斷對實驗設定的可追溯性。"""
    spec = importlib.util.spec_from_file_location(config_path.stem, config_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.get_config()


def latest_checkpoint_step(ckpt_dir: Path) -> int:
    """What: 找最新 checkpoint step。 Why: 預設診斷應對最新已落盤狀態。"""
    steps = []
    for path in ckpt_dir.iterdir():
        if path.is_dir() and path.name.startswith("checkpoint_"):
            steps.append(int(path.name.split("_")[1]))
    if not steps:
        raise FileNotFoundError(f"No checkpoint_* found under {ckpt_dir}")
    return max(steps)


def to_window_local_time(t_window: np.ndarray) -> np.ndarray:
    """What: 絕對時間轉 local time。 Why: 對齊 checkpoint 訓練定義。"""
    t_window = np.asarray(t_window)
    return t_window - float(t_window[0])


def radial_average_spectrum(field2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """What: 2D 場做徑向平均功率譜。 Why: 用低/高 k 彙總比單一方向 FFT 更穩。"""
    nx, ny = field2d.shape
    fft = np.fft.fftshift(np.fft.fft2(field2d))
    power = np.abs(fft) ** 2

    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=1.0 / nx))
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=1.0 / ny))
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    k_mag = np.sqrt(kx_grid**2 + ky_grid**2)
    k_shell = np.floor(k_mag + 0.5).astype(int)

    k_vals = []
    spec_vals = []
    for k in range(1, int(k_shell.max()) + 1):
        mask = k_shell == k
        if not np.any(mask):
            continue
        k_vals.append(k)
        spec_vals.append(float(power[mask].mean()))
    return np.asarray(k_vals), np.asarray(spec_vals)


def build_periodic_neighborhood_mask(indices: np.ndarray, grid_size: int, radius: int) -> np.ndarray:
    """
    What:
        建立以 sensor 為中心的 periodic square neighborhood mask。
    Why:
        「sensor 點附近」不是單一標準；用 grid radius 可以明確定義受 sensor 約束的局部區域。
    """
    mask_2d = np.zeros((grid_size, grid_size), dtype=bool)
    ix = indices // grid_size
    iy = indices % grid_size
    offsets = np.arange(-radius, radius + 1, dtype=int)
    for dx in offsets:
        for dy in offsets:
            mask_2d[(ix + dx) % grid_size, (iy + dy) % grid_size] = True
    return mask_2d.reshape(-1)


def predict_window_fields(
    model,
    params,
    coords: np.ndarray,
    t_window_local: np.ndarray,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    What:
        分塊預測整個 window 的 u/v/w。
    Why:
        `512x512` 場不能假設一次全吃得下；分塊可保留 correctness 且避免記憶體尖峰。
    """
    num_time = len(t_window_local)
    num_space = coords.shape[0]

    u_pred = np.zeros((num_time, num_space), dtype=np.float32)
    v_pred = np.zeros((num_time, num_space), dtype=np.float32)
    w_pred = np.zeros((num_time, num_space), dtype=np.float32)

    for t_idx, t_local in enumerate(t_window_local):
        for start in range(0, num_space, chunk_size):
            end = min(start + chunk_size, num_space)
            coords_chunk = coords[start:end]
            x_chunk = jnp.asarray(coords_chunk[:, 0])
            y_chunk = jnp.asarray(coords_chunk[:, 1])
            u_pred[t_idx, start:end] = np.asarray(
                jax.device_get(model.u_ic_pred_fn(params, float(t_local), x_chunk, y_chunk))
            )
            v_pred[t_idx, start:end] = np.asarray(
                jax.device_get(model.v_ic_pred_fn(params, float(t_local), x_chunk, y_chunk))
            )
            w_pred[t_idx, start:end] = np.asarray(
                jax.device_get(model.w_ic_pred_fn(params, float(t_local), x_chunk, y_chunk))
            )
    return u_pred, v_pred, w_pred


def relative_l2(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> float:
    """What: 計算 masked relative L2。 Why: 讓 sensor 區域與全場可同一口徑比較。"""
    pred_sel = pred[:, mask]
    ref_sel = ref[:, mask]
    numer = float(np.sum((pred_sel - ref_sel) ** 2))
    denom = float(np.sum(ref_sel**2))
    return float(np.sqrt(numer / max(denom, np.finfo(np.float64).eps)))


def per_time_relative_l2(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> list[float]:
    """What: 各時刻 masked relative L2。 Why: 確認誤差是在窗口內累積還是一開始就偏掉。"""
    values = []
    for t_idx in range(pred.shape[0]):
        numer = float(np.sum((pred[t_idx, mask] - ref[t_idx, mask]) ** 2))
        denom = float(np.sum(ref[t_idx, mask] ** 2))
        values.append(float(np.sqrt(numer / max(denom, np.finfo(np.float64).eps))))
    return values


def write_summary(summary_path: Path, summary: dict) -> None:
    """What: 寫文字摘要。 Why: 讓帳本與報告可以直接引用。"""
    lines = [
        "=== Sensor Window Failure Analysis ===",
        f"window = {summary['window']}",
        f"checkpoint = {summary['checkpoint']}",
        f"t_end = {summary['t_end']:.6f}",
        f"sensor_count = {summary['sensor_count']}",
        f"grid_size = {summary['grid_size']}",
        "--- Region Relative L2 ---",
    ]
    for region_name, region_metrics in summary["region_metrics"].items():
        lines.append(
            f"{region_name}: coverage={region_metrics['coverage_fraction']:.6f}, "
            f"u={region_metrics['u_err']:.6f}, v={region_metrics['v_err']:.6f}, "
            f"w={region_metrics['w_err']:.6f}"
        )
    lines.extend(
        [
            "--- Final-Step Vorticity Diagnostics ---",
            f"final_w_err = {summary['final_w_err']:.6f}",
            f"final_corr = {summary['final_corr']:.6f}",
            f"final_std_ratio = {summary['final_std_ratio']:.6f}",
            f"final_enstrophy_ratio = {summary['final_enstrophy_ratio']:.6f}",
            f"low_k_ratio = {summary['low_k_ratio']:.6f}",
            f"high_k_ratio = {summary['high_k_ratio']:.6f}",
        ]
    )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figure(
    output_path: Path,
    grid_size: int,
    sensor_indices: np.ndarray,
    final_w_ref: np.ndarray,
    final_w_pred: np.ndarray,
    region_metrics: dict,
    spectrum_ref: tuple[np.ndarray, np.ndarray],
    spectrum_pred: tuple[np.ndarray, np.ndarray],
) -> None:
    """
    What:
        生成單張診斷圖。
    Why:
        用同一個 artifact 同時回答「sensor 附近是否較準」與「高-k 是否流失」。
    """
    final_w_ref_2d = final_w_ref.reshape(grid_size, grid_size)
    final_w_pred_2d = final_w_pred.reshape(grid_size, grid_size)
    final_w_err_2d = np.abs(final_w_pred_2d - final_w_ref_2d)

    ix = sensor_indices // grid_size
    iy = sensor_indices % grid_size

    k_ref, e_ref = spectrum_ref
    k_pred, e_pred = spectrum_pred

    region_names = list(region_metrics.keys())
    u_vals = [region_metrics[name]["u_err"] for name in region_names]
    v_vals = [region_metrics[name]["v_err"] for name in region_names]
    w_vals = [region_metrics[name]["w_err"] for name in region_names]

    x = np.arange(len(region_names))
    width = 0.24

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.26, wspace=0.24)

    ax0 = fig.add_subplot(gs[0, 0])
    vmax = float(np.max(final_w_err_2d))
    im0 = ax0.imshow(final_w_err_2d.T, origin="lower", cmap="hot", vmin=0.0, vmax=vmax, aspect="equal")
    ax0.scatter(ix, iy, s=18, facecolors="none", edgecolors="cyan", linewidths=0.8, label="Sensors")
    ax0.set_title("Final-Step Vorticity Abs Error")
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.legend(frameon=False, loc="upper right")
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.bar(x - width, u_vals, width=width, label="u", color="#2563eb")
    ax1.bar(x, v_vals, width=width, label="v", color="#0f766e")
    ax1.bar(x + width, w_vals, width=width, label="w", color="#d97706")
    ax1.set_xticks(x)
    ax1.set_xticklabels(region_names, rotation=20, ha="right")
    ax1.set_ylabel("Relative L2")
    ax1.set_title("Region Error Comparison")
    ax1.grid(True, alpha=0.25, axis="y")
    ax1.legend(frameon=False)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.loglog(k_ref, e_ref, label="DNS", color="#2563eb", linewidth=2.0)
    ax2.loglog(k_pred, e_pred, label="Pred", color="#0f766e", linewidth=2.0)
    ax2.set_title("Final-Step Vorticity Spectrum")
    ax2.set_xlabel("Wavenumber k")
    ax2.set_ylabel("Power")
    ax2.grid(True, alpha=0.25, which="both")
    ax2.legend(frameon=False)

    ax3 = fig.add_subplot(gs[1, 1])
    ref_flat = final_w_ref.ravel()
    pred_flat = final_w_pred.ravel()
    lo = float(min(ref_flat.min(), pred_flat.min()))
    hi = float(max(ref_flat.max(), pred_flat.max()))
    ax3.scatter(ref_flat[::64], pred_flat[::64], s=4, alpha=0.20, color="#1d4ed8")
    ax3.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, color="#111827")
    ax3.set_title("Final-Step Vorticity Scatter")
    ax3.set_xlabel("DNS")
    ax3.set_ylabel("Pred")
    ax3.grid(True, alpha=0.25)

    fig.suptitle("Window Failure Analysis", fontsize=15, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze sensor-point vs field error for one checkpoint window.")
    parser.add_argument("--config", required=True, help="Config path.")
    parser.add_argument("--checkpoint-root", required=True, help="Checkpoint root containing time_window_*.")
    parser.add_argument("--window", type=int, required=True, help="1-based time window index.")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step. Default: latest.")
    parser.add_argument("--chunk-size", type=int, default=4096, help="Spatial chunk size.")
    parser.add_argument(
        "--neighbor-radii",
        default="2,4,8",
        help="Comma-separated grid radii for sensor-neighborhood masks.",
    )
    parser.add_argument("--figure", required=True, help="Output figure path.")
    parser.add_argument("--json", required=True, help="Output JSON path.")
    parser.add_argument("--summary", required=True, help="Output text summary path.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config = load_config(config_path)

    dataset_path = Path(config.dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = REPO_ROOT / dataset_path

    sensor_json_path = Path(config.sensor_json)
    if not sensor_json_path.is_absolute():
        sensor_json_path = REPO_ROOT / sensor_json_path
    sensor_payload = json.loads(sensor_json_path.read_text(encoding="utf-8"))
    sensor_indices = np.asarray(sensor_payload["indices"], dtype=int)

    u_ref, v_ref, w_ref, t_star, coords, nu = get_dataset(
        time_fraction=config.time_fraction,
        dataset_path=str(dataset_path),
        time_range=config.get("dns_time_range"),
        time_stride=config.get("dns_time_stride", 1),
    )

    num_time_steps = len(t_star) // int(config.training.num_time_windows)
    window_idx = int(args.window)
    start_idx = (window_idx - 1) * num_time_steps
    end_idx = window_idx * num_time_steps
    if start_idx < 0 or end_idx > len(t_star):
        raise ValueError(f"window {window_idx} out of range")

    t_window_abs = np.asarray(t_star[start_idx:end_idx])
    t_window_local = to_window_local_time(t_window_abs)
    u_ref_window = np.asarray(u_ref[start_idx:end_idx, :])
    v_ref_window = np.asarray(v_ref[start_idx:end_idx, :])
    w_ref_window = np.asarray(w_ref[start_idx:end_idx, :])

    u0 = np.asarray(u_ref[start_idx, :])
    v0 = np.asarray(v_ref[start_idx, :])
    w0 = np.asarray(w_ref[start_idx, :])

    model = models.NavierStokes(
        config,
        t_window_local,
        coords,
        u0,
        v0,
        w0,
        nu,
        replicate_state=False,
    )

    ckpt_dir = Path(args.checkpoint_root) / f"time_window_{window_idx}"
    step = int(args.step) if args.step is not None else latest_checkpoint_step(ckpt_dir)
    model.state = restore_checkpoint(model.state, str(ckpt_dir), step=step)

    u_pred, v_pred, w_pred = predict_window_fields(
        model=model,
        params=model.state.params,
        coords=np.asarray(coords),
        t_window_local=t_window_local,
        chunk_size=int(args.chunk_size),
    )

    grid_size = int(np.sqrt(coords.shape[0]))
    if grid_size * grid_size != coords.shape[0]:
        raise ValueError(f"coords size {coords.shape[0]} is not a square grid")

    region_masks = {
        "sensor_points": np.isin(np.arange(coords.shape[0]), sensor_indices),
        "full_field": np.ones(coords.shape[0], dtype=bool),
    }
    radii = [int(token.strip()) for token in args.neighbor_radii.split(",") if token.strip()]
    for radius in radii:
        region_masks[f"sensor_r{radius}"] = build_periodic_neighborhood_mask(sensor_indices, grid_size, radius)

    region_metrics = {}
    region_time_series = {}
    for region_name, mask in region_masks.items():
        region_metrics[region_name] = {
            "coverage_fraction": float(np.mean(mask)),
            "u_err": relative_l2(u_pred, u_ref_window, mask),
            "v_err": relative_l2(v_pred, v_ref_window, mask),
            "w_err": relative_l2(w_pred, w_ref_window, mask),
        }
        region_time_series[region_name] = {
            "u_err": per_time_relative_l2(u_pred, u_ref_window, mask),
            "v_err": per_time_relative_l2(v_pred, v_ref_window, mask),
            "w_err": per_time_relative_l2(w_pred, w_ref_window, mask),
        }

    final_w_ref = np.asarray(w_ref_window[-1])
    final_w_pred = np.asarray(w_pred[-1])
    final_w_ref_2d = final_w_ref.reshape(grid_size, grid_size)
    final_w_pred_2d = final_w_pred.reshape(grid_size, grid_size)
    k_ref, e_ref = radial_average_spectrum(final_w_ref_2d)
    k_pred, e_pred = radial_average_spectrum(final_w_pred_2d)

    k_cut = max(4, int(0.25 * min(len(k_ref), len(k_pred))))
    low_mask_ref = np.arange(len(k_ref)) < k_cut
    high_mask_ref = np.arange(len(k_ref)) >= k_cut
    low_mask_pred = np.arange(len(k_pred)) < k_cut
    high_mask_pred = np.arange(len(k_pred)) >= k_cut

    summary = {
        "window": window_idx,
        "checkpoint": step,
        "t_end": float(t_window_abs[-1]),
        "grid_size": grid_size,
        "sensor_count": int(sensor_indices.size),
        "region_metrics": region_metrics,
        "final_w_err": float(region_time_series["full_field"]["w_err"][-1]),
        "final_corr": float(np.corrcoef(final_w_ref, final_w_pred)[0, 1]),
        "final_std_ratio": float(np.std(final_w_pred) / max(np.std(final_w_ref), np.finfo(np.float64).eps)),
        "final_enstrophy_ratio": float(
            (0.5 * np.mean(final_w_pred**2)) / max(0.5 * np.mean(final_w_ref**2), np.finfo(np.float64).eps)
        ),
        "low_k_ratio": float(np.sum(e_pred[low_mask_pred]) / max(np.sum(e_ref[low_mask_ref]), np.finfo(np.float64).eps)),
        "high_k_ratio": float(np.sum(e_pred[high_mask_pred]) / max(np.sum(e_ref[high_mask_ref]), np.finfo(np.float64).eps)),
    }

    make_figure(
        output_path=Path(args.figure),
        grid_size=grid_size,
        sensor_indices=sensor_indices,
        final_w_ref=final_w_ref,
        final_w_pred=final_w_pred,
        region_metrics=region_metrics,
        spectrum_ref=(k_ref, e_ref),
        spectrum_pred=(k_pred, e_pred),
    )

    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_payload = {
        "summary": summary,
        "time_series": {
            "t_abs": t_window_abs.tolist(),
            "t_local": t_window_local.tolist(),
            "regions": region_time_series,
        },
    }
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    write_summary(Path(args.summary), summary)

    print(f"saved figure: {args.figure}")
    print(f"saved json: {args.json}")
    print(f"saved summary: {args.summary}")


if __name__ == "__main__":
    main()
