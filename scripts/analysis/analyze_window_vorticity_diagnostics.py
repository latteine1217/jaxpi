#!/usr/bin/env python3
"""
What:
    針對單一 time window / checkpoint 產生 vorticity 細部診斷圖與數值摘要。

Why:
    單一 `w_err` 只能告訴我們渦度誤差有多大，無法回答它是來自
    小尺度能量衰減、整體振幅偏差，還是隨時間累積的導數放大效應。
    這支腳本把 `w(t)`、enstrophy 與 final-step vorticity spectrum 放到同一份輸出，
    讓 `window 12` 這類問題窗口可以被更細地診斷。
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
    """What: 動態載入 config。 Why: 避免把實驗設定硬寫死在腳本。"""
    spec = importlib.util.spec_from_file_location(config_path.stem, config_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.get_config()


def latest_checkpoint_step(ckpt_dir: Path) -> int:
    """What: 解析最新 checkpoint step。 Why: 診斷預設應看最新可驗證狀態。"""
    steps = []
    for path in ckpt_dir.iterdir():
        if path.is_dir() and path.name.startswith("checkpoint_"):
            steps.append(int(path.name.split("_")[1]))
    if not steps:
        raise FileNotFoundError(f"No checkpoint_* found under {ckpt_dir}")
    return max(steps)


def to_window_local_time(t_window: np.ndarray) -> np.ndarray:
    """What: 轉成 window-local time。 Why: 對齊訓練 checkpoint 的時間座標定義。"""
    t_window = np.asarray(t_window)
    return t_window - float(t_window[0])


def radial_average_spectrum(field2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    What:
        對 2D 場做徑向平均 spectrum。
    Why:
        我們關心的是低/高波數能量是否失衡，而不是單一方向 FFT 的細節。
    """
    nx, ny = field2d.shape
    fft = np.fft.fftshift(np.fft.fft2(field2d))
    power = np.abs(fft) ** 2

    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=1.0 / nx))
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=1.0 / ny))
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    k_mag = np.sqrt(kx_grid**2 + ky_grid**2)

    k_shell = np.floor(k_mag + 0.5).astype(int)
    k_max = int(k_shell.max())
    k_vals = []
    spec_vals = []
    for k in range(1, k_max + 1):
        mask = k_shell == k
        count = int(mask.sum())
        if count == 0:
            continue
        k_vals.append(k)
        spec_vals.append(float(power[mask].mean()))
    return np.asarray(k_vals), np.asarray(spec_vals)


def analyze_window(
    model,
    params,
    coords: np.ndarray,
    t_window_local: np.ndarray,
    t_window_abs: np.ndarray,
    w_ref_window: np.ndarray,
    chunk_size: int,
):
    """
    What:
        分塊收集整個 window 的渦度誤差與 enstrophy 指標。
    Why:
        高解析 `512x512` 場若一次性具體化全部 `w_pred[t, x, y]`，
        會吃掉不必要的記憶體；逐時間/空間分塊即可保留診斷所需的統計量。
    """
    num_time = len(t_window_local)
    num_space = coords.shape[0]

    w_rel = np.zeros(num_time, dtype=np.float64)
    enstrophy_ref = np.zeros(num_time, dtype=np.float64)
    enstrophy_pred = np.zeros(num_time, dtype=np.float64)

    final_pred_chunks: list[np.ndarray] = []

    for t_idx, (t_local, t_abs) in enumerate(zip(t_window_local, t_window_abs)):
        numer = 0.0
        denom = 0.0
        pred_sq = 0.0

        for start in range(0, num_space, chunk_size):
            end = min(start + chunk_size, num_space)
            coords_chunk = coords[start:end]
            x_chunk = jnp.asarray(coords_chunk[:, 0])
            y_chunk = jnp.asarray(coords_chunk[:, 1])

            w_pred_chunk = np.asarray(
                jax.device_get(model.w_ic_pred_fn(params, float(t_local), x_chunk, y_chunk))
            )
            w_ref_chunk = np.asarray(w_ref_window[t_idx, start:end])

            diff = w_pred_chunk - w_ref_chunk
            numer += float(np.sum(diff**2))
            denom += float(np.sum(w_ref_chunk**2))
            pred_sq += float(np.sum(w_pred_chunk**2))

            if t_idx == num_time - 1:
                final_pred_chunks.append(w_pred_chunk)

        w_rel[t_idx] = np.sqrt(numer / max(denom, np.finfo(np.float64).eps))
        enstrophy_ref[t_idx] = 0.5 * float(np.sum(np.asarray(w_ref_window[t_idx]) ** 2) / num_space)
        enstrophy_pred[t_idx] = 0.5 * pred_sq / num_space

    final_w_ref = np.asarray(w_ref_window[-1])
    final_w_pred = np.concatenate(final_pred_chunks, axis=0)
    return {
        "t_local": np.asarray(t_window_local),
        "t_abs": np.asarray(t_window_abs),
        "w_rel": w_rel,
        "enstrophy_ref": enstrophy_ref,
        "enstrophy_pred": enstrophy_pred,
        "final_w_ref": final_w_ref,
        "final_w_pred": final_w_pred,
    }


def make_figure(
    output_path: Path,
    window_idx: int,
    step: int,
    grid_size: int,
    diagnostics: dict,
    spectrum_ref: tuple[np.ndarray, np.ndarray],
    spectrum_pred: tuple[np.ndarray, np.ndarray],
) -> None:
    """
    What:
        生成單張綜合診斷圖。
    Why:
        把空間誤差、時間累積與頻譜失衡放在同一個 artifact，比單張場圖更能解釋 `w_err`。
    """
    final_ref_2d = diagnostics["final_w_ref"].reshape(grid_size, grid_size)
    final_pred_2d = diagnostics["final_w_pred"].reshape(grid_size, grid_size)
    final_err_2d = np.abs(final_ref_2d - final_pred_2d)

    k_ref, e_ref = spectrum_ref
    k_pred, e_pred = spectrum_pred

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.9], hspace=0.28, wspace=0.28)

    vmax = max(float(np.abs(final_ref_2d).max()), float(np.abs(final_pred_2d).max()))
    vmin = -vmax

    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(final_ref_2d.T, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="equal")
    ax0.set_title("DNS Vorticity")
    ax0.set_xticks([])
    ax0.set_yticks([])
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(final_pred_2d.T, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="equal")
    ax1.set_title("Predicted Vorticity")
    ax1.set_xticks([])
    ax1.set_yticks([])
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(gs[0, 2])
    im2 = ax2.imshow(final_err_2d.T, origin="lower", cmap="hot", vmin=0.0, vmax=float(final_err_2d.max()), aspect="equal")
    ax2.set_title("Absolute Error")
    ax2.set_xticks([])
    ax2.set_yticks([])
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(diagnostics["t_abs"], diagnostics["w_rel"], color="#d97706", linewidth=2.2)
    ax3.set_title("Relative L2 Error by Time")
    ax3.set_xlabel("Physical Time")
    ax3.set_ylabel("Relative L2")
    ax3.grid(True, alpha=0.25)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(diagnostics["t_abs"], diagnostics["enstrophy_ref"], label="DNS", color="#2563eb", linewidth=2.0)
    ax4.plot(diagnostics["t_abs"], diagnostics["enstrophy_pred"], label="Pred", color="#0f766e", linewidth=2.0)
    ax4.set_title("Enstrophy by Time")
    ax4.set_xlabel("Physical Time")
    ax4.set_ylabel("Enstrophy")
    ax4.grid(True, alpha=0.25)
    ax4.legend(frameon=False)

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.loglog(k_ref, e_ref, label="DNS", color="#2563eb", linewidth=2.0)
    ax5.loglog(k_pred, e_pred, label="Pred", color="#0f766e", linewidth=2.0)
    ax5.set_title("Final-Step Vorticity Spectrum")
    ax5.set_xlabel("Wavenumber k")
    ax5.set_ylabel("Power")
    ax5.grid(True, alpha=0.25, which="both")
    ax5.legend(frameon=False)

    fig.suptitle(
        f"Window {window_idx} Vorticity Diagnostics | checkpoint_{step}",
        fontsize=15,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(summary_path: Path, summary: dict) -> None:
    """What: 輸出文字摘要。 Why: 讓診斷結論可直接被 markdown / demo / record 引用。"""
    lines = [
        "=== Window Vorticity Diagnostics ===",
        f"window = {summary['window']}",
        f"checkpoint = {summary['checkpoint']}",
        f"t_end = {summary['t_end']:.6f}",
        f"mean_w_err = {summary['mean_w_err']:.6f}",
        f"final_w_err = {summary['final_w_err']:.6f}",
        f"peak_w_err = {summary['peak_w_err']:.6f}",
        f"peak_w_err_time = {summary['peak_w_err_time']:.6f}",
        f"final_corr = {summary['final_corr']:.6f}",
        f"final_bias = {summary['final_bias']:.6f}",
        f"final_std_ratio = {summary['final_std_ratio']:.6f}",
        f"final_enstrophy_ratio = {summary['final_enstrophy_ratio']:.6f}",
        f"low_k_ratio = {summary['low_k_ratio']:.6f}",
        f"high_k_ratio = {summary['high_k_ratio']:.6f}",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze vorticity diagnostics for a window checkpoint.")
    parser.add_argument("--config", required=True, help="Config path.")
    parser.add_argument("--checkpoint-root", required=True, help="Checkpoint root containing time_window_*.")
    parser.add_argument("--window", type=int, required=True, help="1-based time window index.")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step. Default: latest.")
    parser.add_argument("--chunk-size", type=int, default=4096, help="Spatial chunk size.")
    parser.add_argument("--figure", required=True, help="Output figure path.")
    parser.add_argument("--json", required=True, help="Output JSON summary path.")
    parser.add_argument("--summary", required=True, help="Output text summary path.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config = load_config(config_path)

    dataset_path = Path(config.dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = REPO_ROOT / dataset_path

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

    diagnostics = analyze_window(
        model=model,
        params=model.state.params,
        coords=np.asarray(coords),
        t_window_local=t_window_local,
        t_window_abs=t_window_abs,
        w_ref_window=w_ref_window,
        chunk_size=int(args.chunk_size),
    )

    grid_size = int(np.sqrt(coords.shape[0]))
    final_w_ref_2d = diagnostics["final_w_ref"].reshape(grid_size, grid_size)
    final_w_pred_2d = diagnostics["final_w_pred"].reshape(grid_size, grid_size)
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
        "mean_w_err": float(np.mean(diagnostics["w_rel"])),
        "final_w_err": float(diagnostics["w_rel"][-1]),
        "peak_w_err": float(np.max(diagnostics["w_rel"])),
        "peak_w_err_time": float(t_window_abs[int(np.argmax(diagnostics["w_rel"]))]),
        "final_corr": float(np.corrcoef(diagnostics["final_w_ref"], diagnostics["final_w_pred"])[0, 1]),
        "final_bias": float(np.mean(diagnostics["final_w_pred"] - diagnostics["final_w_ref"])),
        "final_std_ratio": float(np.std(diagnostics["final_w_pred"]) / max(np.std(diagnostics["final_w_ref"]), np.finfo(np.float64).eps)),
        "final_enstrophy_ratio": float(diagnostics["enstrophy_pred"][-1] / max(diagnostics["enstrophy_ref"][-1], np.finfo(np.float64).eps)),
        "low_k_ratio": float(np.sum(e_pred[low_mask_pred]) / max(np.sum(e_ref[low_mask_ref]), np.finfo(np.float64).eps)),
        "high_k_ratio": float(np.sum(e_pred[high_mask_pred]) / max(np.sum(e_ref[high_mask_ref]), np.finfo(np.float64).eps)),
    }

    make_figure(
        output_path=Path(args.figure),
        window_idx=window_idx,
        step=step,
        grid_size=grid_size,
        diagnostics=diagnostics,
        spectrum_ref=(k_ref, e_ref),
        spectrum_pred=(k_pred, e_pred),
    )

    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_payload = {
        "summary": summary,
        "time_series": {
            "t_abs": diagnostics["t_abs"].tolist(),
            "t_local": diagnostics["t_local"].tolist(),
            "w_rel": diagnostics["w_rel"].tolist(),
            "enstrophy_ref": diagnostics["enstrophy_ref"].tolist(),
            "enstrophy_pred": diagnostics["enstrophy_pred"].tolist(),
        },
    }
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    write_summary(Path(args.summary), summary)

    print(f"saved figure: {args.figure}")
    print(f"saved json: {args.json}")
    print(f"saved summary: {args.summary}")


if __name__ == "__main__":
    main()
