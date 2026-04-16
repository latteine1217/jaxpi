#!/usr/bin/env python3
"""
評估並繪製 window-1 ablation 的 checkpoint sweep。

What:
    對 no-data 與 sensor 兩個版本，在同一個 time window 內評估指定的
    checkpoint steps，並輸出 full-window relative L2 與 vorticity 場圖。

Why:
    這個實驗要比較「加入 sparse data 是否加速收斂」。只看最終 loss
    無法回答收斂速度，因此必須保留 checkpoint step 維度，並用相同
    DNS window、相同 local-time 定義、相同繪圖尺度比較兩個版本。
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_SCRIPT_PATH = Path(__file__).resolve()
if (_SCRIPT_PATH.parent / "examples").exists():
    REPO_ROOT = _SCRIPT_PATH.parent
elif (_SCRIPT_PATH.parent.parent / "examples").exists():
    REPO_ROOT = _SCRIPT_PATH.parent.parent
elif (_SCRIPT_PATH.parent.parent.parent / "examples").exists():
    REPO_ROOT = _SCRIPT_PATH.parent.parent.parent
elif (Path.cwd() / "examples").exists():
    REPO_ROOT = Path.cwd()
else:
    REPO_ROOT = _SCRIPT_PATH.parents[min(2, len(_SCRIPT_PATH.parents) - 1)]
sys.path.insert(0, str(REPO_ROOT))

from examples.kolmogorov_flow import models
from examples.kolmogorov_flow.utils import get_dataset
from jaxpi.utils import restore_checkpoint


DEFAULT_STEPS = tuple(range(10_000, 100_001, 10_000))


@dataclass(frozen=True)
class RunSpec:
    """What: 一組待評估 run 的必要資訊。 Why: 避免兩版本參數在函式間散落。"""

    label: str
    config_path: Path
    checkpoint_root: Path


def parse_steps(raw: str) -> list[int]:
    """What: 解析 checkpoint step list。 Why: 明確定義 sweep 的評估點。"""
    steps = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not steps:
        raise ValueError("steps must not be empty")
    if sorted(set(steps)) != steps:
        raise ValueError("steps must be strictly increasing and unique")
    return steps


def resolve_repo_path(path: str) -> Path:
    """What: 支援絕對路徑與 repo-relative 路徑。 Why: 同一腳本要能本地與 server 共用。"""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def load_config(config_path: Path):
    """What: 動態載入 config。 Why: 評估腳本不應綁死單一實驗模組。"""
    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")
    spec = importlib.util.spec_from_file_location(config_path.stem, config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load config: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()


def resolve_checkpoint_root(path: Path) -> Path:
    """What: 接受 experiment root 或 ckpt root。 Why: 降低提交 Slurm 時填錯層級的風險。"""
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"checkpoint path not found: {path}")
    if any(child.name.startswith("time_window_") for child in path.iterdir() if child.is_dir()):
        return path
    ckpt_dir = path / "ckpt"
    if ckpt_dir.is_dir():
        return ckpt_dir
    raise FileNotFoundError(f"cannot resolve checkpoint root from: {path}")


def require_checkpoints(ckpt_root: Path, window: int, steps: list[int], allow_missing: bool) -> list[int]:
    """What: 檢查 checkpoint 是否存在。 Why: 完整 10 點 sweep 不可被缺檔靜默污染。"""
    window_dir = ckpt_root / f"time_window_{window}"
    if not window_dir.is_dir():
        raise FileNotFoundError(f"window checkpoint dir not found: {window_dir}")

    available: list[int] = []
    missing: list[int] = []
    for step in steps:
        ckpt_dir = window_dir / f"checkpoint_{step}"
        if ckpt_dir.is_dir():
            available.append(step)
        else:
            missing.append(step)

    if missing and not allow_missing:
        raise FileNotFoundError(
            f"missing checkpoints under {window_dir}: "
            + ", ".join(f"checkpoint_{step}" for step in missing)
        )
    return available


def to_window_local_time(t_window: np.ndarray) -> np.ndarray:
    """
    What:
        將單一 window 的 absolute time 轉成以 0 起算的 local time。
    Why:
        訓練時 time-window checkpoint 使用 local-time 定義；評估若餵入
        DNS absolute time，會把 time coordinate mismatch 混進誤差。
    """
    t_window = np.asarray(t_window)
    return t_window - float(t_window[0])


def resolve_eval_chunk_sizes(config, t_values: np.ndarray, time_chunk_size: int | None, space_chunk_size: int | None) -> tuple[int, int]:
    """What: 統一解析 full-window L2 chunk。 Why: 評估要可重跑且避免 OOM。"""
    if time_chunk_size is None:
        configured = getattr(config.logging, "eval_time_chunk_size", None)
        if configured is not None:
            time_chunk_size = int(configured)
        else:
            chunk_seconds = float(getattr(config.logging, "eval_time_chunk_seconds", 1.0))
            if t_values.size > 1:
                dt = float(np.sort(t_values)[1] - np.sort(t_values)[0])
                time_chunk_size = max(1, int(chunk_seconds / dt)) if dt > 0 else 1
            else:
                time_chunk_size = 1

    if space_chunk_size is None:
        space_chunk_size = int(getattr(config.logging, "eval_space_chunk_size", 4096))

    return max(1, int(time_chunk_size)), max(1, int(space_chunk_size))


def predict_uv_direct(model, params, t_eval: float, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    What:
        用單一明確 batch 維直接呼叫 `state.apply_fn` 預測 `u/v`。
    Why:
        `model.u_pred_fn/u_ic_pred_fn` 會在 vmapped scalar function 內部建立 fake
        batch 維，已驗證會造成 batch/chunk-size dependent error；評估必須避開。
    """
    coords_jnp = jnp.asarray(coords)
    t_norm = float(t_eval) / float(model.t_star[-1])
    t_col = jnp.full((coords_jnp.shape[0], 1), t_norm, dtype=coords_jnp.dtype)
    z = jnp.concatenate([t_col, coords_jnp], axis=-1)
    _, outputs = model.state.apply_fn(params, z)
    outputs = jax.device_get(outputs)
    return np.asarray(outputs[:, 0]), np.asarray(outputs[:, 1])


def predict_vorticity_direct(model, params, t_eval: float, coords: np.ndarray) -> np.ndarray:
    """
    What:
        用 direct scalar `apply_fn` 搭配自動微分計算 vorticity。
    Why:
        vorticity 依賴 `dv/dx - du/dy`；為避免 `w_pred_fn/w_ic_pred_fn` 的
        vmap scalar-wrapper bug，梯度也必須走同一條 direct `apply_fn` 路徑。
    """
    t_norm = float(t_eval) / float(model.t_star[-1])

    def uv_single(x, y):
        z = jnp.array([[t_norm, x, y]], dtype=jnp.result_type(x, y))
        _, outputs = model.state.apply_fn(params, z)
        return outputs[0, 0], outputs[0, 1]

    def u_single(x, y):
        u, _ = uv_single(x, y)
        return u

    def v_single(x, y):
        _, v = uv_single(x, y)
        return v

    def w_single(x, y):
        return jax.grad(v_single, argnums=0)(x, y) - jax.grad(u_single, argnums=1)(x, y)

    coords_jnp = jnp.asarray(coords)
    w_values = jax.vmap(w_single)(coords_jnp[:, 0], coords_jnp[:, 1])
    return np.asarray(jax.device_get(w_values))


def predict_vorticity_in_chunks(model, params, t_eval: float, coords: np.ndarray, chunk_size: int) -> np.ndarray:
    """What: 分塊預測 vorticity。 Why: 高解析網格一次 forward 容易造成記憶體峰值過高。"""
    chunks = []
    for start in range(0, coords.shape[0], chunk_size):
        end = min(start + chunk_size, coords.shape[0])
        coords_chunk = coords[start:end]
        pred = predict_vorticity_direct(model, params, t_eval, coords_chunk)
        chunks.append(pred)
    return np.concatenate(chunks, axis=0)


def compute_l2_error_direct(
    model,
    params,
    t_values: np.ndarray,
    coords: np.ndarray,
    u_ref: np.ndarray,
    v_ref: np.ndarray,
    w_ref: np.ndarray,
    *,
    space_chunk_size: int,
) -> tuple[float, float, float]:
    """
    What:
        使用 direct batched `apply_fn` 路徑計算 full-window relative L2。
    Why:
        這是目前已由 `3271/3272` audit 驗證可復現舊 corrected result 的
        batch-invariant 路徑；不要再用 wrapper-based chunked evaluator。
    """
    total_u = 0.0
    total_v = 0.0
    total_w = 0.0
    denom_u = 0.0
    denom_v = 0.0
    denom_w = 0.0
    chunk_size = max(1, int(space_chunk_size))

    for time_idx, t_eval in enumerate(np.asarray(t_values)):
        for start in range(0, coords.shape[0], chunk_size):
            end = min(start + chunk_size, coords.shape[0])
            coords_chunk = coords[start:end]
            u_target = np.asarray(u_ref[time_idx, start:end])
            v_target = np.asarray(v_ref[time_idx, start:end])
            w_target = np.asarray(w_ref[time_idx, start:end])

            u_pred, v_pred = predict_uv_direct(model, params, float(t_eval), coords_chunk)
            w_pred = predict_vorticity_direct(model, params, float(t_eval), coords_chunk)

            total_u += float(np.sum((u_pred - u_target) ** 2))
            total_v += float(np.sum((v_pred - v_target) ** 2))
            total_w += float(np.sum((w_pred - w_target) ** 2))
            denom_u += float(np.sum(u_target**2))
            denom_v += float(np.sum(v_target**2))
            denom_w += float(np.sum(w_target**2))

    eps = np.finfo(np.float64).eps
    return (
        float(np.sqrt(total_u / max(denom_u, eps))),
        float(np.sqrt(total_v / max(denom_v, eps))),
        float(np.sqrt(total_w / max(denom_w, eps))),
    )


def evaluate_run(
    run: RunSpec,
    *,
    window: int,
    steps: list[int],
    allow_missing: bool,
    time_chunk_size: int | None,
    space_chunk_size: int | None,
) -> tuple[list[dict], dict]:
    """What: 評估單一 run 的 full-window relative L2。 Why: 保持 no-data/sensor 對稱流程。"""
    config = load_config(run.config_path)
    dataset_path = resolve_repo_path(str(config.dataset_path))
    ckpt_root = resolve_checkpoint_root(run.checkpoint_root)
    available_steps = require_checkpoints(ckpt_root, window, steps, allow_missing)

    u_ref, v_ref, w_ref, t_star, coords, nu = get_dataset(
        time_fraction=config.time_fraction,
        dataset_path=str(dataset_path),
        time_range=config.get("dns_time_range"),
        time_stride=config.get("dns_time_stride", 1),
    )

    num_windows = int(config.training.num_time_windows)
    num_time_steps = len(t_star) // num_windows
    start_idx = (window - 1) * num_time_steps
    end_idx = window * num_time_steps
    if window < 1 or start_idx < 0 or end_idx > len(t_star):
        raise ValueError(f"window {window} out of range for {num_windows} windows")

    t_window = np.asarray(t_star[start_idx:end_idx])
    t_window_local = to_window_local_time(t_window)
    u_ref_window = np.asarray(u_ref[start_idx:end_idx, :])
    v_ref_window = np.asarray(v_ref[start_idx:end_idx, :])
    w_ref_window = np.asarray(w_ref[start_idx:end_idx, :])

    model = models.NavierStokes(
        config,
        t_window_local,
        coords,
        u_ref[start_idx, :],
        v_ref[start_idx, :],
        w_ref[start_idx, :],
        nu,
        replicate_state=False,
    )
    eval_time_chunk, eval_space_chunk = resolve_eval_chunk_sizes(
        config, t_window_local, time_chunk_size, space_chunk_size
    )

    rows: list[dict] = []
    window_dir = ckpt_root / f"time_window_{window}"
    print(f"=== Evaluate {run.label} ===")
    print(f"config: {run.config_path}")
    print(f"checkpoint_root: {ckpt_root}")
    print(f"window: {window}")
    print(f"steps: {available_steps}")
    print(f"chunk: time={eval_time_chunk}, space={eval_space_chunk}")
    print("forward_path: direct_apply_fn")

    for step in available_steps:
        print(f"--- {run.label} checkpoint_{step} ---", flush=True)
        model.state = restore_checkpoint(model.state, str(window_dir), step=step)
        u_error, v_error, w_error = compute_l2_error_direct(
            model,
            model.state.params,
            t_window_local,
            coords,
            u_ref_window,
            v_ref_window,
            w_ref_window,
            space_chunk_size=eval_space_chunk,
        )
        row = {
            "run": run.label,
            "window": window,
            "checkpoint_step": step,
            "t_start": float(t_window[0]),
            "t_end": float(t_window[-1]),
            "u_error": float(u_error),
            "v_error": float(v_error),
            "w_error": float(w_error),
        }
        rows.append(row)
        print(
            f"u={row['u_error']:.6e}, "
            f"v={row['v_error']:.6e}, "
            f"w={row['w_error']:.6e}",
            flush=True,
        )

    context = {
        "config": config,
        "checkpoint_root": ckpt_root,
        "dataset_path": dataset_path,
        "u_ref": u_ref,
        "v_ref": v_ref,
        "w_ref": w_ref,
        "t_star": t_star,
        "coords": coords,
        "nu": nu,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "t_window": t_window,
        "t_window_local": t_window_local,
    }
    return rows, context


def write_results(output_dir: Path, rows: list[dict], metadata: dict) -> None:
    """What: 輸出 CSV/NPZ/summary。 Why: 後續分析不應依賴 console log。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "checkpoint_sweep_results.csv"
    npz_path = output_dir / "checkpoint_sweep_results.npz"
    txt_path = output_dir / "checkpoint_sweep_summary.txt"

    fieldnames = [
        "run",
        "window",
        "checkpoint_step",
        "t_start",
        "t_end",
        "u_error",
        "v_error",
        "w_error",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    np.savez(
        npz_path,
        run=np.array([row["run"] for row in rows], dtype=object),
        window=np.array([row["window"] for row in rows], dtype=np.int32),
        checkpoint_step=np.array([row["checkpoint_step"] for row in rows], dtype=np.int32),
        u_error=np.array([row["u_error"] for row in rows], dtype=np.float64),
        v_error=np.array([row["v_error"] for row in rows], dtype=np.float64),
        w_error=np.array([row["w_error"] for row in rows], dtype=np.float64),
        metadata=np.array(json.dumps(metadata, indent=2), dtype=object),
    )

    with txt_path.open("w") as f:
        f.write("=== Window Checkpoint Sweep Summary ===\n")
        f.write(json.dumps(metadata, indent=2))
        f.write("\n\n")
        f.write(f"{'run':<12} {'step':>8} {'u_error':>14} {'v_error':>14} {'w_error':>14}\n")
        f.write("-" * 68 + "\n")
        for row in rows:
            f.write(
                f"{row['run']:<12} {row['checkpoint_step']:>8d} "
                f"{row['u_error']:>14.6e} {row['v_error']:>14.6e} "
                f"{row['w_error']:>14.6e}\n"
            )

    print(f"saved: {csv_path}")
    print(f"saved: {npz_path}")
    print(f"saved: {txt_path}")


def plot_error_vs_step(output_dir: Path, rows: list[dict]) -> None:
    """What: 畫出 checkpoint step 對 full-window error。 Why: 直接比較收斂速度。"""
    output_path = output_dir / "checkpoint_sweep_error_vs_step.png"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    components = [("u_error", "u"), ("v_error", "v"), ("w_error", "vorticity")]
    colors = {"no_data": "#2b6cb0", "sensor": "#c2410c"}

    for ax, (key, component_name) in zip(axes, components):
        for run_label in sorted({row["run"] for row in rows}):
            run_rows = sorted(
                [row for row in rows if row["run"] == run_label],
                key=lambda row: row["checkpoint_step"],
            )
            ax.plot(
                [row["checkpoint_step"] for row in run_rows],
                [row[key] for row in run_rows],
                marker="o",
                linewidth=2.0,
                label=run_label,
                color=colors.get(run_label),
            )
        ax.set_title(f"{component_name} Relative L2")
        ax.set_xlabel("Checkpoint step")
        ax.set_ylabel("Relative L2 error")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Window 1 Full-Window Error vs Checkpoint Step")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"saved: {output_path}")


def plot_vorticity_grid(
    run: RunSpec,
    context: dict,
    *,
    window: int,
    steps: list[int],
    time_index: int,
    chunk_size: int,
    plot_stride: int,
    output_dir: Path,
) -> None:
    """What: 針對單一 run 畫 DNS/PINN/abs-error vorticity grid。 Why: error scalar 需要場圖驗證。"""
    config = context["config"]
    coords = np.asarray(context["coords"])
    w_ref = np.asarray(context["w_ref"])
    t_window = np.asarray(context["t_window"])
    t_window_local = np.asarray(context["t_window_local"])
    start_idx = int(context["start_idx"])
    ckpt_root = Path(context["checkpoint_root"])
    nu = context["nu"]

    local_time_idx = int(time_index)
    if local_time_idx < 0:
        local_time_idx = len(t_window) + local_time_idx
    if local_time_idx < 0 or local_time_idx >= len(t_window):
        raise ValueError(f"time-index {time_index} out of range for window length {len(t_window)}")

    global_time_idx = start_idx + local_time_idx
    grid_size = int(np.sqrt(coords.shape[0]))
    if grid_size * grid_size != coords.shape[0]:
        raise ValueError(f"coords size {coords.shape[0]} is not a square grid")

    stride = max(1, int(plot_stride))
    coords_grid = coords.reshape(grid_size, grid_size, 2)
    coords_sub = coords_grid[::stride, ::stride, :].reshape(-1, 2)
    sub_shape = coords_grid[::stride, ::stride, 0].shape

    u0 = np.asarray(context["u_ref"][start_idx, :]).reshape(grid_size, grid_size)[::stride, ::stride].reshape(-1)
    v0 = np.asarray(context["v_ref"][start_idx, :]).reshape(grid_size, grid_size)[::stride, ::stride].reshape(-1)
    w0 = np.asarray(w_ref[start_idx, :]).reshape(grid_size, grid_size)[::stride, ::stride].reshape(-1)
    w_dns = np.asarray(w_ref[global_time_idx, :]).reshape(grid_size, grid_size)[::stride, ::stride]

    model = models.NavierStokes(
        config,
        t_window_local,
        coords_sub,
        u0,
        v0,
        w0,
        nu,
        replicate_state=False,
    )
    t_eval_local = float(t_window_local[local_time_idx])
    t_eval_abs = float(t_window[local_time_idx])
    window_dir = ckpt_root / f"time_window_{window}"

    predictions = []
    error_max = 0.0
    field_min = float(np.min(w_dns))
    field_max = float(np.max(w_dns))

    for step in steps:
        model.state = restore_checkpoint(model.state, str(window_dir), step=step)
        w_pred = predict_vorticity_in_chunks(
            model,
            model.state.params,
            t_eval_local,
            coords_sub,
            chunk_size,
        ).reshape(sub_shape)
        predictions.append((step, w_pred))
        field_min = min(field_min, float(np.min(w_pred)))
        field_max = max(field_max, float(np.max(w_pred)))
        error_max = max(error_max, float(np.max(np.abs(w_pred - w_dns))))

    fig, axes = plt.subplots(
        len(predictions),
        3,
        figsize=(10.5, max(2.1 * len(predictions), 6.0)),
        squeeze=False,
    )
    cmap = "RdBu_r"
    err_cmap = "magma"
    error_max = max(error_max, 1e-12)

    for row_idx, (step, w_pred) in enumerate(predictions):
        err = np.abs(w_pred - w_dns)
        images = [
            axes[row_idx, 0].imshow(w_dns.T, origin="lower", cmap=cmap, vmin=field_min, vmax=field_max),
            axes[row_idx, 1].imshow(w_pred.T, origin="lower", cmap=cmap, vmin=field_min, vmax=field_max),
            axes[row_idx, 2].imshow(err.T, origin="lower", cmap=err_cmap, vmin=0.0, vmax=error_max),
        ]
        for col_idx, ax in enumerate(axes[row_idx]):
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(["DNS", "PINN", "|Error|"][col_idx])
        axes[row_idx, 0].set_ylabel(f"step={step}", rotation=0, ha="right", va="center")

    fig.colorbar(images[1], ax=axes[:, :2], shrink=0.65, label="Vorticity")
    fig.colorbar(images[2], ax=axes[:, 2], shrink=0.65, label="Absolute error")
    fig.suptitle(
        f"{run.label}: Window {window} Vorticity Field "
        f"(t_abs={t_eval_abs:.4f}, t_local={t_eval_local:.4f}, stride={stride})"
    )
    fig.tight_layout(rect=(0, 0, 0.94, 0.98))

    output_path = output_dir / f"vorticity_sweep_{run.label}.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate and plot two window-1 checkpoint sweeps."
    )
    parser.add_argument(
        "--no-data-config",
        default="examples/kolmogorov_flow/configs/paper_repro_soap_window1_ablation.py",
        help="No-data config path.",
    )
    parser.add_argument(
        "--no-data-checkpoint-root",
        default="re1e6_n512_ds4_soap_w1_ablation/ckpt",
        help="No-data checkpoint root or experiment root.",
    )
    parser.add_argument(
        "--sensor-config",
        default="examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50_window1_ablation.py",
        help="Sensor config path.",
    )
    parser.add_argument(
        "--sensor-checkpoint-root",
        default="re1e6_n512_ds4_soap_sensor100_w50_w1_ablation/ckpt",
        help="Sensor checkpoint root or experiment root.",
    )
    parser.add_argument("--window", type=int, default=1, help="1-based time window index.")
    parser.add_argument(
        "--steps",
        default=",".join(str(step) for step in DEFAULT_STEPS),
        help="Comma-separated checkpoint steps.",
    )
    parser.add_argument(
        "--output-dir",
        default="eval_runs/window1_checkpoint_sweep",
        help="Output directory.",
    )
    parser.add_argument(
        "--time-index",
        type=int,
        default=-1,
        help="Window-local time index for vorticity field plots.",
    )
    parser.add_argument(
        "--time-chunk-size",
        type=int,
        default=None,
        help="Override full-window L2 time chunk size.",
    )
    parser.add_argument(
        "--space-chunk-size",
        type=int,
        default=None,
        help="Override full-window L2 space chunk size.",
    )
    parser.add_argument(
        "--field-chunk-size",
        type=int,
        default=4096,
        help="Spatial chunk size for field prediction.",
    )
    parser.add_argument(
        "--plot-stride",
        type=int,
        default=2,
        help="Subsample stride for vorticity grid plots.",
    )
    parser.add_argument(
        "--skip-fields",
        action="store_true",
        help="Only compute scalar errors and skip vorticity field grids.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Run partial sweep instead of failing when requested checkpoints are missing.",
    )
    args = parser.parse_args()

    steps = parse_steps(args.steps)
    output_dir = resolve_repo_path(args.output_dir)
    runs = [
        RunSpec(
            label="no_data",
            config_path=resolve_repo_path(args.no_data_config),
            checkpoint_root=resolve_repo_path(args.no_data_checkpoint_root),
        ),
        RunSpec(
            label="sensor",
            config_path=resolve_repo_path(args.sensor_config),
            checkpoint_root=resolve_repo_path(args.sensor_checkpoint_root),
        ),
    ]

    print("=== Window Checkpoint Sweep ===")
    print(f"repo_root: {REPO_ROOT}")
    print(f"output_dir: {output_dir}")
    print(f"window: {args.window}")
    print(f"steps: {steps}")

    all_rows: list[dict] = []
    contexts: dict[str, dict] = {}
    for run in runs:
        rows, context = evaluate_run(
            run,
            window=args.window,
            steps=steps,
            allow_missing=args.allow_missing,
            time_chunk_size=args.time_chunk_size,
            space_chunk_size=args.space_chunk_size,
        )
        all_rows.extend(rows)
        contexts[run.label] = context

    metadata = {
        "window": args.window,
        "requested_steps": steps,
        "time_index": args.time_index,
        "plot_stride": args.plot_stride,
        "forward_path": "direct_apply_fn",
        "runs": [
            {
                "label": run.label,
                "config_path": str(run.config_path),
                "checkpoint_root": str(resolve_checkpoint_root(run.checkpoint_root)),
            }
            for run in runs
        ],
    }
    write_results(output_dir, all_rows, metadata)
    plot_error_vs_step(output_dir, all_rows)

    if not args.skip_fields:
        for run in runs:
            run_steps = sorted(
                row["checkpoint_step"] for row in all_rows if row["run"] == run.label
            )
            plot_vorticity_grid(
                run,
                contexts[run.label],
                window=args.window,
                steps=run_steps,
                time_index=args.time_index,
                chunk_size=int(args.field_chunk_size),
                plot_stride=int(args.plot_stride),
                output_dir=output_dir,
            )

    print("=== Done ===")


if __name__ == "__main__":
    main()
