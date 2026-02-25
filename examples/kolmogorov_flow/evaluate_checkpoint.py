"""
統一版 Kolmogorov Flow checkpoint 評估腳本

模式：
1. window：評估整個時間窗口（原 evaluate_checkpoint.py）
2. final_step：只評估每個窗口最後一個時間點（整合原 simple/cpu）

設備：
- auto：由 JAX 自動選擇
- cpu：強制使用 CPU
- gpu：強制使用 GPU
"""

import argparse
import os
import sys
from typing import Optional

import numpy as np

# 添加專案路徑
THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)


def configure_device(device: str) -> None:
    """在匯入 JAX 相關模組前設定執行裝置。"""
    if device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    elif device == "gpu":
        os.environ["JAX_PLATFORMS"] = "gpu"
    elif device == "auto":
        # 保持 JAX 預設策略
        pass
    else:
        raise ValueError(f"Unknown device option: {device}")


def load_config(config_name: str):
    """依配置名稱載入實驗 config。"""
    if config_name == "soap":
        from configs import soap as config_module
    elif config_name == "pirate":
        from configs import pirate as config_module
    else:
        raise ValueError(f"Unknown config: {config_name}")

    return config_module.get_config()


def discover_windows(checkpoint_path: str, requested_window: Optional[int]) -> list[int]:
    """取得要評估的時間窗口列表。"""
    if requested_window is not None:
        return [requested_window]

    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint 根目錄不存在: {checkpoint_path}")

    window_dirs = [d for d in os.listdir(checkpoint_path) if d.startswith("time_window_")]
    windows: list[int] = []
    for d in window_dirs:
        try:
            windows.append(int(d.split("_")[-1]))
        except ValueError:
            continue

    windows.sort()
    return windows


def get_latest_checkpoint_step(ckpt_dir: str) -> Optional[int]:
    """回傳指定窗口下最新 checkpoint step，找不到則回傳 None。"""
    if not os.path.isdir(ckpt_dir):
        return None

    steps = []
    for name in os.listdir(ckpt_dir):
        if not name.startswith("checkpoint_"):
            continue
        try:
            steps.append(int(name.split("_")[1]))
        except (IndexError, ValueError):
            continue

    return max(steps) if steps else None


def evaluate_checkpoint(
    config_name: str,
    checkpoint_path: str,
    mode: str,
    device: str,
    time_window_idx: Optional[int] = None,
):
    """統一評估入口。"""
    configure_device(device)

    # 延後匯入，確保 device 設定先套用
    import jax.numpy as jnp

    from jaxpi.utils import restore_checkpoint
    from examples.kolmogorov_flow.utils import get_dataset
    from examples.kolmogorov_flow import models

    config = load_config(config_name)

    print("=== Data Loading ===")
    u_ref, v_ref, w_ref, t_star, coords, nu = get_dataset(time_fraction=config.time_fraction)
    print(f"time steps: {len(t_star)} (t in [{float(t_star[0]):.4f}, {float(t_star[-1]):.4f}])")
    print(
        f"space points: {coords.shape[0]} ({int(np.sqrt(coords.shape[0]))}x{int(np.sqrt(coords.shape[0]))})"
    )
    print(f"u_ref shape: {u_ref.shape}")
    print(f"v_ref shape: {v_ref.shape}")
    print(f"w_ref shape: {w_ref.shape}")
    print(f"Re: {1 / nu:.0f}")
    print()

    num_time_steps = len(t_star) // config.training.num_time_windows
    time_windows = discover_windows(checkpoint_path, time_window_idx)

    if not time_windows:
        raise RuntimeError("找不到可評估的時間窗口，請檢查 checkpoint_path。")

    print("=== Evaluation Setup ===")
    print(f"config: {config_name}")
    print(f"mode: {mode}")
    print(f"device: {device}")
    print(f"windows: {time_windows}")
    print()

    results = []

    for window_idx in time_windows:
        print("=" * 80)
        print(f"Evaluating Time Window {window_idx}")
        print("=" * 80)

        start_idx = (window_idx - 1) * num_time_steps
        end_idx = window_idx * num_time_steps

        if start_idx >= len(t_star) or end_idx > len(t_star):
            print(f"⚠️  跳過窗口 {window_idx}: 超出時間資料範圍")
            print()
            continue

        t_window = t_star[start_idx:end_idx]
        u_ref_window = u_ref[start_idx:end_idx, :]
        v_ref_window = v_ref[start_idx:end_idx, :]
        w_ref_window = w_ref[start_idx:end_idx, :]

        print(f"time range: [{float(t_window[0]):.4f}, {float(t_window[-1]):.4f}] ({len(t_window)} steps)")

        u0 = u_ref[start_idx, :]
        v0 = v_ref[start_idx, :]
        w0 = w_ref[start_idx, :]

        # CPU final_step 模式可只保留窗口首尾時間，降低建模記憶體壓力
        if mode == "final_step" and device == "cpu":
            model_t = jnp.asarray([t_window[0], t_window[-1]])
        else:
            model_t = t_window

        model = models.NavierStokes(config, model_t, coords, u0, v0, w0, nu)

        ckpt_dir = os.path.join(checkpoint_path, f"time_window_{window_idx}")
        max_step = get_latest_checkpoint_step(ckpt_dir)

        if max_step is None:
            print(f"⚠️  跳過窗口 {window_idx}: 找不到 checkpoint ({ckpt_dir})")
            print()
            continue

        print(f"checkpoint: {ckpt_dir}/checkpoint_{max_step}")
        model.state = restore_checkpoint(model.state, ckpt_dir, step=max_step)

        chunk_seconds = getattr(config.logging, "eval_time_chunk_seconds", 1.0)

        if mode == "window":
            print("running full-window L2 error ...")
            u_error, v_error, w_error = model.compute_l2_error_time_chunked(
                model.state.params,
                t_window,
                coords,
                u_ref_window,
                v_ref_window,
                w_ref_window,
                chunk_seconds=chunk_seconds,
            )

            record = {
                "window": window_idx,
                "mode": mode,
                "device": device,
                "checkpoint_step": max_step,
                "t_start": float(t_window[0]),
                "t_end": float(t_window[-1]),
                "u_error": float(u_error),
                "v_error": float(v_error),
                "w_error": float(w_error),
            }
            print(
                f"u={record['u_error']:.6f}, v={record['v_error']:.6f}, w={record['w_error']:.6f}"
            )

        elif mode == "final_step":
            print("running final-step L2 error ...")
            t_final = float(t_window[-1])

            t_eval = jnp.asarray([t_final])
            u_ref_final = u_ref_window[-1:, :]
            v_ref_final = v_ref_window[-1:, :]
            w_ref_final = w_ref_window[-1:, :]

            u_error, v_error, w_error = model.compute_l2_error_time_chunked(
                model.state.params,
                t_eval,
                coords,
                u_ref_final,
                v_ref_final,
                w_ref_final,
                chunk_seconds=chunk_seconds,
            )

            record = {
                "window": window_idx,
                "mode": mode,
                "device": device,
                "checkpoint_step": max_step,
                "t_eval": t_final,
                "u_error": float(u_error),
                "v_error": float(v_error),
                "w_error": float(w_error),
            }
            print(
                f"t={record['t_eval']:.4f}, u={record['u_error']:.6f}, v={record['v_error']:.6f}, w={record['w_error']:.6f}"
            )

        else:
            raise ValueError(f"Unknown mode: {mode}")

        results.append(record)
        print()

    return results


def print_summary(results: list[dict], mode: str) -> None:
    """列印摘要表。"""
    if not results:
        print("=== Summary ===")
        print("沒有可用結果。")
        return

    print("=== Summary ===")
    if mode == "window":
        print(f"{'Window':<10} {'Time Range':<25} {'u_error':<12} {'v_error':<12} {'w_error':<12}")
        print("-" * 80)
        for r in results:
            print(
                f"{r['window']:<10} [{r['t_start']:.4f}, {r['t_end']:.4f}]      "
                f"{r['u_error']:<12.6f} {r['v_error']:<12.6f} {r['w_error']:<12.6f}"
            )
    else:
        print(f"{'Window':<10} {'t_eval':<12} {'u_error':<12} {'v_error':<12} {'w_error':<12}")
        print("-" * 70)
        for r in results:
            print(
                f"{r['window']:<10} {r['t_eval']:<12.4f} "
                f"{r['u_error']:<12.6f} {r['v_error']:<12.6f} {r['w_error']:<12.6f}"
            )

    u_errors = np.array([r["u_error"] for r in results])
    v_errors = np.array([r["v_error"] for r in results])
    w_errors = np.array([r["w_error"] for r in results])

    print("-" * 80)
    print(
        f"mean: u={u_errors.mean():.6f}, v={v_errors.mean():.6f}, w={w_errors.mean():.6f}"
    )
    print(
        f"min : u={u_errors.min():.6f}, v={v_errors.min():.6f}, w={w_errors.min():.6f}"
    )
    print(
        f"max : u={u_errors.max():.6f}, v={v_errors.max():.6f}, w={w_errors.max():.6f}"
    )


def main():
    parser = argparse.ArgumentParser(description="評估 Kolmogorov Flow checkpoint（統一版）")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        choices=["soap", "pirate"],
        help="配置名稱",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Checkpoint 根目錄",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="window",
        choices=["window", "final_step"],
        help="評估模式：window 或 final_step",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="執行裝置",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=None,
        help="只評估指定時間窗口（預設評估全部）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="輸出結果檔案路徑（npz）",
    )

    args = parser.parse_args()

    results = evaluate_checkpoint(
        config_name=args.config,
        checkpoint_path=args.checkpoint_path,
        mode=args.mode,
        device=args.device,
        time_window_idx=args.window,
    )

    if args.output:
        print(f"=== Save Output ===\n{args.output}")
        np.savez(args.output, results=np.array(results, dtype=object))
        print("saved")

    print()
    print_summary(results, args.mode)


if __name__ == "__main__":
    main()
