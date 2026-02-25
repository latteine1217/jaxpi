import os
import sys

# 修復導入路徑
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
"""
評估 Kolmogorov Flow PINN 的 checkpoint

這個腳本會：
1. 載入指定的 checkpoint
2. 在 DNS 數據上計算 L2 相對誤差
3. 生成誤差報告和可視化
"""

import os
import sys
import argparse
import numpy as np
import jax.numpy as jnp
from jax import random
import ml_collections

# 添加專案路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.dirname(__file__))

from jaxpi.utils import restore_checkpoint
from examples.kolmogorov_flow.utils import get_dataset

from jaxpi.utils import restore_checkpoint
from examples.kolmogorov_flow.utils import get_dataset
from examples.kolmogorov_flow import models


def evaluate_checkpoint(config_name, checkpoint_path, time_window_idx=None):
    """
    評估指定的 checkpoint

    Args:
        config_name: 配置文件名稱 ('soap' 或 'pirate')
        checkpoint_path: checkpoint 目錄路徑
        time_window_idx: 時間窗口索引（如果為 None，則評估所有窗口）
    """

    # 載入配置
    if config_name == "soap":
        from configs import soap as config_module
    elif config_name == "pirate":
        from configs import pirate as config_module
    else:
        raise ValueError(f"Unknown config: {config_name}")

    config = config_module.get_config()

    # 載入 DNS 數據
    print("載入 DNS 參考數據...")
    u_ref, v_ref, w_ref, t_star, coords, nu = get_dataset(time_fraction=config.time_fraction)

    print(f"DNS 數據形狀:")
    print(f"  - 時間步數: {len(t_star)} (t ∈ [{t_star[0]:.4f}, {t_star[-1]:.4f}])")
    print(
        f"  - 空間點數: {coords.shape[0]} ({int(np.sqrt(coords.shape[0]))}×{int(np.sqrt(coords.shape[0]))})"
    )
    print(f"  - u_ref: {u_ref.shape}")
    print(f"  - v_ref: {v_ref.shape}")
    print(f"  - w_ref: {w_ref.shape}")
    print(f"  - 雷諾數: Re = {1 / nu:.0f}")
    print()

    # 計算時間窗口
    num_time_steps = len(t_star) // config.training.num_time_windows

    if time_window_idx is None:
        # 評估所有可用的窗口
        window_dirs = sorted(
            [d for d in os.listdir(checkpoint_path) if d.startswith("time_window_")]
        )
        time_windows = [int(d.split("_")[-1]) for d in window_dirs]
    else:
        time_windows = [time_window_idx]

    print(f"將評估 {len(time_windows)} 個時間窗口: {time_windows}")
    print("=" * 80)
    print()

    results = []

    for window_idx in time_windows:
        print(f"{'=' * 80}")
        print(f"評估 Time Window {window_idx}")
        print(f"{'=' * 80}")

        # 確定當前窗口的時間範圍
        start_idx = (window_idx - 1) * num_time_steps
        end_idx = window_idx * num_time_steps

        t = t_star[start_idx:end_idx]
        u_ref_window = u_ref[start_idx:end_idx, :]
        v_ref_window = v_ref[start_idx:end_idx, :]
        w_ref_window = w_ref[start_idx:end_idx, :]

        print(f"時間範圍: t ∈ [{t[0]:.4f}, {t[-1]:.4f}] ({len(t)} 個時間步)")
        print()

        # 初始化模型
        u0 = u_ref[start_idx, :]
        v0 = v_ref[start_idx, :]
        w0 = w_ref[start_idx, :]

        model = models.NavierStokes(config, t, coords, u0, v0, w0, nu)

        # 載入 checkpoint
        ckpt_dir = os.path.join(checkpoint_path, f"time_window_{window_idx}")

        if not os.path.exists(ckpt_dir):
            print(f"⚠️  Checkpoint 不存在: {ckpt_dir}")
            print()
            continue

        # 查找最新的 checkpoint
        ckpt_files = [f for f in os.listdir(ckpt_dir) if f.startswith("checkpoint_")]
        if not ckpt_files:
            print(f"⚠️  沒有找到 checkpoint 文件: {ckpt_dir}")
            print()
            continue

        # 獲取最大的 step number
        steps = [int(f.split("_")[1]) for f in ckpt_files]
        max_step = max(steps)

        print(f"載入 checkpoint: {ckpt_dir}/checkpoint_{max_step}")
        model.state = restore_checkpoint(model.state, ckpt_dir, step=max_step)
        print(f"✓ Checkpoint 載入成功 (step {max_step})")
        print()

        # 計算時間窗整體誤差（chunked 以降低記憶體峰值）
        print("計算時間窗誤差（chunked）...")
        chunk_seconds = getattr(config.logging, "eval_time_chunk_seconds", 1.0)
        u_error, v_error, w_error = model.compute_l2_error_time_chunked(
            model.state.params,
            t,
            coords,
            u_ref_window,
            v_ref_window,
            w_ref_window,
            chunk_seconds=chunk_seconds,
        )

        errors_u = np.array([float(u_error)])
        errors_v = np.array([float(v_error)])
        errors_w = np.array([float(w_error)])

        # 統計結果
        print()
        print(f"{'=' * 80}")
        print(f"Time Window {window_idx} 誤差統計")
        print(f"{'=' * 80}")
        print(f"{'指標':<20} {'平均':<15} {'最小':<15} {'最大':<15} {'最終':<15}")
        print(f"{'-' * 80}")
        print(
            f"{'u_error':<20} {errors_u.mean():<15.6f} {errors_u.min():<15.6f} {errors_u.max():<15.6f} {errors_u[-1]:<15.6f}"
        )
        print(
            f"{'v_error':<20} {errors_v.mean():<15.6f} {errors_v.min():<15.6f} {errors_v.max():<15.6f} {errors_v[-1]:<15.6f}"
        )
        print(
            f"{'w_error':<20} {errors_w.mean():<15.6f} {errors_w.min():<15.6f} {errors_w.max():<15.6f} {errors_w[-1]:<15.6f}"
        )
        print(f"{'=' * 80}")
        print()

        # 保存結果
        results.append(
            {
                "window": window_idx,
                "time_range": (float(t[0]), float(t[-1])),
                "checkpoint_step": max_step,
                "errors_u": errors_u,
                "errors_v": errors_v,
                "errors_w": errors_w,
                "u_error_mean": errors_u.mean(),
                "v_error_mean": errors_v.mean(),
                "w_error_mean": errors_w.mean(),
                "u_error_final": errors_u[-1],
                "v_error_final": errors_v[-1],
                "w_error_final": errors_w[-1],
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="評估 Kolmogorov Flow PINN checkpoint")
    parser.add_argument(
        "--config", type=str, required=True, choices=["soap", "pirate"], help="配置文件名稱"
    )
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Checkpoint 根目錄路徑")
    parser.add_argument(
        "--window", type=int, default=None, help="指定評估的時間窗口（默認評估所有）"
    )
    parser.add_argument("--output", type=str, default=None, help="輸出結果文件路徑（.npz 格式）")

    args = parser.parse_args()

    # 評估 checkpoint
    results = evaluate_checkpoint(args.config, args.checkpoint_path, args.window)

    # 保存結果
    if args.output:
        print(f"保存結果到: {args.output}")
        np.savez(args.output, results=results)
        print("✓ 結果已保存")

    # 打印總結
    if len(results) > 1:
        print()
        print(f"{'=' * 80}")
        print(f"所有時間窗口的總結")
        print(f"{'=' * 80}")
        print(
            f"{'Window':<10} {'時間範圍':<25} {'u_error (平均)':<20} {'v_error (平均)':<20} {'w_error (平均)':<20}"
        )
        print(f"{'-' * 80}")
        for r in results:
            print(
                f"{r['window']:<10} [{r['time_range'][0]:.4f}, {r['time_range'][1]:.4f}]        "
                f"{r['u_error_mean']:<20.6f} {r['v_error_mean']:<20.6f} {r['w_error_mean']:<20.6f}"
            )
        print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
