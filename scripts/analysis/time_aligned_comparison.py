#!/usr/bin/env python3
"""
時間對齊的公平比較：在相同物理時間評估 PIRATE 和 SOAP。
"""

import argparse
import os
import sys
from typing import Optional

import numpy as np

THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, "../.."))
EXAMPLE_DIR = os.path.join(ROOT_DIR, "examples", "kolmogorov_flow")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if EXAMPLE_DIR not in sys.path:
    sys.path.insert(0, EXAMPLE_DIR)


def load_config(config_name: str):
    if config_name == "pirate":
        from examples.kolmogorov_flow.configs import pirate as config_module
    elif config_name == "soap":
        from examples.kolmogorov_flow.configs import soap as config_module
    else:
        raise ValueError(f"Unknown config: {config_name}")
    return config_module.get_config()


def get_latest_checkpoint_step(ckpt_dir: str) -> Optional[int]:
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


def find_window_for_time(t_target, t_star, num_windows):
    """找出包含目標時間的時間窗口（1-based）。"""
    num_steps_per_window = len(t_star) // num_windows

    for w in range(1, num_windows + 1):
        start_idx = (w - 1) * num_steps_per_window
        end_idx = w * num_steps_per_window - 1
        if t_star[start_idx] <= t_target <= t_star[end_idx]:
            return w

    return None


def resolve_space_chunk_size(config) -> int:
    space_chunk_size = int(getattr(config.logging, "eval_space_chunk_size", 4096))
    return space_chunk_size if space_chunk_size > 0 else 4096


def evaluate_at_time(
    config_name,
    checkpoint_base,
    window_idx,
    t_target,
    dns_data,
):
    """
    在指定時間評估模型。
    """
    import jax.numpy as jnp

    from jaxpi.utils import restore_checkpoint
    from examples.kolmogorov_flow import models

    config = load_config(config_name)

    t_star = dns_data["t"]
    coords = dns_data["coords"]
    u_ref_all = dns_data["u_ref"]
    v_ref_all = dns_data["v_ref"]
    w_ref_all = dns_data["w_ref"]
    nu = dns_data["nu"]

    num_steps_per_window = len(t_star) // config.training.num_time_windows
    start_idx = (window_idx - 1) * num_steps_per_window
    end_idx = window_idx * num_steps_per_window

    t_window = t_star[start_idx:end_idx]
    if not (t_window[0] <= t_target <= t_window[-1]):
        raise ValueError(
            f"Target time {t_target:.6f} is outside window {window_idx} "
            f"range [{t_window[0]:.6f}, {t_window[-1]:.6f}]"
        )

    u0 = u_ref_all[start_idx, :]
    v0 = v_ref_all[start_idx, :]
    w0 = w_ref_all[start_idx, :]

    model = models.NavierStokes(config, t_window, coords, u0, v0, w0, nu)

    ckpt_dir = os.path.join(checkpoint_base, f"time_window_{window_idx}")
    max_step = get_latest_checkpoint_step(ckpt_dir)
    if max_step is None:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
    model.state = restore_checkpoint(model.state, ckpt_dir, step=max_step)

    dns_time_idx = int(np.argmin(np.abs(t_star - t_target)))
    t_actual = float(t_star[dns_time_idx])

    u_dns = u_ref_all[dns_time_idx : dns_time_idx + 1, :]
    v_dns = v_ref_all[dns_time_idx : dns_time_idx + 1, :]
    w_dns = w_ref_all[dns_time_idx : dns_time_idx + 1, :]
    t_eval = jnp.asarray([t_actual])

    u_error, v_error, w_error = model.compute_l2_error_time_space_chunked(
        model.state.params,
        t_eval,
        coords,
        u_dns,
        v_dns,
        w_dns,
        time_chunk_size=1,
        space_chunk_size=resolve_space_chunk_size(config),
    )

    return {
        "u_error": float(u_error),
        "v_error": float(v_error),
        "w_error": float(w_error),
        "t_actual": t_actual,
        "window": window_idx,
        "checkpoint_step": int(max_step),
    }


def generate_time_aligned_comparison(args):
    from examples.kolmogorov_flow.utils import get_dataset

    print("=" * 100)
    print("時間對齊的 PIRATE vs SOAP 公平比較")
    print("=" * 100)

    u_ref, v_ref, w_ref, t_star, coords, nu = get_dataset(
        time_fraction=args.time_fraction,
        dataset_path=args.dataset_path,
    )
    dns_data = {
        "t": np.asarray(t_star),
        "coords": np.asarray(coords),
        "u_ref": np.asarray(u_ref),
        "v_ref": np.asarray(v_ref),
        "w_ref": np.asarray(w_ref),
        "nu": float(nu),
    }

    print(
        f"DNS: time_steps={len(t_star)}, space_points={coords.shape[0]}, "
        f"t=[{float(t_star[0]):.6f}, {float(t_star[-1]):.6f}]"
    )

    soap_windows = [
        int(d.split("_")[-1])
        for d in os.listdir(args.soap_checkpoint_base)
        if d.startswith("time_window_")
    ]
    if not soap_windows:
        raise RuntimeError(f"No SOAP windows found in {args.soap_checkpoint_base}")

    soap_max_window = args.soap_max_window or max(soap_windows)

    pirate_num_windows = args.pirate_num_windows
    soap_num_windows = args.soap_num_windows

    print(f"SOAP windows to compare: 1..{min(soap_max_window, soap_num_windows)}")
    print(f"PIRATE total windows: {pirate_num_windows}")

    results = []

    for soap_w in range(1, min(soap_max_window, soap_num_windows) + 1):
        soap_end_idx = soap_w * 2 - 1
        if soap_end_idx >= len(t_star):
            break
        t_soap_end = float(t_star[soap_end_idx])

        pirate_w = find_window_for_time(t_soap_end, t_star, pirate_num_windows)
        if pirate_w is None:
            print(f"skip SOAP window {soap_w}: target time out of PIRATE range")
            continue

        print(
            f"evaluate SOAP window {soap_w} at t={t_soap_end:.6f} "
            f"vs PIRATE window {pirate_w}"
        )

        try:
            soap_result = evaluate_at_time(
                "soap",
                args.soap_checkpoint_base,
                soap_w,
                t_soap_end,
                dns_data,
            )
            pirate_result = evaluate_at_time(
                "pirate",
                args.pirate_checkpoint_base,
                pirate_w,
                t_soap_end,
                dns_data,
            )
        except Exception as exc:
            print(f"  failed: {exc}")
            continue

        results.append(
            {
                "soap_window": soap_w,
                "pirate_window": pirate_w,
                "t": soap_result["t_actual"],
                "soap_u": soap_result["u_error"],
                "soap_v": soap_result["v_error"],
                "soap_w": soap_result["w_error"],
                "pirate_u": pirate_result["u_error"],
                "pirate_v": pirate_result["v_error"],
                "pirate_w": pirate_result["w_error"],
                "soap_step": soap_result["checkpoint_step"],
                "pirate_step": pirate_result["checkpoint_step"],
            }
        )

    print("=" * 100)
    print("比較結果")
    print("=" * 100)

    if not results:
        print("沒有可用結果。")
        return results

    print(
        f"{'Time':<10} {'SOAP W':<8} {'PIRATE W':<10} "
        f"{'SOAP u%':<10} {'PIRATE u%':<12} {'SOAP v%':<10} "
        f"{'PIRATE v%':<12} {'SOAP w%':<10} {'PIRATE w%':<12}"
    )
    print("-" * 110)

    for r in results:
        print(
            f"{r['t']:<10.4f} {r['soap_window']:<8} {r['pirate_window']:<10} "
            f"{r['soap_u']*100:<10.2f} {r['pirate_u']*100:<12.2f} "
            f"{r['soap_v']*100:<10.2f} {r['pirate_v']*100:<12.2f} "
            f"{r['soap_w']*100:<10.2f} {r['pirate_w']*100:<12.2f}"
        )

    if args.output:
        np.savez(args.output, results=np.array(results, dtype=object))
        print(f"saved: {args.output}")

    return results


def main():
    parser = argparse.ArgumentParser(description="時間對齊的 PIRATE vs SOAP 公平比較")
    parser.add_argument("--pirate-checkpoint-base", type=str, required=True)
    parser.add_argument("--soap-checkpoint-base", type=str, required=True)
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="examples/kolmogorov_flow/data/kolmogorov_dns/kolmogorov_dns_10000.npy",
    )
    parser.add_argument("--time-fraction", type=float, default=1.0)
    parser.add_argument("--pirate-num-windows", type=int, default=10)
    parser.add_argument("--soap-num-windows", type=int, default=25)
    parser.add_argument("--soap-max-window", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    generate_time_aligned_comparison(args)


if __name__ == "__main__":
    main()
