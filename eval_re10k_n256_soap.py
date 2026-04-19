"""
評估 paper_repro_soap 訓練結果（window 1-N，評估所有現有 checkpoint）

輸出：
  - eval_paper_repro_soap/l2_errors.npz          → 各窗口 L2 誤差
  - eval_paper_repro_soap/comparison_plots.png   → L2 error / KE / Enstrophy / Spectrum
  - eval_paper_repro_soap/vorticity_fields.png   → 各窗口最終時刻場圖（DNS vs Pred vs |error|）
  - eval_paper_repro_soap/summary.txt            → 文字摘要
"""

import os
import sys
import importlib.util
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── 路徑設定 ────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
EXAMPLE_DIR = os.path.join(ROOT_DIR, "examples", "kolmogorov_flow")
for p in [ROOT_DIR, EXAMPLE_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from examples.kolmogorov_flow.eval_common import (
    compute_energy_spectrum,
    infer_domain_lengths,
    resolve_window_layout_from_config,
    to_window_local_time,
)

# ── 設定 ────────────────────────────────────────────────────────────────────
CONFIG_PATH = os.path.join(EXAMPLE_DIR, "configs", "re10k_soap.py")
CKPT_ROOT = os.environ.get(
    "EVAL_CKPT_ROOT",
    os.path.join(ROOT_DIR, "re10k_n256_soap", "ckpt"),
)
OUTPUT_DIR = os.path.join(ROOT_DIR, "eval_re10k_n256_soap")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 環境設定（在 JAX import 之前） ─────────────────────────────────────────
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")


# ── Helpers ─────────────────────────────────────────────────────────────────


def load_config(config_path: str):
    spec = importlib.util.spec_from_file_location("cfg", config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_config()


def pred_chunked(pred_fn, params, t_val, x, y, chunk_size=65536):
    """分批呼叫 pred_fn 避免 4M 點一次 forward pass 導致 OOM。"""
    n = x.shape[0]
    out = []
    for i in range(0, n, chunk_size):
        out.append(pred_fn(params, t_val, x[i:i+chunk_size], y[i:i+chunk_size]))
    return np.concatenate([np.array(o) for o in out])


def discover_windows(ckpt_root: str) -> list[int]:
    dirs = [d for d in os.listdir(ckpt_root) if d.startswith("time_window_")]
    wins = []
    for d in dirs:
        try:
            wins.append(int(d.split("_")[-1]))
        except ValueError:
            pass
    wins.sort()
    return wins


def latest_step(ckpt_dir: str):
    steps = []
    for f in os.listdir(ckpt_dir):
        if f.startswith("checkpoint_"):
            try:
                steps.append(int(f.split("_")[1]))
            except (IndexError, ValueError):
                pass
    return max(steps) if steps else None


# ── 主評估 ──────────────────────────────────────────────────────────────────


def main():
    import jax.numpy as jnp
    from jaxpi.utils import restore_checkpoint
    from examples.kolmogorov_flow.utils import get_dataset
    from examples.kolmogorov_flow import models as kf_models

    print("=" * 70)
    print("  re10k_n256_soap  |  Evaluation Script")
    print("=" * 70)

    # ── 載入 config ──────────────────────────────────────────────────────────
    config = load_config(CONFIG_PATH)

    # ── 載入資料集 ────────────────────────────────────────────────────────────
    print("\n[1/4] Loading dataset ...")
    u_ref, v_ref, w_ref, t_star, coords, nu = get_dataset(
        time_fraction=config.time_fraction,
        dataset_path=config.dataset_path,
        time_range=config.get("dns_time_range"),
        time_stride=config.get("dns_time_stride", 1),
    )
    N_t = len(t_star)
    N_space = coords.shape[0]
    nx = int(np.sqrt(N_space))
    Re = int(round(1.0 / nu))

    print(f"  t_star : {N_t} steps, [{float(t_star[0]):.4f}, {float(t_star[-1]):.4f}]")
    print(f"  space  : {N_space} ({nx}×{nx}), Re={Re}, nu={nu:.6f}")
    print(f"  u_ref  : {u_ref.shape},  w_ref: {w_ref.shape}")

    # ── 窗口發現 ──────────────────────────────────────────────────────────────
    time_windows = discover_windows(CKPT_ROOT)
    if not time_windows:
        raise RuntimeError(f"No time_window_* directories found under {CKPT_ROOT}")

    layout = resolve_window_layout_from_config(t_star, config)
    num_time_steps = layout.num_time_steps
    lx, ly = infer_domain_lengths(coords)
    print(f"\n  num_time_windows (config): {config.training.num_time_windows}")
    print(f"  steps per window         : {num_time_steps}")
    print(f"  trailing time steps      : {layout.time_remainder}")
    print(f"  windows with checkpoints : {time_windows}")
    print(f"  domain length            : Lx={lx:.6f}, Ly={ly:.6f}")

    # ── 評估 ──────────────────────────────────────────────────────────────────
    print("\n[2/4] Evaluating windows ...")

    space_chunk_size = int(getattr(config.logging, "eval_space_chunk_size", 4096))
    dt = float(t_star[1] - t_star[0]) if N_t > 1 else 1.0
    chunk_sec = float(getattr(config.logging, "eval_time_chunk_seconds", 1.0))
    time_chunk_size = max(1, int(chunk_sec / dt)) if dt > 0 else 1

    records = []  # per-window summary
    ts_all, eu_all, ev_all, ew_all = [], [], [], []
    ke_ref_all, ke_pred_all, ens_ref_all, ens_pred_all = [], [], [], []
    last_w = {"k": None, "E_ref": None, "E_pred": None, "t": None}

    # For field snapshot plots
    snap_data = []  # list of dict per window (last time step)

    for win in time_windows:
        print(f"\n  === Window {win} ===")
        si = (win - 1) * num_time_steps
        ei = win * num_time_steps

        if si >= N_t or ei > N_t:
            print(f"  [SKIP] indices [{si},{ei}) out of range ({N_t})")
            continue

        t_win = t_star[si:ei]
        t_win_local = to_window_local_time(t_win)
        u_ref_win = u_ref[si:ei, :]
        v_ref_win = v_ref[si:ei, :]
        w_ref_win = w_ref[si:ei, :]

        u0 = u_ref[si, :]
        v0 = v_ref[si, :]
        w0 = w_ref[si, :]

        model = kf_models.NavierStokes(config, t_win_local, coords, u0, v0, w0, nu)

        ckpt_dir = os.path.join(CKPT_ROOT, f"time_window_{win}")
        step = latest_step(ckpt_dir)
        if step is None:
            print(f"  [SKIP] no checkpoint in {ckpt_dir}")
            continue

        print(f"  loading ckpt step={step} ...", end=" ", flush=True)
        model.state = restore_checkpoint(model.state, ckpt_dir, step=step)
        print("done")

        # ── full-window L2 error ──────────────────────────────────────────────
        print(f"  computing full-window L2 error ...", end=" ", flush=True)
        u_err, v_err, w_err = model.compute_l2_error_time_space_chunked(
            model.state.params,
            t_win_local,
            coords,
            u_ref_win,
            v_ref_win,
            w_ref_win,
            time_chunk_size=time_chunk_size,
            space_chunk_size=space_chunk_size,
        )
        u_err, v_err, w_err = float(u_err), float(v_err), float(w_err)
        print(f"u={u_err:.4f}  v={v_err:.4f}  w={w_err:.4f}")

        records.append(
            {
                "window": win,
                "step": step,
                "t_start": float(t_win[0]),
                "t_end": float(t_win[-1]),
                "u_error": u_err,
                "v_error": v_err,
                "w_error": w_err,
            }
        )

        # ── time-series metrics (sampled every ~0.1 s) ────────────────────────
        sample_dt = 0.1
        sample_idx = [0]
        for i in range(1, len(t_win)):
            if float(t_win[i]) - float(t_win[sample_idx[-1]]) >= sample_dt:
                sample_idx.append(i)
        if sample_idx[-1] != len(t_win) - 1:
            sample_idx.append(len(t_win) - 1)

        print(f"  time-series metrics ({len(sample_idx)} pts) ...", end=" ", flush=True)
        x_c, y_c = coords[:, 0], coords[:, 1]
        for idx_i in sample_idx:
            t_i_abs = float(t_win[idx_i])
            t_i_local = float(t_win_local[idx_i])
            u_p = pred_chunked(model.u_ic_pred_fn, model.state.params, t_i_local, x_c, y_c)
            v_p = pred_chunked(model.v_ic_pred_fn, model.state.params, t_i_local, x_c, y_c)
            w_p = pred_chunked(model.w_ic_pred_fn, model.state.params, t_i_local, x_c, y_c)

            ur = u_ref_win[idx_i, :]
            vr = v_ref_win[idx_i, :]
            wr = w_ref_win[idx_i, :]

            ts_all.append(t_i_abs)
            eu_all.append(float(jnp.linalg.norm(u_p - ur) / jnp.linalg.norm(ur)))
            ev_all.append(float(jnp.linalg.norm(v_p - vr) / jnp.linalg.norm(vr)))
            ew_all.append(float(jnp.linalg.norm(w_p - wr) / jnp.linalg.norm(wr)))

            ke_ref_all.append(0.5 * float(jnp.mean(ur**2 + vr**2)))
            ke_pred_all.append(0.5 * float(jnp.mean(u_p**2 + v_p**2)))
            ens_ref_all.append(0.5 * float(jnp.mean(wr**2)))
            ens_pred_all.append(0.5 * float(jnp.mean(w_p**2)))
        print("done")

        # ── last time-step field + spectrum (only last available window) ───────
        t_last = float(t_win[-1])
        t_last_local = float(t_win_local[-1])
        u_p_last = pred_chunked(model.u_ic_pred_fn, model.state.params, t_last_local, x_c, y_c)
        v_p_last = pred_chunked(model.v_ic_pred_fn, model.state.params, t_last_local, x_c, y_c)
        w_p_last = pred_chunked(model.w_ic_pred_fn, model.state.params, t_last_local, x_c, y_c)
        w_r_last = np.array(w_ref_win[-1, :])
        u_r_last = np.array(u_ref_win[-1, :])
        v_r_last = np.array(v_ref_win[-1, :])

        snap_data.append(
            {
                "window": win,
                "t": t_last,
                "step": step,
                "u_ref": u_r_last,
                "v_ref": v_r_last,
                "w_ref": w_r_last,
                "u_pred": u_p_last,
                "v_pred": v_p_last,
                "w_pred": w_p_last,
            }
        )

        # energy spectrum at last step
        u2d = u_p_last.reshape(nx, nx)
        v2d = v_p_last.reshape(nx, nx)
        ur2d = u_r_last.reshape(nx, nx)
        vr2d = v_r_last.reshape(nx, nx)
        k, E_pred = compute_energy_spectrum(u2d, v2d, lx=lx, ly=ly)
        _, E_ref = compute_energy_spectrum(ur2d, vr2d, lx=lx, ly=ly)
        last_w = {"k": k, "E_ref": E_ref, "E_pred": E_pred, "t": t_last}

    if not records:
        print("\n[ERROR] No windows evaluated. Check checkpoint path.")
        return

    # ── 儲存 L2 誤差數據 ──────────────────────────────────────────────────────
    print("\n[3/4] Saving numerical results ...")
    np.savez(
        os.path.join(OUTPUT_DIR, "l2_errors.npz"),
        windows=np.array([r["window"] for r in records]),
        steps=np.array([r["step"] for r in records]),
        t_start=np.array([r["t_start"] for r in records]),
        t_end=np.array([r["t_end"] for r in records]),
        u_errors=np.array([r["u_error"] for r in records]),
        v_errors=np.array([r["v_error"] for r in records]),
        w_errors=np.array([r["w_error"] for r in records]),
        ts_all=np.array(ts_all),
        eu_all=np.array(eu_all),
        ev_all=np.array(ev_all),
        ew_all=np.array(ew_all),
        ke_ref=np.array(ke_ref_all),
        ke_pred=np.array(ke_pred_all),
        ens_ref=np.array(ens_ref_all),
        ens_pred=np.array(ens_pred_all),
    )
    print("  -> l2_errors.npz saved")

    # ── 摘要文字 ──────────────────────────────────────────────────────────────
    u_arr = np.array([r["u_error"] for r in records])
    v_arr = np.array([r["v_error"] for r in records])
    w_arr = np.array([r["w_error"] for r in records])

    summary_lines = [
        "=" * 70,
        "  re10k_n256_soap  |  Evaluation Summary",
        "=" * 70,
        f"  config  : {CONFIG_PATH}",
        f"  ckpt    : {CKPT_ROOT}",
        f"  Re      : {Re}",
        f"  windows : {[r['window'] for r in records]}",
        "",
        f"  {'Win':>4}  {'step':>8}  {'t_end':>7}  {'u_err':>9}  {'v_err':>9}  {'w_err':>9}",
        "  " + "-" * 60,
    ]
    for r in records:
        summary_lines.append(
            f"  {r['window']:>4}  {r['step']:>8}  {r['t_end']:>7.4f}  "
            f"{r['u_error']:>9.6f}  {r['v_error']:>9.6f}  {r['w_error']:>9.6f}"
        )
    summary_lines += [
        "  " + "-" * 60,
        f"  {'mean':>4}  {'':>8}  {'':>7}  {u_arr.mean():>9.6f}  {v_arr.mean():>9.6f}  {w_arr.mean():>9.6f}",
        f"  {'max':>4}  {'':>8}  {'':>7}  {u_arr.max():>9.6f}  {v_arr.max():>9.6f}  {w_arr.max():>9.6f}",
        "",
    ]
    summary_text = "\n".join(summary_lines)
    print(summary_text)

    summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text + "\n")
    print(f"  -> summary.txt saved")

    # ── 可視化 ────────────────────────────────────────────────────────────────
    print("\n[4/4] Generating plots ...")
    ts_np = np.array(ts_all)
    eu_np = np.array(eu_all)
    ev_np = np.array(ev_all)
    ew_np = np.array(ew_all)
    ke_r_np = np.array(ke_ref_all)
    ke_p_np = np.array(ke_pred_all)
    en_r_np = np.array(ens_ref_all)
    en_p_np = np.array(ens_pred_all)

    # ── Plot 1: L2 error / KE / Enstrophy / Energy spectrum ──────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        f"re10k_n256_soap  |  Re={Re}  |  Windows {[r['window'] for r in records]}",
        fontsize=13,
        fontweight="bold",
    )

    ax = axes[0, 0]
    ax.semilogy(ts_np, eu_np, "b-", lw=1.8, label="u")
    ax.semilogy(ts_np, ev_np, "r--", lw=1.8, label="v")
    ax.semilogy(ts_np, ew_np, "g:", lw=1.8, label="ω")
    # mark window boundaries
    for r in records[:-1]:
        ax.axvline(r["t_end"], color="gray", ls=":", lw=0.8, alpha=0.6)
    ax.set_xlabel("t")
    ax.set_ylabel("Rel. L2 error")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Relative L2 Error over Time")

    ax = axes[0, 1]
    ax.plot(ts_np, ke_r_np, "b-", lw=1.8, label="DNS")
    ax.plot(ts_np, ke_p_np, "r--", lw=1.8, label="PINN")
    for r in records[:-1]:
        ax.axvline(r["t_end"], color="gray", ls=":", lw=0.8, alpha=0.6)
    ax.set_xlabel("t")
    ax.set_ylabel("Kinetic Energy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Kinetic Energy")

    ax = axes[1, 0]
    ax.plot(ts_np, en_r_np, "b-", lw=1.8, label="DNS")
    ax.plot(ts_np, en_p_np, "r--", lw=1.8, label="PINN")
    for r in records[:-1]:
        ax.axvline(r["t_end"], color="gray", ls=":", lw=0.8, alpha=0.6)
    ax.set_xlabel("t")
    ax.set_ylabel("Enstrophy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Enstrophy")

    ax = axes[1, 1]
    if last_w["k"] is not None:
        k, E_ref, E_pred = last_w["k"], last_w["E_ref"], last_w["E_pred"]
        mask = (k > 0) & (E_ref > 1e-14) & (E_pred > 1e-14)
        k_p, Er_p, Ep_p = k[mask], E_ref[mask], E_pred[mask]
        ax.loglog(k_p, Er_p, "b-", lw=2, label="DNS")
        ax.loglog(k_p, Ep_p, "r--", lw=2, label="PINN")
        if len(k_p) > 8:
            mi = len(k_p) // 4
            k_ref_line = k_p[mi : 3 * len(k_p) // 4]
            ax.loglog(
                k_ref_line,
                Er_p[mi] * (k_ref_line / k_p[mi]) ** (-3),
                "k:",
                lw=1.5,
                label=r"$k^{-3}$",
                alpha=0.7,
            )
        ax.set_xlabel("Wavenumber k")
        ax.set_ylabel("E(k)")
        ax.set_title(f"Energy Spectrum (t={last_w['t']:.4f})")
        ax.legend()
        ax.grid(True, alpha=0.3, which="both")
    else:
        ax.text(0.5, 0.5, "No spectrum data", ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    comp_path = os.path.join(OUTPUT_DIR, "comparison_plots.png")
    fig.savefig(comp_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> comparison_plots.png saved")

    # ── Plot 2: 渦度場快照（每個 window 末 time step）─────────────────────────
    n_snaps = len(snap_data)
    fig2, axes2 = plt.subplots(n_snaps, 3, figsize=(12, 4 * n_snaps))
    if n_snaps == 1:
        axes2 = axes2[np.newaxis, :]

    fig2.suptitle(
        f"Vorticity (ω) Snapshots  |  Re={Re}  |  paper_repro_soap", fontsize=13, fontweight="bold"
    )

    for row, snap in enumerate(snap_data):
        w_r = snap["w_ref"].reshape(nx, nx)
        w_p = snap["w_pred"].reshape(nx, nx)
        w_e = np.abs(w_r - w_p)

        vmax = max(np.abs(w_r).max(), np.abs(w_p).max()) * 1.0
        vmin = -vmax

        ax0, ax1, ax2 = axes2[row, 0], axes2[row, 1], axes2[row, 2]

        im0 = ax0.imshow(w_r.T, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="equal")
        ax0.set_title(f"DNS  (t={snap['t']:.4f})")
        plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

        im1 = ax1.imshow(w_p.T, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="equal")
        ax1.set_title(f"PINN (win={snap['window']}, step={snap['step']})")
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        im2 = ax2.imshow(w_e.T, origin="lower", cmap="hot", vmin=0, vmax=w_e.max(), aspect="equal")
        ax2.set_title("|error|")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        for ax in (ax0, ax1, ax2):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel(f"Win {snap['window']}", fontsize=9)

    plt.tight_layout()
    field_path = os.path.join(OUTPUT_DIR, "vorticity_fields.png")
    fig2.savefig(field_path, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"  -> vorticity_fields.png saved")

    print("\n" + "=" * 70)
    print("  Evaluation complete.")
    print(f"  Output: {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
