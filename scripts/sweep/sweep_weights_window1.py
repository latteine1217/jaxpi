"""
What:
    Optuna sweep：在 window-1 上搜尋最早穩定達標的 data loss weight。
Why:
    先前的 scoring 用單次 threshold crossing，容易被 loss 震盪誤導。
    現在改成比較最早連續 `k` 次滿足 `max(ru_loss, rv_loss, rc_loss) < threshold`
    的步數，讓 objective 直接回答「哪個權重最快穩定收斂」。
    並用 PatientPruner 包住 MedianPruner，避免 trial 因短期停滯就被過早剪掉。
    目標仍是搜尋 u_data / v_data weight 組合，其餘參數固定在預設值。
    使用 K=100 QR-pivot sensor（Re=1e6, N=512）作為 data constraint。
    每個 trial 只跑 window-1（ablation config），
    保留原始短期動態，同時降低誤殺風險。
    一旦 trial 提早穩定達標，就在當下保存唯一保留的 ckpt，並立即結束該 trial。

Usage:
    uv run python scripts/sweep/sweep_weights_window1.py \
        --n-trials 60 \
        --max-steps 100000 \
        --threshold 5e-5 \
        --stable-reports 3 \
        --study-name kf_w1_data_weight_sweep \
        --storage sqlite:///sweep_w1_data.db
"""

import argparse
import copy
import sys
import os
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
sys.path.insert(0, str(_REPO_ROOT))
# train.py 用 `import models`（bare import），需將 kolmogorov_flow 目錄加入 path
sys.path.insert(0, str(_REPO_ROOT / "examples/kolmogorov_flow"))

import importlib.util
import numpy as np
import optuna
from optuna.pruners import MedianPruner, PatientPruner

import jax
import wandb

# ── helpers ───────────────────────────────────────────────────────────────────

DATA_WEIGHT_MIN = 5.0
DATA_WEIGHT_MAX = 80.0
DEFAULT_THRESHOLD = 5e-5
DEFAULT_STABLE_REPORTS = 3
DEFAULT_PATIENCE = 25
DEFAULT_MIN_DELTA = 1e-5
DEFAULT_N_TRIALS = 60
DEFAULT_MAX_STEPS = 100000


def suggest_data_weight(trial: optuna.Trial) -> float:
    """
    What:
        定義 `data_weight` 的 Optuna 搜尋空間。
    Why:
        根據目前 evidence，較有訊號的區域集中在中等到偏高權重；
        先聚焦 `5~80`，保留對 `60` 以上區間的探索，同時避免把 trial
        浪費在更高的外圍區間。
    """

    return trial.suggest_float("data_weight", DATA_WEIGHT_MIN, DATA_WEIGHT_MAX, log=True)


def build_pruner():
    """
    What:
        建立 sweep 用的 Optuna pruner。
    Why:
        `step_callback` 只在每次 log 時回報一次，因此 `patience` 以 report 次數計。
        目前 `log_every_steps=100`，`patience=25` 約對應 2500 個 training steps 的觀察窗口。
    """

    return PatientPruner(
        MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5000,
            interval_steps=500,
        ),
        patience=DEFAULT_PATIENCE,
        min_delta=DEFAULT_MIN_DELTA,
    )

def _load_config(config_path: str):
    spec = importlib.util.spec_from_file_location("_cfg", config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_config()


def _build_config(trial: optuna.Trial, base_config_path: str, max_steps: int):
    """
    What: 從 trial 參數組出訓練 config。
    Why: 透過 ml_collections 的 unlock，原地修改不可變欄位。
         只 sweep u_data/v_data；u_ic/v_ic 固定 100，causal_tol/momentum 固定預設值。
    """
    config = _load_config(base_config_path)

    with config.unlocked():
        # data loss weights（對稱流場，u/v 綁定）
        data_w = suggest_data_weight(trial)
        config.weighting.init_weights.u_data = data_w
        config.weighting.init_weights.v_data = data_w
        # IC 固定，causal_tol / momentum 維持 config 預設（1.0 / 0.9）

        config.training.max_steps = max_steps

        # 只保留最後一個 checkpoint；若 trial 提早穩定達標，就保存停下來那一點。
        config.saving.save_every_steps = max_steps
        config.saving.num_keep_ckpts = 1
        config.saving.ckpt_dir = str(_REPO_ROOT / "sweep_ckpts" / f"trial_{trial.number:04d}")
        config.saving.overwrite = True

        # 關閉 wandb（train_one_window 內的 wandb.log 已加 `if wandb.run` guard）
        config.wandb.log = False

    return config


# ── objective ─────────────────────────────────────────────────────────────────

def make_objective(
    base_config_path: str,
    max_steps: int,
    threshold: float,
    stable_reports: int,
    data_root: str,
    sensor_json: str,
):
    """
    What: 工廠函式，回傳 Optuna objective。
    Why: 閉包捕捉固定參數，讓 objective 簽名符合 Optuna 規範。
    """

    def objective(trial: optuna.Trial) -> float:
        config = _build_config(trial, base_config_path, max_steps)

        # ── 載入資料 ──────────────────────────────────────────────────────
        from examples.kolmogorov_flow.utils import get_dataset
        from examples.kolmogorov_flow import models as kf_models
        from examples.kolmogorov_flow.train import (
            JaxSampler, HostSampler,
            _build_ic_sampler, _build_uniform_sampler, _build_sensor_sampler,
            _adjust_batch_sizes_for_causal,
            train_one_window,
        )
        from jaxpi.dataio.sensors_format import attach_dns_values, load_sensor_json
        from jax import random

        dataset_path = config.get(
            "dataset_path",
            "examples/kolmogorov_flow/data/kolmogorov_dns/kolmogorov_dns_10000.npy",
        )
        u_ref, v_ref, w_ref, t_star, coords, nu = get_dataset(
            time_fraction=config.time_fraction,
            dataset_path=str(_REPO_ROOT / dataset_path) if not os.path.isabs(dataset_path) else dataset_path,
            time_range=config.get("dns_time_range"),
            time_stride=int(config.get("dns_time_stride", 1)),
        )

        # Window-0 時間切片（遵循 train_and_evaluate 的相同邏輯）
        num_time_steps = len(t_star) // config.training.num_time_windows
        t_win  = t_star[:num_time_steps]
        u_win  = u_ref[0:num_time_steps, :]
        v_win  = v_ref[0:num_time_steps, :]
        w_win  = w_ref[0:num_time_steps, :]
        u0_win = u_ref[0, :]
        v0_win = v_ref[0, :]
        w0_win = w_ref[0, :]

        # 網格邊界
        dt = t_win[1] - t_win[0]
        dom = np.array([[t_win[0], t_win[-1] + 1.1 * dt], [0.0, 1.0], [0.0, 1.0]])

        # 強制單卡 parallel state（sweep 不得使用多 GPU，否則每個 trial 會 OOM）
        parallel_state = {
            "num_devices": 1,
            "mesh": None,
            "data_sharding": None,
            "replicated_sharding": None,
        }

        # 調整 batch size（causal weighting 整除性）
        config = _adjust_batch_sizes_for_causal(config, num_devices=1)
        global_batch = config.training.batch_size_per_device  # 單卡，per_device == global

        # ── 建立 model + samplers ─────────────────────────────────────────
        model = kf_models.NavierStokes(
            config, t_win, coords, u0_win, v0_win, w0_win, nu, replicate_state=False
        )

        ics_fn  = _build_ic_sampler(coords, u0_win, v0_win, w0_win, global_batch * 2,
                                    rng_seed=config.seed + 1)
        res_fn  = _build_uniform_sampler(np.asarray(dom), global_batch)
        samplers = {
            "ics": HostSampler(ics_fn),
            "res": JaxSampler(res_fn, random.PRNGKey(config.seed + 2)),
        }

        # ── K=100 QR-pivot sensor sampler ─────────────────────────────────
        sensor = load_sensor_json(sensor_json)
        # dns_values_npz 存在 metadata；路徑可能是相對路徑（相對於 repo root）
        npz_rel = sensor.metadata.get("dns_values_npz", "")
        npz_path = npz_rel if os.path.isabs(npz_rel) else str(_REPO_ROOT / npz_rel)
        sensor = attach_dns_values(sensor, npz_path)
        dns_v = sensor.dns_values
        # 全程使用 numpy，避免在 CPU 預處理觸發 JAX concrete value 限制
        s_time   = np.array(dns_v["time"])
        s_coords = np.array(sensor.coords)
        s_u      = np.array(dns_v["u"])
        s_v      = np.array(dns_v["v"])
        s_w      = np.array(dns_v.get("omega", np.zeros_like(dns_v["u"])))

        # 篩選落在 window-0 時間範圍內的 sensor 點
        time_mask = (s_time >= t_win[0]) & (s_time <= t_win[-1])
        if time_mask.any():
            time_window = s_time[time_mask]
            # 對齊 train.py 的 sensor_time_shift（預設 True）：時間從 window 起點算
            if config.get("sensor_time_shift", True):
                time_window = time_window - t_win[0]
            sensor_fn = _build_sensor_sampler(
                time_window,
                s_coords,
                s_u[:, time_mask],
                s_v[:, time_mask],
                s_w[:, time_mask],
                global_batch,
                rng_seed=config.seed + 3,
            )
            samplers["data"] = HostSampler(sensor_fn)

        # ── scoring / pruner callback ─────────────────────────────────────
        stable_steps: list[int] = []

        def step_callback(step: int, log_dict: dict) -> str | None:
            ru = float(log_dict.get("ru_loss", np.inf))
            rv = float(log_dict.get("rv_loss", np.inf))
            rc = float(log_dict.get("rc_loss", np.inf))
            worst = max(ru, rv, rc)

            # 向 Optuna 回報 raw worst value（供 PatientPruner + MedianPruner 判斷）
            trial.report(float(worst), step)

            if worst < threshold:
                stable_steps.append(step)
            else:
                stable_steps.clear()

            if len(stable_steps) >= stable_reports:
                # 以首次穩定達標的步數作為 objective，越小越好。
                raise _StableConvergenceReached(stable_steps[0])

            if trial.should_prune():
                raise optuna.TrialPruned()

            return None

        # ── 執行訓練 ─────────────────────────────────────────────────────
        try:
            train_one_window(
                config=config,
                workdir="/tmp/sweep_dummy",
                model=model,
                samplers=samplers,
                t=t_win,
                coords=coords,
                u_ref=u_win,
                v_ref=v_win,
                w_ref=w_win,
                idx=0,
                parallel_state=parallel_state,
                step_callback=step_callback,
            )
        except _StableConvergenceReached as stable_hit:
            from jaxpi.utils import save_checkpoint

            ckpt_path = os.path.join(config.saving.ckpt_dir, "time_window_1")
            save_checkpoint(
                model.state,
                ckpt_path,
                keep=config.saving.num_keep_ckpts,
                overwrite=config.saving.overwrite,
            )
            return float(stable_hit.first_step)
        except optuna.TrialPruned:
            raise
        except FloatingPointError:
            # NaN/Inf → 給出最差分數
            return float("inf")

        # 未在 max_steps 內穩定達標，回傳上限步數。
        return float(max_steps)

    return objective


class _StableConvergenceReached(RuntimeError):
    """Internal control-flow exception used to short-circuit finished trials."""

    def __init__(self, first_step: int):
        super().__init__(f"stable convergence reached at step {first_step}")
        self.first_step = first_step


# ── main ──────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    """
    What:
        建立 sweep CLI parser。
    Why:
        讓預設 threshold 與其他 CLI 行為可被測試直接驗證。
    """

    parser = argparse.ArgumentParser(description="Optuna weight sweep for Kolmogorov window-1")
    parser.add_argument("--n-trials",    type=int,   default=DEFAULT_N_TRIALS)
    parser.add_argument("--max-steps",   type=int,   default=DEFAULT_MAX_STEPS,
                        help="每個 trial 的最大訓練步數")
    parser.add_argument("--threshold",   type=float, default=DEFAULT_THRESHOLD,
                        help="stable convergence threshold for max(ru, rv, rc)")
    parser.add_argument("--stable-reports", type=int, default=DEFAULT_STABLE_REPORTS,
                        help="需要連續幾次 report 低於 threshold 才算穩定達標")
    parser.add_argument("--study-name",  type=str,   default="kf_w1_data_weight_sweep")
    parser.add_argument("--storage",     type=str,   default=None,
                        help="Optuna storage URI，e.g. sqlite:///sweep.db")
    parser.add_argument(
        "--config",
        type=str,
        default=str(_REPO_ROOT / "examples/kolmogorov_flow/configs/paper_repro_soap_window1_ablation.py"),
        help="Base config 路徑",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(_REPO_ROOT / "examples/kolmogorov_flow/data/kolmogorov_dns"),
        help="DNS 資料根目錄",
    )
    parser.add_argument(
        "--sensor-json",
        type=str,
        default="examples/kolmogorov_flow/data/kolmogorov_sensors/re1000000/sensors_qrpivot_K100_N512_t0-5.json",
        help="K=100 QR-pivot sensor JSON 路徑（相對於 repo root 或絕對路徑）",
    )

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.stable_reports <= 0:
        raise ValueError(f"--stable-reports must be positive, got {args.stable_reports}")

    # sensor_json 解析為絕對路徑
    sensor_json = args.sensor_json if os.path.isabs(args.sensor_json) else str(_REPO_ROOT / args.sensor_json)

    # 強制關閉 wandb（覆蓋 Slurm common.sh 可能設定的 offline）
    os.environ["WANDB_MODE"] = "disabled"

    pruner = build_pruner()

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",
        pruner=pruner,
        load_if_exists=True,  # 支援斷點續跑
    )

    print(f"=== Sweep: {args.study_name} ===")
    print("target   : minimize earliest stable crossing step of max(ru,rv,rc)")
    print(f"max_steps: {args.max_steps}")
    print(f"n_trials : {args.n_trials}")
    print(f"range    : data_weight in [{DATA_WEIGHT_MIN}, {DATA_WEIGHT_MAX}] (log scale)")
    print(f"threshold: {args.threshold}")
    print(f"stable_k : {args.stable_reports}")
    print(f"pruner   : Patient(Median), patience={DEFAULT_PATIENCE}, min_delta={DEFAULT_MIN_DELTA}")
    print(f"config   : {args.config}")
    print(f"sensor   : {sensor_json}")
    print("score    : earliest step with k consecutive threshold hits")

    objective = make_objective(
        base_config_path=args.config,
        max_steps=args.max_steps,
        threshold=args.threshold,
        stable_reports=args.stable_reports,
        data_root=args.data_root,
        sensor_json=sensor_json,
    )

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    print("\n=== Best Trial ===")
    best = study.best_trial
    print(f"  earliest_stable_step : {best.value}")
    print(f"  params:")
    for k, v in best.params.items():
        print(f"    {k} = {v:.4g}")

    # 儲存摘要 CSV（需要 pandas，失敗時跳過）
    try:
        df = study.trials_dataframe()
        out_csv = f"{args.study_name}_results.csv"
        df.to_csv(out_csv, index=False)
        print(f"\nresults saved to {out_csv}")
    except ImportError:
        print("\n(pandas not installed, skipping CSV export)")


if __name__ == "__main__":
    main()
