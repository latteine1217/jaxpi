import time
import os
from contextlib import nullcontext

from absl import logging

import jax
from jax import random, jit
import jax.numpy as jnp
from jax.tree_util import tree_map
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as P

import numpy as np

import ml_collections
import wandb

from jaxpi.logging import Logger
from jaxpi.utils import init_parallel, save_checkpoint, restore_checkpoint
from jaxpi.models import _create_train_state

import models
from utils import get_dataset


class JaxSampler:
    def __init__(self, sample_fn, rng_key=random.PRNGKey(1234)):
        self.sample_fn = sample_fn
        self.key = rng_key

    def sample(self):
        self.key, subkey = random.split(self.key)
        return self.sample_fn(subkey)


class HostSampler:
    def __init__(self, sample_fn):
        self.sample_fn = sample_fn

    def sample(self):
        return self.sample_fn()


def _build_uniform_sampler(dom, batch_size):
    dom = jnp.asarray(dom)

    @jit
    def _sample(key):
        return random.uniform(
            key,
            shape=(batch_size, dom.shape[0]),
            minval=dom[:, 0],
            maxval=dom[:, 1],
        )

    return _sample


def _build_ic_sampler(coords, u, v, w, batch_size, rng_seed):
    coords = np.asarray(coords)
    u = np.asarray(u)
    v = np.asarray(v)
    w = np.asarray(w)

    rng = np.random.default_rng(rng_seed)

    def _sample():
        idx = rng.integers(0, coords.shape[0], size=batch_size)
        coords_batch = coords[idx, :]
        u_batch = u[idx]
        v_batch = v[idx]
        w_batch = w[idx]
        return coords_batch, u_batch, v_batch, w_batch

    return _sample


def _build_sensor_sampler(
    time_values,
    coords,
    u_values,
    v_values,
    w_values,
    batch_size,
    rng_seed,
    sampling_mode="random",
    pad_to_multiple=1,
):
    """
    What:
        建立 sensor data sampler，支援隨機抽樣或固定全覆蓋。

    Why:
        `random` 模式只保證每步看到部分 `(sensor, time)` 點，對 exact sensor fit 夠用，
        但不足以保證每一步都覆蓋所有 sensor constraints。
        `all_points` 模式會在每一步固定返回整個 `(sensor × time)` 笛卡兒積，
        讓 sensor loss 真正覆蓋當前 window 的全部 sensor points。
    """
    time_values = np.asarray(time_values)
    coords = np.asarray(coords)
    u_values = np.asarray(u_values)
    v_values = np.asarray(v_values)
    w_values = np.asarray(w_values)

    rng = np.random.default_rng(rng_seed)

    if sampling_mode == "all_points":
        if time_values.ndim != 1:
            raise ValueError(f"time_values 必須是一維，實際 shape={time_values.shape}")
        if coords.ndim != 2:
            raise ValueError(f"coords 必須是二維，實際 shape={coords.shape}")

        num_sensors = coords.shape[0]
        num_time = time_values.shape[0]
        if num_sensors == 0 or num_time == 0:
            raise ValueError("all_points 模式需要非空的 sensor/time 維度")

        time_grid = np.broadcast_to(time_values[None, :], (num_sensors, num_time)).reshape(-1)
        coords_grid = np.repeat(coords, num_time, axis=0)
        u_grid = u_values.reshape(-1)
        v_grid = v_values.reshape(-1)
        w_grid = w_values.reshape(-1)

        if pad_to_multiple > 1:
            remainder = time_grid.shape[0] % pad_to_multiple
            if remainder != 0:
                pad_count = pad_to_multiple - remainder
                time_grid = np.concatenate([time_grid, time_grid[:pad_count]], axis=0)
                coords_grid = np.concatenate([coords_grid, coords_grid[:pad_count]], axis=0)
                u_grid = np.concatenate([u_grid, u_grid[:pad_count]], axis=0)
                v_grid = np.concatenate([v_grid, v_grid[:pad_count]], axis=0)
                w_grid = np.concatenate([w_grid, w_grid[:pad_count]], axis=0)

        def _sample_all_points():
            return time_grid, coords_grid, u_grid, v_grid, w_grid

        return _sample_all_points

    if sampling_mode != "random":
        raise ValueError(f"未知 sensor sampling_mode={sampling_mode}")

    def _sample():
        time_idx = rng.integers(0, time_values.shape[0], size=batch_size)
        sensor_idx = rng.integers(0, coords.shape[0], size=batch_size)

        t_batch = time_values[time_idx]
        coords_batch = coords[sensor_idx, :]
        u_batch = u_values[sensor_idx, time_idx]
        v_batch = v_values[sensor_idx, time_idx]
        w_batch = w_values[sensor_idx, time_idx]

        return t_batch, coords_batch, u_batch, v_batch, w_batch

    return _sample


def train_one_window(
    config, workdir, model, samplers, t, coords, u_ref, v_ref, w_ref, idx, parallel_state,
    step_callback=None,
):
    """
    What: 訓練單一 time window。
    Why:  step_callback 供 Optuna sweep 早停使用：每次 log 時呼叫
          step_callback(step, log_dict)，回傳 "stop" 則提前結束訓練。
          預設為 None，不影響既有行為。
    """
    step_offset = idx * config.training.max_steps

    # Logger
    logger = Logger()

    # Initialize evaluator
    evaluator = models.NavierStokesEvaluator(config, model)

    eval_time_samples = getattr(config.logging, "eval_time_samples", None)
    eval_space_samples = getattr(config.logging, "eval_space_samples", None)
    rng = np.random.default_rng(config.seed + idx)

    def _sample_eval_data(t_eval, coords_eval, u_ref_eval, v_ref_eval, w_ref_eval):
        # 優化：設定合理的預設採樣數量，避免完整資料集載入
        # 原始邏輯：當未設定時返回完整資料（記憶體密集）
        # 新邏輯：自動應用合理採樣策略
        time_size = len(t_eval)
        space_size = coords_eval.shape[0]

        # 預設採樣策略：最多 100 時間點 × 4096 空間點
        # 這對於 logging 已經足夠準確，且大幅減少記憶體開銷
        default_time_samples = min(100, time_size)
        default_space_samples = min(4096, space_size)

        time_keep = min(time_size, eval_time_samples or default_time_samples)
        space_keep = min(space_size, eval_space_samples or default_space_samples)

        # 如果採樣大小等於原始大小，直接返回（避免不必要的複製）
        if time_keep == time_size and space_keep == space_size:
            return t_eval, coords_eval, u_ref_eval, v_ref_eval, w_ref_eval

        time_idx = rng.choice(time_size, size=time_keep, replace=False)
        space_idx = rng.choice(space_size, size=space_keep, replace=False)

        # 優化：使用 jnp.take 的高級索引，減少中間複製
        t_sample = jnp.take(t_eval, time_idx, axis=0)
        coords_sample = jnp.take(coords_eval, space_idx, axis=0)

        # 優化：直接在兩個維度上取子集，避免產生完整中間陣列
        u_sample = jnp.take(jnp.take(u_ref_eval, time_idx, axis=0), space_idx, axis=1)
        v_sample = jnp.take(jnp.take(v_ref_eval, time_idx, axis=0), space_idx, axis=1)
        w_sample = jnp.take(jnp.take(w_ref_eval, time_idx, axis=0), space_idx, axis=1)

        return t_sample, coords_sample, u_sample, v_sample, w_sample

    num_devices = parallel_state["num_devices"]
    mesh = parallel_state["mesh"]
    data_sharding = parallel_state["data_sharding"]
    replicated_sharding = parallel_state["replicated_sharding"]
    weighting_start_step = int(config.weighting.get("start_step", 0))

    def _split_and_put_batch(batch):
        """
        優化：合併 batch 分片與設備傳輸，減少一次記憶體複製

        原始流程：
        1. _split_batch: host 上 reshape (複製)
        2. _device_put: host → device (複製)
        總計：2 次完整 batch 複製

        優化流程：
        1. 直接 reshape + device_put (單次複製)
        總計：1 次完整 batch 複製

        記憶體節省：~50% (對於大 batch)
        """
        if num_devices <= 1:
            return batch

        def _reshape_and_put(x):
            if x.shape[0] % num_devices != 0:
                raise ValueError("batch_size 必須能被 num_devices 整除")
            local_size = x.shape[0] // num_devices
            # 直接 reshape + device_put，避免中間 host 複製
            reshaped = x.reshape((num_devices, local_size) + x.shape[1:])
            return jax.device_put(reshaped, data_sharding)

        return tree_map(_reshape_and_put, batch)

    def _to_host_state(state):
        if num_devices <= 1:
            return state
        return jax.device_get(state)

    def _to_host_batch(batch):
        """
        將多 GPU 分片的 batch 轉回 host 並展平為原始形狀

        多 GPU 時：(num_devices, local_size, ...) -> (global_size, ...)
        單 GPU 時：(global_size, ...) -> (global_size, ...)
        """
        if num_devices <= 1:
            return batch

        # 先用 device_get 取回 host
        batch_host = jax.device_get(batch)

        # 展平分片的維度：將第一維 (num_devices, local_size) 合併為 (global_size,)
        def _flatten_sharded_dim(x):
            if x.ndim >= 2 and x.shape[0] == num_devices:
                # 形狀：(num_devices, local_size, ...) -> (global_size, ...)
                return x.reshape(-1, *x.shape[2:])
            return x

        return tree_map(_flatten_sharded_dim, batch_host)

    def _replicated_state(state):
        if num_devices <= 1:
            return state
        return jax.device_put(state, replicated_sharding)

    if num_devices > 1:
        model.state = _replicated_state(model.state)

    def _first_replica(tree):
        if num_devices <= 1:
            return tree

        def _take_leaf(x):
            if hasattr(x, "addressable_data"):
                return x.addressable_data(0)
            if hasattr(x, "addressable_shards") and x.addressable_shards:
                return x.addressable_shards[0].data
            if hasattr(x, "shape") and x.shape and x.shape[0] == num_devices:
                return x[0]
            return x

        return jax.tree_util.tree_map(_take_leaf, tree)

    def _jit_step(state, batch):
        grads = jax.grad(model.loss)(state.params, state.weights, batch)

        def _leaf_is_finite(x):
            return jnp.all(jnp.isfinite(x))

        grads_finite = jax.tree_util.tree_reduce(
            lambda acc, x: jnp.logical_and(acc, _leaf_is_finite(x)),
            grads,
            initializer=jnp.array(True),
        )
        safe_grads = tree_map(lambda g: jnp.where(jnp.isfinite(g), g, jnp.zeros_like(g)), grads)
        next_state = state.apply_gradients(grads=safe_grads)
        return jax.lax.cond(grads_finite, lambda _: next_state, lambda _: state, operand=None)

    if num_devices > 1:
        in_shardings = (
            replicated_sharding,
            {
                "ics": data_sharding,
                "res": data_sharding,
                "data": data_sharding,
            },
        )
        if "data" not in samplers:
            in_shardings = (
                replicated_sharding,
                {
                    "ics": data_sharding,
                    "res": data_sharding,
                },
            )
        out_shardings = replicated_sharding
        # donate_argnums=(0,) 允許 XLA 重用 state buffer，省一次 state 大小的 HBM 配置；
        # 純記憶體別名優化，計算結果 bit-identical。安全前提：呼叫端
        # `model.state = jit_step(model.state, ...)` 立即覆蓋舊 reference（已符合）。
        jit_step = jit(
            _jit_step,
            in_shardings=in_shardings,
            out_shardings=out_shardings,
            donate_argnums=(0,),
        )

        def _jit_update_weights(state, batch):
            weights = model.compute_weights(state.params, batch)
            return state.apply_weights(weights=weights)

        jit_update_weights = jit(
            _jit_update_weights,
            in_shardings=in_shardings,
            out_shardings=out_shardings,
            donate_argnums=(0,),
        )
    else:
        # Why: 單卡模式必須顯式 jit，否則 SOAP lax.cond 兩個 branch 無法 operator-fuse，
        #      導致第一步 XLA 編譯時間爆增（數小時）。Multi-device 路徑已有 jit wrapper。
        # donate_argnums=(0,) 同樣套用，回收 state buffer，計算結果不變。
        jit_step = jit(_jit_step, donate_argnums=(0,))
        jit_update_weights = None  # 單卡 weight update 走 line 383-384 的 eager 路徑

    if num_devices > 1:
        mesh_context = mesh
    else:
        mesh_context = nullcontext()

    # jit warm up
    print("Waiting for JIT...")
    for step in range(config.training.max_steps):
        start_time = time.time()

        # Sample mini-batch
        batch = {}
        for key, sampler in samplers.items():
            batch[key] = sampler.sample()

        # 優化：合併 split + device_put，減少一次記憶體複製
        batch = _split_and_put_batch(batch)
        with mesh_context:
            model.state = jit_step(model.state, batch)

        # Update weights if necessary
        if config.weighting.scheme in ["grad_norm", "ntk"]:
            should_update_weights = (
                step >= weighting_start_step
                and (step - weighting_start_step) % config.weighting.update_every_steps == 0
            )
            if should_update_weights:
                if num_devices > 1 and jit_update_weights is not None:
                    with mesh_context:
                        model.state = jit_update_weights(model.state, batch)
                else:
                    weights = model.compute_weights(model.state.params, batch)
                    model.state = model.state.apply_weights(weights=weights)

        # Log training metrics, only use host 0 to record results
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                # 優化：在 GPU 上計算 metrics，只傳輸標量結果
                # 獲取第一個 replica 的 state（多 GPU 時）
                if num_devices > 1:
                    state_for_eval = _first_replica(model.state)
                    batch_for_eval = _first_replica(batch)
                else:
                    state_for_eval = model.state
                    batch_for_eval = batch

                t_eval, coords_eval, u_eval, v_eval, w_eval = _sample_eval_data(
                    t, coords, u_ref, v_ref, w_ref
                )

                # 在 GPU 上計算 metrics（evaluator 內部都是 JAX 操作）
                log_dict_device = evaluator(
                    state_for_eval,
                    batch_for_eval,
                    t_eval,
                    coords_eval,
                    u_eval,
                    v_eval,
                    w_eval,
                )

                # 僅傳輸標量結果到 host（<1 KB vs 數十 MB）
                log_dict = jax.device_get(log_dict_device)

                non_finite_metrics = {}
                for key, value in log_dict.items():
                    value_array = np.asarray(value)
                    if np.issubdtype(value_array.dtype, np.number) and not np.all(
                        np.isfinite(value_array)
                    ):
                        non_finite_metrics[key] = value_array.tolist()

                if non_finite_metrics:
                    raise FloatingPointError(
                        "偵測到非有限訓練指標: "
                        f"time_window={idx + 1}, step={step}, metrics={non_finite_metrics}"
                    )

                # 添加時間窗口資訊到 log
                log_dict["time_window"] = idx + 1

                # 記憶體監控：追蹤 GPU 記憶體使用情況
                try:
                    for device_idx, device in enumerate(jax.devices()):
                        memory_stats = device.memory_stats()
                        if memory_stats:
                            device_kind = device.device_kind
                            log_dict[f"memory/{device_kind}_{device_idx}_bytes_in_use_MB"] = (
                                memory_stats.get("bytes_in_use", 0) / (1024 ** 2)
                            )
                            log_dict[f"memory/{device_kind}_{device_idx}_peak_bytes_in_use_MB"] = (
                                memory_stats.get("peak_bytes_in_use", 0) / (1024 ** 2)
                            )
                except Exception as e:
                    # 某些後端可能不支援記憶體統計，靜默失敗
                    pass

                if wandb.run:
                    wandb.log(log_dict, step + step_offset)

                end_time = time.time()
                logger.log_iter(
                    step,
                    start_time,
                    end_time,
                    log_dict,
                    max_steps=config.training.max_steps,
                    num_time_windows=config.training.num_time_windows,
                )

                # Sweep early-stop hook
                if step_callback is not None:
                    if step_callback(step, log_dict) == "stop":
                        return model

        # Saving
        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (
                step + 1
            ) == config.training.max_steps:
                ckpt_root = config.saving.ckpt_dir
                if ckpt_root is None:
                    ckpt_root = os.path.join(os.getcwd(), config.wandb.name, "ckpt")
                ckpt_path = os.path.join(ckpt_root, "time_window_{}".format(idx + 1))
                save_checkpoint(
                    model.state,
                    ckpt_path,
                    keep=config.saving.num_keep_ckpts,
                    overwrite=config.saving.overwrite,
                )

    return model


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    wandb_config = config.wandb

    parallel_state = init_parallel(config)
    num_devices = parallel_state["num_devices"]

    if num_devices > 1:
        logging.info("多 GPU 模式: 使用 jax.sharding + jit")
    else:
        logging.info("單 GPU 模式: 使用單卡 jit")

    # 驗證 batch size 配置（特別是在多 GPU 環境下）
    _validate_batch_size_config(config, num_devices)

    # 準備 wandb.init 參數
    wandb_init_kwargs = {
        "project": wandb_config.project,
        "name": wandb_config.name,
    }

    # 添加 group（如果有設定）
    if wandb_config.get("group") is not None:
        wandb_init_kwargs["group"] = wandb_config.group

    # 添加 tags（如果有設定）
    if wandb_config.get("tags") is not None and len(wandb_config.tags) > 0:
        wandb_init_kwargs["tags"] = wandb_config.tags

    # 添加 notes（如果有設定）
    if wandb_config.get("notes") is not None:
        wandb_init_kwargs["notes"] = wandb_config.notes

    # 如果是 Sweep run，使用 wandb.init() 不帶參數（sweep agent 會自動配置）
    if wandb_config.get("sweep_id") is not None:
        # Sweep mode: wandb agent 會自動注入配置
        wandb.init()
        # 從 wandb.config 更新超參數到 config
        config = update_config_from_sweep(config, wandb.config)
    else:
        # Normal mode: 手動配置
        wandb.init(**wandb_init_kwargs, config=config.to_dict())

    logging.info(f"Wandb run initialized: {wandb.run.name} (ID: {wandb.run.id})")
    if wandb_config.get("group"):
        logging.info(f"  Group: {wandb_config.group}")
    if wandb_config.get("tags"):
        logging.info(f"  Tags: {wandb_config.tags}")

    data_constraint = config.get("data_constraint")
    windowed_data_dir = config.get("windowed_data_dir")
    use_windowed_data = bool(windowed_data_dir)

    window_files = None
    if use_windowed_data:
        from pathlib import Path

        window_dir = Path(windowed_data_dir)
        window_files = sorted(window_dir.glob("window_*.npy"))
        if not window_files:
            raise FileNotFoundError(f"windowed_data_dir is empty: {windowed_data_dir}")
        if len(window_files) != config.training.num_time_windows:
            raise ValueError(
                "windowed_data_dir files count does not match num_time_windows: "
                f"{len(window_files)} vs {config.training.num_time_windows}"
            )

        first_window = str(window_files[0])
        u_ref, v_ref, w_ref, t_star, coords, nu = get_dataset(
            time_fraction=1.0,
            dataset_path=first_window,
            time_range=config.get("dns_time_range"),
            time_stride=config.get("dns_time_stride", 1),
        )
    else:
        u_ref, v_ref, w_ref, t_star, coords, nu = get_dataset(
            time_fraction=config.time_fraction,
            dataset_path=config.get(
                "dataset_path",
                "examples/kolmogorov_flow/data/kolmogorov_dns/kolmogorov_dns_10000.npy",
            ),
            time_range=config.get("dns_time_range"),
            time_stride=config.get("dns_time_stride", 1),
        )

    sensor_data = None
    if data_constraint != "dense_les":
        sensor_json = config.get("sensor_json")
        if sensor_json:
            from jaxpi.dataio.sensors_format import attach_dns_values, load_sensor_json

            sensor = load_sensor_json(sensor_json)
            sensor_values = config.get("sensor_values")
            if not sensor_values:
                metadata_values = sensor.metadata.get("dns_values_npz")
                if metadata_values:
                    candidate_paths = []
                    if os.path.isabs(metadata_values):
                        candidate_paths.append(metadata_values)
                    else:
                        sensor_dir = os.path.dirname(sensor_json)
                        example_dir = os.path.dirname(__file__)
                        data_dns_dir = os.path.join(example_dir, "data_dns")
                        candidate_paths.extend(
                            [
                                os.path.join(sensor_dir, metadata_values),
                                os.path.join(example_dir, metadata_values),
                                os.path.join(data_dns_dir, metadata_values),
                            ]
                        )

                    sensor_values = next(
                        (path for path in candidate_paths if os.path.exists(path)),
                        None,
                    )

            sensor = attach_dns_values(sensor, sensor_values)
            if sensor.dns_values is not None:
                dns_values = sensor.dns_values
                sensor_data = {
                    "time": jnp.array(dns_values["time"]),
                    "coords": jnp.array(sensor.coords),
                    "u": jnp.array(dns_values["u"]),
                    "v": jnp.array(dns_values["v"]),
                    "w": jnp.array(dns_values.get("omega", dns_values["u"] * 0.0)),
                }

    # Initial condition of the first time window
    u0 = u_ref[0, :]
    v0 = v_ref[0, :]
    w0 = w_ref[0, :]

    # Get the time domain for each time window
    if use_windowed_data:
        num_time_steps = len(t_star)
        t = t_star
    else:
        num_time_steps = len(t_star) // config.training.num_time_windows
        t = t_star[:num_time_steps]

    # Define the time and space domain
    dt = t[1] - t[0]
    t0 = t[0]
    t1 = t[-1] + 1.1 * dt

    x0 = 0.0
    x1 = 1.0

    y0 = 0.0
    y1 = 1.0

    dom = jnp.array([[t0, t1], [x0, x1], [y0, y1]])

    # 自動調整 batch size 以符合 causal weighting 的整除性要求
    config = _adjust_batch_sizes_for_causal(config, parallel_state["num_devices"])

    global_batch_size = config.training.batch_size_per_device * parallel_state["num_devices"]
    sensor_batch_per_device = config.get("sensor_batch_size_per_device")
    if sensor_batch_per_device is None:
        sensor_batch_per_device = config.training.batch_size_per_device
    sensor_global_batch_size = sensor_batch_per_device * parallel_state["num_devices"]
    sensor_sampling_mode = str(config.get("sensor_sampling", "random"))

    # 接續訓練：從指定 window 開始（預設 0，即從頭訓練）
    # 若 start_window > 0，需從前一個 window 的 ckpt 重建 IC（PINN 預測值，非 DNS 資料）
    start_window = int(config.get("start_window", 0))
    if start_window < 0 or start_window >= config.training.num_time_windows:
        raise ValueError(
            f"start_window={start_window} 超出範圍 [0, {config.training.num_time_windows})"
        )
    max_windows_to_run = config.training.get("max_windows_to_run", None)
    if max_windows_to_run is not None:
        max_windows_to_run = int(max_windows_to_run)
        if max_windows_to_run < 1:
            raise ValueError(f"max_windows_to_run={max_windows_to_run} 必須 >= 1")
        end_window_exclusive = min(
            config.training.num_time_windows,
            start_window + max_windows_to_run,
        )
    else:
        end_window_exclusive = config.training.num_time_windows
    effective_window_count = end_window_exclusive - start_window
    if effective_window_count < 1:
        raise ValueError(
            "本次訓練沒有任何 time window 可執行；請檢查 start_window 與 max_windows_to_run。"
        )
    if start_window > 0:
        logging.info(f"接續訓練：從 window {start_window + 1}/{config.training.num_time_windows} 開始")
        # IC 使用 DNS 參考資料在 window 邊界的值（避免對 4M 空間點做 forward pass 導致 OOM）
        # window 1 訓練後 PINN 應已緊密擬合 DNS，兩者差異極小
        u0 = u_ref[num_time_steps * start_window, :]
        v0 = v_ref[num_time_steps * start_window, :]
        w0 = w_ref[num_time_steps * start_window, :]
        logging.info(f"IC 設定完成（使用 DNS t_idx={num_time_steps * start_window}），準備從 window {start_window + 1} 開始訓練")

    def _predict_next_window_ic(model, current_t_star, current_coords):
        """
        What:
            以目前 window 訓練完成的模型，預測下一個 window 的初始條件。
        Why:
            多時間窗口訓練必須把前一窗口末端的 PINN 狀態傳遞到下一窗口；
            否則後續窗口會一直使用舊初值，破壞 time marching 機制。
            對高解析度網格（例如 Re=1e6, N=512），渦度預測需要空間導數，
            必須分塊計算以避免一次對整個空間場做 autodiff 而 OOM。
        """
        state = jax.device_get(model.state) if parallel_state["num_devices"] > 1 else model.state
        params = state.params

        if use_windowed_data:
            if current_t_star.shape[0] < 2:
                raise ValueError("windowed_data_dir 模式至少需要 2 個時間點，才能推進下一個 window IC。")
            dt_next = current_t_star[1] - current_t_star[0]
            next_ic_time = current_t_star[-1] + dt_next
        else:
            next_ic_time = t_star[num_time_steps]

        chunk_size = int(config.logging.get("ic_propagation_chunk_size", config.logging.get("eval_space_chunk_size", 4096)))
        if chunk_size < 1:
            chunk_size = 4096

        coords_np = np.asarray(current_coords)
        next_u0_chunks = []
        next_v0_chunks = []
        next_w0_chunks = []

        for start in range(0, coords_np.shape[0], chunk_size):
            end = min(start + chunk_size, coords_np.shape[0])
            coords_chunk = coords_np[start:end]
            x_chunk = jnp.asarray(coords_chunk[:, 0])
            y_chunk = jnp.asarray(coords_chunk[:, 1])

            next_u0_chunks.append(np.asarray(jax.device_get(model.u_ic_pred_fn(params, next_ic_time, x_chunk, y_chunk))))
            next_v0_chunks.append(np.asarray(jax.device_get(model.v_ic_pred_fn(params, next_ic_time, x_chunk, y_chunk))))
            next_w0_chunks.append(np.asarray(jax.device_get(model.w_ic_pred_fn(params, next_ic_time, x_chunk, y_chunk))))

        next_u0 = jnp.asarray(np.concatenate(next_u0_chunks, axis=0))
        next_v0 = jnp.asarray(np.concatenate(next_v0_chunks, axis=0))
        next_w0 = jnp.asarray(np.concatenate(next_w0_chunks, axis=0))
        return next_u0, next_v0, next_w0

    if end_window_exclusive < config.training.num_time_windows:
        logging.info(
            "限制訓練窗口數：僅執行 window %d 到 %d（原始總窗口數=%d）",
            start_window + 1,
            end_window_exclusive,
            config.training.num_time_windows,
        )

    for idx in range(start_window, end_window_exclusive):
        logging.info("Training time window {}".format(idx + 1))
        if use_windowed_data and window_files is not None:
            window_path = str(window_files[idx])
            u_ref, v_ref, w_ref, t_star, coords, nu = get_dataset(
                time_fraction=1.0,
                dataset_path=window_path,
                time_range=config.get("dns_time_range"),
                time_stride=config.get("dns_time_stride", 1),
            )
            t = t_star
            u_star, v_star, w_star = u_ref, v_ref, w_ref
        else:
            # Get the reference solution for the current time window
            u_star = u_ref[num_time_steps * idx : num_time_steps * (idx + 1), :]
            v_star = v_ref[num_time_steps * idx : num_time_steps * (idx + 1), :]
            w_star = w_ref[num_time_steps * idx : num_time_steps * (idx + 1), :]

        # Initialize the model
        model = models.NavierStokes(
            config,
            t,
            coords,
            u0,
            v0,
            w0,
            nu,
            replicate_state=False,
        )
        if parallel_state["num_devices"] > 1:
            model.state = jax.device_put(model.state, parallel_state["replicated_sharding"])

        if config.transfer_learning:
            if idx > 0:
                # restore the checkpoint from the previous time window
                ckpt_path = os.path.join(
                    os.getcwd(), config.wandb.name, "ckpt", "time_window_{}".format(idx)
                )
                state = restore_checkpoint(model.state, ckpt_path)

                if parallel_state["num_devices"] > 1:
                    unreplicated_state = jax.device_get(state)
                else:
                    unreplicated_state = state

                # 檢查是否傳遞完整優化器狀態
                if hasattr(config, "transfer_optimizer_state") and config.transfer_optimizer_state:
                    logging.info("傳遞完整優化器狀態（params + opt_state）")
                    model.state = _create_train_state(
                        config,
                        params=unreplicated_state.params,
                        weights=unreplicated_state.weights,
                        opt_state=unreplicated_state.opt_state,
                        step=0,  # 重置step以獲得獨立的LR schedule
                        replicate=False,
                    )
                    logging.info(
                        f"從時間窗口 {idx - 1} 恢復checkpoint（包含優化器狀態），步數: {unreplicated_state.step}"
                    )
                else:
                    logging.info("僅傳遞params（opt_state重置）")
                    # 標準params-only遷移（向後相容）
                    model.state = _create_train_state(
                        config,
                        params=unreplicated_state.params,
                        weights=unreplicated_state.weights,
                        replicate=False,
                    )
                    logging.info(
                        f"從時間窗口 {idx - 1} 恢復checkpoint（僅params），步數: {unreplicated_state.step}"
                    )

                if parallel_state["num_devices"] > 1:
                    model.state = jax.device_put(model.state, parallel_state["replicated_sharding"])

        # Initialize the samplers
        ics_sample_fn = _build_ic_sampler(
            coords,
            u0,
            v0,
            w0,
            global_batch_size * 2,
            rng_seed=config.seed + idx * 10 + 1,
        )
        res_sample_fn = _build_uniform_sampler(dom, global_batch_size)

        samplers = {
            "ics": HostSampler(ics_sample_fn),
            "res": JaxSampler(res_sample_fn, random.PRNGKey(config.seed + idx * 10 + 2)),
        }

        if data_constraint == "dense_les":
            sensor_sample_fn = _build_sensor_sampler(
                t,
                coords,
                u_star.T,
                v_star.T,
                w_star.T,
                sensor_global_batch_size,
                rng_seed=config.seed + idx * 10 + 3,
            )
            samplers["data"] = HostSampler(sensor_sample_fn)
        elif sensor_data is not None:
            if use_windowed_data:
                window_start = t_star[0]
                window_end = t_star[-1]
            else:
                window_start = t_star[num_time_steps * idx]
                window_end = t_star[num_time_steps * (idx + 1) - 1]
            time_mask = (sensor_data["time"] >= window_start) & (sensor_data["time"] <= window_end)

            if time_mask.any():
                time_window = sensor_data["time"][time_mask]
                if config.get("sensor_time_shift", True):
                    time_window = time_window - window_start

                sensor_sample_fn = _build_sensor_sampler(
                    time_window,
                    sensor_data["coords"],
                    sensor_data["u"][:, time_mask],
                    sensor_data["v"][:, time_mask],
                    sensor_data["w"][:, time_mask],
                    sensor_global_batch_size,
                    rng_seed=config.seed + idx * 10 + 4,
                    sampling_mode=sensor_sampling_mode,
                    pad_to_multiple=parallel_state["num_devices"],
                )
                samplers["data"] = HostSampler(sensor_sample_fn)

        # Training the current time window
        model = train_one_window(
            config,
            workdir,
            model,
            samplers,
            t,
            coords,
            u_star,
            v_star,
            w_star,
            idx,
            parallel_state,
        )

        # 每個 window 結束後，立即把 PINN 預測的邊界狀態傳給下一個 window。
        if idx < end_window_exclusive - 1:
            u0, v0, w0 = _predict_next_window_ic(model, t_star, coords)

    if jax.process_index() == 0 and config.logging.log_errors:
        if use_windowed_data:
            logging.warning(
                "windowed_data_dir enabled; skip full-series error logging to avoid loading full data."
            )
            return
        state = jax.device_get(model.state) if num_devices > 1 else model.state
        chunk_seconds = float(config.logging.get("eval_time_chunk_seconds", 1.0))
        time_chunk_size = config.logging.get("eval_time_chunk_size", None)
        if time_chunk_size is None:
            if t_star.shape[0] > 1:
                dt = float(np.asarray(t_star[1] - t_star[0]))
                if dt > 0:
                    time_chunk_size = max(1, int(chunk_seconds / dt))
                else:
                    time_chunk_size = 1
            else:
                time_chunk_size = 1
        time_chunk_size = int(time_chunk_size)
        if time_chunk_size < 1:
            time_chunk_size = 1
        space_chunk = int(config.logging.get("eval_space_chunk_size", 4096))
        if space_chunk < 1:
            space_chunk = 4096

        if num_devices > 1:
            mesh = parallel_state["mesh"]
            coords_sharding = jax.sharding.NamedSharding(mesh, P("data", None))
            space_sharding = jax.sharding.NamedSharding(mesh, P(None, "data"))

            def _eval_full(params, t_all, coords_all, u_all, v_all, w_all):
                return model._compute_l2_error_time_space_chunked_impl(
                    params,
                    t_all,
                    coords_all,
                    u_all,
                    v_all,
                    w_all,
                    time_chunk_size=time_chunk_size,
                    space_chunk_size=space_chunk,
                )

            pjit_eval = pjit(
                _eval_full,
                in_shardings=(
                    parallel_state["replicated_sharding"],
                    parallel_state["replicated_sharding"],
                    coords_sharding,
                    space_sharding,
                    space_sharding,
                    space_sharding,
                ),
                out_shardings=(
                    parallel_state["replicated_sharding"],
                    parallel_state["replicated_sharding"],
                    parallel_state["replicated_sharding"],
                ),
            )

            t_dev = jax.device_put(t_star, parallel_state["replicated_sharding"])
            coords_dev = jax.device_put(coords, coords_sharding)
            u_dev = jax.device_put(u_ref, space_sharding)
            v_dev = jax.device_put(v_ref, space_sharding)
            w_dev = jax.device_put(w_ref, space_sharding)

            with mesh:
                u_error, v_error, w_error = pjit_eval(
                    state.params,
                    t_dev,
                    coords_dev,
                    u_dev,
                    v_dev,
                    w_dev,
                )
            u_error, v_error, w_error = jax.device_get((u_error, v_error, w_error))
        else:
            u_error, v_error, w_error = model.compute_l2_error_time_space_chunked(
                state.params,
                t_star,
                coords,
                u_ref,
                v_ref,
                w_ref,
                time_chunk_size=time_chunk_size,
                space_chunk_size=space_chunk,
            )

        wandb.log(
            {
                "u_error_full": u_error,
                "v_error_full": v_error,
                "w_error_full": w_error,
            },
            config.training.max_steps * effective_window_count,
        )


def update_config_from_sweep(
    config: ml_collections.ConfigDict, sweep_config
) -> ml_collections.ConfigDict:
    """
    從 wandb sweep config 更新訓練配置

    Sweep 會自動設定以下參數（example）:
    - learning_rate
    - hidden_dim
    - batch_size_per_device
    - warmup_steps
    等等
    """
    # 更新 optimizer 參數
    if hasattr(sweep_config, "learning_rate"):
        config.optim.learning_rate = sweep_config.learning_rate
        logging.info(f"Sweep: learning_rate = {sweep_config.learning_rate}")

    if hasattr(sweep_config, "warmup_steps"):
        config.optim.warmup_steps = sweep_config.warmup_steps
        logging.info(f"Sweep: warmup_steps = {sweep_config.warmup_steps}")

    # 更新網路架構參數
    if hasattr(sweep_config, "hidden_dim"):
        config.arch.hidden_dim = sweep_config.hidden_dim
        config.arch.fourier_emb.embed_dim = sweep_config.hidden_dim
        logging.info(f"Sweep: hidden_dim = {sweep_config.hidden_dim}")

    if hasattr(sweep_config, "num_layers"):
        config.arch.num_layers = sweep_config.num_layers
        logging.info(f"Sweep: num_layers = {sweep_config.num_layers}")

    # 更新訓練參數
    if hasattr(sweep_config, "batch_size_per_device"):
        config.training.batch_size_per_device = sweep_config.batch_size_per_device
        logging.info(f"Sweep: batch_size_per_device = {sweep_config.batch_size_per_device}")

    if hasattr(sweep_config, "max_steps"):
        config.training.max_steps = sweep_config.max_steps
        logging.info(f"Sweep: max_steps = {sweep_config.max_steps}")

    # 更新 weighting 參數
    if hasattr(sweep_config, "causal_tol"):
        config.weighting.causal_tol = sweep_config.causal_tol
        logging.info(f"Sweep: causal_tol = {sweep_config.causal_tol}")

    return config


def _adjust_batch_sizes_for_causal(
    config: ml_collections.ConfigDict, num_devices: int
) -> ml_collections.ConfigDict:
    """
    自動調整 batch size 以符合 causal weighting 的整除性要求

    如果 batch_size_per_device 或 sensor_batch_size_per_device 不能被 num_chunks 整除，
    自動向下調整到最接近的整除值。

    Returns:
        調整後的 config（會修改原地，同時返回引用）
    """
    if not config.weighting.use_causal:
        return config

    num_chunks = config.weighting.num_chunks
    adjusted = False

    # 調整主 batch size
    batch_size_per_device = config.training.batch_size_per_device
    if batch_size_per_device % num_chunks != 0:
        old_size = batch_size_per_device
        new_size = (batch_size_per_device // num_chunks) * num_chunks
        config.training.batch_size_per_device = new_size
        adjusted = True
        logging.warning(
            f"自動調整 batch_size_per_device: {old_size} -> {new_size} "
            f"(must be divisible by num_chunks={num_chunks})"
        )

    # 調整 sensor batch size（如果有設定）
    sensor_batch_per_device = config.get("sensor_batch_size_per_device")
    sensor_sampling_mode = str(config.get("sensor_sampling", "random"))
    if (
        sensor_sampling_mode != "all_points"
        and sensor_batch_per_device is not None
        and sensor_batch_per_device % num_chunks != 0
    ):
        old_size = sensor_batch_per_device
        new_size = (sensor_batch_per_device // num_chunks) * num_chunks
        config.sensor_batch_size_per_device = new_size
        adjusted = True
        logging.warning(
            f"自動調整 sensor_batch_size_per_device: {old_size} -> {new_size} "
            f"(must be divisible by num_chunks={num_chunks})"
        )

    if adjusted:
        logging.info("✓ Batch sizes 已自動調整為符合 causal weighting 要求")

    return config


def _validate_batch_size_config(config: ml_collections.ConfigDict, num_devices: int):
    """
    驗證並輸出 batch size 配置摘要

    注意：batch size 的自動調整已在 _adjust_batch_sizes_for_causal 中完成
    此函數僅用於最終驗證和輸出配置摘要
    """
    batch_size_per_device = config.training.batch_size_per_device
    global_batch_size = batch_size_per_device * num_devices

    # 輸出配置摘要
    if config.weighting.use_causal:
        num_chunks = config.weighting.num_chunks
        chunk_size_per_device = batch_size_per_device // num_chunks

        # 最終驗證（理論上已經被 _adjust_batch_sizes_for_causal 保證）
        assert batch_size_per_device % num_chunks == 0, (
            f"Internal error: batch_size_per_device ({batch_size_per_device}) "
            f"is not divisible by num_chunks ({num_chunks}) after adjustment"
        )

        logging.info(
            f"✓ Batch size configuration:\n"
            f"  - batch_size_per_device: {batch_size_per_device}\n"
            f"  - num_devices: {num_devices}\n"
            f"  - global_batch_size: {global_batch_size}\n"
            f"  - num_chunks: {num_chunks}\n"
            f"  - chunk_size_per_device: {chunk_size_per_device}\n"
            f"  - weighting.start_step: {int(config.weighting.get('start_step', 0))}"
        )

        # 檢查 sensor batch size（如果有）
        sensor_batch_per_device = config.get("sensor_batch_size_per_device")
        sensor_sampling_mode = str(config.get("sensor_sampling", "random"))
        if sensor_sampling_mode == "all_points":
            logging.info("  - sensor_sampling: all_points (sensor_batch_size_per_device ignored)")
        elif sensor_batch_per_device is not None:
            assert sensor_batch_per_device % num_chunks == 0, (
                f"Internal error: sensor_batch_size_per_device ({sensor_batch_per_device}) "
                f"is not divisible by num_chunks ({num_chunks}) after adjustment"
            )
            logging.info(f"  - sensor_batch_size_per_device: {sensor_batch_per_device}")
