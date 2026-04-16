import ml_collections

import jax.numpy as jnp


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.mode = "train"

    config.transfer_learning = True  # 論文配置：使用 transfer learning 初始化參數
    config.transfer_optimizer_state = False  # 是否傳遞完整優化器狀態（預設False保持向後相容）

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "PINN-Kolmogorov_flow"
    wandb.name = "soap_Re10000"
    wandb.group = "optimizer_comparison"  # 與 PIRATE 同組方便比較
    wandb.tags = ["soap", "2gpu", "Re10000", "paper_aligned", "L=4", "hidden_dim=384"]  # 標籤
    wandb.notes = "實驗配置：SOAP optimizer, L=4, hidden_dim=384, swish activation, transfer_learning=True"  # 備註
    wandb.sweep_id = None  # Sweep ID (超參數調優時自動填入)

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "PirateNet"
    arch.num_layers = 4  # 調整為 4 層
    arch.hidden_dim = 384  # 調整為 384 hidden dim
    arch.out_dim = 3
    arch.activation = "swish"  # 論文配置：使用 Swish activation
    arch.periodicity = ml_collections.ConfigDict(
        {"period": (2 * jnp.pi, 2 * jnp.pi), "axis": (1, 2), "trainable": (False, False)}
    )
    arch.fourier_emb = ml_collections.ConfigDict(
        {"embed_scale": 2.0, "embed_dim": 192}
    )  # embed_dim = hidden_dim // 2
    arch.reparam = ml_collections.ConfigDict({"type": "weight_fact", "mean": 1.0, "stddev": 0.1})
    arch.nonlinearity = 0.0
    arch.pi_init = None

    # Body force: f(x,y) = [A * sin(2π * k * y), 0]
    # 論文 2507.08972 設定：A=0.1，k=2（在 [0,1]² 域）
    config.body_force = body_force = ml_collections.ConfigDict()
    body_force.amplitude = 0.1  # 強迫振幅 A
    body_force.wavenumber = 2.0  # 注入能量波數 k（[0,1] 域下的模態數）

    config.time_fraction = 1.0
    config.dataset_path = "examples/kolmogorov_flow/data/kolmogorov_dns/kolmogorov_dns_10000.npy"
    config.dns_time_range = None
    config.dns_time_stride = 1
    config.sensor_json = None
    config.sensor_values = None
    config.sensor_batch_size_per_device = None
    config.sensor_time_shift = True

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "Soap"
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.learning_rate = 1e-3
    optim.decay_rate = 0.9
    optim.decay_steps = 2000
    optim.staircase = False
    optim.warmup_steps = 2000
    optim.grad_accum_steps = 0
    optim.schedule_free = False  # SOAP 本身就是 schedule-free，不需要額外 wrapper

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 20000
    training.batch_size_per_device = 4096  # 降低以避免 OOM
    training.num_time_windows = 25

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"
    weighting.init_weights = ml_collections.ConfigDict(
        {
            "u_ic": 100.0,
            "v_ic": 100.0,
            "ru": 1.0,
            "rv": 1.0,
            "rc": 1.0,
            "u_data": 0.0,
            "v_data": 0.0,
            "w_data": 0.0,
        }
    )
    weighting.momentum = 0.9
    weighting.update_every_steps = 1000

    weighting.use_causal = True
    weighting.causal_tol = 1.0
    weighting.num_chunks = 16

    # Memory Optimization（記憶體優化配置）
    config.optimization = optimization = ml_collections.ConfigDict()
    optimization.use_vmap_chunking = False  # 啟用 vmap 分塊優化（建議 GPU < 16GB 時啟用）
    optimization.vmap_chunk_size = 512  # vmap 分塊大小（僅當 use_vmap_chunking=True 時生效）
    optimization.use_eval_checkpoint = (
        False  # 評估時使用 gradient checkpointing（極度記憶體受限時啟用）
    )

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 100
    logging.log_errors = False  # 訓練時關閉以提升效能（評估時可開啟）
    logging.log_losses = True
    logging.log_weights = True
    logging.log_lr = False
    logging.log_preds = False
    logging.log_grads = False
    logging.log_ntk = False
    logging.log_nonlinearities = False
    logging.log_cossim = False
    logging.eval_time_samples = 8  # 已優化：預設會使用 min(100, actual_size)
    logging.eval_space_samples = 8192  # 已優化：預設會使用 min(4096, actual_size)
    logging.eval_time_chunk_seconds = 1.0
    logging.eval_space_chunk_size = 4096

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 5000
    saving.num_keep_ckpts = None
    saving.ckpt_dir = None
    saving.overwrite = True

    # # Input shape for initializing Flax models
    config.input_dim = 3

    # Integer for PRNG random seed.
    config.seed = 42

    # 驗證配置的一致性
    _validate_config(config)

    return config


def _validate_config(config):
    """
    驗證配置參數的一致性，特別是多 GPU 環境下的 batch size 和 num_chunks 的關係

    注意：從嚴格錯誤改為警告，因為 train.py 會自動調整 batch size
    """
    import warnings

    batch_size_per_device = config.training.batch_size_per_device
    num_chunks = config.weighting.num_chunks

    # 檢查 1：batch_size_per_device 最好能被 num_chunks 整除
    if batch_size_per_device % num_chunks != 0:
        warnings.warn(
            f"batch_size_per_device ({batch_size_per_device}) is not divisible by "
            f"num_chunks ({num_chunks}). It will be auto-adjusted during training to "
            f"{(batch_size_per_device // num_chunks) * num_chunks} for proper causal weighting."
        )

    # 檢查 2：只有在 use_causal = True 時才需要檢查 num_chunks
    if not config.weighting.use_causal and num_chunks > 1:
        import warnings

        warnings.warn(
            f"use_causal is False but num_chunks is {num_chunks}. "
            "num_chunks will be ignored since causal weighting is disabled."
        )

    # 檢查 3：Fourier embedding dimension 應該是 hidden_dim 的一半
    if config.arch.fourier_emb is not None:
        embed_dim = config.arch.fourier_emb.embed_dim
        hidden_dim = config.arch.hidden_dim
        if embed_dim != hidden_dim // 2:
            import warnings

            warnings.warn(
                f"Fourier embed_dim ({embed_dim}) is not hidden_dim // 2 ({hidden_dim // 2}). "
                "This may affect network performance."
            )

    print(
        f"✓ Configuration validated: batch_size_per_device={batch_size_per_device}, num_chunks={num_chunks}"
    )
    print(f"  Each chunk will contain {batch_size_per_device // num_chunks} samples per device")
