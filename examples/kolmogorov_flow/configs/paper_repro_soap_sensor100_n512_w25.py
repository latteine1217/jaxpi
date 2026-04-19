import ml_collections

import jax.numpy as jnp


def get_config():
    """SOAP + 100 QR-pivot sensors, 25 windows, aligned to current N512 ds4 run."""
    config = ml_collections.ConfigDict()

    config.mode = "train"
    config.transfer_learning = True
    config.transfer_optimizer_state = False

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "PINN-Kolmogorov_flow"
    wandb.name = "re1e6_n512_ds4_soap_sensor100_w25"
    wandb.group = "re1e6_training"
    wandb.tags = [
        "soap",
        "kolmogorov",
        "sensor_constraint",
        "qr_pivot",
        "K100",
        "w25",
        "Re1e6",
        "N512",
        "ds4",
        "hidden_dim=768",
    ]
    wandb.notes = (
        "SOAP + 100 QR-pivot sensors on the currently available Re=1e6, N=512, T=5, ds4 DNS. "
        "Built from the legacy sensor100_w25 recipe, but aligned to the current unit-domain N512 dataset. "
        "Uses 25 windows (0.2s/window) and enables u/v sensor data loss to suppress degenerate time-stationary solutions."
    )
    wandb.sweep_id = None

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "PirateNet"
    arch.num_layers = 2
    arch.hidden_dim = 768
    arch.out_dim = 3
    arch.activation = "swish"
    arch.periodicity = ml_collections.ConfigDict(
        {"period": (2 * jnp.pi, 2 * jnp.pi), "axis": (1, 2), "trainable": (False, False)}
    )
    arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 2.0, "embed_dim": 384})
    arch.reparam = ml_collections.ConfigDict({"type": "weight_fact", "mean": 1.0, "stddev": 0.1})
    arch.nonlinearity = 0.0
    arch.pi_init = None

    # Body force
    config.body_force = body_force = ml_collections.ConfigDict()
    body_force.amplitude = 0.1
    body_force.wavenumber = 2.0

    # Data
    config.time_fraction = 1.0
    config.dataset_path = "examples/kolmogorov_flow/data/kolmogorov_dns/kolmogorov_Re1e6_N512_T5_ds4.npy"
    config.dns_time_range = None
    config.dns_time_stride = 1

    # Sensor constraint: N512 QR-pivot sensors aligned with the current ds4 dataset.
    config.sensor_json = (
        "examples/kolmogorov_flow/data/kolmogorov_sensors/re1000000/"
        "sensors_qrpivot_K100_N512_t0-5.json"
    )
    config.sensor_values = (
        "examples/kolmogorov_flow/data/kolmogorov_sensors/re1000000/"
        "sensors_qrpivot_K100_N512_t0-5_dns_values.npz"
    )
    # 保留舊欄位作為 random sampling fallback；all_points 模式下會被忽略。
    config.sensor_batch_size_per_device = 48
    config.sensor_sampling = "all_points"
    config.sensor_time_shift = True
    config.use_vorticity_data_loss = False

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
    optim.grad_clip_norm = 1.0
    optim.grad_accum_steps = 0
    optim.schedule_free = True

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 100000
    training.batch_size_per_device = 4096
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
            "u_data": 100.0,
            "v_data": 100.0,
            "w_data": 0.0,
        }
    )
    weighting.momentum = 0.9
    weighting.update_every_steps = 1000
    weighting.use_causal = True
    weighting.causal_tol = 1.0
    weighting.num_chunks = 16

    config.optimization = optimization = ml_collections.ConfigDict()
    optimization.use_vmap_chunking = False
    optimization.vmap_chunk_size = 512
    optimization.use_eval_checkpoint = False

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 100
    logging.log_errors = False
    logging.log_losses = True
    logging.log_weights = True
    logging.log_lr = False
    logging.log_preds = False
    logging.log_grads = False
    logging.log_ntk = False
    logging.log_nonlinearities = False
    logging.log_cossim = False
    logging.eval_time_samples = 8
    logging.eval_space_samples = 8192
    logging.eval_time_chunk_seconds = 1.0
    logging.eval_space_chunk_size = 4096

    config.eval = eval_cfg = ml_collections.ConfigDict()
    eval_cfg.expected_time_remainder = 1

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 10000
    saving.num_keep_ckpts = None
    saving.ckpt_dir = None
    saving.overwrite = True

    config.input_dim = 3
    config.seed = 42
    config.start_window = 0

    _validate_config(config)
    return config


def _validate_config(config):
    import warnings

    batch_size_per_device = config.training.batch_size_per_device
    num_chunks = config.weighting.num_chunks

    if batch_size_per_device % num_chunks != 0:
        warnings.warn(
            f"batch_size_per_device ({batch_size_per_device}) is not divisible by "
            f"num_chunks ({num_chunks}). Will be auto-adjusted during training."
        )

    if config.use_vorticity_data_loss:
        raise ValueError("paper_repro_soap_sensor100_n512_w25 禁止啟用 w_data loss")

    print(
        f"✓ Sensor-100 N512 w25 config validated: batch_size_per_device={batch_size_per_device}, "
        f"sensor_sampling={config.sensor_sampling}, "
        f"sensor_batch_size_per_device={config.sensor_batch_size_per_device}, "
        f"num_chunks={num_chunks}"
    )
