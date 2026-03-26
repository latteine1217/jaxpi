import ml_collections

import jax.numpy as jnp


def get_config():
    """SOAP + 100 QR-pivot sensors, 25 time windows, for Kolmogorov Re=1e6.

    Same as paper_repro_soap_sensor100 but uses 25 windows (window size=0.2s)
    so each window contains 4 native DNS time points (dt=0.05s × 4 = 0.2s).
    Richer sensor constraints without interpolation.
    """
    config = ml_collections.ConfigDict()

    config.mode = "train"
    config.transfer_learning = True
    config.transfer_optimizer_state = False

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "PINN-Kolmogorov_flow"
    wandb.name = "re1e6_n2048_ke024_soap_sensor100_w25"
    wandb.group = "re1e6_training"
    wandb.tags = [
        "soap",
        "kolmogorov",
        "sensor_constraint",
        "qr_pivot",
        "K100",
        "w25",
        "Re1e6",
        "N2048",
        "hidden_dim=768",
    ]
    wandb.notes = (
        "SOAP + 100 QR-pivot sensors. Re=1e6, N=2048, T=5, ke024 DNS. "
        "Sensor positions chosen via QR column pivoting (u,v,p,omega,|∇u|,|∇v| at 0.1s stride). "
        "Sensor data loss added to break degenerate time-stationary solution."
    )
    wandb.sweep_id = None

    # Arch (identical to paper_repro_soap)
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
    config.dataset_path = "examples/kolmogorov_flow/data/kolmogorov_dns/kolmogorov_Re1e6_N2048_T5_ke024.npy"
    config.dns_time_range = None
    config.dns_time_stride = 1

    # Sensor constraint: 100 QR-pivot positions, u+v data loss only (no vorticity to save compute)
    config.sensor_json = (
        "examples/kolmogorov_flow/data/kolmogorov_sensors/re1000000/"
        "sensors_qrpivot_K100_N2048_t0-5.json"
    )
    config.sensor_values = (
        "examples/kolmogorov_flow/data/kolmogorov_sensors/re1000000/"
        "sensors_qrpivot_K100_N2048_t0-5_dns_values.npz"
    )
    # 50 per GPU × 2 GPUs = 100 sensor data points per training step
    config.sensor_batch_size_per_device = 50
    config.sensor_time_shift = True
    config.use_vorticity_data_loss = False

    # Optim (identical to paper_repro_soap)
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
    # 25 windows × 0.2s/window = T=5s; 101//25=4 DNS time points per window
    training.num_time_windows = 25

    # Weighting — u_data / v_data enabled; w_data disabled (use_vorticity_data_loss=False)
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"
    weighting.init_weights = ml_collections.ConfigDict(
        {
            "u_ic":    100.0,
            "v_ic":    100.0,
            "ru":        1.0,
            "rv":        1.0,
            "rc":        1.0,
            "u_data":    1.0,   # sensor u loss enabled
            "v_data":    1.0,   # sensor v loss enabled
            "w_data":    0.0,   # vorticity data loss disabled
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

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 10000
    saving.num_keep_ckpts = 2
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

    print(
        f"✓ Sensor-100 config validated: batch_size_per_device={batch_size_per_device}, "
        f"sensor_batch_size_per_device={config.sensor_batch_size_per_device}, "
        f"num_chunks={num_chunks}"
    )
