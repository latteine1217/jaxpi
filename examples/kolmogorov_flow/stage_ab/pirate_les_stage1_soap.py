"""Stage A: Dense LES data constraint (full dataset, SOAP optimizer)."""

import ml_collections

import jax.numpy as jnp


def get_config():
    """Stage A: LES prefit with SOAP optimizer."""
    config = ml_collections.ConfigDict()

    config.mode = "train"
    config.transfer_learning = True
    config.transfer_optimizer_state = False

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "PINN-Kolmogorov_flow"
    wandb.name = "pirate_les_stage1_soap"
    wandb.group = "les_then_sensor"
    wandb.tags = ["les", "stage1", "soap", "no_sensor", "Re100000"]
    wandb.notes = "Stage A: LES prefit with SOAP optimizer"
    wandb.sweep_id = None

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "PirateNet"
    arch.num_layers = 4
    arch.hidden_dim = 384
    arch.out_dim = 3
    arch.activation = "swish"
    arch.periodicity = ml_collections.ConfigDict(
        {"period": (2 * jnp.pi, 2 * jnp.pi), "axis": (1, 2), "trainable": (False, False)}
    )
    arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 2.0, "embed_dim": 384})
    arch.reparam = ml_collections.ConfigDict(
        {"type": "weight_fact", "mean": 1.0, "stddev": 0.1}
    )
    arch.nonlinearity = 0.0
    arch.pi_init = None

    # Data
    config.time_fraction = 1.0
    config.dataset_path = "examples/kolmogorov_flow/data/kolmogorov_les/kolmogorov_les_re100000.npy"
    config.dns_time_range = None
    config.dns_time_stride = 1
    config.data_constraint = "dense_les"
    config.windowed_data_dir = None
    config.sensor_json = None
    config.sensor_values = None
    config.use_vorticity_data_loss = False
    config.sensor_batch_size_per_device = 512
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
    optim.schedule_free = False

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 20000
    training.batch_size_per_device = 1024
    training.num_time_windows = 10

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"
    weighting.init_weights = ml_collections.ConfigDict(
        {
            "u_ic": 100.0,
            "v_ic": 100.0,
            "ru": 0.5,
            "rv": 0.5,
            "rc": 0.5,
            "u_data": 1.0,
            "v_data": 1.0,
            "w_data": 0.0,
        }
    )
    weighting.momentum = 0.9
    weighting.start_step = 1000
    weighting.update_every_steps = 1000

    weighting.use_causal = False
    weighting.causal_tol = 1.0
    weighting.num_chunks = 1

    # Memory Optimization
    config.optimization = optimization = ml_collections.ConfigDict()
    optimization.use_vmap_chunking = True
    optimization.vmap_chunk_size = 256
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

    _validate_config(config)
    return config


def _validate_config(config):
    """維持與既有流程相容的檢查。"""
    import warnings

    batch_size_per_device = config.training.batch_size_per_device
    num_chunks = config.weighting.num_chunks

    if batch_size_per_device % num_chunks != 0:
        warnings.warn(
            f"batch_size_per_device ({batch_size_per_device}) is not divisible by "
            f"num_chunks ({num_chunks}). It will be auto-adjusted during training to "
            f"{(batch_size_per_device // num_chunks) * num_chunks} for proper causal weighting."
        )

    if not config.weighting.use_causal and num_chunks > 1:
        warnings.warn(
            f"use_causal is False but num_chunks is {num_chunks}. "
            "num_chunks will be ignored since causal weighting is disabled."
        )

    if config.arch.fourier_emb is not None:
        embed_dim = config.arch.fourier_emb.embed_dim
        hidden_dim = config.arch.hidden_dim
        if embed_dim != hidden_dim // 2:
            warnings.warn(
                f"Fourier embed_dim ({embed_dim}) is not hidden_dim // 2 ({hidden_dim // 2}). "
                "This may affect network performance."
            )

    print(
        f"✓ Configuration validated: batch_size_per_device={batch_size_per_device}, "
        f"num_chunks={num_chunks}"
    )
    print(f"  Each chunk will contain {batch_size_per_device // num_chunks} samples per device")
