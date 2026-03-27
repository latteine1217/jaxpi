import ml_collections
import jax.numpy as jnp


def get_config():
    """
    Re=10000, N=256 的 SOAP 訓練 config。
    對齊原始作者 soap.py 設定：hidden_dim=384, period=(2π,2π), transfer_learning=False。
    """
    config = ml_collections.ConfigDict()

    config.mode = "train"

    config.transfer_learning = False
    config.transfer_optimizer_state = False

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "PINN-Kolmogorov_flow"
    wandb.name = "re10k_n256_soap"
    wandb.group = "re10k_training"
    wandb.tags = ["soap", "kolmogorov", "pure_pinn", "Re10000", "N256", "hidden_dim=384"]
    wandb.notes = (
        "Kolmogorov Re=10000, N=256. "
        "PirateNet 3 layers hidden_dim=384 tanh, SOAP. "
        "Aligned with upstream original soap.py."
    )
    wandb.sweep_id = None

    # Arch — 對齊原作者 upstream/pirate 預設
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "PirateNet"
    arch.num_layers = 3
    arch.hidden_dim = 384
    arch.out_dim = 3
    arch.activation = "tanh"
    arch.periodicity = ml_collections.ConfigDict(
        {"period": (2 * jnp.pi, 2 * jnp.pi), "axis": (1, 2), "trainable": (False, False)}
    )
    arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 2.0, "embed_dim": 384})
    arch.reparam = ml_collections.ConfigDict({"type": "weight_fact", "mean": 1.0, "stddev": 0.1})
    arch.nonlinearity = 0.0
    arch.pi_init = None

    # Body force: coords ∈ [0,1]²，對齊 upstream 原始設定 2*sin(4πy)
    # 公式 A * sin(2π * k * y)，A=2, k=2 → 2*sin(4πy)
    config.body_force = body_force = ml_collections.ConfigDict()
    body_force.amplitude = 2.0
    body_force.wavenumber = 2.0

    # Data
    config.time_fraction = 1.0
    config.dataset_path = (
        "examples/kolmogorov_flow/data/kolmogorov_flow_Re10000_256.npy"
    )
    config.dns_time_range = None
    config.dns_time_stride = 1
    config.sensor_json = None
    config.sensor_values = None
    config.sensor_batch_size_per_device = None
    config.sensor_time_shift = True

    # Optim — 保留 SOAP
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

    # Training — 對齊原作者設定
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 20000
    training.batch_size_per_device = 8192
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

    # Memory
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
    logging.eval_space_samples = 4096
    logging.eval_time_chunk_seconds = 1.0
    logging.eval_space_chunk_size = 4096

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 5000
    saving.num_keep_ckpts = 2
    saving.ckpt_dir = None
    saving.overwrite = True

    config.input_dim = 3
    config.seed = 42
    config.start_window = 0

    return config
