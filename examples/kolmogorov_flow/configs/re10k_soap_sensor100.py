import ml_collections
import jax.numpy as jnp


def get_config():
    """
    Re=10000, N=256 的 SOAP 訓練 config（含 100 QR-pivot sensors）。
    與 re10k_soap.py（pure PINN）使用相同資料，僅增加 sensor 約束。
    資料：kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_ds4.npy
      - domain [0,1]², T=5, 41 snapshots
      - A=0.1, k_f=2
    num_time_windows=20: 41//20=2 steps/window
    Sensor: 100 QR-pivot positions (from same DNS t0-5), u+v+w data loss enabled.
    """
    config = ml_collections.ConfigDict()

    config.mode = "train"

    config.transfer_learning = False
    config.transfer_optimizer_state = False

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "PINN-Kolmogorov_flow"
    wandb.name = "re10k_n256_soap_sensor100"
    wandb.group = "re10k_training"
    wandb.tags = ["soap", "kolmogorov", "sensor_constraint", "qr_pivot", "K100",
                  "Re10000", "N256", "hidden_dim=384"]
    wandb.notes = (
        "Kolmogorov Re=10000, N=256, SOAP + 100 QR-pivot sensors. "
        "Same DNS as pure PINN baseline (kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_ds4.npy). "
        "Sensor u+v data loss to compare with pure PINN re10k_soap.py."
    )
    wandb.sweep_id = None

    # Arch — 與 re10k_soap.py 完全相同
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

    # Body force: 與 re10k_soap.py 相同
    config.body_force = body_force = ml_collections.ConfigDict()
    body_force.amplitude = 0.1
    body_force.wavenumber = 2.0

    # Data — 與 re10k_soap.py 相同
    config.time_fraction = 1.0
    config.dataset_path = (
        "examples/kolmogorov_flow/data/kolmogorov_dns/"
        "kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_ds4.npy"
    )
    config.dns_time_range = None
    config.dns_time_stride = 1

    # Sensor: 100 QR-pivot positions，由相同 DNS 生成
    # sensor NPZ shape: (100, 41) — K sensors × 41 DNS time points
    config.sensor_json = (
        "examples/kolmogorov_flow/data/kolmogorov_sensors/re10000/"
        "sensors_qrpivot_K100_N256_t0-5.json"
    )
    config.sensor_values = (
        "examples/kolmogorov_flow/data/kolmogorov_sensors/re10000/"
        "sensors_qrpivot_K100_N256_t0-5_dns_values.npz"
    )
    config.sensor_batch_size_per_device = 48
    config.sensor_time_shift = True
    config.use_vorticity_data_loss = True

    # Optim — 與 re10k_soap.py 完全相同
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

    # Training — 與 re10k_soap.py 相同
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 20000
    training.batch_size_per_device = 8192
    training.num_time_windows = 20

    # Weighting — u_data / v_data 啟用（100.0），其餘與 re10k_soap.py 相同
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"
    weighting.init_weights = ml_collections.ConfigDict(
        {
            "u_ic":   100.0,
            "v_ic":   100.0,
            "ru":       1.0,
            "rv":       1.0,
            "rc":       1.0,
            "u_data": 100.0,
            "v_data": 100.0,
            "w_data": 100.0,
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

    config.eval = eval_cfg = ml_collections.ConfigDict()
    eval_cfg.expected_time_remainder = 1

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 5000
    saving.num_keep_ckpts = None
    saving.ckpt_dir = None
    saving.overwrite = True

    config.input_dim = 3
    config.seed = 42
    config.start_window = 0

    return config
