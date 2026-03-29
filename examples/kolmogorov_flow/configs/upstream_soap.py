import ml_collections
import jax.numpy as jnp


def get_config():
    """
    What:
        建立一份和 upstream `pirate` branch `soap.py` 對齊的 Kolmogorov config。
    Why:
        保留原作者 SOAP baseline，避免和本地客製版 `soap.py` 混淆。

    說明：
        與 upstream 重疊的欄位皆逐值對齊。
        只補本地 branch 執行必需、但不改變 upstream SOAP 行為的相容欄位：
        1. body_force：把原本寫死在 `models.py` 的 2*sin(4πy) 顯式寫回 config
        2. dataset_path：指向 repo 內可用的本地等價 Re10000 資料檔
        3. u_data / v_data / w_data：補成 0.0，讓本地 loss dict 結構完整
        4. saving.ckpt_dir / saving.overwrite：補齊本地 train.py 直接存取的欄位
    """
    config = ml_collections.ConfigDict()

    config.mode = "train"
    config.transfer_learning = False

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "PINN-Kolmogorov_flow"
    wandb.name = "soap_Re10000"
    wandb.tag = None

    # Arch
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
    arch.reparam = ml_collections.ConfigDict(
        {"type": "weight_fact", "mean": 1.0, "stddev": 0.1}
    )
    arch.nonlinearity = 0.0
    arch.pi_init = None

    config.time_fraction = 1.0

    # Compatibility-only: preserve upstream forcing 2 * sin(4πy) on the local branch.
    config.body_force = body_force = ml_collections.ConfigDict()
    body_force.amplitude = 2.0
    body_force.wavenumber = 2.0

    # Compatibility-only: bind to the equivalent local Re10000 dataset that exists in this repo.
    config.dataset_path = "examples/kolmogorov_flow/data/kolmogorov_flow_Re10000_256.npy"

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
    optim.schedule_free = True

    # Training
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

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 100
    logging.log_errors = True
    logging.log_losses = True
    logging.log_weights = True
    logging.log_lr = False
    logging.log_preds = False
    logging.log_grads = False
    logging.log_ntk = False
    logging.log_nonlinearities = False
    logging.log_cossim = False

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 5000
    saving.num_keep_ckpts = 2
    saving.ckpt_dir = None
    saving.overwrite = True

    config.input_dim = 3
    config.seed = 42

    return config
