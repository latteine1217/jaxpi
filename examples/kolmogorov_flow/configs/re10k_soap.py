import ml_collections
import jax.numpy as jnp


def get_config():
    """
    Re=10000, N=256 的 SOAP 訓練 config。

    What:
        盡量對齊 upstream `examples/kolmogorov_flow/configs/soap.py` 的 SOAP 設定。

    Why:
        本 branch 的 `re10k` 資料只有 41 個時間點；若完全照 upstream 設成
        `num_time_windows=25`，每個 window 只剩 1 個時間點，現有 `train.py`
        會在計算 `dt = t[1] - t[0]` 時失敗。
        因此此檔僅保留一個必要差異：`training.num_time_windows = 20`。
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
    arch.reparam = ml_collections.ConfigDict({"type": "weight_fact", "mean": 1.0, "stddev": 0.1})
    arch.nonlinearity = 0.0
    arch.pi_init = None

    config.time_fraction = 1.0

    # Re10k case 仍需顯式指定資料來源；其餘共同超參數盡量與 upstream 對齊。
    config.dataset_path = (
        "examples/kolmogorov_flow/data/kolmogorov_dns/"
        "kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_ds4.npy"
    )

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

    # 與 upstream 唯一保留差異：re10k 資料僅 41 個時間點，必須至少保留 2 steps/window。
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 20000
    training.batch_size_per_device = 8192
    training.num_time_windows = 20

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
            # 本 branch 的 loss dict 固定包含 data 項；保留 0.0 以維持相容性。
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
    saving.num_keep_ckpts = None

    config.input_dim = 3
    config.seed = 42

    return config
