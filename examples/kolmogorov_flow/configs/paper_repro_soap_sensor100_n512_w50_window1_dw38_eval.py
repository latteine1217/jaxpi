import examples.kolmogorov_flow.configs.paper_repro_soap_sensor100_n512_w50_window1_ablation as base_config


def get_config():
    """
    What:
        固定 `data_weight=38.0614` 的 window-1 驗證 config。
    Why:
        `3400` sweep（`5to80_thr5e5_k3_100k_stopckpt`，60 trials 全 COMPLETE）
        以「earliest k=3 stable threshold-crossing」為 score，rank 1 trial #0
        在 step 49,400 達標，對應 `data_weight = 38.0614`。
        sweep 只回答 residual health signal；要回答 corrected field error
        是否真的加速，必須單獨重跑並保留 1k step 密度的 checkpoint 軌跡，
        與 `no-data` (`paper_repro_soap_window1_ablation`) 與
        `dw=23.1429` (`3324`) 對照。
    """

    config = base_config.get_config()

    config.wandb.name = "re1e6_n512_ds4_soap_sensor100_w50_w1_dw38_eval"
    config.wandb.group = "re1e6_window1_fixed_weight_eval"
    config.wandb.tags = list(config.wandb.tags) + [
        "fixed_data_weight",
        "dw=38.0614",
        "sweep_3400_rank1",
        "checkpoint_eval",
    ]
    config.wandb.notes = (
        "Fixed-weight validation rerun after sweep 3400. "
        "Use the rank-1 window-1 data weight (38.0614, first_stable_step=49400 in 3400), "
        "save checkpoints every 1000 steps, and stop after window 1 for corrected "
        "checkpoint evaluation comparing no-data vs with-data convergence."
    )

    config.training.max_windows_to_run = 1
    config.training.max_steps = 50000

    config.weighting.init_weights.u_data = 38.0614
    config.weighting.init_weights.v_data = 38.0614

    config.saving.save_every_steps = 1000
    config.saving.num_keep_ckpts = None

    return config
