import examples.kolmogorov_flow.configs.paper_repro_soap_sensor100_n512_w50_window1_ablation as base_config


def get_config():
    """
    What:
        固定 `data_weight=23.1429` 的 window-1 驗證 config。
    Why:
        `3318` sweep 只提供 threshold-crossing 證據，沒有保留 checkpoint；
        這次單獨重跑要把最佳權重固定下來，並保留足夠密度的 checkpoint
        供後續 corrected evaluation 驗證是否真的收斂。
    """

    config = base_config.get_config()

    config.wandb.name = "re1e6_n512_ds4_soap_sensor100_w50_w1_dw231429_eval"
    config.wandb.group = "re1e6_window1_fixed_weight_eval"
    config.wandb.tags = list(config.wandb.tags) + ["fixed_data_weight", "dw=23.1429", "checkpoint_eval"]
    config.wandb.notes = (
        "Fixed-weight validation rerun after sweep 3318. "
        "Use the best observed window-1 data weight (23.1429), save checkpoints every 1000 steps, "
        "and stop after window 1 for corrected checkpoint evaluation."
    )

    config.training.max_windows_to_run = 1
    config.training.max_steps = 50000

    config.weighting.init_weights.u_data = 23.1429
    config.weighting.init_weights.v_data = 23.1429

    config.saving.save_every_steps = 1000
    config.saving.num_keep_ckpts = None

    return config
