import examples.kolmogorov_flow.configs.paper_repro_soap_sensor100_n512_w50_window1_ablation as base_config


def get_config():
    """
    What:
        固定 `data_weight=38.0614` 的 window-1 100k-step 延伸驗證 config。
    Why:
        `3481` (50k 步) 的 corrected eval 顯示 sensor (dw=38) 與 no-data
        在 10k~50k 步區間幾乎無差別；且 sensor 50k (u=1.21e-3, v=1.11e-3,
        w=0.82e-3) **未達** no-data 100k (u=1.08e-3, v=1.09e-3, w=0.70e-3)
        的精度。要回答「sensor 是否在長期仍能超越 no-data」必須把 sensor
        也訓練到 100k 並用同樣 1k step 密度的 checkpoint 軌跡比較。
        若 sensor 100k 仍劣於 no-data 100k → sparse sensor 對 window-1
        確認沒有加速益處；若 sensor 100k 顯著贏 → 存在後期加速效應。
    """

    config = base_config.get_config()

    config.wandb.name = "re1e6_n512_ds4_soap_sensor100_w50_w1_dw38_100k_eval"
    config.wandb.group = "re1e6_window1_fixed_weight_eval"
    config.wandb.tags = list(config.wandb.tags) + [
        "fixed_data_weight",
        "dw=38.0614",
        "sweep_3400_rank1",
        "checkpoint_eval",
        "max_steps_100k",
    ]
    config.wandb.notes = (
        "100k-step extension of 3481 (dw=38.0614). "
        "Save checkpoints every 1000 steps (100 ckpts total) to enable "
        "apples-to-apples comparison with no-data 100k baseline at full "
        "training horizon."
    )

    config.training.max_windows_to_run = 1
    config.training.max_steps = 100000

    config.weighting.init_weights.u_data = 38.0614
    config.weighting.init_weights.v_data = 38.0614

    config.saving.save_every_steps = 1000
    config.saving.num_keep_ckpts = None

    return config
