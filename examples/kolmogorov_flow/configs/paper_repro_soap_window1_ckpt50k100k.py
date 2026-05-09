import examples.kolmogorov_flow.configs.paper_repro_soap_window1_ablation as base_config


def get_config():
    """Window-1 no-data rerun that keeps all checkpoints every 10000 steps."""
    config = base_config.get_config()

    config.wandb.name = "re1e6_n512_ds4_soap_w1_ckpt50k100k"
    config.wandb.group = "re1e6_window1_ckpt_validation"
    config.wandb.tags = list(config.wandb.tags) + ["save_every_10000", "keep_all_ckpts"]
    config.wandb.notes = (
        "No-data window-1 rerun for checkpoint validation. "
        "Save every 10000 steps and keep all checkpoints."
    )

    config.training.max_windows_to_run = 1
    config.training.max_steps = 100000
    config.saving.save_every_steps = 10000
    config.saving.num_keep_ckpts = None

    return config
