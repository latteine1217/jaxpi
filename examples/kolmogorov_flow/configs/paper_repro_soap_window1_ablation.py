import examples.kolmogorov_flow.configs.paper_repro_soap as base_config


def get_config():
    """Window-1-only no-data ablation for convergence-speed comparison."""
    config = base_config.get_config()

    config.wandb.name = "re1e6_n512_ds4_soap_w1_ablation"
    config.wandb.group = "re1e6_window1_convergence"
    config.wandb.tags = list(config.wandb.tags) + ["window1_only", "convergence_ablation"]
    config.wandb.notes = (
        "No-data baseline rerun for convergence-speed comparison. "
        "Keep the original 50-window time discretization, but stop after window 1."
    )

    config.training.max_windows_to_run = 1
    return config
