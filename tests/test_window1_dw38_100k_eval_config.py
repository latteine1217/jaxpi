"""
What:
    驗證 window-1 `dw=38.0614` 100k 步延伸 config 的關鍵行為。
Why:
    此 config 用來回答「sensor 是否在後期長期訓練下能超越 no-data 100k」。
    若 max_steps 被誤改回 50k 或 save_every_steps 被改變，會破壞與
    既有 no-data 100k baseline 的 apples-to-apples 比較。
"""

import sys
import types
import unittest


def _install_stub_modules():
    if "jax" not in sys.modules:
        jax_mod = types.ModuleType("jax")
        jnp_mod = types.ModuleType("jax.numpy")
        jnp_mod.pi = 3.141592653589793
        jax_mod.numpy = jnp_mod
        sys.modules["jax"] = jax_mod
        sys.modules["jax.numpy"] = jnp_mod

    if "ml_collections" not in sys.modules:
        mlc = types.ModuleType("ml_collections")

        class ConfigDict(dict):
            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError as exc:
                    raise AttributeError(name) from exc

            def __setattr__(self, name, value):
                self[name] = value

        mlc.ConfigDict = ConfigDict
        sys.modules["ml_collections"] = mlc


_install_stub_modules()


class Window1Dw38_100kEvalConfigTest(unittest.TestCase):
    def test_config_extends_to_100k_with_dense_checkpoints(self):
        from examples.kolmogorov_flow.configs import (
            paper_repro_soap_sensor100_n512_w50_window1_dw38_100k_eval as cfg_mod,
        )

        config = cfg_mod.get_config()

        self.assertEqual(config.training.max_windows_to_run, 1)
        self.assertEqual(config.training.max_steps, 100000)
        self.assertEqual(config.weighting.init_weights.u_data, 38.0614)
        self.assertEqual(config.weighting.init_weights.v_data, 38.0614)
        self.assertEqual(config.saving.save_every_steps, 1000)
        self.assertIsNone(config.saving.num_keep_ckpts)
        self.assertEqual(
            config.wandb.name,
            "re1e6_n512_ds4_soap_sensor100_w50_w1_dw38_100k_eval",
        )
        self.assertIn("max_steps_100k", config.wandb.tags)
        self.assertIn("sweep_3400_rank1", config.wandb.tags)


if __name__ == "__main__":
    unittest.main()
