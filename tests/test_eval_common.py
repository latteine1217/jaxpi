import unittest

import numpy as np

from examples.kolmogorov_flow import eval_common


class EvalCommonTest(unittest.TestCase):
    def test_to_window_local_time_rebases_to_zero(self):
        t_abs = np.array([2.5, 2.75, 3.0], dtype=float)
        t_local = eval_common.to_window_local_time(t_abs)
        np.testing.assert_allclose(t_local, np.array([0.0, 0.25, 0.5]))

    def test_resolve_window_layout_rejects_implicit_trailing_steps(self):
        t_values = np.linspace(0.0, 5.0, 41)
        with self.assertRaisesRegex(ValueError, "trailing time steps"):
            eval_common.resolve_window_layout(t_values, num_time_windows=20)

    def test_resolve_window_layout_accepts_explicit_remainder_contract(self):
        t_values = np.linspace(0.0, 5.0, 41)
        layout = eval_common.resolve_window_layout(
            t_values,
            num_time_windows=20,
            expected_time_remainder=1,
        )
        self.assertEqual(layout.num_time_steps, 2)
        self.assertEqual(layout.time_remainder, 1)

    def test_infer_domain_lengths_reads_unit_square_from_coords(self):
        x = np.linspace(0.0, 1.0, 4, endpoint=False)
        y = np.linspace(0.0, 1.0, 4, endpoint=False)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        coords = np.stack([xx.ravel(), yy.ravel()], axis=1)

        lx, ly = eval_common.infer_domain_lengths(coords)
        self.assertAlmostEqual(lx, 1.0)
        self.assertAlmostEqual(ly, 1.0)

    def test_compute_energy_spectrum_uses_physical_domain_lengths(self):
        n = 32
        x = np.linspace(0.0, 1.0, n, endpoint=False)
        y = np.linspace(0.0, 1.0, n, endpoint=False)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        u = np.sin(2 * np.pi * xx)
        v = np.zeros_like(u)

        k, e_k = eval_common.compute_energy_spectrum(u, v, lx=1.0, ly=1.0)
        dominant_k = int(k[np.argmax(e_k)])
        self.assertEqual(dominant_k, 1)


if __name__ == "__main__":
    unittest.main()
