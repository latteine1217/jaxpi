import jax.numpy as jnp
import numpy as np
from jax import random, vmap, jit


def get_dataset(
    Re=1e4,
    time_fraction=1.0,
    dataset_path="examples/kolmogorov_flow/data/kolmogorov_dns/kolmogorov_dns_10000.npy",
    time_range=None,
    time_stride=1,
):
    from jaxpi.dataio.loaders import load_kolmogorov_dns

    u_ref, v_ref, _w_unused, omega_ref, t, coords, config = load_kolmogorov_dns(
        dataset_path,
        time_range=time_range,
        time_stride=time_stride,
        normalize_time=True,
    )
    nu = float(config.get("nu", 1.0 / Re))

    # 2D Kolmogorov flow 的第三個回傳值 `w` 不是渦度；此處應該對齊 omega_ref。
    # 為了維持既有呼叫介面，仍沿用變數名 `w_ref`，但其內容實際上是 vorticity reference。
    w_ref = omega_ref
    del _w_unused, omega_ref

    if time_fraction < 1.0:
        num_steps = int(time_fraction * t.shape[0])
        u_ref = u_ref[:num_steps]
        v_ref = v_ref[:num_steps]
        w_ref = w_ref[:num_steps]
        t = t[:num_steps]

    # 優化：原地 reshape 避免額外複製（NumPy reshape 預設返回 view）
    def _flatten_field(field):
        if field is None:
            return None
        return field.reshape((field.shape[0], -1))

    u_ref = _flatten_field(u_ref)
    v_ref = _flatten_field(v_ref)
    w_ref = _flatten_field(w_ref)

    print(
        "t.shape",
        t.shape,
        "u_ref.shape",
        u_ref.shape,
        "v_ref.shape",
        v_ref.shape,
        "omega_ref.shape",
        w_ref.shape,
    )

    return u_ref, v_ref, w_ref, t, coords, nu


@jit
def relative_l2_error_periodic(field1, field2):
    """
    Compute relative L2 error with optimal periodic shift
    Returns: min ||field1 - shifted_field2||_2 / ||field1||_2
    """
    # Compute L2 norm of reference field
    norm_ref = jnp.sqrt(jnp.sum(field1**2))

    shape = field1.shape
    min_error = jnp.inf

    def compute_error_for_shift(shift_x, shift_y):
        shifted = jnp.roll(jnp.roll(field2, shift_x, axis=0), shift_y, axis=1)
        diff = field1 - shifted
        return jnp.sqrt(jnp.sum(diff**2)) / norm_ref

    # Vectorize over all possible shifts
    shifts_x, shifts_y = jnp.meshgrid(jnp.arange(shape[0]), jnp.arange(shape[1]))
    errors = vmap(lambda x, y: compute_error_for_shift(x, y))(shifts_x.ravel(), shifts_y.ravel())

    return jnp.min(errors)
