from functools import partial

import jax
import jax.numpy as jnp
from jax import lax, jit, grad, vmap, jacrev, hessian

import numpy as np

from jaxpi.models import ForwardIVP
from jaxpi.evaluator import BaseEvaluator


class NavierStokes(ForwardIVP):
    def __init__(
        self,
        config,
        t_star,
        coords,
        u0,
        v0,
        w0,
        nu,
        replicate_state=True,
    ):
        super().__init__(config, replicate_state=replicate_state)

        self.u0 = u0
        self.v0 = v0
        self.w0 = w0

        self.t_star = t_star
        self.coords = coords

        self.nu = nu

        # 從 config 讀取體積力參數，預設值對齊論文 2507.08972：
        #   f(x, y) = [A * sin(2π * k * y), 0]，A=0.1，k=2（在 [0,1]² 域）
        # Why: 避免硬編碼導致不同 Re/domain 實驗使用錯誤的強迫振幅與波數。
        _bf = getattr(config, "body_force", None)
        if _bf is not None:
            A = float(getattr(_bf, "amplitude", 0.1))
            k = float(getattr(_bf, "wavenumber", 2.0))
        else:
            A = 0.1
            k = 2.0
        self.body_force_fn = lambda x, y: A * jnp.sin(2 * jnp.pi * k * y)

        # 檢查是否啟用 vmap 分塊優化
        self.use_vmap_chunking = getattr(config, "optimization", None) is not None and getattr(
            config.optimization, "use_vmap_chunking", False
        )
        self.vmap_chunk_size = (
            getattr(getattr(config, "optimization", None), "vmap_chunk_size", 512)
            if self.use_vmap_chunking
            else None
        )

        # Predictions over a grid
        self.u_ic_pred_fn = vmap(self.u_net, (None, None, 0, 0))
        self.v_ic_pred_fn = vmap(self.v_net, (None, None, 0, 0))
        self.w_ic_pred_fn = vmap(self.w_net, (None, None, 0, 0))

        # 根據配置選擇預測函數版本
        if self.use_vmap_chunking:
            # 分塊版本：記憶體優化
            self.u_pred_fn = self._create_chunked_pred_fn(self.u_net)
            self.v_pred_fn = self._create_chunked_pred_fn(self.v_net)
            self.w_pred_fn = self._create_chunked_pred_fn(self.w_net)
            print(f"✅ 啟用 vmap 分塊優化 (chunk_size={self.vmap_chunk_size})")
        else:
            # 原始版本：雙層 vmap（更快但記憶體密集）
            self.u_pred_fn = vmap(vmap(self.u_net, (None, None, 0, 0)), (None, 0, None, None))
            self.v_pred_fn = vmap(vmap(self.v_net, (None, None, 0, 0)), (None, 0, None, None))
            self.w_pred_fn = vmap(vmap(self.w_net, (None, None, 0, 0)), (None, 0, None, None))

        self.r_pred_fn = vmap(self.r_net, (None, 0, 0, 0))

        self.u_data_pred_fn = vmap(self.u_net, (None, 0, 0, 0))
        self.v_data_pred_fn = vmap(self.v_net, (None, 0, 0, 0))
        self.w_data_pred_fn = vmap(self.w_net, (None, 0, 0, 0))

    def _create_chunked_pred_fn(self, net_fn):
        """
        創建記憶體優化的分塊預測函數

        策略：
        1. 時間維度：使用 lax.scan 逐時間步處理
        2. 空間維度：分塊處理，避免一次性載入所有空間點

        原始 vmap(vmap(...)):
          記憶體：O(num_time × num_space) - 完整具體化
          時間：O(1) - 無循環開銷

        分塊版本：
          記憶體：O(num_time × chunk_size) - 僅具體化分塊
          時間：O(num_chunks) - 有循環開銷

        Parameters
        ----------
        net_fn : callable
            單點預測函數 (params, t, x, y) -> scalar

        Returns
        -------
        chunked_fn : callable
            分塊預測函數 (params, t_array, x_array, y_array) -> array
        """
        chunk_size = self.vmap_chunk_size

        def chunked_fn(params, t_array, x_array, y_array):
            """
            分塊預測函數

            Parameters
            ----------
            params : pytree
                模型參數
            t_array : array, shape (num_time,)
                時間點陣列
            x_array : array, shape (num_space,)
                x 座標陣列
            y_array : array, shape (num_space,)
                y 座標陣列

            Returns
            -------
            predictions : array, shape (num_time, num_space)
                預測結果
            """
            num_space = x_array.shape[0]

            # 使用固定分塊形狀，避免 traced 區域的 Python for-loop
            num_chunks = (num_space + chunk_size - 1) // chunk_size
            pad_space = num_chunks * chunk_size - num_space

            if pad_space > 0:
                x_padded = jnp.pad(x_array, (0, pad_space), mode="constant")
                y_padded = jnp.pad(y_array, (0, pad_space), mode="constant")
            else:
                x_padded = x_array
                y_padded = y_array

            x_chunks = jnp.reshape(x_padded, (num_chunks, chunk_size))
            y_chunks = jnp.reshape(y_padded, (num_chunks, chunk_size))

            valid_mask = jnp.concatenate(
                [
                    jnp.ones((num_space,), dtype=jnp.bool_),
                    jnp.zeros((pad_space,), dtype=jnp.bool_),
                ]
            )
            mask_chunks = jnp.reshape(valid_mask, (num_chunks, chunk_size))

            def process_time_step(t_single):
                """處理單個時間步的所有空間點（分塊）"""

                def space_scan(_, xs):
                    x_chunk, y_chunk, mask_chunk = xs
                    pred_chunk = vmap(net_fn, (None, None, 0, 0))(
                        params, t_single, x_chunk, y_chunk
                    )
                    pred_chunk = pred_chunk * mask_chunk.astype(pred_chunk.dtype)
                    return None, pred_chunk

                _, pred_chunks = lax.scan(space_scan, None, (x_chunks, y_chunks, mask_chunks))
                return jnp.reshape(pred_chunks, (-1,))[:num_space]

            # 使用 lax.scan 處理時間維度（記憶體高效）
            def scan_body(carry, t_single):
                pred_t = process_time_step(t_single)
                return carry, pred_t

            _, predictions = jax.lax.scan(scan_body, None, t_array)

            return predictions

        return chunked_fn

    def neural_net(self, params, t, x, y):
        t = t / self.t_star[-1]
        if jnp.ndim(t) == 0:
            t = jnp.broadcast_to(t, x.shape)
        z = jnp.stack([t, x, y], axis=-1)

        if z.ndim == 1:
            z_in = z[None, :]
            _, outputs = self.state.apply_fn(params, z_in)
            outputs = outputs[0]
        else:
            z_in = jnp.reshape(z, (-1, z.shape[-1]))
            _, outputs = self.state.apply_fn(params, z_in)
            outputs = jnp.reshape(outputs, z.shape[:-1] + (-1,))

        u = outputs[..., 0]
        v = outputs[..., 1]
        p = outputs[..., 2]
        return u, v, p

    def u_net(self, params, t, x, y):
        u, _, _ = self.neural_net(params, t, x, y)
        return u

    def v_net(self, params, t, x, y):
        _, v, _ = self.neural_net(params, t, x, y)
        return v

    def p_net(self, params, t, x, y):
        _, _, p = self.neural_net(params, t, x, y)
        return p

    def w_net(self, params, t, x, y):
        """
        What:
            計算單一時空點的渦度 w = dv/dx - du/dy。
        Why:
            `w_net` 必須維持和 `u_net` / `v_net` 相同的單點函數介面，
            讓外層 `vmap` 統一負責 batch 向量化；若在此再次 `vmap`，
            會在外層已傳入 scalar 時觸發 shape 錯誤。
        """
        u_y = grad(self.u_net, argnums=3)(params, t, x, y)
        v_x = grad(self.v_net, argnums=2)(params, t, x, y)
        return v_x - u_y

    def r_net(self, params, t, x, y):
        u, v, p = self.neural_net(params, t, x, y)

        (u_t, u_x, u_y), (v_t, v_x, v_y), (_, p_x, p_y) = jacrev(
            self.neural_net, argnums=(1, 2, 3)
        )(params, t, x, y)

        u_hessian = hessian(self.u_net, argnums=(2, 3))(params, t, x, y)
        v_hessian = hessian(self.v_net, argnums=(2, 3))(params, t, x, y)

        u_xx = u_hessian[0][0]
        u_yy = u_hessian[1][1]

        v_xx = v_hessian[0][0]
        v_yy = v_hessian[1][1]

        body_force = self.body_force_fn(x, y)

        # PDE residual
        ru = u_t + u * u_x + v * u_y + p_x - self.nu * (u_xx + u_yy) - body_force
        rv = v_t + u * v_x + v * v_y + p_y - self.nu * (v_xx + v_yy)
        rc = u_x + v_y

        return ru, rv, rc

    @partial(jit, static_argnums=(0,))
    def res_and_w(self, params, batch):
        batch = jnp.reshape(batch, (-1, batch.shape[-1]))
        batch_size = batch.shape[0]

        # 安全的 batch size 處理：如果不能被 num_chunks 整除，裁切到最接近的整除值
        # 這樣可以避免因 batch size 微小差異導致的訓練中斷
        safe_batch_size = (batch_size // self.num_chunks) * self.num_chunks

        if safe_batch_size < batch_size:
            # 裁切 batch 到安全大小
            # 警告：這會在每次 JIT 編譯時打印，但不會影響性能
            batch = batch[:safe_batch_size, :]
            batch_size = safe_batch_size

        # 在多 GPU 環境下，每個設備獨立計算 causal weight，因此每個設備上的
        # batch_size_per_device 必須能被 num_chunks 整除
        # 上面的裁切邏輯已經確保了這一點
        chunk_size = batch_size // self.num_chunks

        # Sort temporal coordinates
        t_sorted = jnp.sort(batch[:, 0])
        ru_pred, rv_pred, rc_pred = self.r_pred_fn(params, t_sorted, batch[:, 1], batch[:, 2])

        # 現在可以安全地 reshape，因為 batch_size 已經是 num_chunks 的整數倍
        ru_pred = ru_pred.reshape(self.num_chunks, -1)
        rv_pred = rv_pred.reshape(self.num_chunks, -1)
        rc_pred = rc_pred.reshape(self.num_chunks, -1)

        ru_l = jnp.mean(ru_pred**2, axis=1)
        rv_l = jnp.mean(rv_pred**2, axis=1)
        rc_l = jnp.mean(rc_pred**2, axis=1)

        ru_gamma = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ ru_l)))
        rv_gamma = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ rv_l)))
        rc_gamma = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ rc_l)))

        # Take minimum of the causal weights
        gamma = jnp.vstack([ru_gamma, rv_gamma, rc_gamma])
        gamma = gamma.min(0)

        return ru_l, rv_l, rc_l, gamma

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        # Unpack batch
        ics_batch = batch["ics"]
        res_batch = batch["res"]

        # Initial condition loss
        coords_batch, u_batch, v_batch, w_batch = ics_batch

        # Initial conditions loss
        u_ic_pred = self.u_ic_pred_fn(params, 0.0, coords_batch[..., 0], coords_batch[..., 1])
        v_ic_pred = self.v_ic_pred_fn(params, 0.0, coords_batch[..., 0], coords_batch[..., 1])
        # w_ic_pred = self.w_ic_pred_fn(params, 0.0, coords_batch[..., 0], coords_batch[..., 1])

        u_ic_loss = jnp.mean((u_ic_pred - u_batch) ** 2)
        v_ic_loss = jnp.mean((v_ic_pred - v_batch) ** 2)
        # w_ic_loss = jnp.mean((w_ic_pred - w_batch) ** 2)

        # residual loss
        if self.config.weighting.use_causal == True:
            ru_l, rv_l, rc_l, gamma = self.res_and_w(params, res_batch)
            ru_loss = jnp.mean(gamma * ru_l)
            rv_loss = jnp.mean(gamma * rv_l)
            rc_loss = jnp.mean(gamma * rc_l)
        else:
            # 非 causal 路徑直接對殘差 batch 做逐點 residual 計算。
            res_batch = jnp.reshape(res_batch, (-1, res_batch.shape[-1]))
            ru_pred, rv_pred, rc_pred = self.r_pred_fn(
                params, res_batch[:, 0], res_batch[:, 1], res_batch[:, 2]
            )

            ru_loss = jnp.mean(ru_pred**2)
            rv_loss = jnp.mean(rv_pred**2)
            rc_loss = jnp.mean(rc_pred**2)

        data_batch = batch.get("data")
        if data_batch is None:
            u_data_loss = jnp.array(0.0)
            v_data_loss = jnp.array(0.0)
            w_data_loss = jnp.array(0.0)
        else:
            t_data, coords_data, u_data, v_data, w_data = data_batch
            u_pred = self.u_data_pred_fn(params, t_data, coords_data[..., 0], coords_data[..., 1])
            v_pred = self.v_data_pred_fn(params, t_data, coords_data[..., 0], coords_data[..., 1])

            u_data_loss = jnp.mean((u_pred - u_data) ** 2)
            v_data_loss = jnp.mean((v_pred - v_data) ** 2)
            if self.config.get("use_vorticity_data_loss", True):
                w_pred = self.w_data_pred_fn(
                    params, t_data, coords_data[..., 0], coords_data[..., 1]
                )
                w_data_loss = jnp.mean((w_pred - w_data) ** 2)
            else:
                w_data_loss = jnp.array(0.0)

        loss_dict = {
            "u_ic": u_ic_loss,
            "v_ic": v_ic_loss,
            "ru": ru_loss,
            "rv": rv_loss,
            "rc": rc_loss,
            "u_data": u_data_loss,
            "v_data": v_data_loss,
            "w_data": w_data_loss,
        }
        return loss_dict

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, t, coords, u_ref, v_ref, w_ref):
        u_pred = self.u_pred_fn(params, t, coords[:, 0], coords[:, 1])
        v_pred = self.v_pred_fn(params, t, coords[:, 0], coords[:, 1])
        w_pred = self.w_pred_fn(params, t, coords[:, 0], coords[:, 1])

        u_error = jnp.linalg.norm(u_pred - u_ref) / jnp.linalg.norm(u_ref)
        v_error = jnp.linalg.norm(v_pred - v_ref) / jnp.linalg.norm(v_ref)
        w_error = jnp.linalg.norm(w_pred - w_ref) / jnp.linalg.norm(w_ref)

        return u_error, v_error, w_error

    def _compute_l2_error_time_space_chunked_impl(
        self,
        params,
        t,
        coords,
        u_ref,
        v_ref,
        w_ref,
        time_chunk_size,
        space_chunk_size,
    ):
        """
        以固定 time/space 分塊計算全時序 L2 誤差。

        Why:
            以固定尺寸 chunk + lax.scan 取代 Python while/for，
            讓評估路徑可 JIT/pjit 化並減少 host-device 來回。
        """
        num_time = t.shape[0]
        num_space = coords.shape[0]

        num_time_chunks = (num_time + time_chunk_size - 1) // time_chunk_size
        num_space_chunks = (num_space + space_chunk_size - 1) // space_chunk_size
        pad_time = num_time_chunks * time_chunk_size - num_time
        pad_space = num_space_chunks * space_chunk_size - num_space

        if pad_time > 0:
            t_padded = jnp.pad(t, (0, pad_time), mode="edge")
        else:
            t_padded = t

        if pad_time > 0 or pad_space > 0:
            u_padded = jnp.pad(u_ref, ((0, pad_time), (0, pad_space)), mode="constant")
            v_padded = jnp.pad(v_ref, ((0, pad_time), (0, pad_space)), mode="constant")
            w_padded = jnp.pad(w_ref, ((0, pad_time), (0, pad_space)), mode="constant")
            coords_padded = jnp.pad(coords, ((0, pad_space), (0, 0)), mode="constant")
        else:
            u_padded = u_ref
            v_padded = v_ref
            w_padded = w_ref
            coords_padded = coords

        t_chunks = jnp.reshape(t_padded, (num_time_chunks, time_chunk_size))
        coords_chunks = jnp.reshape(coords_padded, (num_space_chunks, space_chunk_size, 2))

        u_time_space_chunks = jnp.reshape(
            u_padded,
            (num_time_chunks, time_chunk_size, num_space_chunks, space_chunk_size),
        )
        v_time_space_chunks = jnp.reshape(
            v_padded,
            (num_time_chunks, time_chunk_size, num_space_chunks, space_chunk_size),
        )
        w_time_space_chunks = jnp.reshape(
            w_padded,
            (num_time_chunks, time_chunk_size, num_space_chunks, space_chunk_size),
        )

        time_mask = jnp.concatenate(
            [
                jnp.ones((num_time,), dtype=jnp.bool_),
                jnp.zeros((pad_time,), dtype=jnp.bool_),
            ]
        )
        space_mask = jnp.concatenate(
            [
                jnp.ones((num_space,), dtype=jnp.bool_),
                jnp.zeros((pad_space,), dtype=jnp.bool_),
            ]
        )
        time_mask_chunks = jnp.reshape(time_mask, (num_time_chunks, time_chunk_size))
        space_mask_chunks = jnp.reshape(space_mask, (num_space_chunks, space_chunk_size))

        dtype = u_ref.dtype
        init_carry = (
            jnp.array(0.0, dtype=dtype),
            jnp.array(0.0, dtype=dtype),
            jnp.array(0.0, dtype=dtype),
            jnp.array(0.0, dtype=dtype),
            jnp.array(0.0, dtype=dtype),
            jnp.array(0.0, dtype=dtype),
        )

        def time_scan(carry, xs):
            total_u, total_v, total_w, denom_u, denom_v, denom_w = carry
            t_chunk, time_mask_chunk, u_chunk, v_chunk, w_chunk = xs

            # 將空間塊移到第一維，便於 scan
            u_space_first = jnp.swapaxes(u_chunk, 0, 1)
            v_space_first = jnp.swapaxes(v_chunk, 0, 1)
            w_space_first = jnp.swapaxes(w_chunk, 0, 1)

            def space_scan(space_carry, space_xs):
                su, sv, sw, du, dv, dw = space_carry
                coords_chunk, space_mask_chunk, u_local, v_local, w_local = space_xs

                u_pred = self.u_pred_fn(params, t_chunk, coords_chunk[:, 0], coords_chunk[:, 1])
                v_pred = self.v_pred_fn(params, t_chunk, coords_chunk[:, 0], coords_chunk[:, 1])
                w_pred = self.w_pred_fn(params, t_chunk, coords_chunk[:, 0], coords_chunk[:, 1])

                valid = jnp.logical_and(time_mask_chunk[:, None], space_mask_chunk[None, :])
                valid = valid.astype(u_pred.dtype)

                u_diff = (u_pred - u_local) * valid
                v_diff = (v_pred - v_local) * valid
                w_diff = (w_pred - w_local) * valid
                u_valid = u_local * valid
                v_valid = v_local * valid
                w_valid = w_local * valid

                su = su + jnp.sum(u_diff**2)
                sv = sv + jnp.sum(v_diff**2)
                sw = sw + jnp.sum(w_diff**2)
                du = du + jnp.sum(u_valid**2)
                dv = dv + jnp.sum(v_valid**2)
                dw = dw + jnp.sum(w_valid**2)
                return (su, sv, sw, du, dv, dw), None

            (total_u, total_v, total_w, denom_u, denom_v, denom_w), _ = lax.scan(
                space_scan,
                (total_u, total_v, total_w, denom_u, denom_v, denom_w),
                (
                    coords_chunks,
                    space_mask_chunks,
                    u_space_first,
                    v_space_first,
                    w_space_first,
                ),
            )
            return (total_u, total_v, total_w, denom_u, denom_v, denom_w), None

        (total_u, total_v, total_w, denom_u, denom_v, denom_w), _ = lax.scan(
            time_scan,
            init_carry,
            (
                t_chunks,
                time_mask_chunks,
                u_time_space_chunks,
                v_time_space_chunks,
                w_time_space_chunks,
            ),
        )

        eps = jnp.array(jnp.finfo(dtype).eps, dtype=dtype)
        u_error = jnp.sqrt(total_u / jnp.maximum(denom_u, eps))
        v_error = jnp.sqrt(total_v / jnp.maximum(denom_v, eps))
        w_error = jnp.sqrt(total_w / jnp.maximum(denom_w, eps))
        return u_error, v_error, w_error

    @partial(jit, static_argnums=(0, 7, 8))
    def compute_l2_error_time_space_chunked(
        self,
        params,
        t,
        coords,
        u_ref,
        v_ref,
        w_ref,
        time_chunk_size,
        space_chunk_size,
    ):
        return self._compute_l2_error_time_space_chunked_impl(
            params,
            t,
            coords,
            u_ref,
            v_ref,
            w_ref,
            time_chunk_size,
            space_chunk_size,
        )


class NavierStokesEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def _resolve_eval_chunk_sizes(self, t):
        """
        解析評估 chunk 大小，統一使用固定 time/space chunk 路徑。
        """
        time_chunk_size = getattr(self.config.logging, "eval_time_chunk_size", None)
        if time_chunk_size is None:
            chunk_seconds = float(getattr(self.config.logging, "eval_time_chunk_seconds", 1.0))
            t_values = np.asarray(t)
            if t_values.size > 1:
                t_sorted = np.sort(t_values)
                dt = float(t_sorted[1] - t_sorted[0])
                if dt > 0:
                    time_chunk_size = max(1, int(chunk_seconds / dt))
                else:
                    time_chunk_size = 1
            else:
                time_chunk_size = 1
        time_chunk_size = int(time_chunk_size)
        if time_chunk_size < 1:
            time_chunk_size = 1

        space_chunk_size = int(getattr(self.config.logging, "eval_space_chunk_size", 4096))
        if space_chunk_size < 1:
            space_chunk_size = 4096

        return time_chunk_size, space_chunk_size

    def log_errors(self, params, t, coords, u_ref, v_ref, w_ref):
        time_chunk_size, space_chunk_size = self._resolve_eval_chunk_sizes(t)
        u_error, v_error, w_error = self.model.compute_l2_error_time_space_chunked(
            params,
            t,
            coords,
            u_ref,
            v_ref,
            w_ref,
            time_chunk_size=time_chunk_size,
            space_chunk_size=space_chunk_size,
        )
        self.log_dict["u_error"] = u_error
        self.log_dict["v_error"] = v_error
        self.log_dict["w_error"] = w_error

    def __call__(self, state, batch, t, coords, u_ref, v_ref, w_ref):
        self.log_dict = super().__call__(state, batch)

        if self.config.logging.log_errors:
            self.log_errors(state.params, t, coords, u_ref, v_ref, w_ref)

        if self.config.weighting.use_causal:
            _, _, _, causal_weight = self.model.res_and_w(state.params, batch["res"])
            self.log_dict["cas_weight"] = causal_weight.min()

        return self.log_dict
