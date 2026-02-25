from functools import partial

import jax
import jax.numpy as jnp
from jax import lax, jit, grad, vmap, jacrev, hessian
from jax.tree_util import tree_map

import numpy as np
import optax

from jaxpi import archs
from jaxpi.models import ForwardIVP
from jaxpi.evaluator import BaseEvaluator
from jaxpi.utils import ntk_fn


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

        self.body_force_fn = lambda x, y: 2 * jnp.sin(4 * jnp.pi * y)

        # 檢查是否啟用 vmap 分塊優化
        self.use_vmap_chunking = getattr(config, 'optimization', None) is not None and \
                                  getattr(config.optimization, 'use_vmap_chunking', False)
        self.vmap_chunk_size = getattr(getattr(config, 'optimization', None), 'vmap_chunk_size', 512) \
                               if self.use_vmap_chunking else None

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
            num_time = t_array.shape[0]
            num_space = x_array.shape[0]

            # 計算需要的分塊數
            num_chunks = (num_space + chunk_size - 1) // chunk_size

            def process_time_step(t_single):
                """處理單個時間步的所有空間點（分塊）"""
                # 初始化結果陣列
                results = []

                for chunk_idx in range(num_chunks):
                    start_idx = chunk_idx * chunk_size
                    end_idx = jnp.minimum(start_idx + chunk_size, num_space)

                    # 取得當前分塊的空間座標
                    x_chunk = x_array[start_idx:end_idx]
                    y_chunk = y_array[start_idx:end_idx]

                    # 使用單層 vmap 處理空間分塊（可控記憶體）
                    chunk_pred = vmap(net_fn, (None, None, 0, 0))(
                        params, t_single, x_chunk, y_chunk
                    )

                    results.append(chunk_pred)

                # 合併所有分塊結果
                return jnp.concatenate(results, axis=0)

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
        def _u_y(t_scalar, x_scalar, y_scalar):
            return grad(self.u_net, argnums=3)(params, t_scalar, x_scalar, y_scalar)

        def _v_x(t_scalar, x_scalar, y_scalar):
            return grad(self.v_net, argnums=2)(params, t_scalar, x_scalar, y_scalar)

        u_y = vmap(_u_y)(t, x, y)
        v_x = vmap(_v_x)(t, x, y)
        w = v_x - u_y
        return w

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
            ru_loss = jnp.mean(ru_l)
            rv_loss = jnp.mean(rv_l)
            rc_loss = jnp.mean(rc_l)
        else:
            # For rare cases where weighting_scheme is not grad_norm
            # and res_and_w is not used
            ru_pred, rv_pred, rc_pred = self.res_fn(
                params, res_batch[..., 0], res_batch[..., 1], res_batch[..., 2]
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
            w_pred = self.w_data_pred_fn(params, t_data, coords_data[..., 0], coords_data[..., 1])

            u_data_loss = jnp.mean((u_pred - u_data) ** 2)
            v_data_loss = jnp.mean((v_pred - v_data) ** 2)
            w_data_loss = jnp.mean((w_pred - w_data) ** 2)

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

    def compute_l2_error_time_chunked(
        self,
        params,
        t,
        coords,
        u_ref,
        v_ref,
        w_ref,
        chunk_seconds=1.0,
    ):
        t_values = np.asarray(t)
        total_u = jnp.array(0.0)
        total_v = jnp.array(0.0)
        total_w = jnp.array(0.0)
        denom_u = jnp.array(0.0)
        denom_v = jnp.array(0.0)
        denom_w = jnp.array(0.0)

        start = 0
        time_count = t_values.shape[0]
        while start < time_count:
            t0 = t_values[start]
            end = start + 1
            while end < time_count and (t_values[end] - t0) < chunk_seconds:
                end += 1

            t_chunk = t[start:end]
            u_ref_chunk = u_ref[start:end, :]
            v_ref_chunk = v_ref[start:end, :]
            w_ref_chunk = w_ref[start:end, :]

            u_pred = self.u_pred_fn(params, t_chunk, coords[:, 0], coords[:, 1])
            v_pred = self.v_pred_fn(params, t_chunk, coords[:, 0], coords[:, 1])
            w_pred = self.w_pred_fn(params, t_chunk, coords[:, 0], coords[:, 1])

            total_u = total_u + jnp.sum((u_pred - u_ref_chunk) ** 2)
            total_v = total_v + jnp.sum((v_pred - v_ref_chunk) ** 2)
            total_w = total_w + jnp.sum((w_pred - w_ref_chunk) ** 2)
            denom_u = denom_u + jnp.sum(u_ref_chunk**2)
            denom_v = denom_v + jnp.sum(v_ref_chunk**2)
            denom_w = denom_w + jnp.sum(w_ref_chunk**2)

            start = end

        u_error = jnp.sqrt(total_u) / jnp.sqrt(denom_u)
        v_error = jnp.sqrt(total_v) / jnp.sqrt(denom_v)
        w_error = jnp.sqrt(total_w) / jnp.sqrt(denom_w)
        return u_error, v_error, w_error


class NavierStokesEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, t, coords, u_ref, v_ref, w_ref):
        chunk_seconds = getattr(self.config.logging, "eval_time_chunk_seconds", 1.0)
        u_error, v_error, w_error = self.model.compute_l2_error_time_chunked(
            params,
            t,
            coords,
            u_ref,
            v_ref,
            w_ref,
            chunk_seconds=chunk_seconds,
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
