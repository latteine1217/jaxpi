from functools import partial
from typing import Any, Callable, Sequence, Tuple, Optional, Dict

from flax.training import train_state
from flax import jax_utils

import jax.numpy as jnp
from jax import lax, jit, grad, pmap, random, jacfwd, jacrev
from jax.tree_util import tree_map, tree_reduce, tree_leaves

import optax

from jaxpi import archs
from jaxpi.utils import flatten_pytree

# Apply compatibility patch for soap_jax before importing it
try:
    from jaxpi import optax_soap_patch  # Patch for optax 0.1.9 compatibility
except ImportError:
    pass  # Patch module not found, continue anyway

try:
    from soap_jax import soap  # Install from https://github.com/haydn-jones/SOAP_JAX
except ImportError:  # Optional dependency
    soap = None

try:
    from psgd_jax.kron import kron
except ImportError:  # Optional dependency
    kron = None


class TrainState(train_state.TrainState):
    weights: Dict
    momentum: float

    def apply_weights(self, weights, **kwargs):
        """Updates `weights` using running average  in return value.

        Returns:
          An updated instance of `self` with new weights updated by applying `running_average`,
          and additional attributes replaced as specified by `kwargs`.
        """

        def running_average(old_w, new_w):
            candidate = old_w * self.momentum + (1 - self.momentum) * new_w
            # 非有限值代表 adaptive weighting 已失穩，保留舊權重避免整個 state 被污染。
            return jnp.where(jnp.isfinite(candidate), candidate, old_w)

        weights = tree_map(running_average, self.weights, weights)
        weights = lax.stop_gradient(weights)

        return self.replace(
            step=self.step,
            params=self.params,
            opt_state=self.opt_state,
            weights=weights,
            **kwargs,
        )


def _create_arch(config):
    if config.arch_name == "Mlp":
        arch = archs.Mlp(**config)

    elif config.arch_name == "ResNet":
        arch = archs.ResNet(**config)

    elif config.arch_name == "ModifiedMlp":
        arch = archs.ModifiedMlp(**config)

    elif config.arch_name == "PIResNet":
        arch = archs.PIResNet(**config)

    elif config.arch_name == "PirateNet":
        arch = archs.PirateNet(**config)

    elif config.arch_name == "DeepONet":
        arch = archs.DeepONet(**config)

    else:
        raise NotImplementedError(f"Arch {config.arch_name} not supported yet!")

    return arch


def _create_optimizer(config):

    lr = optax.exponential_decay(
        init_value=config.learning_rate,
        transition_steps=config.decay_steps,
        decay_rate=config.decay_rate,
        staircase=config.staircase
        )

    if config.warmup_steps > 0:
        warmup = optax.linear_schedule(init_value=0.0, end_value=config.learning_rate,
                                       transition_steps=config.warmup_steps)

        lr = optax.join_schedules([warmup, lr], [config.warmup_steps])

    if config.optimizer == "Adam":
        tx = optax.adam(
            learning_rate=lr, b1=config.beta1, b2=config.beta2, eps=config.eps
        )

    elif config.optimizer == "Soap":

        tx = soap(
            learning_rate=lr, b1=config.beta1, b2=config.beta2, weight_decay=0.0, precondition_frequency=2
            )


    elif config.optimizer == "Kron":
            tx = kron(
                learning_rate=lr, b1=config.beta1
                )

    elif config.optimizer == "Muon":
        tx = optax.contrib.muon(
            learning_rate=lr,
            ns_coeffs=(2, -1.5, 0.5),
            ns_steps=10,
            beta=0.99,
            adam_b1=0.99
        )

    elif config.optimizer == "Lamb":
        tx = optax.lamb(
            learning_rate=lr, b1=config.beta1, b2=config.beta2, eps=config.eps
        )

    elif config.optimizer == "Adagrad":
        tx = optax.adagrad(
            learning_rate=lr, eps=config.eps
        )

    elif config.optimizer == "RMSProp":
        tx = optax.rmsprop(
            learning_rate=lr
        )

    if config.schedule_free:
        tx = optax.contrib.schedule_free(tx, lr, b1=config.beta1)

    grad_clip_norm = getattr(config, "grad_clip_norm", None)
    if grad_clip_norm is not None and float(grad_clip_norm) > 0:
        tx = optax.chain(optax.clip_by_global_norm(float(grad_clip_norm)), tx)

    # Gradient accumulation
    if config.grad_accum_steps > 1:
        tx = optax.MultiSteps(tx, every_k_schedule=config.grad_accum_steps)

    return lr, tx


def _create_train_state(
    config, params=None, weights=None, opt_state=None, step=None, replicate=True
):
    """
    建立TrainState，支援可選的優化器狀態遷移
    
    Parameters
    ----------
    config : ml_collections.ConfigDict
        訓練配置
    params : Optional[FrozenDict]
        模型參數（若為None則隨機初始化）
    weights : Optional[Dict]
        損失權重
    opt_state : Optional[optax.OptState]
        優化器狀態（若提供則直接使用，否則初始化）
    step : Optional[int]
        訓練步數（若為None則從0開始）
    
    Returns
    -------
    TrainState (replicated across devices)
    """
    # Initialize network
    arch = _create_arch(config.arch)
    x = jnp.ones((1, config.input_dim))

    # Initialize optax optimizer
    lr, tx = _create_optimizer(config.optim)

    if params is None:
        params = arch.init(random.PRNGKey(config.seed), x)

    if weights is None:
        weights = dict(config.weighting.init_weights)

    # 核心分支：根據是否提供opt_state選擇建立方式
    if opt_state is not None:
        # 路徑A：使用提供的opt_state（遷移學習）
        import logging
        logger = logging.getLogger(__name__)
        logger.info("建立TrainState並使用提供的opt_state（遷移學習模式）")
        
        # 方案1：嘗試直接構造
        try:
            state = TrainState(
                step=step if step is not None else 0,
                apply_fn=arch.apply,
                params=params,
                tx=tx,
                opt_state=opt_state,
                weights=weights,
                momentum=config.weighting.momentum,
            )
        except TypeError as e:
            # 方案2：若不支援直接構造，使用replace()
            logger.warning(f"直接構造TrainState失敗: {e}")
            logger.info("回退至TrainState.create() + replace()")
            temp_state = TrainState.create(
                apply_fn=arch.apply,
                params=params,
                tx=tx,
                weights=weights,
                momentum=config.weighting.momentum,
            )
            state = temp_state.replace(
                opt_state=opt_state,
                step=step if step is not None else 0
            )
    else:
        # 路徑B：標準初始化（現有行為）
        state = TrainState.create(
            apply_fn=arch.apply,
            params=params,
            tx=tx,
            weights=weights,
            momentum=config.weighting.momentum,
        )

    if replicate:
        return jax_utils.replicate(state)
    return state


class PINN:
    def __init__(self, config, replicate_state=True):
        self.config = config
        self.state = _create_train_state(config, replicate=replicate_state)

    def u_net(self, params, *args):
        raise NotImplementedError("Subclasses should implement this!")

    def r_net(self, params, *args):
        raise NotImplementedError("Subclasses should implement this!")

    def losses(self, params, batch, *args):
        raise NotImplementedError("Subclasses should implement this!")

    def compute_diag_ntk(self, params, batch, *args):
        raise NotImplementedError("Subclasses should implement this!")

    @partial(jit, static_argnums=(0,))
    def loss(self, params, weights, batch, *args):
        # Compute losses
        losses = self.losses(params, batch, *args)
        # Compute weighted loss
        weighted_losses = tree_map(lambda x, y: x * y, losses, weights)
        # Sum weighted losses
        loss = tree_reduce(lambda x, y: x + y, weighted_losses)
        return loss

    @partial(jit, static_argnums=(0,))
    def compute_weights(self, params, batch, *args):
        if self.config.weighting.scheme == "grad_norm":
            # Compute the grad norm of each loss w.r.t. the parameters serially
            # to avoid materializing the full Jacobian of all losses at once.
            loss_dict = self.losses(params, batch, *args)
            loss_keys = tuple(loss_dict.keys())

            grad_norm_dict = {}
            for key in loss_keys:
                loss_fn = lambda p, key=key: self.losses(p, batch, *args)[key]
                g = grad(loss_fn)(params)
                flattened_grad = flatten_pytree(g)
                grad_norm_dict[key] = jnp.linalg.norm(flattened_grad)

            # Compute the mean of grad norms over all losses
            mean_grad_norm = jnp.mean(jnp.stack(tree_leaves(grad_norm_dict)))
            # Grad Norm Weighting
            w = tree_map(
                lambda x: (mean_grad_norm / (x + 1e-5 * mean_grad_norm)), grad_norm_dict
            )

        elif self.config.weighting.scheme == "ntk":
            # Compute the diagonal of the NTK of each loss
            ntk = self.compute_diag_ntk(params, batch, *args)

            # Compute the mean of the diagonal NTK corresponding to each loss
            mean_ntk_dict = tree_map(lambda x: jnp.mean(x), ntk)

            # Compute the average over all ntk means
            mean_ntk = jnp.mean(jnp.stack(tree_leaves(mean_ntk_dict)))
            # NTK Weighting
            w = tree_map(lambda x: (mean_ntk / (x + 1e-5 * mean_ntk)), mean_ntk_dict)

        return w

    @partial(pmap, axis_name="batch", static_broadcasted_argnums=(0,))
    def update_weights(self, state, batch, *args):
        weights = self.compute_weights(state.params, batch, *args)
        weights = lax.pmean(weights, "batch")
        state = state.apply_weights(weights=weights)
        return state

    @partial(pmap, axis_name="batch", static_broadcasted_argnums=(0,))
    def step(self, state, batch, *args):
        grads = grad(self.loss)(state.params, state.weights, batch, *args)
        grads = lax.pmean(grads, "batch")
        state = state.apply_gradients(grads=grads)
        return state


class ForwardIVP(PINN):
    def __init__(self, config, replicate_state=True):
        super().__init__(config, replicate_state=replicate_state)

        if config.weighting.use_causal:
            self.tol = config.weighting.causal_tol
            self.num_chunks = config.weighting.num_chunks
            self.M = jnp.triu(jnp.ones((self.num_chunks, self.num_chunks)), k=1).T


class ForwardBVP(PINN):
    def __init__(self, config):
        super().__init__(config)
