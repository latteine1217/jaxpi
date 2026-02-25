import os
import os
import json

from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
from jax import jit, grad
from jax.tree_util import tree_map, tree_leaves
from jax.flatten_util import ravel_pytree

from flax.training import checkpoints


def flatten_pytree(pytree):
    return ravel_pytree(pytree)[0]


def init_parallel(config=None):
    num_devices = jax.device_count()
    if num_devices <= 1:
        return {
            "num_devices": num_devices,
            "mesh": None,
            "data_sharding": None,
            "replicated_sharding": None,
        }

    devices = np.array(jax.devices()).reshape((num_devices,))
    mesh = jax.sharding.Mesh(devices, ("data",))
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    return {
        "num_devices": num_devices,
        "mesh": mesh,
        "data_sharding": data_sharding,
        "replicated_sharding": replicated_sharding,
    }


@partial(jit, static_argnums=(0,))
def jacobian_fn(apply_fn, params, *args):
    # apply_fn needs to be a scalar function
    J = grad(apply_fn, argnums=0)(params, *args)
    J, _ = ravel_pytree(J)
    return J


@partial(jit, static_argnums=(0,))
def ntk_fn(apply_fn, params, *args):
    # apply_fn needs to be a scalar function
    J = jacobian_fn(apply_fn, params, *args)
    K = jnp.dot(J, J)
    return K


def save_checkpoint(state, workdir, keep=5, name=None, overwrite=False):
    # Create the workdir if it doesn't exist.
    if not os.path.isdir(workdir):
        os.makedirs(workdir)

    # Save the checkpoint.
    if jax.process_index() == 0:
        # Get the first replica's state and save it.
        leaf_sharding = tree_map(lambda x: x.sharding, tree_leaves(state.params))[0]
        if isinstance(leaf_sharding, jax.sharding.PmapSharding):
            state = jax.device_get(tree_map(lambda x: x[0], state))
        elif isinstance(leaf_sharding, jax.sharding.NamedSharding):
            state = jax.device_get(state)
        else:
            state = jax.device_get(state)

        step = int(state.step)
        checkpoints.save_checkpoint(workdir, state, step=step, keep=keep, overwrite=overwrite)


def _extract_leaf_sharding(params):
    for leaf in tree_leaves(params):
        if hasattr(leaf, "sharding"):
            return leaf.sharding
    return None


def restore_checkpoint(state, workdir, step=None):
    # check if passed state is in a sharded state
    # if so, reduce to a single device sharding
    leaf_sharding = _extract_leaf_sharding(state.params)
    if isinstance(leaf_sharding, jax.sharding.PmapSharding):
        state = tree_map(lambda x: x[0], state)
    elif isinstance(leaf_sharding, jax.sharding.NamedSharding):
        state = jax.device_get(state)

    state = checkpoints.restore_checkpoint(workdir, state, step=step)
    return state


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, o):
        # Custom serialization for JAX numpy arrays
        if isinstance(o, jnp.ndarray):
            return o.tolist()  # Convert JAX numpy array to a list
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, o)


def save_config(config, workdir, name=None):
    # Create the workdir if it doesn't exist.
    if not os.path.isdir(workdir):
        os.makedirs(workdir)

    # Set default name if not provided
    if name is None:
        name = "config"
    # Correctly append the '.json' extension to the filename
    config_path = os.path.join(workdir, name + ".json")

    # Write the config to a JSON file
    with open(config_path, "w") as config_file:
        json.dump(config.to_dict(), config_file, cls=CustomJSONEncoder, indent=4)
