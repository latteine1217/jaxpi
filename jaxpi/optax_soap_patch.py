"""
Compatibility patch for soap_jax with optax 0.1.9

This module provides missing functions that soap_jax expects from newer optax versions.
Import this module before importing soap_jax to apply the patches.

Usage:
    import optax_soap_patch  # Must be imported first!
    from soap_jax import soap
"""

import optax
import jax.tree_util as jtu
import jax.numpy as jnp


def tree_update_moment(updates, moments, decay, order):
    """
    Update momentum with exponential moving average.
    
    This function is expected by soap_jax but missing in optax 0.1.9.
    
    Args:
        updates: The updates (gradients) pytree
        moments: The current momentum pytree  
        decay: Decay rate (beta1 or beta2)
        order: Moment order (1 for first moment, 2 for second moment)
    
    Returns:
        Updated momentum pytree
    """
    return jtu.tree_map(
        lambda m, u: decay * m + (1 - decay) * (u ** order),
        moments,
        updates
    )


def tree_update_moment_per_elem_norm(updates, moments, decay, order):
    """
    Update momentum per element with exponential moving average.
    
    Similar to tree_update_moment but computes element-wise norm.
    
    Args:
        updates: The updates pytree
        moments: The current momentum pytree
        decay: Decay rate
        order: Norm order
    
    Returns:
        Updated momentum pytree
    """
    return jtu.tree_map(
        lambda m, u: decay * m + (1 - decay) * (jnp.abs(u) ** order),
        moments,
        updates
    )


# Apply the patch by adding missing functions to optax.tree_utils
if not hasattr(optax.tree_utils, 'tree_update_moment'):
    optax.tree_utils.tree_update_moment = tree_update_moment
    print("[optax_soap_patch] Added tree_update_moment to optax.tree_utils")

if not hasattr(optax.tree_utils, 'tree_update_moment_per_elem_norm'):
    optax.tree_utils.tree_update_moment_per_elem_norm = tree_update_moment_per_elem_norm
    print("[optax_soap_patch] Added tree_update_moment_per_elem_norm to optax.tree_utils")

print("[optax_soap_patch] SOAP compatibility patch applied successfully!")
