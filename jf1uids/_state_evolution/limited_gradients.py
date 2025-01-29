from typing import Union
from jf1uids._geometry.geometry import CARTESIAN, STATE_TYPE


import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped


from functools import partial

from jf1uids._state_evolution.limiters import _minmod


@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['geometry'])
def _calculate_limited_gradients(primitive_states: Float[Array, "num_vars num_cells"], dx: Union[float, Float[Array, ""]], geometry: int, rv: Float[Array, "num_cells"]) -> Float[Array, "num_vars num_cells-2"]:
    """
    Calculate the limited gradients of the primitive variables.

    Args:
        primitive_states: The primitive state array.
        dx: The cell width.
        geometry: The geometry of the domain.
        rv: The volumetric centers of the cells.

    Returns:
        The limited gradients of the primitive variables.

    """
    if geometry == CARTESIAN:
        cell_distances_left = dx # distances r_i - r_{i-1}
        cell_distances_right = dx # distances r_{i+1} - r_i
    else:
        # calculate the distances
        cell_distances_left = rv[1:-1] - rv[:-2]
        cell_distances_right = rv[2:] - rv[1:-1]

    # formulation 1:
    epsilon = 1e-11  # Small constant to prevent division by zero
    a = (primitive_states[:, 1:-1] - primitive_states[:, :-2]) / cell_distances_left
    b = (primitive_states[:, 2:] - primitive_states[:, 1:-1]) / cell_distances_right
    g = jnp.where(
        jnp.abs(a) > epsilon,  # Avoid division if `a` is very small
        b / (a + epsilon),  # Add epsilon to `a` for numerical stability
        jnp.zeros_like(a)
    )
    # slope_limited = jnp.maximum(0, jnp.minimum(1, g))  # Minmod limiter
    slope_limited = jnp.maximum(0, jnp.minimum(1.3, g))  # Osher limiter with beta = 1.3
    # ospre limiter
    # slope_limited = (1.5 * (g ** 2 + g)) / (g ** 2 + g + 1)
    limited_gradients = slope_limited * a

    # # formulation 2:
    # limited_gradients = _minmod(
    #     (primitive_states[:, 1:-1] - primitive_states[:, :-2]) / cell_distances_left,
    #     (primitive_states[:, 2:] - primitive_states[:, 1:-1]) / cell_distances_right
    # )

    return limited_gradients


# TODO: improve shape annotations, two smaller in the flux_direction_index dimension
# or maybe better: equal shapes everywhere
@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['axis'])
def _calculate_limited_gradients3D(primitive_states: STATE_TYPE, dx: Union[float, Float[Array, ""]], axis: int) -> Float[Array, "num_vars num_cells_a num_cells_b num_cells_c"]:
    # axis = 1 for x (0th axis are the primitive variables)

    # formulation 1:
    epsilon = 1e-11  # Small constant to prevent division by zero

    # get array sizee along the axis
    num_cells = primitive_states.shape[axis]

    a = (jax.lax.slice_in_dim(primitive_states, 1, num_cells - 1, axis = axis) - jax.lax.slice_in_dim(primitive_states, 0, num_cells - 2, axis = axis)) / dx
    b = (jax.lax.slice_in_dim(primitive_states, 2, num_cells, axis = axis) - jax.lax.slice_in_dim(primitive_states, 1, num_cells - 1, axis = axis)) / dx
    # g = jnp.where(
    #     jnp.abs(a) > epsilon,  # Avoid division if `a` is very small
    #     b / (a + epsilon),  # Add epsilon to `a` for numerical stability
    #     jnp.zeros_like(a)
    # )
    # # slope_limited = jnp.maximum(0, jnp.minimum(1, g))  # Minmod limiter
    # slope_limited = jnp.maximum(0, jnp.minimum(1.3, g))  # Osher limiter with beta = 1.3
    # # ospre limiter
    # # slope_limited = (1.5 * (g ** 2 + g)) / (g ** 2 + g + 1)
    # limited_gradients = slope_limited * a

    limited_gradients = _minmod(a, b)

    return limited_gradients