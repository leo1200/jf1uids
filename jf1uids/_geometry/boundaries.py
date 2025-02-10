from typing import NamedTuple
import jax.numpy as jnp
from functools import partial
import jax

from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

from jf1uids.option_classes.simulation_config import OPEN_BOUNDARY, PERIODIC_BOUNDARY, REFLECTIVE_BOUNDARY, STATE_TYPE, SimulationConfig

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['axis', 'set_index', 'get_index'])
def _set_along_axis(primitive_state, axis: int, set_index: int, get_index: int):

    s_set = (slice(None),) * axis + (set_index,) + (slice(None),)*(primitive_state.ndim - axis - 1)
    s_get = (slice(None),) * axis + (get_index,) + (slice(None),)*(primitive_state.ndim - axis - 1)

    primitive_state = primitive_state.at[s_set].set(primitive_state[s_get])

    return primitive_state

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['axis', 'set_index', 'get_index', 'var_index_a', 'var_index_b', 'factor'])
def _set_along_axis_with_selection(primitive_state, axis: int, set_index: int, get_index: int, var_index_a: int, var_index_b: int, factor: float):

    s_set = (slice(var_index_a, var_index_b),) * axis + (set_index,) + (slice(None),)*(primitive_state.ndim - axis - 1)
    s_get = (slice(var_index_a, var_index_b),) * axis + (get_index,) + (slice(None),)*(primitive_state.ndim - axis - 1)

    primitive_state = primitive_state.at[s_set].set(factor * primitive_state[s_get])

    return primitive_state

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config'])
def _boundary_handler(primitive_state: STATE_TYPE, config: SimulationConfig) -> STATE_TYPE:
    """Apply the boundary conditions to the primitive states.

    Args:
        primitive_state: The primitive state array.
        left_boundary: The left boundary condition.
        right_boundary: The right boundary condition.

    Returns:
        The primitive state array with the boundary conditions applied.
    """

    # TODO: implement reflective boundaries for 2D and 3D
    # / throw error here

    # jax.debug.print("boundary handled, left {lb}, right {rb}", lb = config.boundary_settings.left_boundary, rb = config.boundary_settings.right_boundary)

    if config.dimensionality == 1:
        if config.boundary_settings.left_boundary == OPEN_BOUNDARY:
            primitive_state = _open_left_boundary(primitive_state, axis = 1)
        elif config.boundary_settings.left_boundary == REFLECTIVE_BOUNDARY:
            primitive_state = _reflective_left_boundary1d(primitive_state)

        if config.boundary_settings.right_boundary == OPEN_BOUNDARY:
            primitive_state = _open_right_boundary(primitive_state, axis = 1)
        elif config.boundary_settings.right_boundary == REFLECTIVE_BOUNDARY:
            primitive_state = _reflective_right_boundary1d(primitive_state)

        if config.boundary_settings.left_boundary == PERIODIC_BOUNDARY and config.boundary_settings.right_boundary == PERIODIC_BOUNDARY:
            primitive_state = _periodic_boundaries(primitive_state, axis = 1)

    if config.dimensionality == 2:
        if config.boundary_settings.x.left_boundary == OPEN_BOUNDARY:
            primitive_state = _open_left_boundary(primitive_state, axis = 1)

        if config.boundary_settings.x.right_boundary == OPEN_BOUNDARY:
            primitive_state = _open_right_boundary(primitive_state, axis = 1)

        if config.boundary_settings.y.left_boundary == OPEN_BOUNDARY:
            primitive_state = _open_left_boundary(primitive_state, axis = 2)

        if config.boundary_settings.y.right_boundary == OPEN_BOUNDARY:
            primitive_state = _open_right_boundary(primitive_state, axis = 2)

        if config.boundary_settings.x.left_boundary == PERIODIC_BOUNDARY and config.boundary_settings.x.right_boundary == PERIODIC_BOUNDARY:
            primitive_state = _periodic_boundaries(primitive_state, axis = 1)

        if config.boundary_settings.y.left_boundary == PERIODIC_BOUNDARY and config.boundary_settings.y.right_boundary == PERIODIC_BOUNDARY:
            primitive_state = _periodic_boundaries(primitive_state, axis = 2)

    if config.dimensionality == 3:
        if config.boundary_settings.x.left_boundary == OPEN_BOUNDARY:
            primitive_state = _open_left_boundary(primitive_state, axis = 1)

        if config.boundary_settings.x.right_boundary == OPEN_BOUNDARY:
            primitive_state = _open_right_boundary(primitive_state, axis = 1)

        if config.boundary_settings.y.left_boundary == OPEN_BOUNDARY:
            primitive_state = _open_left_boundary(primitive_state, axis = 2)

        if config.boundary_settings.y.right_boundary == OPEN_BOUNDARY:
            primitive_state = _open_right_boundary(primitive_state, axis = 2)

        if config.boundary_settings.z.left_boundary == OPEN_BOUNDARY:
            primitive_state = _open_left_boundary(primitive_state, axis = 3)

        if config.boundary_settings.z.right_boundary == OPEN_BOUNDARY:
            primitive_state = _open_right_boundary(primitive_state, axis = 3)

        if config.boundary_settings.x.left_boundary == PERIODIC_BOUNDARY and config.boundary_settings.x.right_boundary == PERIODIC_BOUNDARY:
            primitive_state = _periodic_boundaries(primitive_state, axis = 1)

        if config.boundary_settings.y.left_boundary == PERIODIC_BOUNDARY and config.boundary_settings.y.right_boundary == PERIODIC_BOUNDARY:
            primitive_state = _periodic_boundaries(primitive_state, axis = 2)

        if config.boundary_settings.z.left_boundary == PERIODIC_BOUNDARY and config.boundary_settings.z.right_boundary == PERIODIC_BOUNDARY:
            primitive_state = _periodic_boundaries(primitive_state, axis = 3)

    return primitive_state


@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['axis'])
def _open_right_boundary(primitive_state: STATE_TYPE, axis: int) -> STATE_TYPE:
    """Apply the open boundary condition to the right boundary.
    
    Args:
        primitive_state: The primitive state array.

    Returns:
        The primitive state array with the open boundary condition applied.
    """

    primitive_state = _set_along_axis(primitive_state, axis, -1, -3)
    primitive_state = _set_along_axis(primitive_state, axis, -2, -3)

    return primitive_state

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['axis'])
def _open_left_boundary(primitive_state: STATE_TYPE, axis: int) -> STATE_TYPE:
    """Apply the open boundary condition to the left boundary.

    Args:
        primitive_state: The primitive state array.

    Returns:
        The primitive state array with
        the open boundary condition applied.

    """

    primitive_state = _set_along_axis(primitive_state, axis, 0, 2)
    primitive_state = _set_along_axis(primitive_state, axis, 1, 2)

    return primitive_state

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['axis'])
def _periodic_boundaries(primitive_state: STATE_TYPE, axis: int) -> STATE_TYPE:
    """Apply the periodic boundary condition to the primitive states.

    Args:
        primitive_state: The primitive state array.

    Returns:
        The primitive state array with the periodic boundary condition applied.

    """

    # jax.debug.print("Setting boundary conditions")

    primitive_state = _set_along_axis(primitive_state, axis, 0, -4)
    primitive_state = _set_along_axis(primitive_state, axis, 1, -3)

    primitive_state = _set_along_axis(primitive_state, axis, -2, 2)
    primitive_state = _set_along_axis(primitive_state, axis, -1, 3)

    return primitive_state

@jaxtyped(typechecker=typechecker)
@jax.jit
def _reflective_left_boundary1d(primitive_state: Float[Array, "num_vars num_cells"]) -> Float[Array, "num_vars num_cells"]:
    """Apply the reflective boundary condition to the left boundary.

    Args:
        primitive_state: The primitive state array.

    Returns:
        The primitive state array with
        the reflective boundary condition applied.

    """

    primitive_state = primitive_state.at[:, 0].set(primitive_state[:, 2])
    primitive_state = primitive_state.at[1, 0].set(-primitive_state[1, 2])

    primitive_state = primitive_state.at[:, 1].set(primitive_state[:, 2])
    primitive_state = primitive_state.at[1, 1].set(-primitive_state[1, 2])

    return primitive_state

@jaxtyped(typechecker=typechecker)
@jax.jit
def _reflective_right_boundary1d(primitive_state: Float[Array, "num_vars num_cells"]) -> Float[Array, "num_vars num_cells"]:
    """Apply the reflective boundary condition to the right boundary.

    Args:
        primitive_state: The primitive state array.

    Returns:
        The primitive state array with
        the reflective boundary condition applied.

    """

    primitive_state = primitive_state.at[:, -1].set(primitive_state[:, -3])
    primitive_state = primitive_state.at[1, -1].set(-primitive_state[1, -3])

    primitive_state = primitive_state.at[:, -2].set(primitive_state[:, -3])
    primitive_state = primitive_state.at[1, -2].set(-primitive_state[1, -3])

    return primitive_state