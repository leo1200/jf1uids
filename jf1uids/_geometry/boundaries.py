from typing import NamedTuple
import jax.numpy as jnp
from functools import partial
import jax

from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

from jf1uids.option_classes.simulation_config import OPEN_BOUNDARY, PERIODIC_BOUNDARY, REFLECTIVE_BOUNDARY, STATE_TYPE, SimulationConfig

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['axis', 'set_index', 'get_index'])
def _set_along_axis(
    primitive_state: STATE_TYPE,
    axis: int,
    set_index: int,
    get_index: int
) -> STATE_TYPE:

    s_set = (slice(None),) * axis + (set_index,) + (slice(None),)*(primitive_state.ndim - axis - 1)
    s_get = (slice(None),) * axis + (get_index,) + (slice(None),)*(primitive_state.ndim - axis - 1)

    primitive_state = primitive_state.at[s_set].set(primitive_state[s_get])

    return primitive_state

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['axis', 'set_index', 'get_index', 'var_index', 'factor'])
def _set_specific_var_along_axis(
    primitive_state: STATE_TYPE,
    axis: int,
    set_index: int,
    get_index: int,
    var_index: int,
    factor: float
) -> STATE_TYPE:

    s_set = (var_index,) + (slice(None),) * (axis - 1) + (set_index,) + (slice(None),)*(primitive_state.ndim - axis - 1)
    s_get = (var_index,) + (slice(None),) * (axis - 1) + (get_index,) + (slice(None),)*(primitive_state.ndim - axis - 1)

    primitive_state = primitive_state.at[s_set].set(factor * primitive_state[s_get])

    return primitive_state

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config'])
def _boundary_handler(
    primitive_state: STATE_TYPE,
    config: SimulationConfig
) -> STATE_TYPE:
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
            primitive_state = _open_left_boundary(primitive_state, config.num_ghost_cells, axis = 1)
        elif config.boundary_settings.left_boundary == REFLECTIVE_BOUNDARY:
            primitive_state = _reflective_left_boundary1d(primitive_state)

        if config.boundary_settings.right_boundary == OPEN_BOUNDARY:
            primitive_state = _open_right_boundary(primitive_state, config.num_ghost_cells, axis = 1)
        elif config.boundary_settings.right_boundary == REFLECTIVE_BOUNDARY:
            primitive_state = _reflective_right_boundary1d(primitive_state)

        if config.boundary_settings.left_boundary == PERIODIC_BOUNDARY and config.boundary_settings.right_boundary == PERIODIC_BOUNDARY:
            primitive_state = _periodic_boundaries(primitive_state, config.num_ghost_cells, axis = 1)

    if config.dimensionality == 2:
        if config.boundary_settings.x.left_boundary == OPEN_BOUNDARY:
            primitive_state = _open_left_boundary(primitive_state, config.num_ghost_cells, axis = 1)

        if config.boundary_settings.x.right_boundary == OPEN_BOUNDARY:
            primitive_state = _open_right_boundary(primitive_state, config.num_ghost_cells, axis = 1)

        if config.boundary_settings.y.left_boundary == OPEN_BOUNDARY:
            primitive_state = _open_left_boundary(primitive_state, config.num_ghost_cells, axis = 2)

        if config.boundary_settings.y.right_boundary == OPEN_BOUNDARY:
            primitive_state = _open_right_boundary(primitive_state, config.num_ghost_cells, axis = 2)

        if config.boundary_settings.x.left_boundary == REFLECTIVE_BOUNDARY:
            primitive_state = _reflective_left_boundary(primitive_state, config.num_ghost_cells, axis = 1)

        if config.boundary_settings.x.right_boundary == REFLECTIVE_BOUNDARY:
            primitive_state = _reflective_right_boundary(primitive_state, config.num_ghost_cells, axis = 1)

        if config.boundary_settings.y.left_boundary == REFLECTIVE_BOUNDARY:
            primitive_state = _reflective_left_boundary(primitive_state, config.num_ghost_cells, axis = 2)

        if config.boundary_settings.y.right_boundary == REFLECTIVE_BOUNDARY:
            primitive_state = _reflective_right_boundary(primitive_state, config.num_ghost_cells, axis = 2)

        if config.boundary_settings.x.left_boundary == PERIODIC_BOUNDARY and config.boundary_settings.x.right_boundary == PERIODIC_BOUNDARY:
            primitive_state = _periodic_boundaries(primitive_state, config.num_ghost_cells, axis = 1)

        if config.boundary_settings.y.left_boundary == PERIODIC_BOUNDARY and config.boundary_settings.y.right_boundary == PERIODIC_BOUNDARY:
            primitive_state = _periodic_boundaries(primitive_state, config.num_ghost_cells, axis = 2)

    if config.dimensionality == 3:
        if config.boundary_settings.x.left_boundary == OPEN_BOUNDARY:
            primitive_state = _open_left_boundary(primitive_state, config.num_ghost_cells, axis = 1)

        if config.boundary_settings.x.right_boundary == OPEN_BOUNDARY:
            primitive_state = _open_right_boundary(primitive_state, config.num_ghost_cells, axis = 1)

        if config.boundary_settings.y.left_boundary == OPEN_BOUNDARY:
            primitive_state = _open_left_boundary(primitive_state, config.num_ghost_cells, axis = 2)

        if config.boundary_settings.y.right_boundary == OPEN_BOUNDARY:
            primitive_state = _open_right_boundary(primitive_state, config.num_ghost_cells, axis = 2)

        if config.boundary_settings.z.left_boundary == OPEN_BOUNDARY:
            primitive_state = _open_left_boundary(primitive_state, config.num_ghost_cells, axis = 3)

        if config.boundary_settings.z.right_boundary == OPEN_BOUNDARY:
            primitive_state = _open_right_boundary(primitive_state, config.num_ghost_cells, axis = 3)

        if config.boundary_settings.x.left_boundary == REFLECTIVE_BOUNDARY:
            primitive_state = _reflective_left_boundary(primitive_state, config.num_ghost_cells, axis = 1)

        if config.boundary_settings.x.right_boundary == REFLECTIVE_BOUNDARY:
            primitive_state = _reflective_right_boundary(primitive_state, config.num_ghost_cells, axis = 1)

        if config.boundary_settings.y.left_boundary == REFLECTIVE_BOUNDARY:
            primitive_state = _reflective_left_boundary(primitive_state, config.num_ghost_cells, axis = 2)

        if config.boundary_settings.y.right_boundary == REFLECTIVE_BOUNDARY:
            primitive_state = _reflective_right_boundary(primitive_state, config.num_ghost_cells, axis = 2)

        if config.boundary_settings.z.left_boundary == REFLECTIVE_BOUNDARY:
            primitive_state = _reflective_left_boundary(primitive_state, config.num_ghost_cells, axis = 3)

        if config.boundary_settings.z.right_boundary == REFLECTIVE_BOUNDARY:
            primitive_state = _reflective_right_boundary(primitive_state, config.num_ghost_cells, axis = 3)

        if config.boundary_settings.x.left_boundary == PERIODIC_BOUNDARY and config.boundary_settings.x.right_boundary == PERIODIC_BOUNDARY:
            primitive_state = _periodic_boundaries(primitive_state, config.num_ghost_cells, axis = 1)

        if config.boundary_settings.y.left_boundary == PERIODIC_BOUNDARY and config.boundary_settings.y.right_boundary == PERIODIC_BOUNDARY:
            primitive_state = _periodic_boundaries(primitive_state, config.num_ghost_cells, axis = 2)

        if config.boundary_settings.z.left_boundary == PERIODIC_BOUNDARY and config.boundary_settings.z.right_boundary == PERIODIC_BOUNDARY:
            primitive_state = _periodic_boundaries(primitive_state, config.num_ghost_cells, axis = 3)

    return primitive_state


@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['axis', 'num_ghost_cells'])
def _open_right_boundary(
    primitive_state: STATE_TYPE,
    num_ghost_cells: int,
    axis: int
) -> STATE_TYPE:
    """Apply the open boundary condition to the right boundary.
    
    Args:
        primitive_state: The primitive state array.

    Returns:
        The primitive state array with the open boundary condition applied.
    """

    get_index = -num_ghost_cells - 1
    
    for set_index in range(-num_ghost_cells, 0):
        primitive_state = _set_along_axis(primitive_state, axis, set_index, get_index)

    return primitive_state

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['axis', 'num_ghost_cells'])
def _open_left_boundary(
    primitive_state: STATE_TYPE,
    num_ghost_cells: int,
    axis: int
) -> STATE_TYPE:
    """Apply the open boundary condition to the left boundary.

    Args:
        primitive_state: The primitive state array.

    Returns:
        The primitive state array with
        the open boundary condition applied.

    """

    get_index = num_ghost_cells

    for set_index in range(num_ghost_cells):
        primitive_state = _set_along_axis(primitive_state, axis, set_index, get_index)

    return primitive_state

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['axis', 'num_ghost_cells'])
def _periodic_boundaries(
    primitive_state: STATE_TYPE,
    num_ghost_cells: int,
    axis: int
) -> STATE_TYPE:
    """Apply the periodic boundary condition to the primitive states.

    Args:
        primitive_state: The primitive state array.

    Returns:
        The primitive state array with the periodic boundary condition applied.

    """

    # primitive_state = _set_along_axis(primitive_state, axis, 0, -4)
    # primitive_state = _set_along_axis(primitive_state, axis, 1, -3)

    # primitive_state = _set_along_axis(primitive_state, axis, -2, 2)
    # primitive_state = _set_along_axis(primitive_state, axis, -1, 3)

    for i in range(num_ghost_cells):

        # left boundary
        set_index = i
        get_index = i - 2 * num_ghost_cells
        primitive_state = _set_along_axis(primitive_state, axis, set_index, get_index)

        # right boundary
        set_index = -i - 1
        get_index = 2 * num_ghost_cells - i - 1
        primitive_state = _set_along_axis(primitive_state, axis, set_index, get_index)

    return primitive_state

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['axis', 'num_ghost_cells'])
def _reflective_left_boundary(
    primitive_state: STATE_TYPE,
    num_ghost_cells: int,
    axis: int
) -> STATE_TYPE:

    # reflect the velocity component along the axis
    for i in range(num_ghost_cells):
        get_index = i + num_ghost_cells
        set_index = num_ghost_cells - i - 1

        primitive_state = _set_along_axis(primitive_state, axis, set_index, get_index)
        primitive_state = _set_specific_var_along_axis(primitive_state, axis, set_index, get_index, axis, -1.0)

    return primitive_state

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['axis', 'num_ghost_cells'])
def _reflective_right_boundary(
    primitive_state: STATE_TYPE,
    num_ghost_cells: int,
    axis: int
) -> STATE_TYPE:

    # reflect the velocity component along the axis
    for i in range(num_ghost_cells):
        get_index = -i - num_ghost_cells - 1
        set_index = -num_ghost_cells + i

        primitive_state = _set_along_axis(primitive_state, axis, set_index, get_index)
        primitive_state = _set_specific_var_along_axis(primitive_state, axis, set_index, get_index, axis, -1.0)
      
    return primitive_state

@jaxtyped(typechecker=typechecker)
@jax.jit
def _reflective_left_boundary1d(
    primitive_state: Float[Array, "num_vars num_cells"]
) -> Float[Array, "num_vars num_cells"]:
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
def _reflective_right_boundary1d(
    primitive_state: Float[Array, "num_vars num_cells"]
) -> Float[Array, "num_vars num_cells"]:
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