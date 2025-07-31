# general imports
from functools import partial
import jax
import jax.numpy as jnp

# typechecking imports
from beartype import beartype as typechecker
from jaxtyping import jaxtyped
from typing import Union

# general jf1uids imports
from jf1uids._stencil_operations._stencil_operations import _stencil_add
from jf1uids.data_classes.simulation_helper_data import HelperData
from jf1uids.option_classes.simulation_config import CARTESIAN, MINMOD, OSHER, DOUBLE_MINMOD, SUPERBEE, SPHERICAL, STATE_TYPE, STATE_TYPE_ALTERED, SimulationConfig

# limiter imports
from jf1uids._state_evolution.limiters import _double_minmod, _minmod, _superbee


@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'axis'])
def _calculate_limited_gradients(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    helper_data: HelperData,
    axis: int
) -> STATE_TYPE_ALTERED:
    """
    Calculate the limited gradients of the primitive variables.

    Args:
        primitive_state: The primitive state array.
        grid_spacing_or_rv: Usually the cell width, for spherical 
        geometry the volumetric centers of the cells.
        axis: The array axis along which the gradients are calculated,
        = 1 for x (0th axis are the variables).
        geometry: The geometry of the domain.

    Returns:
        The limited gradients of the primitive variables.

    """

    # TODO: improve shape annotations, two smaller 
    # in the flux_direction_index dimension
    # or maybe better: equal shapes everywhere

    # get array sizee along the axis
    num_cells = primitive_state.shape[axis]

    # We first need to calculate the distances between the cells.
    # For 1D simulations in spherical geometry, we have to mind
    # that the distances of the volumetric centers to the cell 
    # interfaces are not equal.

    if config.geometry == CARTESIAN:
        cell_distances_left = config.grid_spacing # distances r_i - r_{i-1}
        cell_distances_right = config.grid_spacing # distances r_{i+1} - r_i
    elif config.geometry == SPHERICAL and config.dimensionality == 1:
        # calculate the distances
        cell_distances_left = _stencil_add(helper_data.volumetric_centers, indices = (0, -1), factors = (1.0, -1.0), axis = 0)
        cell_distances_right = _stencil_add(helper_data.volumetric_centers, indices = (1, 0), factors = (1.0, -1.0), axis = 0)
    else:
        raise ValueError("Geometry and dimensionality combination not supported.")

    
    # Next we calculate the finite differences of consecutive cells.
    # a is the left difference, b the right difference for cells
    # 1 to num_cells - 1.

    # backward
    a = _stencil_add(primitive_state, indices = (0, -1), factors = (1.0, -1.0), axis = axis) / cell_distances_left

    # forward
    b = _stencil_add(primitive_state, indices = (1, 0), factors = (1.0, -1.0), axis = axis) / cell_distances_right

    # We apply limiting to not create new extrema in regions where consecutive finite
    # differences differ strongly.

    if config.limiter == MINMOD:
        # Limited average formulations:
        limited_gradients = _minmod(a, b)
    elif config.limiter == SUPERBEE:
        limited_gradients = _superbee(a, b)
    elif config.limiter == DOUBLE_MINMOD:
        limited_gradients = _double_minmod(a, b)
    elif config.limiter == OSHER:
        # Quotient formulation:
        epsilon = 1e-11  # Small constant to prevent division by zero
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
    else:
        raise ValueError("Unknown limiter.")

    return limited_gradients