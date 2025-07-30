# general imports
import jax
import jax.numpy as jnp
from functools import partial

# typing imports
from typing import Union
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped

# general jf1uids imports
from jf1uids._physics_modules._cosmic_rays.cr_fluid_equations import speed_of_sound_crs
from jf1uids._physics_modules._self_gravity._poisson_solver import _compute_gravitational_potential
from jf1uids._state_evolution.limiters import _minmod
from jf1uids._stencil_operations._stencil_operations import _stencil_add
from jf1uids.option_classes.simulation_config import CARTESIAN, STATE_TYPE, STATE_TYPE_ALTERED, SimulationConfig
from jf1uids.data_classes.simulation_helper_data import HelperData
from jf1uids.fluid_equations.registered_variables import RegisteredVariables

# speed of sound calculation
from jf1uids.fluid_equations.fluid import speed_of_sound

# limited gradients
from jf1uids._state_evolution.limited_gradients import _calculate_limited_gradients
from jf1uids.option_classes.simulation_params import SimulationParams

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'registered_variables', 'axis'])
def _reconstruct_at_interface(
    primitive_state: STATE_TYPE,
    dt: Union[float, Float[Array, ""]],
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
    axis: int
)-> tuple[STATE_TYPE_ALTERED, STATE_TYPE_ALTERED]:
    
    """
    Limited linear reconstruction of the primitive variables at the interfaces.

    Args:
        primitive_state: The primitive state array.
        dt: The time step.
        grid_spacing: The cell width.
        gamma: The adiabatic index.

    Returns:
        The primitive variables at both sides of the interfaces.
    """

    num_cells = primitive_state.shape[axis]

    # get fluid variables for convenience
    rho = primitive_state[registered_variables.density_index]
    p = primitive_state[registered_variables.pressure_index]
    u = primitive_state[axis]

    # get the limited gradients on the cells
    limited_gradients = _calculate_limited_gradients(primitive_state, config, helper_data, axis = axis)

    # calculate the sound speed
    if not config.cosmic_ray_config.cosmic_rays:
        c = speed_of_sound(rho, p, gamma)
    else:
        c = speed_of_sound_crs(primitive_state, registered_variables)

    # ================ construct A_W, the "primitive Jacabian" (not an actual Jacabian) ================
    # see https://diglib.uibk.ac.at/download/pdf/4422963.pdf, 2.11

    # calculate the vectors making up A_W
    A_W = jnp.zeros((registered_variables.num_vars,) + primitive_state.shape)

    # set u diagonal, this way all quantities are automatically advected
    A_W = A_W.at[jnp.arange(registered_variables.num_vars), jnp.arange(registered_variables.num_vars)].set(u)

    # set rest
    A_W = A_W.at[registered_variables.density_index, axis].set(rho)
    A_W = A_W.at[registered_variables.pressure_index, 1].set(rho * c ** 2)
    A_W = A_W.at[axis, registered_variables.pressure_index].set(1 / rho)

    A_W = jax.lax.slice_in_dim(A_W, 1, num_cells - 1, axis = axis + 1)

    # ====================================================================================================

    # project the gradients
    if config.dimensionality == 1:
        projected_gradients = jnp.einsum('bax, ax -> bx', A_W, limited_gradients)
    elif config.dimensionality == 2:
        projected_gradients = jnp.einsum('baxy, axy -> bxy', A_W, limited_gradients)
    elif config.dimensionality == 3:
        projected_gradients = jnp.einsum('baxyz, axyz -> bxyz', A_W, limited_gradients)

    # predictor step
    predictors = jax.lax.slice_in_dim(primitive_state, 1, num_cells - 1, axis = axis) - dt / 2 * projected_gradients

    # did not seem to help
    # in the case of self-gravity, include the gravitational acceleration
    # in the half-step predictor
    # if config.self_gravity:

    #     # TODO: THIS SHOULD NOT BE CALUCLATED HERE BUT PASSED, NOW FOR TESTING
    #     G = 1.0
    #     gravitational_potential = _compute_gravitational_potential(rho, config.grid_spacing, config, G)

    #     # a_i = - (phi_{i+1} - phi_{i-1}) / (2 * dx)
    #     acceleration = -_stencil_add(gravitational_potential, indices = (1, -1), factors = (1.0, -1.0), axis = axis - 1, zero_pad = False) / (2 * config.grid_spacing)

    #     # add the gravitational acceleration to the predictor
    #     predictors = predictors.at[axis].add(dt / 2 * acceleration)

    # compute primitives at the interfaces
    if config.geometry == CARTESIAN:
        distances_to_left_interfaces = config.grid_spacing / 2 # distances r_i - r_{i-1/2}
        distances_to_right_interfaces = config.grid_spacing / 2 # distances r_{i+1/2} - r_i
    else:
        r = helper_data.geometric_centers
        rv = helper_data.volumetric_centers

        distances_to_left_interfaces = rv[1:-1] - (r[1:-1] - config.grid_spacing / 2)
        distances_to_right_interfaces = (r[1:-1] + config.grid_spacing / 2) - rv[1:-1]

    primitives_left = predictors - distances_to_left_interfaces * limited_gradients
    primitives_right = predictors + distances_to_right_interfaces * limited_gradients

    # the first entries are the state to the left and right
    # of the interface between cell 1 and 2
    num_prim = primitives_left.shape[axis]
    return jax.lax.slice_in_dim(primitives_right, 0, num_prim - 1, axis = axis), jax.lax.slice_in_dim(primitives_left, 1, num_prim, axis = axis)

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'registered_variables'])
def _reconstruct_at_interface_pp(
    primitive_state: STATE_TYPE,
    dt: Union[float, Float[Array, ""]],
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
):
    """
    Positivity preserving reconstruction

    TODO: better notation with left_center, left_interface
    """
    
    # limited gradients: dimensionality x state_shape
    limited_gradients = jnp.zeros((config.dimensionality,) + primitive_state.shape)

    for axis in range(1, config.dimensionality + 1):
        limited_gradients = limited_gradients.at[axis - 1].set(
            _calculate_limited_gradients_pp(primitive_state, config, helper_data, axis = axis)
        )

    # ensure positivity

    eps = 1e-14
    
    rho = primitive_state[registered_variables.density_index]
    p = primitive_state[registered_variables.pressure_index]
    c = speed_of_sound(rho, p, gamma)
    alpha_lax = jnp.zeros((config.dimensionality,))
    for axis in range(1, config.dimensionality + 1):
        u = primitive_state[axis]
        alpha_lax_i = jnp.max(jnp.abs(u) + c)
        alpha_lax = alpha_lax.at[axis - 1].set(alpha_lax_i)

    # NOTE: formula will change for different grid spacings along dimensions!!!
    
    C = alpha_lax / jnp.sum(alpha_lax)
    q = 2 / params.C_cfl # 2 because Delta t half steps ?

    differences = limited_gradients * config.grid_spacing / 2

    density_diff_protected = jnp.where(
        jnp.abs(differences[:, registered_variables.density_index]) > eps,
        differences[:, registered_variables.density_index],
        eps
    )

    pressure_diff_protected = jnp.where(
        jnp.abs(differences[:, registered_variables.pressure_index]) > eps,
        differences[:, registered_variables.pressure_index],
        eps
    )

    alpha_density = jnp.where(
        jnp.abs(differences[:, registered_variables.density_index]) > eps, 
        jnp.minimum(
            primitive_state[registered_variables.density_index] / (jnp.abs(density_diff_protected) * (1 + eps)),
            1
        ),
        1
    )

    kappa_pressure = jnp.where(
        jnp.abs(differences[:, registered_variables.pressure_index]) > eps,
        jnp.minimum(
            primitive_state[registered_variables.pressure_index] / (jnp.abs(pressure_diff_protected) * (1 + eps)),
            1
        ),
        1
    )

    if config.dimensionality == 1:
        A1 = jnp.sum(jnp.sum(C[:, None] * alpha_density * differences[:, registered_variables.density_index] * differences[:, registered_variables.velocity_index.x: registered_variables.velocity_index.x + config.dimensionality], axis = 0) ** 2, axis = 0)
        A2 = jnp.sum(C[:, None] * jnp.sum(differences[:, registered_variables.velocity_index.x: registered_variables.velocity_index.x + config.dimensionality] ** 2, axis = 1), axis = 0)
    elif config.dimensionality == 2:
        A1 = jnp.sum(jnp.sum(C[:, None, None] * alpha_density * differences[:, registered_variables.density_index] * differences[:, registered_variables.velocity_index.x: registered_variables.velocity_index.x + config.dimensionality], axis = 0) ** 2, axis = 0)
        A2 = jnp.sum(C[:, None, None] * jnp.sum(differences[:, registered_variables.velocity_index.x: registered_variables.velocity_index.x + config.dimensionality] ** 2, axis = 1), axis = 0)
    elif config.dimensionality == 3:
        A1 = jnp.sum(jnp.sum(C[:, None, None, None] * alpha_density * differences[:, registered_variables.density_index] * differences[:, registered_variables.velocity_index.x: registered_variables.velocity_index.x + config.dimensionality], axis = 0) ** 2, axis = 0)
        A2 = jnp.sum(C[:, None, None, None] * jnp.sum(differences[:, registered_variables.velocity_index.x: registered_variables.velocity_index.x + config.dimensionality] ** 2, axis = 1), axis = 0)
    vsum = jnp.sum(jnp.sum(differences[:, registered_variables.velocity_index.x: registered_variables.velocity_index.x + config.dimensionality] ** 2, axis = 1), axis = 0)
    A1 = jnp.where(vsum > eps, A1, eps)
    A2 = jnp.where(vsum > eps, A2, eps)

    beta = jnp.where(vsum > eps, jnp.minimum(jnp.sqrt(((q - 2) ** 2 * primitive_state[registered_variables.density_index] * primitive_state[registered_variables.pressure_index]) / ((gamma - 1) * (2 * A1 + (q - 2) * primitive_state[registered_variables.density_index] ** 2 * A2))), 1), 1)

    primitives_left_interface = jnp.zeros((config.dimensionality,) + primitive_state.shape)
    primitives_right_interface = jnp.zeros((config.dimensionality,) + primitive_state.shape)

    differences_pp = differences

    differences_pp = differences_pp.at[:, registered_variables.density_index].set(
        differences[:, registered_variables.density_index] * alpha_density
    )

    differences_pp = differences_pp.at[:, registered_variables.pressure_index].set(
        differences[:, registered_variables.pressure_index] * kappa_pressure
    )

    differences_pp = differences_pp.at[:, registered_variables.velocity_index.x: registered_variables.velocity_index.x + config.dimensionality].set(
        differences[:, registered_variables.velocity_index.x: registered_variables.velocity_index.x + config.dimensionality] * beta
    )

    for axis in range(1, config.dimensionality + 1):


        # i-1/2R, ...
        primitives_left_center = primitive_state - differences_pp[axis - 1] # left of the cell center but the right of the interface
        # i+1/2L, ...
        primitives_right_center = primitive_state + differences_pp[axis - 1] # right of the cell center but the left of the interface

        # primitives left at i is the left state at the interface
        # between i-1 and i so the right extrapolation from the cell i-1
        p_left_interface = jnp.roll(primitives_right_center, shift = 1, axis = axis)

        # primitives right at i is the right state at the interface
        # between i-1 and i so the left extrapolation from the cell i
        p_right_interface = primitives_left_center

        # set the values
        primitives_left_interface = primitives_left_interface.at[axis - 1].set(p_left_interface)
        primitives_right_interface = primitives_right_interface.at[axis - 1].set(p_right_interface)

    return primitives_left_interface, primitives_right_interface

def _calculate_limited_gradients_pp(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    helper_data: HelperData,
    axis: int
) -> STATE_TYPE_ALTERED:
    """
    van Albada limited gradients along an axis
    """

    grid_spacing = config.grid_spacing
    epsilon = 3 * grid_spacing
    forward_difference = _stencil_add(primitive_state, indices = (1, 0), factors = (1.0, -1.0), axis = axis, zero_pad = True) / grid_spacing
    backward_difference = _stencil_add(primitive_state, indices = (0, -1), factors = (1.0, -1.0), axis = axis, zero_pad = True) / grid_spacing

    limited_gradients = ((forward_difference ** 2 + epsilon) * backward_difference + (backward_difference ** 2 + epsilon) * forward_difference) / (forward_difference ** 2 + backward_difference ** 2 + 2 * epsilon)

    return limited_gradients

# @jaxtyped(typechecker=typechecker)
# @partial(jax.jit, static_argnames=['config', 'axis'])
# def _calculate_limited_gradients_pp(
#     primitive_state: STATE_TYPE,
#     config: SimulationConfig,
#     helper_data: HelperData,
#     axis: int
# ) -> STATE_TYPE_ALTERED:
#     """
#     Calculate the limited gradients of the primitive variables.

#     Args:
#         primitive_state: The primitive state array.
#         grid_spacing_or_rv: Usually the cell width, for spherical 
#         geometry the volumetric centers of the cells.
#         axis: The array axis along which the gradients are calculated,
#         = 1 for x (0th axis are the variables).
#         geometry: The geometry of the domain.

#     Returns:
#         The limited gradients of the primitive variables.

#     """

#     # Next we calculate the finite differences of consecutive cells.
#     # a is the left difference, b the right difference for cells
#     # 1 to num_cells - 1.
#     # a = (jax.lax.slice_in_dim(primitive_state, 1, num_cells - 1, axis = axis) - jax.lax.slice_in_dim(primitive_state, 0, num_cells - 2, axis = axis)) / cell_distances_left
#     # b = (jax.lax.slice_in_dim(primitive_state, 2, num_cells, axis = axis) - jax.lax.slice_in_dim(primitive_state, 1, num_cells - 1, axis = axis)) / cell_distances_right

#     a = _stencil_add(primitive_state, indices = (0, -1), factors = (1.0, -1.0), axis = axis, zero_pad = True) / config.grid_spacing
#     b = _stencil_add(primitive_state, indices = (1, 0), factors = (1.0, -1.0), axis = axis, zero_pad = True) / config.grid_spacing
#     # We apply limiting to not create new extrema in regions where consecutive finite
#     # differences differ strongly.

#     limited_gradients = _minmod(a, b)

#     return limited_gradients