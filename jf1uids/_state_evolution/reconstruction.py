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
from jf1uids._stencil_operations._stencil_operations import _stencil_add
from jf1uids.option_classes.simulation_config import CARTESIAN, STATE_TYPE, STATE_TYPE_ALTERED, SimulationConfig
from jf1uids.data_classes.simulation_helper_data import HelperData
from jf1uids.fluid_equations.registered_variables import RegisteredVariables

# speed of sound calculation
from jf1uids.fluid_equations.fluid import speed_of_sound

# limited gradients
from jf1uids._state_evolution.limited_gradients import _calculate_limited_gradients

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