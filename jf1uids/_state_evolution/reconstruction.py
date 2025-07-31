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
from jf1uids._state_evolution.limiters import _van_albada_limiter, _minmod
from jf1uids._stencil_operations._stencil_operations import _stencil_add
from jf1uids.option_classes.simulation_config import CARTESIAN, MUSCL, STATE_TYPE, STATE_TYPE_ALTERED, VAN_ALBADA, VAN_ALBADA_PP, SimulationConfig
from jf1uids.data_classes.simulation_helper_data import HelperData
from jf1uids.fluid_equations.registered_variables import RegisteredVariables

# speed of sound calculation
from jf1uids.fluid_equations.fluid import speed_of_sound

# limited gradients
from jf1uids._state_evolution.limited_gradients import _calculate_limited_gradients
from jf1uids.option_classes.simulation_params import SimulationParams

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'registered_variables', 'axis'])
def _reconstruct_at_interface_split(
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

    # get fluid variables for convenience
    rho = primitive_state[registered_variables.density_index]
    p = primitive_state[registered_variables.pressure_index]
    u = primitive_state[axis]

    # get the limited gradients on the cells
    limited_gradients = _calculate_limited_gradients(primitive_state, config, helper_data, axis = axis)

    if config.time_integrator == MUSCL:
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

        # ====================================================================================================

        # project the gradients
        if config.dimensionality == 1:
            projected_gradients = jnp.einsum('bax, ax -> bx', A_W, limited_gradients)
        elif config.dimensionality == 2:
            projected_gradients = jnp.einsum('baxy, axy -> bxy', A_W, limited_gradients)
        elif config.dimensionality == 3:
            projected_gradients = jnp.einsum('baxyz, axyz -> bxyz', A_W, limited_gradients)

        # predictor step
        predictors = primitive_state - dt / 2 * projected_gradients
    else:
        raise ValueError(f"Time integrator {config.time_integrator} not supported for split reconstruction. Only MUSCL is supported.")

    # compute primitives at the interfaces
    if config.geometry == CARTESIAN:
        distances_to_left_interfaces = config.grid_spacing / 2 # distances r_i - r_{i-1/2}
        distances_to_right_interfaces = config.grid_spacing / 2 # distances r_{i+1/2} - r_i
    else:
        r = helper_data.geometric_centers
        rv = helper_data.volumetric_centers

        distances_to_left_interfaces = rv - (r - config.grid_spacing / 2)
        distances_to_right_interfaces = (r + config.grid_spacing / 2) - rv

    primitives_left = predictors - distances_to_left_interfaces * limited_gradients
    primitives_right = predictors + distances_to_right_interfaces * limited_gradients

    # primitives left at i is the left state at the interface
    # between i-1 and i so the right extrapolation from the cell i-1
    p_left_interface = jnp.roll(primitives_right, shift = 1, axis = axis)

    # primitives right at i is the right state at the interface
    # between i-1 and i so the left extrapolation from the cell i
    p_right_interface = primitives_left

    return p_left_interface, p_right_interface

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'registered_variables'])
def _reconstruct_at_interface_unsplit(
    primitive_state: STATE_TYPE,
    dt: Union[float, Float[Array, ""]],
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
):
    """
    Unsplit reconstruction.
    """
    
    # limited gradients: dimensionality x state_shape
    limited_gradients = jnp.zeros((config.dimensionality,) + primitive_state.shape)

    for axis in range(1, config.dimensionality + 1):
        if config.limiter == VAN_ALBADA or config.limiter == VAN_ALBADA_PP:
            limited_gradients = limited_gradients.at[axis - 1].set(
                _van_albada_limiter(primitive_state, config, helper_data, axis = axis)
            )
        else:
            raise ValueError(f"Limiter {config.limiter} currently not supported for unsplit reconstruction.")

    differences = limited_gradients * config.grid_spacing / 2

    if config.limiter == VAN_ALBADA_PP:
        # positivity preserving reconstruction

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

        if config.mhd:
            # in the MHD case we go half-time steps as of
            # the strang splitting, so effectively, for the
            # hydro part we use C_cfl / 2, so 1 / (C_cfl / 2)
            # = 2 / C_cfl
            q = 2 / params.C_cfl
        else:
            q = 1 / params.C_cfl

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

        differences = differences_pp

    for axis in range(1, config.dimensionality + 1):


        # i-1/2R, ...
        primitives_left_center = primitive_state - differences[axis - 1] # left of the cell center but the right of the interface
        # i+1/2L, ...
        primitives_right_center = primitive_state + differences[axis - 1] # right of the cell center but the left of the interface

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