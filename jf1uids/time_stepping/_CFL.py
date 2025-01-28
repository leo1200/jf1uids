import jax.numpy as jnp
from jf1uids._geometry.geometry import STATE_TYPE
from jf1uids._physics_modules._cosmic_rays.cr_fluid_equations import gas_pressure_from_primitives_with_crs
from jf1uids.data_classes.simulation_helper_data import HelperData
from jf1uids.fluid_equations.euler import _euler_flux
from jf1uids.fluid_equations.fluid import speed_of_sound
import jax
from functools import partial

from jf1uids.fluid_equations.registered_variables import RegisteredVariables
from jf1uids.option_classes.simulation_config import SimulationConfig
from jf1uids.option_classes.simulation_params import SimulationParams
from jf1uids._physics_modules.run_physics_modules import _run_physics_modules

from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

from typing import Union

# TODO: merge duplicate code in this and hll.py
@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['registered_variables', 'flux_direction_index'])
def get_wave_speeds3D(primitives_left: STATE_TYPE, primitives_right: STATE_TYPE, gamma: Union[float, Float[Array, ""]], registered_variables: RegisteredVariables, flux_direction_index: int) -> Union[float, Float[Array, ""]]:
    """
    Returns the conservative fluxes.

    Args:
        primitives_left: States left of the interfaces.
        primitives_right: States right of the interfaces.
        gamma: The adiabatic index.

    Returns:
        The conservative fluxes at the interfaces.

    """
    
    rho_L = primitives_left[registered_variables.density_index]
    u_L = primitives_left[flux_direction_index]

    rho_R = primitives_right[registered_variables.density_index]
    u_R = primitives_right[flux_direction_index]
    
    p_L = primitives_left[registered_variables.pressure_index]
    p_R = primitives_right[registered_variables.pressure_index]

    # calculate the sound speeds
    c_L = speed_of_sound(rho_L, p_L, gamma)
    c_R = speed_of_sound(rho_R, p_R, gamma)

    # very simple approach for the wave velocities
    # wave_speeds_right_plus = jnp.maximum(jnp.maximum(u_L + c_L, u_R + c_R), 0)
    # wave_speeds_left_minus = jnp.minimum(jnp.minimum(u_L - c_L, u_R - c_R), 0)

    wave_speeds_right_plus = jnp.abs(u_L) + c_L
    wave_speeds_left_minus = jnp.abs(u_R) + c_R

    max_wave_speed = jnp.maximum(jnp.max(jnp.abs(wave_speeds_right_plus)), jnp.max(jnp.abs(wave_speeds_left_minus)))

    return max_wave_speed

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['registered_variables'])
def _cfl_time_step(primitive_states: STATE_TYPE, dx: Union[float, Float[Array, ""]], dt_max: Union[float, Float[Array, ""]], gamma: Union[float, Float[Array, ""]], registered_variables: RegisteredVariables, C_CFL: Union[float, Float[Array, ""]] = 0.8) -> Float[Array, ""]:

    """Calculate the time step based on the CFL condition.

    Args:
        primitive_states: The primitive state array.
        dx: The cell width.
        dt_max: The maximum time step.
        gamma: The adiabatic index.
        C_CFL: The CFL number.

    Returns:
        The time step.
    
    """

    primitives_left = primitive_states[:, :-1]
    primitives_right = primitive_states[:, 1:]

    rho_L = primitives_left[registered_variables.density_index]
    u_L = primitives_left[registered_variables.velocity_index]

    rho_R = primitives_right[registered_variables.density_index]
    u_R = primitives_right[registered_variables.velocity_index]

    # if registered_variables.cosmic_ray_n_active:
    #     p_L = gas_pressure_from_primitives_with_crs(primitives_left, registered_variables)
    #     p_R = gas_pressure_from_primitives_with_crs(primitives_left, registered_variables)
    # else:
    #     p_L = primitives_left[registered_variables.pressure_index]
    #     p_R = primitives_right[registered_variables.pressure_index]

    p_L = primitives_left[registered_variables.pressure_index]
    p_R = primitives_right[registered_variables.pressure_index]

    # calculate the sound speeds
    c_L = speed_of_sound(rho_L, p_L, gamma)
    c_R = speed_of_sound(rho_R, p_R, gamma)
    
    # very simple approach for the wave velocities
    wave_speeds_right_plus = jnp.maximum(jnp.maximum(u_L + c_L, u_R + c_R), 0)
    wave_speeds_left_minus = jnp.minimum(jnp.minimum(u_L - c_L, u_R - c_R), 0)
    # wave_speeds_right_plus = jnp.abs(u_L) + c_L
    # wave_speeds_left_minus = jnp.abs(u_R) - c_R

    # calculate the maximum wave speed
    # get the maximum wave speed
    max_wave_speed = jnp.maximum(jnp.max(jnp.abs(wave_speeds_right_plus)), jnp.max(jnp.abs(wave_speeds_left_minus)))

    # calculate the time step
    dt = C_CFL * dx / max_wave_speed

    return jnp.minimum(dt, dt_max)

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['registered_variables'])
def _cfl_time_step3D(primitive_states: STATE_TYPE, dx: Union[float, Float[Array, ""]], dt_max: Union[float, Float[Array, ""]], gamma: Union[float, Float[Array, ""]], registered_variables: RegisteredVariables, C_CFL: Union[float, Float[Array, ""]] = 0.8) -> Float[Array, ""]:

    """Calculate the time step based on the CFL condition.

    Args:
        primitive_states: The primitive state array.
        dx: The cell width.
        dt_max: The maximum time step.
        gamma: The adiabatic index.
        C_CFL: The CFL number.

    Returns:
        The time step.
    
    """

    # wave speeds in x direction
    primitive_states_left = primitive_states[:, :-1, :, :]
    primitive_states_right = primitive_states[:, 1:, :, :]
    max_wave_speed_x = get_wave_speeds3D(primitive_states_left, primitive_states_right, gamma, registered_variables, registered_variables.velocity_index.x)

    # wave speeds in y direction
    primitive_states_left = primitive_states[:, :, :-1, :]
    primitive_states_right = primitive_states[:, :, 1:, :]
    max_wave_speed_y = get_wave_speeds3D(primitive_states_left, primitive_states_right, gamma, registered_variables, registered_variables.velocity_index.y)

    # wave speeds in z direction
    primitive_states_left = primitive_states[:, :, :, :-1]
    primitive_states_right = primitive_states[:, :, :, 1:]
    max_wave_speed_z = get_wave_speeds3D(primitive_states_left, primitive_states_right, gamma, registered_variables, registered_variables.velocity_index.z)

    # get the maximum wave speed
    max_wave_speed = jnp.maximum(jnp.maximum(max_wave_speed_x, max_wave_speed_y), max_wave_speed_z)

    # calculate the time step
    dt = C_CFL * dx / max_wave_speed

    return jnp.minimum(dt, dt_max)

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'registered_variables'])
def _source_term_aware_time_step(primitive_states: STATE_TYPE, config: SimulationConfig, params: SimulationParams, helper_data: HelperData, registered_variables: RegisteredVariables) -> Float[Array, ""]:
    """
    Calculate the time step based on the CFL condition and the source terms. What timestep
    would be chosen if the source terms were added under the current CFL time step?

    Args:
        state: The state array.
        config: The configuration.
        params: The parameters.
        helper_data: The helper data.

    Returns:
        The time step.
    """

    # == experimental: correct the CFL time step based on the physical sources ==
    
    # calculate the time step based on the CFL condition
    if config.dimensionality == 3:
        dt = _cfl_time_step3D(primitive_states, config.dx, params.dt_max, params.gamma, registered_variables, params.C_cfl)
    else:
        dt = _cfl_time_step(primitive_states, config.dx, params.dt_max, params.gamma, registered_variables, params.C_cfl)

    hypothetical_new_state = _run_physics_modules(primitive_states, dt, config, params, helper_data, registered_variables)

    if config.dimensionality == 3:
        dt = _cfl_time_step3D(hypothetical_new_state, config.dx, params.dt_max, params.gamma, registered_variables, params.C_cfl)
    else:
        dt = _cfl_time_step(hypothetical_new_state, config.dx, params.dt_max, params.gamma, registered_variables, params.C_cfl)
    
    # ===========================================================================

    return dt
