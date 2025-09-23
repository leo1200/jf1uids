# general
import jax.numpy as jnp
import jax
from functools import partial

# type checking
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker
from typing import Union
from jf1uids._riemann_solver.hll import _get_wave_speeds
from jf1uids._state_evolution.reconstruction import _reconstruct_at_interface_unsplit
from jf1uids.option_classes.simulation_config import STATE_TYPE, WITH_RECONSTRUCTION, WITHOUT_RECONSTRUCTION

# jf1uids containers
from jf1uids.data_classes.simulation_helper_data import HelperData
from jf1uids.option_classes.simulation_config import SimulationConfig
from jf1uids.option_classes.simulation_params import SimulationParams
from jf1uids.fluid_equations.registered_variables import RegisteredVariables

# jf1uids functions
from jf1uids.fluid_equations.fluid import speed_of_sound
from jf1uids._physics_modules._cosmic_rays.cr_fluid_equations import speed_of_sound_crs
from jf1uids._physics_modules.run_physics_modules import _run_physics_modules


@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'registered_variables'])
def _cfl_time_step(
    primitive_state: STATE_TYPE,
    grid_spacing: Union[float, Float[Array, ""]],
    dt_max: Union[float, Float[Array, ""]],
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    params: SimulationParams,
    helper_data: HelperData,
    C_CFL: Union[float, Float[Array, ""]] = 0.8
) -> Float[Array, ""]:

    """
    Calculate the time step based on the CFL condition.

    Args:
        primitive_state: The primitive state array.
        grid_spacing: The cell width.
        dt_max: The maximum time step.
        gamma: The adiabatic index.
        config: The simulation configuration.
        registered_variables: The registered variables.
        C_CFL: The CFL number.

    Returns:
        The time step.
    
    """

    # MISSING TEST: test for unsplit simulations

    if config.timestep_estimator == WITHOUT_RECONSTRUCTION:

        # timestep criterion as in Pang and Wu (2024) 
        # (https://arxiv.org/abs/2410.05173)

        # calculate the sound speed
        if not config.cosmic_ray_config.cosmic_rays:
            rho = primitive_state[registered_variables.density_index]
            p = primitive_state[registered_variables.pressure_index]
            c = speed_of_sound(rho, p, gamma)
        else:
            c = speed_of_sound_crs(primitive_state, registered_variables)

        alpha_lax = jnp.zeros((config.dimensionality,))
        for axis in range(1, config.dimensionality + 1):
            u = primitive_state[axis]
            alpha_lax_i = jnp.max(jnp.abs(u) + c)
            alpha_lax = alpha_lax.at[axis - 1].set(alpha_lax_i)

        dt = C_CFL * 1 / jnp.sum(alpha_lax / grid_spacing)

    # alternatively use a reconstruction of the left
    # and right states and wave speed estimates
    # as in the Riemann solvers

    elif config.timestep_estimator == WITH_RECONSTRUCTION:
        # we cannot use the MUSCL reconstruction as it requires
        # the timestep to be known beforehand

        primitives_left_interface, primitives_right_interface = _reconstruct_at_interface_unsplit(
            primitive_state,
            0.0, # not used
            gamma,
            config,
            params,
            helper_data,
            registered_variables
        )

        alpha_lax = jnp.zeros((config.dimensionality,))

        for axis in range(1, config.dimensionality + 1):
            S_L, S_R = _get_wave_speeds(
                primitives_left_interface[axis - 1],
                primitives_right_interface[axis - 1],
                gamma,
                config,
                registered_variables,
                axis
            )
            alpha_lax_i = jnp.max(jnp.maximum(jnp.abs(S_L), jnp.abs(S_R)))
            alpha_lax = alpha_lax.at[axis - 1].set(alpha_lax_i)

        dt = C_CFL * 1 / jnp.sum(alpha_lax / grid_spacing)

    return jnp.minimum(dt, dt_max)

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'registered_variables'])
def _source_term_aware_time_step(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
    current_time: Union[float, Float[Array, ""]]
) -> Float[Array, ""]:
    """
    Calculate the time step based on the CFL condition and the source terms. 
    What timestep would be chosen if the source terms were 
    added under the current CFL time step?

    Args:
        state: The state array.
        config: The configuration.
        params: The parameters.
        helper_data: The helper data.

    Returns:
        The time step.
    """
    
    # calculate the time step based on the CFL condition
    dt = _cfl_time_step(
        primitive_state,
        config.grid_spacing,
        params.dt_max,
        params.gamma,
        config,
        registered_variables,
        params,
        helper_data,
        params.C_cfl
    )

    hypothetical_new_state = _run_physics_modules(
        primitive_state,
        dt,
        config,
        params,
        helper_data,
        registered_variables,
        current_time
    )

    dt = _cfl_time_step(
        hypothetical_new_state,
        config.grid_spacing,
        params.dt_max,
        params.gamma,
        config,
        registered_variables,
        params,
        helper_data,
        params.C_cfl
    )

    return dt