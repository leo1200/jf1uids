import jax.numpy as jnp
from jf1uids.fluid_equations.euler import _euler_flux
from jf1uids.fluid_equations.fluid import speed_of_sound
import jax
from functools import partial

from jf1uids.physics_modules.run_physics_modules import _run_physics_modules

@jax.jit
def _cfl_time_step(primitive_states, dx, dt_max, gamma, C_CFL = 0.8):
    primitives_left = primitive_states[:, :-1]
    primitives_right = primitive_states[:, 1:]

    rho_L, u_L, p_L = primitives_left
    rho_R, u_R, p_R = primitives_right

    # calculate the sound speeds
    c_L = speed_of_sound(rho_L, p_L, gamma)
    c_R = speed_of_sound(rho_R, p_R, gamma)
    
    # very simple approach for the wave velocities
    wave_speeds_right_plus = jnp.maximum(jnp.maximum(u_L + c_L, u_R + c_R), 0)
    wave_speeds_left_minus = jnp.minimum(jnp.minimum(u_L - c_L, u_R - c_R), 0)

    # calculate the maximum wave speed
    # get the maximum wave speed
    max_wave_speed = jnp.maximum(jnp.max(jnp.abs(wave_speeds_right_plus)), jnp.max(jnp.abs(wave_speeds_left_minus)))

    # calculate the time step
    dt = C_CFL * dx / max_wave_speed

    return jnp.minimum(dt, dt_max)

@partial(jax.jit, static_argnames=['config'])
def _source_term_aware_time_step(state, config, params, helper_data):
    """
    Calculate the time step based on the CFL condition and the source terms
    """

    # calculate the time step based on the CFL condition
    dt = _cfl_time_step(state, config.dx, params.dt_max, params.gamma, params.C_cfl)

    # == experimental: correct the CFL time step based on the physical sources ==
    hypothetical_new_state = _run_physics_modules(state, dt, config, params, helper_data)
    dt = _cfl_time_step(hypothetical_new_state, config.dx, params.dt_max, params.gamma, params.C_cfl)
    # ===========================================================================

    return dt
