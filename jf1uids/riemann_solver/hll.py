import jax.numpy as jnp
import jax
from jf1uids.fluid_equations.euler import _euler_flux
from jf1uids.fluid_equations.fluid import conserved_state, speed_of_sound

@jax.jit
def _hll_solver(primitives_left, primitives_right, gamma):
    """
    Returns the conservative fluxes.
    """
    rho_L, u_L, p_L = primitives_left
    rho_R, u_R, p_R = primitives_right

    # calculate the sound speeds
    c_L = speed_of_sound(rho_L, p_L, gamma)
    c_R = speed_of_sound(rho_R, p_R, gamma)

    # get the left and right states and fluxes
    fluxes_left = _euler_flux(primitives_left, gamma)
    fluxes_right = _euler_flux(primitives_right, gamma)
    
    # very simple approach for the wave velocities
    wave_speeds_right_plus = jnp.maximum(jnp.maximum(u_L + c_L, u_R + c_R), 0)
    wave_speeds_left_minus = jnp.minimum(jnp.minimum(u_L - c_L, u_R - c_R), 0)

    # get the left and right conserved variables
    conservatives_left = conserved_state(primitives_left, gamma)
    conservatives_right = conserved_state(primitives_right, gamma)

    # calculate the interface HLL fluxes
    # F = (S_R * F_L - S_L * F_R + S_L * S_R * (U_R - U_L)) / (S_R - S_L)
    fluxes = (wave_speeds_right_plus * fluxes_left - wave_speeds_left_minus * fluxes_right + wave_speeds_left_minus * wave_speeds_right_plus * (conservatives_right - conservatives_left)) / (wave_speeds_right_plus - wave_speeds_left_minus)

    return fluxes