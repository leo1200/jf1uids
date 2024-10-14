import jax.numpy as jnp
import jax
from jf1uids.fluid_equations.euler import _euler_flux
from jf1uids.fluid_equations.fluid import speed_of_sound, conserved_state_from_primitive

from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

from typing import Union

@jaxtyped(typechecker=typechecker)
@jax.jit
def _hll_solver(primitives_left: Float[Array, "num_vars num_interfaces"], primitives_right: Float[Array, "num_vars num_interfaces"], gamma: Union[float, Float[Array, ""]]) -> Float[Array, "num_vars num_interfaces"]:
    """
    Returns the conservative fluxes.

    Args:
        primitives_left: States left of the interfaces.
        primitives_right: States right of the interfaces.
        gamma: The adiabatic index.

    Returns:
        The conservative fluxes at the interfaces.

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
    conservatives_left = conserved_state_from_primitive(primitives_left, gamma)
    conservatives_right = conserved_state_from_primitive(primitives_right, gamma)

    # calculate the interface HLL fluxes
    # F = (S_R * F_L - S_L * F_R + S_L * S_R * (U_R - U_L)) / (S_R - S_L)
    fluxes = (wave_speeds_right_plus * fluxes_left - wave_speeds_left_minus * fluxes_right + wave_speeds_left_minus * wave_speeds_right_plus * (conservatives_right - conservatives_left)) / (wave_speeds_right_plus - wave_speeds_left_minus)

    return fluxes