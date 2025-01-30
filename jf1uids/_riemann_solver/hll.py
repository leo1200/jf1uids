# general imports
from functools import partial
import jax.numpy as jnp
import jax

# typing
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker
from typing import Union

# general jf1uids
from jf1uids._geometry.geometry import STATE_TYPE
from jf1uids.fluid_equations.registered_variables import RegisteredVariables

# fluid stuff
from jf1uids.fluid_equations.fluid import conserved_state_from_primitive, speed_of_sound
from jf1uids.fluid_equations.euler import _euler_flux
from jf1uids.option_classes.simulation_config import SimulationConfig


@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'registered_variables', 'flux_direction_index'])
def _hll_solver(primitives_left: STATE_TYPE, primitives_right: STATE_TYPE, gamma: Union[float, Float[Array, ""]], config: SimulationConfig, registered_variables: RegisteredVariables, flux_direction_index: int) -> STATE_TYPE:
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

    # get the left and right states and fluxes
    fluxes_left = _euler_flux(primitives_left, gamma, config, registered_variables, flux_direction_index)
    fluxes_right = _euler_flux(primitives_right, gamma, config, registered_variables, flux_direction_index)
    
    # very simple approach for the wave velocities
    wave_speeds_right_plus = jnp.maximum(jnp.maximum(u_L + c_L, u_R + c_R), 0)
    wave_speeds_left_minus = jnp.minimum(jnp.minimum(u_L - c_L, u_R - c_R), 0)

    # wave_speeds_right_plus = jnp.maximum(u_L - c_L, 0)
    # wave_speeds_left_minus = jnp.minimum(u_R + c_R, 0)

    # get the left and right conserved variables
    conservatives_left = conserved_state_from_primitive(primitives_left, gamma, config, registered_variables)
    conservatives_right = conserved_state_from_primitive(primitives_right, gamma, config, registered_variables)

    # calculate the interface HLL fluxes
    # F = (S_R * F_L - S_L * F_R + S_L * S_R * (U_R - U_L)) / (S_R - S_L)
    fluxes = (wave_speeds_right_plus * fluxes_left - wave_speeds_left_minus * fluxes_right + wave_speeds_left_minus * wave_speeds_right_plus * (conservatives_right - conservatives_left)) / (wave_speeds_right_plus - wave_speeds_left_minus)

    return fluxes