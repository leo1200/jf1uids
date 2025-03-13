# general imports
from functools import partial
import jax.numpy as jnp
import jax

# typing
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker
from typing import Union

# general jf1uids
from jf1uids._physics_modules._cosmic_rays.cr_fluid_equations import speed_of_sound_crs
from jf1uids.fluid_equations.registered_variables import RegisteredVariables

# fluid stuff
from jf1uids.fluid_equations.fluid import conserved_state_from_primitive, speed_of_sound
from jf1uids.fluid_equations.euler import _euler_flux
from jf1uids.option_classes.simulation_config import HLLC_LM, STATE_TYPE, SimulationConfig


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

    p_L = primitives_left[registered_variables.pressure_index]
    p_R = primitives_right[registered_variables.pressure_index]

    # calculate the sound speeds
    if not config.cosmic_ray_config.cosmic_rays:
        c_L = speed_of_sound(rho_L, p_L, gamma)
        c_R = speed_of_sound(rho_R, p_R, gamma)
    else:
        c_L = speed_of_sound_crs(primitives_left, registered_variables)
        c_R = speed_of_sound_crs(primitives_right, registered_variables)

    # get the left and right states and fluxes
    fluxes_left = _euler_flux(primitives_left, gamma, config, registered_variables, flux_direction_index)
    fluxes_right = _euler_flux(primitives_right, gamma, config, registered_variables, flux_direction_index)
    
    # very simple approach for the wave velocities
    wave_speeds_right_plus = jnp.maximum(jnp.maximum(u_L + c_L, u_R + c_R), 0)
    wave_speeds_left_minus = jnp.minimum(jnp.minimum(u_L - c_L, u_R - c_R), 0)

    # wave_speeds_right_plus = jnp.maximum(u_L - c_L, 0)
    # wave_speeds_left_minus = jnp.minimum(u_R + c_R, 0)

    # get the left and right conserved variables
    conserved_left = conserved_state_from_primitive(primitives_left, gamma, config, registered_variables)
    conserved_right = conserved_state_from_primitive(primitives_right, gamma, config, registered_variables)

    # calculate the interface HLL fluxes
    # F = (S_R * F_L - S_L * F_R + S_L * S_R * (U_R - U_L)) / (S_R - S_L)
    fluxes = (wave_speeds_right_plus * fluxes_left - wave_speeds_left_minus * fluxes_right + wave_speeds_left_minus * wave_speeds_right_plus * (conserved_right - conserved_left)) / (wave_speeds_right_plus - wave_speeds_left_minus)

    return fluxes

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'registered_variables', 'flux_direction_index'])
def _hllc_solver(primitives_left: STATE_TYPE, primitives_right: STATE_TYPE, gamma: Union[float, Float[Array, ""]], config: SimulationConfig, registered_variables: RegisteredVariables, flux_direction_index: int) -> STATE_TYPE:
    """
    Returns the conservative fluxes.

    There seem to be problems for 1d radial, maybe because the same stuff that has to be
    taken into consideration in the general scheme (averaging based on density might be
    problematic, as the surface increases radially, ...) is not taken into account here.
    Maybe interesting for future research, for now HLL works fine.

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
    if not config.cosmic_ray_config.cosmic_rays:
        c_L = speed_of_sound(rho_L, p_L, gamma)
        c_R = speed_of_sound(rho_R, p_R, gamma)
    else:
        c_L = speed_of_sound_crs(primitives_left, registered_variables)
        c_R = speed_of_sound_crs(primitives_right, registered_variables)

    # get the left and right states and fluxes
    F_L = _euler_flux(primitives_left, gamma, config, registered_variables, flux_direction_index)
    F_R = _euler_flux(primitives_right, gamma, config, registered_variables, flux_direction_index)

    # Roe average of the velocity
    u_hat = (jnp.sqrt(rho_L) * u_L + jnp.sqrt(rho_R) * u_R) / (jnp.sqrt(rho_L) + jnp.sqrt(rho_R))

    # Roe average of the sound speed
    c_hat_squared = (c_L ** 2 * jnp.sqrt(rho_L) + c_R ** 2 * jnp.sqrt(rho_R)) / (jnp.sqrt(rho_L) + jnp.sqrt(rho_R)) + 0.5 * (jnp.sqrt(rho_L) * jnp.sqrt(rho_R) / (jnp.sqrt(rho_L) + jnp.sqrt(rho_R)) ** 2) * (u_R - u_L) ** 2
    c_hat = jnp.sqrt(c_hat_squared)

    # Einfeldt estimates of maximum left and right signal speeds
    S_L = jnp.minimum(u_L - c_L, u_hat - c_hat)
    S_R = jnp.maximum(u_R + c_R, u_hat + c_hat)

    # contact wave signal speed
    S_star = (p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R)) / (rho_L * (S_L - u_L) - rho_R * (S_R - u_R))

    # intermediate states
    U_L = conserved_state_from_primitive(primitives_left, gamma, config, registered_variables)
    U_R = conserved_state_from_primitive(primitives_right, gamma, config, registered_variables)

    U_star_L = U_L.at[flux_direction_index].set(rho_L * S_star)
    U_star_L = U_star_L.at[registered_variables.pressure_index].add((S_star - u_L) * (rho_L * S_star + p_L / (S_L - u_L)))
    U_star_L = U_star_L * (S_L - u_L) / (S_L - S_star)

    U_star_R = U_R.at[flux_direction_index].set(rho_R * S_star)
    U_star_R = U_star_R.at[registered_variables.pressure_index].add((S_star - u_R) * (rho_R * S_star + p_R / (S_R - u_R)))
    U_star_R = U_star_R * (S_R - u_R) / (S_R - S_star)

    # HLLC-LM adaptation
    # following
    # https://doi.org/10.1016/j.jcp.2020.109762
    if config.riemann_solver == HLLC_LM:
        Ma_limit = 0.1
        Ma_local = jnp.maximum(jnp.abs(u_L / c_L), jnp.abs(u_R / c_R))
        phi = jnp.sin(jnp.minimum(1, Ma_local / Ma_limit) * jnp.pi / 2)
        S_L = S_L * phi
        S_R = S_R * phi

    # calculate the interface HLLC fluxes
    F_star = 0.5 * (F_L + F_R) + 0.5 * (S_L * (U_star_L - U_L) + jnp.abs(S_star) * (U_star_L - U_star_R) + S_R * (U_star_R - U_R))
    fluxes = jnp.where(S_L >= 0, F_L, F_star)
    fluxes = jnp.where(S_R <= 0, F_R, fluxes)

    return fluxes