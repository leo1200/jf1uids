from functools import partial
import jax.numpy as jnp
import jax

from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker
from typing import Union

from jf1uids._physics_modules._cosmic_rays.cr_fluid_equations import total_energy_from_primitives_with_crs
from jf1uids.fluid_equations.fluid import get_absolute_velocity, total_energy_from_primitives
from jf1uids.fluid_equations.registered_variables import RegisteredVariables
from jf1uids.option_classes.simulation_config import STATE_TYPE, SimulationConfig

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'registered_variables', 'flux_direction_index'])
def _euler_flux(
    primitive_state: STATE_TYPE,
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    flux_direction_index: int
) -> STATE_TYPE:
    """Compute the Euler fluxes for the given primitive states.

    Args:
        primitive_state: The primitive state of the fluid on all cells.
        gamma: The adiabatic index of the fluid.
        registered_variables: The registered variables.
        flux_direction_index: The index of the velocity component in the flux direction of interest.

    Returns:
        The Euler fluxes for the given primitive states.
    """
    rho = primitive_state[registered_variables.density_index]
    p = primitive_state[registered_variables.pressure_index]
    
    # start with a copy of the primitive states
    flux_vector = primitive_state.copy() # copy should not be necessary (no side effects)

    # calculate the total energy
    utotal = get_absolute_velocity(primitive_state, config, registered_variables)

    if registered_variables.cosmic_ray_n_active:
        E = total_energy_from_primitives_with_crs(primitive_state, registered_variables)
    else:
        E = total_energy_from_primitives(rho, utotal, p, gamma)


    # add the total energy to the pressure_index of the flux vector
    flux_vector = flux_vector.at[registered_variables.pressure_index].add(E)

    # scale the velocity components with the density
    if config.dimensionality == 1:
        flux_vector = flux_vector.at[registered_variables.velocity_index].set(primitive_state[registered_variables.velocity_index] * rho)
    elif config.dimensionality == 2:
        flux_vector = flux_vector.at[registered_variables.velocity_index.x].set(primitive_state[registered_variables.velocity_index.x] * rho)
        flux_vector = flux_vector.at[registered_variables.velocity_index.y].set(primitive_state[registered_variables.velocity_index.y] * rho)
    elif config.dimensionality == 3:
        flux_vector = flux_vector.at[registered_variables.velocity_index.x].set(primitive_state[registered_variables.velocity_index.x] * rho)
        flux_vector = flux_vector.at[registered_variables.velocity_index.y].set(primitive_state[registered_variables.velocity_index.y] * rho)
        flux_vector = flux_vector.at[registered_variables.velocity_index.z].set(primitive_state[registered_variables.velocity_index.z] * rho)

    # multiply the whole vector with the velocity component in the flux direction
    flux_vector = primitive_state[flux_direction_index] * flux_vector
    
    # add the pressure to the velocity component in the flux direction
    flux_vector = flux_vector.at[flux_direction_index].add(p)

    return flux_vector
