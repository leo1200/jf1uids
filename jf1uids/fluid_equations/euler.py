from functools import partial
import jax.numpy as jnp
from jf1uids._geometry.geometry import STATE_TYPE
from jf1uids._physics_modules._cosmic_rays.cr_fluid_equations import total_energy_from_primitives_with_crs
from jf1uids.fluid_equations.fluid import total_energy_from_primitives
import jax

from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

from typing import Union

from jf1uids.fluid_equations.registered_variables import RegisteredVariables

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['registered_variables'])
def _euler_flux(primitive_states: STATE_TYPE, gamma: Union[float, Float[Array, ""]], registered_variables: RegisteredVariables) -> STATE_TYPE:
    """Compute the Euler fluxes for the given primitive states.

    Args:
        primitive_states: The primitive state of the fluid on all cells.
        gamma: The adiabatic index of the fluid.

    Returns:
        The Euler fluxes for the given primitive states.

    """
    rho = primitive_states[registered_variables.density_index]
    u = primitive_states[registered_variables.velocity_index]
    p = primitive_states[registered_variables.pressure_index]


    m = rho * u

    if registered_variables.cosmic_ray_n_active:
        E = total_energy_from_primitives_with_crs(primitive_states, registered_variables)
    else:
        E = total_energy_from_primitives(rho, u, p, gamma)

    # write flux vector
    flux_vector = jnp.zeros_like(primitive_states)
    flux_vector = flux_vector.at[registered_variables.density_index].set(m)
    flux_vector = flux_vector.at[registered_variables.velocity_index].set(m * u + p)
    flux_vector = flux_vector.at[registered_variables.pressure_index].set(u * (E + p))

    # for possible additional variables, check if they are registered and add fluxes
    # as needed
    if registered_variables.wind_density_active:
        flux_vector = flux_vector.at[registered_variables.wind_density_index].set(u * primitive_states[registered_variables.wind_density_index])

    if registered_variables.cosmic_ray_n_active:
        flux_vector = flux_vector.at[registered_variables.cosmic_ray_n_index].set(u * primitive_states[registered_variables.cosmic_ray_n_index])

    return flux_vector


@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['registered_variables', 'flux_direction_index'])
def _euler_flux3D(primitive_states: STATE_TYPE, gamma: Union[float, Float[Array, ""]], registered_variables: RegisteredVariables, flux_direction_index: int) -> STATE_TYPE:
    """Compute the Euler fluxes for the given primitive states.

    Args:
        primitive_states: The primitive state of the fluid on all cells.
        gamma: The adiabatic index of the fluid.
        registered_variables: The registered variables.
        flux_direction_index: The index of the velocity component in the flux direction of interest.

    Returns:
        The Euler fluxes for the given primitive states.
    """
    rho = primitive_states[registered_variables.density_index]
    p = primitive_states[registered_variables.pressure_index]
    u = jnp.sqrt(primitive_states[registered_variables.velocity_index.x]**2 + primitive_states[registered_variables.velocity_index.y]**2 + primitive_states[registered_variables.velocity_index.z]**2)

    # start with a copy of the primitive states
    flux_vector = primitive_states.copy() # copy should not be necessary (no side effects)

    # calculate the total energy
    E = total_energy_from_primitives(rho, u, p, gamma)

    # add the total energy to the pressure_index of the flux vector
    flux_vector = flux_vector.at[registered_variables.pressure_index].add(E)

    # scale the velocity components with the density
    flux_vector = flux_vector.at[registered_variables.velocity_index.x].set(primitive_states[registered_variables.velocity_index.x] * rho)
    flux_vector = flux_vector.at[registered_variables.velocity_index.y].set(primitive_states[registered_variables.velocity_index.y] * rho)
    flux_vector = flux_vector.at[registered_variables.velocity_index.z].set(primitive_states[registered_variables.velocity_index.z] * rho)

    # multiply the whole vector with the velocity component in the flux direction
    flux_vector = primitive_states[flux_direction_index] * flux_vector
    
    # add the pressure to the velocity component in the flux direction
    flux_vector = flux_vector.at[flux_direction_index].add(p)

    return flux_vector
