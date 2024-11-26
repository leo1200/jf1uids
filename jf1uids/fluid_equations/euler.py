from functools import partial
import jax.numpy as jnp
from jf1uids.fluid_equations.fluid import total_energy_from_primitives
import jax

from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

from typing import Union

from jf1uids.fluid_equations.registered_variables import RegisteredVariables

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['registered_variables'])
def _euler_flux(primitive_states: Float[Array, "num_vars num_cells"], gamma: Union[float, Float[Array, ""]], registered_variables: RegisteredVariables) -> Float[Array, "num_vars num_cells"]:
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
    E = total_energy_from_primitives(rho, u, p, gamma)

    # write flux vector
    flux_vector = jnp.zeros_like(primitive_states)
    flux_vector = flux_vector.at[registered_variables.density_index].set(m)
    flux_vector = flux_vector.at[registered_variables.velocity_index].set(m * u + p)
    flux_vector = flux_vector.at[registered_variables.pressure_index].set(u * (E + p))

    # for possible additional variables, check if they are registered and add fluxes
    # as needed
    if registered_variables.wind_density_active:
        flux_vector = flux_vector.at[registered_variables.wind_density_index].set(m)

    return flux_vector