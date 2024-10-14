import jax.numpy as jnp
from jf1uids.fluid_equations.fluid import total_energy_from_primitives
import jax

from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

from typing import Union

@jaxtyped(typechecker=typechecker)
@jax.jit
def _euler_flux(primitive_states: Float[Array, "num_vars num_cells"], gamma: Union[float, Float[Array, ""]]) -> Float[Array, "num_vars num_cells"]:
    """Compute the Euler fluxes for the given primitive states.

    Args:
        primitive_states: The primitive state of the fluid on all cells.
        gamma: The adiabatic index of the fluid.

    Returns:
        The Euler fluxes for the given primitive states.

    """
    rho, u, p = primitive_states
    m = rho * u
    E = total_energy_from_primitives(rho, u, p, gamma)
    return jnp.stack([m, m * u + p, u * (E + p)], axis=0)