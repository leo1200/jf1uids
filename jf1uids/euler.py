import jax.numpy as jnp
from jf1uids.fluid import total_energy_from_primitives
import jax

from functools import partial

@jax.jit
def euler_flux(primitive_states, gamma):
    rho, u, p = primitive_states
    m = rho * u
    E = total_energy_from_primitives(rho, u, p, gamma)
    return jnp.stack([m, m * u + p, u * (E + p)], axis=0)

