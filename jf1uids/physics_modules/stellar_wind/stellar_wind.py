from typing import NamedTuple
import jax.numpy as jnp
import jax
from jf1uids.fluid_equations.fluid import pressure_from_energy

class WindParams(NamedTuple):
    wind_mass_loss_rate: float = 0.0
    wind_final_velocity: float = 0.0

@jax.jit
def wind_source(wind_params, primitive_state, r, dr):

    source_term = jnp.zeros_like(primitive_state)

    num_injection_cells = 10

    r_inj = r[num_injection_cells + 2]
    V = 4/3 * jnp.pi * r_inj**3

    drho_dt = wind_params.wind_mass_loss_rate / V
    dE_dt = 0.5 * wind_params.wind_final_velocity**2 * wind_params.wind_mass_loss_rate / V

    # better encompass the change in rho
    gamma = 5/3
    dp_dt = pressure_from_energy(dE_dt, primitive_state[0, 2:num_injection_cells + 2], primitive_state[1, 2:num_injection_cells + 2], gamma)

    source_term = source_term.at[0, 2:num_injection_cells + 2].set(drho_dt)
    source_term = source_term.at[2, 2:num_injection_cells + 2].set(dp_dt)

    return source_term