from typing import NamedTuple
import jax.numpy as jnp
import jax
from jf1uids.fluid_equations.fluid import conserved_state, pressure_from_energy, primitive_state_from_conserved

from functools import partial

# wind injection schemes
MEO = 0 # momentum and energy overwrite
EI = 1 # thermal energy injection
MEI = 2 # momentum and energy injection

class WindConfig(NamedTuple):
    stellar_wind: bool = False
    num_injection_cells: int = 10
    wind_injection_scheme: int = EI

class WindParams(NamedTuple):
    wind_mass_loss_rate: float = 0.0
    wind_final_velocity: float = 0.0

    # only necesarry for the MEO injection scheme
    pressure_floor: float = 100000.0

@partial(jax.jit, static_argnames=['config'])
def _wind_injection(primitive_state, dt, config, params, helper_data):
        if config.wind_config.wind_injection_scheme == MEO:
            primitive_state = _wind_meo(params.wind_params, primitive_state, dt, helper_data, config.num_ghost_cells, config.wind_config.num_injection_cells, params.gamma)
        elif config.wind_config.wind_injection_scheme == MEI:
            primitive_state = _wind_mei(params.wind_params, primitive_state, dt, helper_data, config.num_ghost_cells, config.wind_config.num_injection_cells, params.gamma)
        elif config.wind_config.wind_injection_scheme == EI:
            primitive_state = _wind_ei(params.wind_params, primitive_state, dt, helper_data, config.num_ghost_cells, config.wind_config.num_injection_cells, params.gamma)
        else:
            raise ValueError("Invalid wind injection scheme")
    
        return primitive_state

# ================= Wind injection schemes =================

# here we implement all the injection schemes from
# https://arxiv.org/abs/2107.14673

@partial(jax.jit, static_argnames=['num_ghost_cells', 'num_injection_cells'])
def _wind_meo(wind_params, primitive_state, dt, helper_data, num_ghost_cells, num_injection_cells, gamma):

    # set density
    density_overwrite = wind_params.wind_mass_loss_rate / helper_data.cell_volumes[num_ghost_cells:num_injection_cells + num_ghost_cells] / wind_params.wind_final_velocity * (helper_data.outer_cell_boundaries[num_ghost_cells:num_injection_cells + num_ghost_cells] - helper_data.inner_cell_boundaries[num_ghost_cells:num_injection_cells + num_ghost_cells])
    primitive_state = primitive_state.at[0, num_ghost_cells:num_injection_cells + num_ghost_cells].set(density_overwrite)

    # set velocity
    primitive_state = primitive_state.at[1, num_ghost_cells:num_injection_cells + num_ghost_cells].set(wind_params.wind_final_velocity)

    # set pressure to the floor value
    primitive_state = primitive_state.at[2, num_ghost_cells:num_injection_cells + num_ghost_cells].set(wind_params.pressure_floor)

    return primitive_state

@partial(jax.jit, static_argnames=['num_ghost_cells', 'num_injection_cells'])
def _wind_mei(wind_params, primitive_state, dt, helper_data, num_ghost_cells, num_injection_cells, gamma):

    conservative_state = conserved_state(primitive_state, gamma)

    V_inj = 4/3 * jnp.pi * helper_data.outer_cell_boundaries[num_injection_cells + num_ghost_cells]**3

    drho = wind_params.wind_mass_loss_rate * dt / V_inj
    dmomentum = wind_params.wind_final_velocity * drho
    denergy = 0.5 * wind_params.wind_final_velocity**2 * drho

    conservative_state = conservative_state.at[0, num_ghost_cells:num_injection_cells + num_ghost_cells].add(drho)
    conservative_state = conservative_state.at[1, num_ghost_cells:num_injection_cells + num_ghost_cells].add(dmomentum)
    conservative_state = conservative_state.at[2, num_ghost_cells:num_injection_cells + num_ghost_cells].add(denergy)

    primitive_state = primitive_state_from_conserved(conservative_state, gamma)

    return primitive_state

# not really ei
@partial(jax.jit, static_argnames=['num_ghost_cells', 'num_injection_cells'])
def _wind_ei(wind_params, primitive_state, dt, helper_data, num_ghost_cells, num_injection_cells, gamma):

    source_term = jnp.zeros_like(primitive_state)
    
    r = helper_data.volumetric_centers
    r_inj = r[num_injection_cells + 2]
    V = 4/3 * jnp.pi * r_inj**3

    # V = jnp.sum(helper_data.cell_volumes[num_ghost_cells:num_injection_cells + num_ghost_cells])

    # mass injection
    drho_dt = wind_params.wind_mass_loss_rate / V
    source_term = source_term.at[0, num_ghost_cells:num_injection_cells + num_ghost_cells].set(drho_dt)
    updated_density = primitive_state[0, num_ghost_cells:num_injection_cells + num_ghost_cells] + drho_dt * dt

    # energy injection
    dE_dt = 0.5 * wind_params.wind_final_velocity**2 * wind_params.wind_mass_loss_rate / V

    dp_dt = pressure_from_energy(dE_dt, updated_density, primitive_state[1, num_ghost_cells:num_injection_cells + num_ghost_cells], gamma)

    source_term = source_term.at[2, num_ghost_cells:num_injection_cells + num_ghost_cells].set(dp_dt)

    primitive_state = primitive_state + source_term * dt

    return primitive_state

# ======================================================