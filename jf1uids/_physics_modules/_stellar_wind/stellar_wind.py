
import jax.numpy as jnp
import jax
from jf1uids.data_classes.simulation_helper_data import HelperData
from jf1uids.fluid_equations.fluid import pressure_from_energy, primitive_state_from_conserved, conserved_state_from_primitive

from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

from functools import partial

from jf1uids.option_classes.simulation_config import SimulationConfig
from jf1uids.option_classes.simulation_params import SimulationParams

from jf1uids._physics_modules._stellar_wind.stellar_wind_options import WindParams
from jf1uids._physics_modules._stellar_wind.stellar_wind_options import WindConfig, MEO, MEI, EI

from typing import Union

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config'])
def _wind_injection(primitive_state: Float[Array, "num_vars num_cells"], dt: Float[Array, ""], config: SimulationConfig, params: SimulationParams, helper_data: HelperData) -> Float[Array, "num_vars num_cells"]:
    """Inject stellar wind into the simulation.

    Args:
        primitive_state: The primitive state array.
        dt: The time step.
        config: The simulation configuration.
        params: The simulation parameters.
        helper_data: The helper data.

    Returns:
        The primitive state array with the stellar wind injected.
    """
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

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['num_ghost_cells', 'num_injection_cells'])
def _wind_meo(wind_params: WindParams, primitive_state: Float[Array, "num_vars num_cells"], dt: Float[Array, ""], helper_data: HelperData, num_ghost_cells: int, num_injection_cells: int, gamma: Union[float, Float[Array, ""]]) -> Float[Array, "num_vars num_cells"]:
    """Inject stellar wind into the simulation by a mass-and-energy-overwrite scheme (MEO).

    Args:
        wind_params: The wind parameters.
        primitive_state: The primitive state array.
        dt: The time step.
        helper_data: The helper data.
        num_ghost_cells: The number of ghost cells.
        num_injection_cells: The number of injection cells.
        gamma: The adiabatic index.

    Returns:
        The primitive state array with the stellar wind injected.
    """

    # set density
    density_overwrite = wind_params.wind_mass_loss_rate / helper_data.cell_volumes[num_ghost_cells:num_injection_cells + num_ghost_cells] / wind_params.wind_final_velocity * (helper_data.outer_cell_boundaries[num_ghost_cells:num_injection_cells + num_ghost_cells] - helper_data.inner_cell_boundaries[num_ghost_cells:num_injection_cells + num_ghost_cells])
    primitive_state = primitive_state.at[0, num_ghost_cells:num_injection_cells + num_ghost_cells].set(density_overwrite)

    # set velocity
    primitive_state = primitive_state.at[1, num_ghost_cells:num_injection_cells + num_ghost_cells].set(wind_params.wind_final_velocity)

    # set pressure to the floor value
    primitive_state = primitive_state.at[2, num_ghost_cells:num_injection_cells + num_ghost_cells].set(wind_params.pressure_floor)

    return primitive_state

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['num_ghost_cells', 'num_injection_cells'])
def _wind_mei(wind_params: WindParams, primitive_state: Float[Array, "num_vars num_cells"], dt: Float[Array, ""], helper_data: HelperData, num_ghost_cells: int, num_injection_cells: int, gamma: Union[float, Float[Array, ""]]) -> Float[Array, "num_vars num_cells"]:
    """Inject stellar wind into the simulation by a momentum-and-energy-injection scheme (MEI).
    
    Args:
        wind_params: The wind parameters.
        primitive_state: The primitive state array.
        dt: The time step.
        helper_data: The helper data.
        num_ghost_cells: The number of ghost cells.
        num_injection_cells: The number of injection cells.
        gamma: The adiabatic index.
        
    Returns:
        The primitive state array with the stellar wind injected.
    """

    conservative_state = conserved_state_from_primitive(primitive_state, gamma)

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
@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['num_ghost_cells', 'num_injection_cells'])
def _wind_ei(wind_params: WindParams, primitive_state: Float[Array, "num_vars num_cells"], dt: Float[Array, ""], helper_data: HelperData, num_ghost_cells: int, num_injection_cells: int, gamma: Union[float, Float[Array, ""]]) -> Float[Array, "num_vars num_cells"]:
    """Inject stellar wind into the simulation by an thermal-energy-injection scheme (EI).

    Args:
        wind_params: The wind parameters.
        primitive_state: The primitive state array.
        dt: The time step.
        helper_data: The helper data.
        num_ghost_cells: The number of ghost cells.
        num_injection_cells: The number of injection cells.
        gamma: The adiabatic index.

    Returns:
        The primitive state array with the stellar wind injected.
    """

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