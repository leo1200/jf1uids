import jax.numpy as jnp
import jax
from functools import partial

from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

from typing import Union

from jf1uids.fluid_equations.registered_variables import RegisteredVariables

# TODO: also write CR init function
@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['registered_variables'])
def get_primitive_state_with_crs(
        gas_density: Float[Array, "num_cells"],
        gas_velocity: Float[Array, "num_cells"],
        gas_pressure: Float[Array, "num_cells"],
        cosmic_ray_pressure: Float[Array, "num_cells"],
        registered_variables: RegisteredVariables
    ) -> Float[Array, "num_vars num_cells"]:

    # TODO: get from configs
    gamma_cr = 4/3

    state = jnp.zeros((registered_variables.num_vars, gas_density.shape[0]))
    state = state.at[registered_variables.density_index].set(gas_density)
    state = state.at[registered_variables.velocity_index].set(gas_velocity)
    state = state.at[registered_variables.pressure_index].set(gas_pressure + cosmic_ray_pressure)
    state = state.at[registered_variables.cosmic_ray_n_index].set(cosmic_ray_pressure ** (1/gamma_cr))

    return state

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['registered_variables'])
def total_energy_from_primitives_with_crs(
        primitive_states: Float[Array, "num_vars num_cells"],
        registered_variables: RegisteredVariables
    ) -> Float[Array, "num_cells"]:

    # TODO: get from configs
    gamma_cr = 4/3
    gamma_gas = 5/3

    # get the cosmic ray pressure
    cosmic_ray_pressure = primitive_states[registered_variables.cosmic_ray_n_index] ** gamma_cr

    # get the cosmic ray energy (density)
    cosmic_ray_energy = cosmic_ray_pressure / (gamma_cr - 1)

    # get the gas pressure
    gas_pressure = primitive_states[registered_variables.pressure_index] - cosmic_ray_pressure

    # get the gas energy
    rho_gas = primitive_states[registered_variables.density_index]
    velocity = primitive_states[registered_variables.velocity_index]
    gas_energy = gas_pressure / (gamma_gas - 1) + 0.5 * rho_gas * velocity**2

    # total energy
    E_tot = gas_energy + cosmic_ray_energy

    return E_tot

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['registered_variables'])
def gas_pressure_from_primitives_with_crs(
        primitive_states: Float[Array, "num_vars num_cells"],
        registered_variables: RegisteredVariables
    ) -> Float[Array, "num_cells"]:

    # TODO: get from configs
    gamma_cr = 4/3
    gamma_gas = 5/3

    # get the cosmic ray pressure
    cosmic_ray_pressure = primitive_states[registered_variables.cosmic_ray_n_index] ** gamma_cr

    # return the gas pressure
    return primitive_states[registered_variables.pressure_index] - cosmic_ray_pressure

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['registered_variables'])
def total_pressure_from_conserved_with_crs(
        conserved_states: Float[Array, "num_vars num_cells"],
        registered_variables: RegisteredVariables
    ) -> Float[Array, "num_cells"]:

    # TODO: get from configs
    gamma_cr = 4/3
    gamma_gas = 5/3

    # get the cosmic ray pressure
    cosmic_ray_pressure = conserved_states[registered_variables.cosmic_ray_n_index] ** gamma_cr

    # get the cosmic ray energy (density)
    cosmic_ray_energy = cosmic_ray_pressure / (gamma_cr - 1)

    # get the gas energy
    gas_energy = conserved_states[registered_variables.pressure_index] - cosmic_ray_energy

    # get the gas pressure
    rho_gas = conserved_states[registered_variables.density_index]
    velocity = conserved_states[registered_variables.velocity_index] / rho_gas
    gas_pressure = (gas_energy - 0.5 * rho_gas * velocity**2) * (gamma_gas - 1)

    # get the total pressure
    total_pressure = cosmic_ray_pressure + gas_pressure

    return total_pressure