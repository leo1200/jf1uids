import jax.numpy as jnp
import jax
from functools import partial

from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

from typing import Union

from jf1uids.fluid_equations.registered_variables import RegisteredVariables


# TODO: make 2D and 3D ready
@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['registered_variables'])
def total_energy_from_primitives_with_crs(
        primitive_state: Float[Array, "num_vars num_cells"],
        registered_variables: RegisteredVariables
    ) -> Float[Array, "num_cells"]:

    # TODO: get from params
    gamma_cr = 4/3
    gamma_gas = 5/3

    # get the cosmic ray pressure
    cosmic_ray_pressure = primitive_state[registered_variables.cosmic_ray_n_index] ** gamma_cr

    # get the cosmic ray energy (density)
    cosmic_ray_energy = cosmic_ray_pressure / (gamma_cr - 1)

    # get the gas pressure
    gas_pressure = primitive_state[registered_variables.pressure_index] - cosmic_ray_pressure

    # get the gas energy
    rho_gas = primitive_state[registered_variables.density_index]
    velocity = primitive_state[registered_variables.velocity_index]
    gas_energy = gas_pressure / (gamma_gas - 1) + 0.5 * rho_gas * velocity**2

    # total energy
    E_tot = gas_energy + cosmic_ray_energy

    return E_tot

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['registered_variables'])
def gas_pressure_from_primitives_with_crs(
        primitive_state: Float[Array, "num_vars num_cells"],
        registered_variables: RegisteredVariables
    ) -> Float[Array, "num_cells"]:

    # TODO: get from configs
    gamma_cr = 4/3
    gamma_gas = 5/3

    # get the cosmic ray pressure
    cosmic_ray_pressure = primitive_state[registered_variables.cosmic_ray_n_index] ** gamma_cr

    # return the gas pressure
    return primitive_state[registered_variables.pressure_index] - cosmic_ray_pressure

# TODO: make 2D and 3D ready
@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['registered_variables'])
def total_pressure_from_conserved_with_crs(
        conserved_state: Float[Array, "num_vars num_cells"],
        registered_variables: RegisteredVariables
    ) -> Float[Array, "num_cells"]:

    # TODO: get from configs
    gamma_cr = 4/3
    gamma_gas = 5/3

    # get the cosmic ray pressure
    cosmic_ray_pressure = conserved_state[registered_variables.cosmic_ray_n_index] ** gamma_cr

    # get the cosmic ray energy (density)
    cosmic_ray_energy = cosmic_ray_pressure / (gamma_cr - 1)

    # get the gas energy
    gas_energy = conserved_state[registered_variables.pressure_index] - cosmic_ray_energy

    # get the gas pressure
    rho_gas = conserved_state[registered_variables.density_index]
    velocity = conserved_state[registered_variables.velocity_index] / rho_gas
    gas_pressure = (gas_energy - 0.5 * rho_gas * velocity**2) * (gamma_gas - 1)

    # get the total pressure
    total_pressure = cosmic_ray_pressure + gas_pressure

    return total_pressure