import jax.numpy as jnp
import jax
from functools import partial

from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

from typing import Union

from jf1uids.data_classes.simulation_helper_data import HelperData

# The default state jf1uids operates on 
# are the primitive variables rho, u, p.

# ======= Create the primitive state ========

@jaxtyped(typechecker=typechecker)
@jax.jit
def construct_primitive_state(rho: Float[Array, "num_cells"], u: Float[Array, "num_cells"], p: Float[Array, "num_cells"]) -> Float[Array, "num_vars num_cells"]:
    """Stack the primitive variables into the state array.
    
    Args:
        rho: The density of the fluid.
        u: The velocity of the fluid.
        p: The pressure of the fluid.
        
    Returns:
        The state array.
    """
    return jnp.stack([rho, u, p], axis = 0)

@jaxtyped(typechecker=typechecker)
@jax.jit
def density(primitive_states: Float[Array, "num_vars num_cells"]) -> Float[Array, "num_cells"]:
    """Extract the density from the primitive state array.

    Args:
        primitive_states: The primitive state array.

    Returns:
        The density.
    """
    return primitive_states[0]

@jaxtyped(typechecker=typechecker)
@jax.jit
def velocity(primitive_states: Float[Array, "num_vars num_cells"]) -> Float[Array, "num_cells"]:
    """Extract the velocity from the primitive state array.

    Args:
        primitive_states: The primitive state array.

    Returns:
        The velocity.
    """
    return primitive_states[1]

@jaxtyped(typechecker=typechecker)
@jax.jit
def pressure(primitive_states: Float[Array, "num_vars num_cells"]) -> Float[Array, "num_cells"]:
    """Extract the pressure from the primitive state array.

    Args:
        primitive_states: The primitive state array.

    Returns:
        The pressure.
    """

    return primitive_states[2]

@jaxtyped(typechecker=typechecker)
@jax.jit
def primitive_state_from_conserved(conserved_state: Float[Array, "num_vars num_cells"], gamma: Union[float, Float[Array, ""]]) -> Float[Array, "num_vars num_cells"]:
    """Convert the conserved state to the primitive state.

    Args:
        conserved_state: The conserved state.
        gamma: The adiabatic index of the fluid.

    Returns:
        The primitive state.
    """
    rho, m, E = conserved_state
    u = m / rho
    p = pressure_from_energy(E, rho, u, gamma)
    return jnp.stack([rho, u, p], axis = 0)

# ===========================================

# ======= Create the conserved state ========

@jaxtyped(typechecker=typechecker)
@jax.jit
def conserved_state_from_primitive(primitive_states: Float[Array, "num_vars num_cells"], gamma: Union[float, Float[Array, ""]]) -> Float[Array, "num_vars num_cells"]:
    """Convert the primitive state to the conserved state.

    Args:
        primitive_state: The primitive state.
        gamma: The adiabatic index of the fluid.

    Returns:
        The conserved state.
    """

    rho, u, p = primitive_states
    E = total_energy_from_primitives(rho, u, p, gamma)
    return jnp.array([rho, rho * u, E])

# ===========================================

# =============== Fluid physics ===============

@jax.jit
def pressure_from_internal_energy(e, rho, gamma):
  """
  Calculate the pressure from the internal energy.
  
  Args:
      e: The internal energy.
      rho: The density.
      gamma: The adiabatic index.
      
  Returns:
      The pressure.
  """
  return (gamma - 1) * rho * e

@jax.jit
def internal_energy_from_energy(E, rho, u):
    """ Calculate the internal energy from the total energy.

    Args:
        E: The total energy.
        rho: The density.
        u: The velocity.

    Returns:
        The internal energy.
    """
    return E / rho - 0.5 * u**2

@jax.jit
def pressure_from_energy(E, rho, u, gamma):
    """ Calculate the pressure from the total energy.

    Args:
        E: The total energy.
        rho: The density.
        u: The velocity.
        gamma: The adiabatic index.

    Returns:
        The pressure.
    """

    e = internal_energy_from_energy(E, rho, u)
    return pressure_from_internal_energy(e, rho, gamma)

@jax.jit
def total_energy_from_primitives(rho, u, p, gamma):
    """Calculate the total energy from the primitive variables.

    Args:
        rho: The density.
        u: The velocity.
        p: The pressure.
        gamma: The adiabatic index.

    Returns:
        The total energy.
    """

    return p / (gamma - 1) + 0.5 * rho * u**2

@jax.jit
def speed_of_sound(rho, p, gamma):
    """Calculate the speed of sound.

    Args:
        rho: The density.
        p: The pressure.
        gamma: The adiabatic index.

    Returns:
        The speed of sound.
    """

    # produces nans if p or rho are negative
    return jnp.sqrt(gamma * p / rho)

# ===========================================

# ============= Total Quantities =============

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['num_ghost_cells'])
def calculate_total_mass(primitive_states: Float[Array, "num_vars num_cells"], helper_data: HelperData, num_ghost_cells: int) -> Float[Array, ""]:
    """
    Calculate the total mass in the domain.

    Args:
        primitive_states: The primitive state array.
        helper_data: The helper data.
        num_ghost_cells: The number of ghost cells.

    Returns:
        The total mass.
    """
    return jnp.sum(primitive_states[0, num_ghost_cells:-num_ghost_cells] * helper_data.cell_volumes[num_ghost_cells:-num_ghost_cells])

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['num_ghost_cells'])
def calculate_total_energy(primitive_states: Float[Array, "num_vars num_cells"], helper_data: HelperData, gamma: Union[float, Float[Array, ""]], num_ghost_cells: int) -> Float[Array, ""]:
    """
    Calculate the total energy in the domain.

    Args:
        primitive_states: The primitive state array.
        helper_data: The helper data.
        gamma: The adiabatic index.
        num_ghost_cells: The number of ghost cells.

    Returns:
        The total energy.
    """
    energy = total_energy_from_primitives(primitive_states[0, num_ghost_cells:-num_ghost_cells], primitive_states[1, num_ghost_cells:-num_ghost_cells], primitive_states[2, num_ghost_cells:-num_ghost_cells], gamma)
    return jnp.sum(energy * helper_data.cell_volumes[num_ghost_cells:-num_ghost_cells])