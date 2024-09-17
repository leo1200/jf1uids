import jax.numpy as jnp
import jax
from functools import partial

# The default state jf1uids operates on 
# are the primitive variables rho, u, p.

# ======= Create the primitive state ========

@jax.jit
def primitive_state(rho, u, p):
    return jnp.stack([rho, u, p], axis = 0)

@jax.jit
def density(state):
    return state[0]

@jax.jit
def velocity(state):
    return state[1]

@jax.jit
def pressure(state):
    return state[2]

@jax.jit
def primitive_state_from_conserved(conserved_state, gamma):
    rho, m, E = conserved_state
    u = m / rho
    p = pressure_from_energy(E, rho, u, gamma)
    return jnp.stack([rho, u, p], axis = 0)

# ===========================================

# ======= Create the conserved state ========

@jax.jit
def conserved_state(primitive_state, gamma):
    rho, u, p = primitive_state
    E = total_energy_from_primitives(rho, u, p, gamma)
    return jnp.array([rho, rho * u, E])

# ===========================================

# =============== Fluid physics ===============

@jax.jit
def pressure_from_internal_energy(e, rho, gamma):
  return (gamma - 1) * rho * e

@jax.jit
def internal_energy_from_energy(E, rho, u):
    return E / rho - 0.5 * u**2

@jax.jit
def pressure_from_energy(E, rho, u, gamma):
    e = internal_energy_from_energy(E, rho, u)
    return pressure_from_internal_energy(e, rho, gamma)

@jax.jit
def total_energy_from_primitives(rho, u, p, gamma):
    return p / (gamma - 1) + 0.5 * rho * u**2

@jax.jit
def speed_of_sound(rho, p, gamma):
    return jnp.sqrt(gamma * p / rho)

# ===========================================

# ============= Total Quantities =============

@partial(jax.jit, static_argnames=['num_ghost_cells'])
def calculate_total_mass_proxy(primitive_state, helper_data, dx, num_ghost_cells):
    """
    Calculate the total mass in the domain.
    """
    return jnp.sum(primitive_state[0, num_ghost_cells:-num_ghost_cells] * helper_data.r_hat_alpha[num_ghost_cells:-num_ghost_cells] * dx)

@partial(jax.jit, static_argnames=['num_ghost_cells'])
def calculate_total_energy_proxy(primitive_state, helper_data, dx, gamma, num_ghost_cells):
    """
    Calculate the total energy in the domain.
    """
    energy = total_energy_from_primitives(primitive_state[0, num_ghost_cells:-num_ghost_cells], primitive_state[1, num_ghost_cells:-num_ghost_cells], primitive_state[2, num_ghost_cells:-num_ghost_cells], gamma)
    return jnp.sum(energy * helper_data.r_hat_alpha[num_ghost_cells:-num_ghost_cells] * dx)