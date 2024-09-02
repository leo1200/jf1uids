import jax.numpy as jnp

# The default state jf1uids operates on 
# are the primitive variables rho, u, p.

# The conserved variables are (rho, rho*u, E) * r^\alpha
# where \alpha specifies the geometry.

# ======= Create the primitive state ========

def primitive_state(rho, u, p):
    return jnp.stack([rho, u, p], axis = 0)

def density(state):
    return state[0]

def velocity(state):
    return state[1]

def pressure(state):
    return state[2]

def primitive_state_from_conserved(conserved_state, gamma): # , r, alpha):
    rho, m, E = conserved_state # * r**(-alpha)
    u = m / rho
    p = pressure_from_energy(E, rho, u, gamma)
    return jnp.stack([rho, u, p], axis = 0)

# ===========================================

# ======= Create the conserved state ========

def conserved_state(primitive_state, gamma): #, r, alpha):
    rho, u, p = primitive_state
    E = total_energy_from_primitives(rho, u, p, gamma)
    return jnp.array([rho, rho * u, E]) # * r**alpha

# ===========================================

# =============== Fluid physics ===============

def pressure_from_internal_energy(e, rho, gamma):
  return (gamma - 1) * rho * e

def internal_energy_from_energy(E, rho, u):
    return E / rho - 0.5 * u**2

def pressure_from_energy(E, rho, u, gamma):
    e = internal_energy_from_energy(E, rho, u)
    return pressure_from_internal_energy(e, rho, gamma)

def total_energy_from_primitives(rho, u, p, gamma):
    return p / (gamma - 1) + 0.5 * rho * u**2

def speed_of_sound(rho, p, gamma):
    return jnp.sqrt(gamma * p / rho)

# ===========================================