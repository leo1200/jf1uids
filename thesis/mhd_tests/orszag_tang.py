# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus = 1)
# =======================

# numerics
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jax.random import PRNGKey, uniform

# plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# fluids
from jf1uids import SimulationConfig
from jf1uids import get_helper_data
from jf1uids import SimulationParams
from jf1uids import time_integration
from jf1uids.fluid_equations.fluid import construct_primitive_state
from jf1uids import get_registered_variables
from jf1uids.option_classes.simulation_config import finalize_config

# jf1uids constants
from jf1uids.option_classes.simulation_config import (
    BACKWARDS, FORWARDS, HLL, HLLC, MINMOD,
    OSHER, PERIODIC_BOUNDARY, BoundarySettings, BoundarySettings1D
)

# simulation settings
gamma = 5/3

# spatial domain
box_size = 1.0
num_cells = 64

# setup simulation config
config = SimulationConfig(
    progress_bar = True,
    mhd = True,
    dimensionality = 2,
    box_size = box_size, 
    num_cells = num_cells,
    differentiation_mode = BACKWARDS,
    limiter = MINMOD,
    riemann_solver = HLL,
    boundary_settings = BoundarySettings(
        BoundarySettings1D(left_boundary = PERIODIC_BOUNDARY, right_boundary = PERIODIC_BOUNDARY),
        BoundarySettings1D(left_boundary = PERIODIC_BOUNDARY, right_boundary = PERIODIC_BOUNDARY),
        BoundarySettings1D(left_boundary = PERIODIC_BOUNDARY, right_boundary = PERIODIC_BOUNDARY))
)

helper_data = get_helper_data(config)

params = SimulationParams(
    t_end = 0.5,
    C_cfl = 0.4
)

registered_variables = get_registered_variables(config)


# Set the random seed for reproducibility
key = PRNGKey(0)

# Grid size and configuration
num_cells = config.num_cells
grid_spacing = box_size / num_cells
x = jnp.linspace(grid_spacing / 2, box_size - grid_spacing / 2, num_cells)
y = jnp.linspace(grid_spacing / 2, box_size - grid_spacing / 2, num_cells)
X, Y = jnp.meshgrid(x, y, indexing="ij")

# Initialize state
rho = jnp.ones_like(X) * gamma ** 2 / (4 * jnp.pi)
P = jnp.ones_like(X) * gamma / (4 * jnp.pi)

V_x = -jnp.sin(2 * jnp.pi * Y)
V_y = jnp.sin(2 * jnp.pi * X)

B_0 = 1 / jnp.sqrt(4 * jnp.pi)
B_x = -B_0 * jnp.sin(2 * jnp.pi * Y)
B_y = B_0 * jnp.sin(4 * jnp.pi * X)
# B_x = jnp.zeros_like(X)
# B_y = jnp.zeros_like(X)
B_z = jnp.zeros_like(X)

initial_magnetic_field = jnp.stack([B_x, B_y, B_z], axis=0)

dx = 1 / (num_cells - 1)

initial_state = construct_primitive_state(
    config = config,
    registered_variables = registered_variables,
    density = rho,
    velocity_x = V_x,
    velocity_y = V_y,
    magnetic_field_x = B_x,
    magnetic_field_y = B_y,
    magnetic_field_z = B_z,
    gas_pressure = P
)

config = finalize_config(config, initial_state.shape)

final_state = time_integration(initial_state, config, params, helper_data, registered_variables)

# plot the final state, just an imshow of the density
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
im = ax.imshow(final_state[registered_variables.density_index].T, origin='lower', extent=(0, box_size, 0, box_size), aspect='auto', cmap='viridis')
ax.set_title('Density')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(im, ax=ax, label='Density')
plt.tight_layout()
plt.savefig('density_orszag_tang.png', dpi=300)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
y_index = jnp.argmin(jnp.abs(y - 0.3125))
print(min(jnp.abs(y - 0.3125)), y_index)
ax.plot(x, final_state[0, :, y_index], label = "Pressure")
ax.set_xlabel("x")
ax.set_ylabel("Pressure")
plt.savefig("pressure_orszag_tang.png", dpi=300)