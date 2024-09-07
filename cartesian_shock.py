# =============== Imports ===============
import jax
import jax.numpy as jnp
# plotting
import matplotlib.pyplot as plt

from jf1uids.geometry import center_of_volume, r_hat_alpha
from jf1uids.simulation_config import SimulationConfig
from jf1uids.simulation_helper_data import SimulationHelperData
from jf1uids.simulation_params import SimulationParams
from jf1uids.time_integration import time_integration
from jf1uids.fluid import primitive_state

# for now using CPU as of outdated NVIDIA Driver
jax.config.update('jax_platform_name', 'cpu')

# ========================================

## simulation settings

gamma = 5/3

# spatial domain
alpha = 0 # 0 -> cartesian, 1 -> cylindrical, 2 -> spherical
L = 1.0
N_grid = 500
dx = L / N_grid
r = jnp.linspace(0, L, N_grid)

# introduce constants to 
# make this more readable
left_boundary = 0
right_boundary = 0

config = SimulationConfig(alpha_geom = alpha, left_boundary = left_boundary, right_boundary = right_boundary)

# time domain
dt_max = 0.001
C_CFL = 0.8
t_end = 0.2

# SOD shock tube
shock_pos = 0.5

helper_data = SimulationHelperData()

rho = jnp.where(r < shock_pos, 1.0, 0.125)
u = jnp.zeros_like(r)
p = jnp.where(r < shock_pos, 1.0, 0.1)

# get initial state
initial_state = primitive_state(rho, u, p)

params = SimulationParams(C_cfl = C_CFL, dt_max = dt_max, dx = dx, gamma = gamma, t_end = t_end)

## run the simulation
final_state = time_integration(initial_state, config, params, helper_data)

rho, u, p = final_state

# print(rho)

## plot the results
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].plot(r, rho)
axs[0].set_title("Density")

axs[1].plot(r, u)
axs[1].set_title("Velocity")

axs[2].plot(r, p)
axs[2].set_title("Pressure")

plt.savefig("figures/cartesian_shock.png")