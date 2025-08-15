import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

# numerics
import jax
import jax.numpy as jnp

# timing
from timeit import default_timer as timer

# plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# jf1uids classes
from jf1uids import SimulationConfig
from jf1uids import SimulationParams
from jf1uids.option_classes.simulation_config import CONSERVATIVE_SOURCE_TERM, DOUBLE_MINMOD, LAX_FRIEDRICHS, MUSCL, RK2_SSP, SIMPLE_SOURCE_TERM, SPLIT, UNSPLIT, BoundarySettings, BoundarySettings1D
from jf1uids._physics_modules._binary._binary_options import BinaryParams

from jf1uids.option_classes.simulation_config import BoundarySettings, BoundarySettings1D

# jf1uids functions
from jf1uids import get_helper_data
from jf1uids import time_integration
from jf1uids.fluid_equations.fluid import construct_primitive_state
from jf1uids.option_classes.simulation_config import finalize_config
from jf1uids import get_registered_variables
from jf1uids.option_classes.simulation_config import BinaryConfig


# jf1uids constants
from jf1uids.option_classes.simulation_config import (
    BACKWARDS, FORWARDS, HLL, HLLC, MINMOD, OSHER, 
    PERIODIC_BOUNDARY, OPEN_BOUNDARY, REFLECTIVE_BOUNDARY, 
    BoundarySettings, BoundarySettings1D
)
#jax.config.update('jax_enable_x64', True)

print("Setting up simulation...")

# simulation settings
gamma = 5/3

# spatial domain
box_size = 4.0
num_cells = 32  #128

fixed_timestep = False
dt_max = 0.001

# self_gravity_version = SIMPLE_SOURCE_TERM
self_gravity_version = CONSERVATIVE_SOURCE_TERM


# setup simulation config
config = SimulationConfig(
    runtime_debugging = False,
    progress_bar = True,
    binary_config = BinaryConfig(
        binary = True,
        deposit_particles = "ngp"  # Options: "ngp", "cic", "tsc"
    ),
    self_gravity = True,
    self_gravity_version = self_gravity_version,
    dimensionality = 3,
    box_size = box_size,
    split = UNSPLIT, 
    num_cells = num_cells,
    fixed_timestep = fixed_timestep,
    differentiation_mode = FORWARDS,
    limiter = MINMOD,
    time_integrator = RK2_SSP,
    # time_integrator = MUSCL,
    riemann_solver = HLL,
    return_snapshots = True,
    num_snapshots = 10,
    boundary_settings = BoundarySettings(
        BoundarySettings1D(
            left_boundary = OPEN_BOUNDARY,
            right_boundary = OPEN_BOUNDARY
        ),
        BoundarySettings1D(
            left_boundary = OPEN_BOUNDARY,
            right_boundary = OPEN_BOUNDARY
        ),
        BoundarySettings1D(
            left_boundary = OPEN_BOUNDARY,
            right_boundary = OPEN_BOUNDARY
        )
    )
)

#### from helper_data
x = jnp.linspace(0, config.box_size, config.num_cells)
y = jnp.linspace(0, config.box_size, config.num_cells)
z = jnp.linspace(0, config.box_size, config.num_cells)
geometric_centers = jnp.meshgrid(x, y, z)
box_center = jnp.zeros(config.dimensionality) + config.box_size / 2
geometric_centers = jnp.array(geometric_centers)
geometric_centers = jnp.moveaxis(geometric_centers, 0, -1)
###############  ADDITIONAL HELPER VARIABLES FOR CYL COORDS ###################
box_center_cyl = jnp.zeros(config.dimensionality - 1) + config.box_size / 2
xy = geometric_centers[..., :2]
r_cyl = jnp.linalg.norm(xy - box_center_cyl, axis=-1) 
x_cyl = geometric_centers[..., 0] - config.box_size / 2
y_cyl = geometric_centers[..., 1] - config.box_size / 2
z_cyl = geometric_centers[..., 2] - config.box_size / 2
################################################################################

helper_data = get_helper_data(config)
x1=0.5
x2=-0.5
## initial state of each point mass [t,x,y,z,vx,vy,vz]
txv1 = jnp.array([0.0, x1, 0.0, 0.0, 0.0, 0.5, 0.0])   
txv2 = jnp.array([0.0, x2, 0.0, 0.0, 0.0,-0.5, 0.0])
masses = jnp.array([0.5, 0.5])  
binary_state = jnp.concatenate([txv1, txv2])

binary_params = BinaryParams(
    masses = masses,
    binary_state = binary_state
)

params = SimulationParams(
    C_cfl = 0.4,
    dt_max = dt_max,
    gamma = gamma,
    t_end = 10,
    binary_params = binary_params
)

registered_variables = get_registered_variables(config)


# initialize density field
##### spherical density profile
# num_injection_cells = jnp.sum(helper_data.r <= R)
# rho = jnp.where(helper_data.r <= R, M / (2 * jnp.pi * R ** 2 * helper_data.r), 1e-4)
# total_injected_mass = jnp.sum(jnp.where(helper_data.r <= R, rho, 0)) * dx ** 3
#########################
#  ####uniform density field
# M=0.1
# rho = jnp.ones_like(helper_data.r) * M / (4/3 * jnp.pi * config.box_size ** 3)


########### FROM PAPER ######################
# === Circumbinary disc initial conditions (from Shamrock §4.7) ===

# Physical parameters
G = 1.0                   # in code units where G M⊙ = 1
M1 = 0.5              
M2 = 0.5                 
Mtot = M1 + M2

A0=2
w0=0.12
r0=0.7

R_o = 1.25
R_i = 0.12
M = 0.001

dx = config.box_size / (config.num_cells - 1)

mask = (r_cyl <= R_o) & (r_cyl >= R_i) & (z_cyl >= -0.3) & (z_cyl <= 0.3)
rho = jnp.where(
    mask,
    M / (4/3 * jnp.pi * config.box_size ** 3), #(1.0 + A0 * jnp.exp(-0.5 * ((r_cyl - r0) / w0) ** 2.0))/1000,    ####/241.677 to get 0.01M
    1e-5)

########## KEPLERIAN
# r_safe = 1e-6
# v_x = jnp.where(
#     mask,
#     jnp.sqrt(Mtot / (r_cyl + r_safe) ** 3) * (-y_cyl),
#     1e-5
# )
# v_y = jnp.where(
#     mask,
#     jnp.sqrt(Mtot / (r_cyl + r_safe) ** 3) * (x_cyl),
#     1e-5
# )
v_z = jnp.zeros_like(rho)
# v_x = jnp.ones_like(rho)*(Mtot/(r_cyl+r_safe)**3)**0.5*(-y_cyl)
# v_y = jnp.ones_like(rho)*(Mtot/(r_cyl+r_safe)**3)**0.5*(x_cyl)

####### ZEROES
v_x = jnp.zeros_like(rho)
v_y = jnp.zeros_like(rho)
v_z = jnp.zeros_like(rho)

# (Optionally) print total mass to check normalization
total_mass = jnp.sum(rho) * dx**3
print(f"Total disc mass: {total_mass:.3e}")
# =========================================


# initial thermal energy per unit mass = 0.05
e = 0.05
p = (gamma - 1) * rho * e

# Construct the initial primitive state for the 3D simulation.
initial_state = construct_primitive_state(
    config = config,
    registered_variables = registered_variables,
    density = rho,
    velocity_x = v_y,
    velocity_y = v_x,
    velocity_z = v_z,
    gas_pressure = p,
)


config = finalize_config(config, initial_state.shape)

from matplotlib.colors import LogNorm

a = num_cells // 2 - 30
b = num_cells // 2 + 30

c = num_cells // 2 + 20
d = num_cells // 2 + 50

save_path = os.path.join('Binary')
plt.imshow(jnp.abs(initial_state[registered_variables.density_index, :, :, num_cells // 2].T), cmap = "jet", origin = "lower", extent=[0, box_size, 0, box_size], norm = LogNorm())
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.savefig(save_path+"/initial_xy.png")
plt.close()

plt.imshow(jnp.abs(initial_state[registered_variables.density_index, :, num_cells // 2, :].T), cmap = "jet", origin = "lower", extent=[0, box_size, 0, box_size], norm = LogNorm())
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.savefig(save_path+"/initial_xz.png")
plt.close()


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

ax1.scatter(r_cyl.flatten(), initial_state[registered_variables.density_index].flatten(), label="Final Density", s = 1)
ax1.set_xlabel("r")
ax1.set_ylabel("Density")

# velocity profile
v_r = jnp.sqrt(initial_state[registered_variables.velocity_index.x] ** 2 + initial_state[registered_variables.velocity_index.y] ** 2 + initial_state[registered_variables.velocity_index.z] ** 2)

ax2.scatter(r_cyl.flatten(), v_r.flatten(), label="Radial Velocity", s = 1)
# ax2.set_xscale("log")
ax2.set_xlabel("r")
ax2.set_ylabel("Velocity")

# plot P / rho^gamma

ax3.scatter(r_cyl.flatten(), initial_state[registered_variables.pressure_index].flatten(), label="P / rho^gamma", s = 1)
ax3.set_xlabel("r")
ax3.set_ylabel("Pressure")
# ax3.set_xscale("log")

fig.suptitle("Binary")
plt.tight_layout()
fig.savefig(save_path+"/initial_variables.png")
plt.close()

result = time_integration(initial_state, config, params, helper_data, registered_variables)
final_state = result.states[-1]

from matplotlib.colors import LogNorm

a = num_cells // 2 - 30
b = num_cells // 2 + 30

c = num_cells // 2 + 20
d = num_cells // 2 + 50

plt.imshow(jnp.abs(final_state[registered_variables.pressure_index, :, :, num_cells // 2]), cmap = "jet", origin = "lower", extent=[0, box_size, 0, box_size], norm = LogNorm())
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.savefig(save_path+"/endstate.png")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

ax1.scatter(r_cyl.flatten(), final_state[registered_variables.density_index].flatten(), label="Final Density", s = 1)
# ax1.set_xscale("log")
# ax1.set_yscale("log")
ax1.set_xlabel("r")
ax1.set_ylabel("Density")

# velocity profile
v_r = jnp.sqrt(final_state[registered_variables.velocity_index.x] ** 2 + final_state[registered_variables.velocity_index.y] ** 2 + final_state[registered_variables.velocity_index.z] ** 2)

ax2.scatter(r_cyl.flatten(), v_r.flatten(), label="Radial Velocity", s = 1)
ax2.set_xlabel("r")
ax2.set_ylabel("Velocity")

# plot P / rho^gamma

ax3.scatter(r_cyl.flatten(), final_state[registered_variables.pressure_index].flatten(), label="P / rho^gamma", s = 1)
ax3.set_xlabel("r")
ax3.set_ylabel("Pressure")

fig.suptitle("Binary")

plt.tight_layout()
fig.savefig(save_path+"/var_finals.png")



# ## Conservational properties
# config = config._replace(return_snapshots = True, num_snapshots = 60)
# params = params._replace(t_end = 3.0)

# snapshots = time_integration(initial_state, config, params, helper_data, registered_variables)
# total_energy = snapshots.total_energy
# internal_energy = snapshots.internal_energy
# kinetic_energy = snapshots.kinetic_energy
# gravitational_energy = snapshots.gravitational_energy
# total_mass = snapshots.total_mass
# time = snapshots.time_points
# t_end = 3.0
# plt.plot(time, total_energy, label="Total Energy", color = "black")
# plt.plot(time, internal_energy, label="Internal Energy", color = "green")
# plt.plot(time, kinetic_energy, label="Kinetic Energy", color = "red")
# plt.plot(time, gravitational_energy, label="Gravitational Energy", color = "blue")
# plt.xlabel("Time")
# plt.ylabel("Energy")
# plt.legend()
# plt.savefig("Binary/energy_conservation.png")

import matplotlib.animation as animation

time = result.time_points
fig, ax = plt.subplots()

def animate(i):
    ax.clear()
    state = result.states[i]
    im = ax.imshow(state[0, :, :, num_cells // 2].T, cmap="jet", origin="lower", extent=[0, box_size, 0, box_size], norm=LogNorm())
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Density at time {time[i]:.2f}")
    return [im]

ani = animation.FuncAnimation(fig, animate, frames=len(result.states), interval=100)
plt.colorbar(ax.imshow(result.states[0][0, :, :, num_cells // 2].T, cmap="jet", origin="lower", extent=[0, box_size, 0, box_size], norm=LogNorm()), ax=ax)
# save to gif
ani.save(save_path+"/ani_rho_xy.gif")
plt.close(fig)

fig, ax = plt.subplots()
def animate(i):
    ax.clear()
    state = result.states[i]
    im = ax.imshow(state[0, num_cells // 2, :, :].T, cmap="jet", origin="lower", extent=[0, box_size, 0, box_size], norm=LogNorm())
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Density at time {time[i]:.2f}")
    return [im]

ani = animation.FuncAnimation(fig, animate, frames=len(result.states), interval=100)
plt.colorbar(ax.imshow(result.states[0][0, num_cells // 2, :, :].T, cmap="jet", origin="lower", extent=[0, box_size, 0, box_size], norm=LogNorm()), ax=ax)
ani.save(save_path+"/ani_rho_xz.gif")
plt.close(fig)
