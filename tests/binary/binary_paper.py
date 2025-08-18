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
# jax.config.update('jax_enable_x64', True)

print("Setting up simulation...")

# simulation settings
gamma = 5/3

# spatial domain
box_size = 5.6
num_cells = 50  #128

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
    num_snapshots = 50,
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

cell_width = config.box_size / config.num_cells
mass_source = 4e-3 # works 
rho_source = mass_source / (cell_width ** 3)
print("mass both stars:", 2*mass_source, "density star: ", rho_source)
masses = jnp.array([mass_source, mass_source])  

x1=0.5
x2=-0.5
v_phi0 = jnp.sqrt(mass_source/2)
print("Azimuthal vel of each star: ", v_phi0)
txv1 = jnp.array([0.0, x1, 0.0, 0.0, 0.0, -v_phi0, 0.0])    #this is clockwise rotation in the xy plane
txv2 = jnp.array([0.0, x2, 0.0, 0.0, 0.0, v_phi0, 0.0])
binary_state = jnp.concatenate([txv1, txv2])

binary_params = BinaryParams(
    masses = masses,
    binary_state = binary_state
)

T_orb = 2*jnp.pi*(x1-x2) / 2 / v_phi0   #T_orb at r=0.5
params = SimulationParams(
    C_cfl = 0.4,
    dt_max = dt_max,
    gamma = gamma,
    t_end = 20,  # 0.8,
    binary_params = binary_params
)

registered_variables = get_registered_variables(config)


from jf1uids.fluid_equations.fluid import construct_primitive_state3D

dx = config.box_size / (config.num_cells - 1)

# initialize density field
#########################
# uniform density field
# rho = jnp.ones_like(helper_data.r) * M / (4/3 * jnp.pi * config.box_size ** 3)
# total_injected_mass = jnp.sum(rho) * dx ** 3
# print(f"Injected mass: {total_injected_mass}")


########### FROM PAPER https://arxiv.org/abs/2503.09713 ######################
# === Circumbinary disc initial conditions (from §4.7) ===

# Physical parameters
G = 1.0                
Mtot = 2 * mass_source

R_cav = 2.5          # cavity radius [au]
delta_0    = 0.5e-7  #1e-5        # floor parameter
sigma_0    = 1e-5    # choose normalization for Σ(r)
rsafe = 1e-6

# Compute surface density 
sigma = sigma_0 * ((1.0 - delta_0) * jnp.exp(- (R_cav / (r_cyl))**12) + delta_0) 
# sigma = jnp.ones_like(r_cyl) * 1e-5  # uniform surface density
# Disc scale height H(r) = 0.1 r
H = 0.1 * (r_cyl)  # :contentReference[oaicite:1]{index=1}

# 3D density: vertical Gaussian
R_o = 2.5
R_i = 0.2
mask = True   #(r_cyl <= R_o) & (r_cyl >= R_i) & (z_cyl >= -0.3) & (z_cyl <= 0.3),
# rho = jnp.where(
#     mask,
#     jnp.clip(sigma / (H * jnp.sqrt(2 * jnp.pi)) * jnp.exp(-0.5 * (z_cyl / H)**2),a_min=1e-8),
#     1e-4)
rho = jnp.clip(sigma / (H * jnp.sqrt(2 * jnp.pi)) * jnp.exp(-0.5 * (z_cyl / H)**2),a_min=1e-9)
# rho = sigma / (H * jnp.sqrt(2 * jnp.pi)) * jnp.exp(-0.5 * (z_cyl / H)**2)


#### UNIFORM DENSITY
# M=0.1
# rho = jnp.ones_like(helper_data.r) * M / (4/3 * jnp.pi * config.box_size ** 3)


# Sound speed (locally isothermal)
cs = (H / (r_cyl + rsafe)) * jnp.sqrt(G * Mtot / (r_cyl + rsafe))

# Gas pressure 
P = jnp.clip(rho * cs**2,a_min=1e-9)

# Cartesian gradients of P
dP_dx = jnp.gradient(P, dx, axis=0)
dP_dy = jnp.gradient(P, dx, axis=1)

# radial unit vectors:
ux = x_cyl / (r_cyl + rsafe)
uy = y_cyl / (r_cyl + rsafe)

# project the pressure gradient onto the radial direction:
dP_dr = dP_dx * ux + dP_dy * uy

#subkeplerian velocity profile
v_phi = jnp.sqrt(
    jnp.clip(G * Mtot / (r_cyl + rsafe) + (r_cyl / (rho)) * dP_dr, a_min=0.0)  # avoid negative values
)

### kelperian velocity profile
# v_phi = jnp.sqrt(
#     jnp.clip(G * Mtot / (r_cyl + rsafe), a_min=0.0)  # avoid negative values
# )

# decompose into cartesian velocities
v_x = -v_phi *  y_cyl / (r_cyl + rsafe)   ## this is clockwise rotation in the xy plane
v_y =  v_phi *  x_cyl / (r_cyl + rsafe)
v_z = jnp.zeros_like(v_x)

# zero velocities
# v_x=jnp.zeros_like(rho)
# v_y=jnp.zeros_like(rho)
# v_z=jnp.zeros_like(rho)

disc_mass=jnp.sum(rho) * cell_width**3     ## use the cell_width definition  (WHY dx???)
# print(f"Total disc mass: {total_mass:.3e}")
print(f"Total disc mass: {disc_mass:.3e}")
print("M_disc/M_star: ", disc_mass/Mtot)
print("vx_max: ", jnp.max(v_x), "vy_max: ", jnp.max(v_y))

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
    gas_pressure = P,
)

"""
import numpy as np

def report(name, arr, near_zero_tol=0.0, max_neighborhood=3):
    
    # Print diagnostics for `arr` (a JAX or numpy array).
    # - near_zero_tol: if >0, treat abs(x) <= near_zero_tol as "zero" (useful for underflow).
    #                  if ==0 (default), checks exact zeros.
    # - max_neighborhood: number of cells per axis to show around the first zero (<=3 recommended).
    
    a = jax.device_get(arr)           # bring to host
    a = np.asarray(a)                 # ensure numpy
    size = a.size

    n_nan = int(np.isnan(a).sum())
    n_inf = int(np.isinf(a).sum())
    n_finite = int(np.isfinite(a).sum())

    if near_zero_tol == 0.0:
        zeros_mask = (a == 0)
    else:
        zeros_mask = np.isclose(a, 0.0, atol=near_zero_tol)

    n_zero = int(np.count_nonzero(zeros_mask))

    print(f"{name}: shape={a.shape}, dtype={a.dtype}, size={size}")
    print(f"  finite: {n_finite}, nan: {n_nan}, inf: {n_inf}, zeros (tol={near_zero_tol}): {n_zero}")

    # min/max ignoring NaNs (if any finite values exist)
    if n_finite:
        try:
            amin = np.nanmin(a)
            amax = np.nanmax(a)
            print(f"  min/max (nan-ignored): {amin} {amax}")
        except ValueError:
            print("  min/max: could not compute (no finite values)")
    else:
        print("  no finite values present -> min/max unavailable")

    # first NaN
    if n_nan:
        idx = np.argwhere(np.isnan(a))[0]
        print("  first NaN index:", tuple(idx), "value:", a[tuple(idx)])

    # first Inf
    if n_inf:
        idx = np.argwhere(np.isinf(a))[0]
        print("  first Inf index:", tuple(idx), "value:", a[tuple(idx)])

    # first zero (or near-zero)
    if n_zero:
        idx = np.argwhere(zeros_mask)[0]
        idx_t = tuple(idx)
        print("  first zero index:", idx_t, "value:", a[idx_t])

        # show a small neighborhood around the index (if array has >=1 dimension)
        if a.ndim >= 1:
            slices = []
            for axis, coord in enumerate(idx):
                half = max(1, max_neighborhood // 2)
                start = max(0, int(coord) - half)
                stop = min(a.shape[axis], int(coord) + half + 1)
                slices.append(slice(start, stop))
            neighborhood = a[tuple(slices)]
            print(f"  neighborhood around first zero (shape={neighborhood.shape}):\n{neighborhood}")

    print("")  


# check the suspect intermediates:
report("r_cyl (original)", r_cyl)        
report("H", H)
report("rho", rho)
report("sigma", sigma)
report("cs", cs)
report("P", P)
report("dP_dx", dP_dx)
report("dP_dy", dP_dy)
report("dP_dr", dP_dr)
report("v_phi", v_phi)
report("initial_state", initial_state)
"""

config = finalize_config(config, initial_state.shape)

from matplotlib.colors import LogNorm

a = num_cells // 2 - 30
b = num_cells // 2 + 30

c = num_cells // 2 + 20
d = num_cells // 2 + 50

save_path = os.path.join('Binary_paper')
plt.imshow(jnp.abs(initial_state[registered_variables.density_index, :, :, num_cells // 2].T), cmap = "jet", origin = "lower", extent=[0, box_size, 0, box_size], norm = LogNorm())
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.savefig(save_path+"/initial_xy.png")
plt.close()

plt.imshow(jnp.abs(initial_state[registered_variables.density_index, num_cells // 2, :, :].T), cmap = "jet", origin = "lower", extent=[0, box_size, 0, box_size], norm = LogNorm())
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.savefig(save_path+"/initial_xz.png")
plt.close()


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))


ax1.scatter(r_cyl.flatten(), initial_state[registered_variables.density_index].flatten(), label="Final Density", s = 1)
ax1.set_xlabel("r")
ax1.set_yscale("log")
ax1.set_ylabel("Density")

# velocity profile
v_r = jnp.sqrt(initial_state[registered_variables.velocity_index.x] ** 2 + initial_state[registered_variables.velocity_index.y] ** 2 + initial_state[registered_variables.velocity_index.z] ** 2)

ax2.scatter(r_cyl.flatten(), v_r.flatten(), label="Radial Velocity", s = 1)
# ax2.set_xscale("log")
ax2.set_ylim(0,0.1)
ax2.set_xlabel("r")
ax2.set_ylabel("Velocity")

# plot P / rho^gamma

ax3.scatter(r_cyl.flatten(), initial_state[registered_variables.pressure_index].flatten(), label="P / rho^gamma", s = 1)
ax3.set_ylim(0,1e-7)
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
idx_rho=registered_variables.density_index
idx_p=registered_variables.pressure_index
plt.imshow(jnp.abs(final_state[idx_rho, :, :, num_cells // 2]), cmap = "jet", origin = "lower", extent=[0, box_size, 0, box_size], norm = LogNorm())
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.savefig(save_path+"/endstate_rho.png")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

ax1.scatter(r_cyl[:, :, num_cells // 2].flatten(), final_state[registered_variables.density_index, :, :, num_cells // 2].flatten(), label="Final Density", s = 1)
# ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("r")
ax1.set_ylabel("Density")

# velocity profile
# v_r = jnp.sqrt(final_state[registered_variables.velocity_index.x] ** 2 + final_state[registered_variables.velocity_index.y] ** 2 + final_state[registered_variables.velocity_index.z] ** 2)
v_r = jnp.sqrt(final_state[registered_variables.velocity_index.x, :, :, num_cells // 2] ** 2 + final_state[registered_variables.velocity_index.y, :, :, num_cells // 2] ** 2 + final_state[registered_variables.velocity_index.z, :, :, num_cells // 2] ** 2)

ax2.scatter(r_cyl[:, :, num_cells // 2].flatten(), v_r.flatten(), label="Radial Velocity", s = 1)
ax2.set_xlabel("r")
ax2.set_ylabel("Velocity")

# plot P / rho^gamma

ax3.scatter(r_cyl[:, :, num_cells // 2].flatten(), final_state[registered_variables.pressure_index, :, :, num_cells // 2].flatten(), label="P / rho^gamma", s = 1)
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
    ax.set_ylabel("z")
    ax.set_title(f"Density at time {time[i]:.2f}")
    return [im]

ani = animation.FuncAnimation(fig, animate, frames=len(result.states), interval=100)
plt.colorbar(ax.imshow(result.states[0][0, num_cells // 2, :, :].T, cmap="jet", origin="lower", extent=[0, box_size, 0, box_size], norm=LogNorm()), ax=ax)
ani.save(save_path+"/ani_rho_xz.gif")
plt.close(fig)

fig, ax = plt.subplots()
def animate(i):
    ax.clear()
    state = result.states[i]
    im = ax.scatter(r_cyl[:, :, num_cells // 2].flatten(), state[0, :, :, num_cells // 2].flatten(), s = 1)
    ax.set_yscale("log")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Density at time {time[i]:.2f}")
    return [im]

ani = animation.FuncAnimation(fig, animate, frames=len(result.states), interval=100)
ani.save(save_path+"/ani_rho_flat.gif")
plt.close(fig)

fig, ax = plt.subplots()
def animate(i):
    ax.clear()
    state = result.states[i]
    im = ax.scatter(r_cyl[:, :, num_cells // 2].flatten(), jnp.sqrt(state[registered_variables.velocity_index.x, :, :, num_cells // 2] ** 2 + state[registered_variables.velocity_index.y, :, :, num_cells // 2] ** 2 + state[registered_variables.velocity_index.z, :, :, num_cells // 2] ** 2).flatten(), s = 1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Velocity at time {time[i]:.2f}")
    return [im]

ani = animation.FuncAnimation(fig, animate, frames=len(result.states), interval=100)
ani.save(save_path+"/ani_vel.gif")
plt.close(fig)