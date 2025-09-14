import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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
from jf1uids.option_classes.simulation_config import DOUBLE_MINMOD, LAX_FRIEDRICHS, MUSCL, RK2_SSP, SIMPLE_SOURCE_TERM, SPLIT, UNSPLIT, BoundarySettings, BoundarySettings1D
from jf1uids._physics_modules._binary._binary_options import BinaryParams

from jf1uids.option_classes.simulation_config import BoundarySettings, BoundarySettings1D

# jf1uids functions
from jf1uids import get_helper_data
from jf1uids import time_integration
from jf1uids.fluid_equations.fluid import construct_primitive_state
from jf1uids.option_classes.simulation_config import finalize_config
from jf1uids import get_registered_variables
from jf1uids.option_classes.simulation_config import BinaryConfig
from jf1uids._physics_modules._binary._nbody_sampler import plummer_sampler, virialized_sphere_sampler


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
box_size = 4.0
num_cells = 140  #128

fixed_timestep = False
dt_max = 0.001

self_gravity_version = SIMPLE_SOURCE_TERM
# boundary = REFLECTIVE_BOUNDARY
boundary = OPEN_BOUNDARY

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
    first_order_fallback = True,
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
    num_snapshots = 300,
    boundary_settings = BoundarySettings(
        BoundarySettings1D(
            left_boundary = boundary,
            right_boundary = boundary
        ),
        BoundarySettings1D(
            left_boundary = boundary,
            right_boundary = boundary
        ),
        BoundarySettings1D(
            left_boundary = boundary,
            right_boundary = boundary
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
mass_source = 1e-3 # works 
# rho_source = mass_source / (cell_width ** 3)
# masses = jnp.array([mass_source, mass_source])  
# # print("mass both stars:", 2*mass_source, "density star: ", rho_source)
#masses = jnp.array([1, 0.001, 0.003, 0.008, 0.002]) * mass_source # (2,)
# M_central=masses[0]
r1=3.8
r2=1
r3=1.9
r4=2.7
def orbital_velocity(mass, radius):
    v = jnp.sqrt(mass/radius)
    return v

#star system; one plane
# orbit1 = jnp.array([0.0, 0., 0.0, 0.0, 0.0, 0., 0.0])
# orbit2 = jnp.array([0.0, r1, 0.0, 0.0, 0.0,-orbital_velocity(M_central,r1), 0.0])
# orbit3 = jnp.array([0.0, 0, r2, 0, orbital_velocity(M_central,r2) ,-0.0, 0.0])
# orbit4 = jnp.array([0.0, 0, -r3, 0.0, -orbital_velocity(M_central,r3),-0.0, 0.0])
# orbit5 = jnp.array([0.0, -r4, 0, 0, 0.0, orbital_velocity(M_central,r4), 0.0])

# orbit1 = jnp.array([0.0,  0.991448574176414,  0.029880705565651,  0.010875687404770,  0.084207016573161,  0.022741169835371,  0.000000000000000])
# orbit2 = jnp.array([0.0,  0.991448574176414, -0.031453374279633, -0.011448092005020, -0.088638964813854,  0.022741169835371,  0.000000000000000])
# orbit3 = jnp.array([0.0, -0.949478527562011,  0.000000000000000,  0.024541055790300,  0.077085096380257, -0.022172640589486, -0.053975565569078])
# orbit4 = jnp.array([0.0, -0.983172316307095,  0.000000000000000, -0.023578661445583, -0.074062151424169, -0.022172640589486,  0.051858876723232])   
# orbit5 = jnp.array([0.0,  0.000000000000000,  2.500000000000000,  2.097749077943200,  0.039063544778898,  0.000000000000000,  0.000000000000000])   
# masses = jnp.array([1, 0.95, 0.98, 1.02, 1.03])*mass_source
# nbody_state = jnp.concatenate([orbit1, orbit2, orbit3, orbit4, orbit5]) 
low_mass = 0.1*mass_source
high_mass =10*low_mass
orbits, masses = plummer_sampler(n=20, M1=low_mass, M2=high_mass, a=box_size/2)
# orbits, masses = virialized_sphere_sampler(20, low_mass, high_mass, a=1.4, seed=0)

orbits = orbits.reshape(-1)
masses = masses.reshape(-1)
print("low density: ", low_mass/(cell_width**3), "high density: ", high_mass/(cell_width**3))

binary_params = BinaryParams(
    masses = masses,
    binary_state = orbits
)

# T_orb = 2*jnp.pi*(x1-x2) / 2 / v_phi0   #T_orb at r=0.5
params = SimulationParams(
    C_cfl = 0.4,
    dt_max = dt_max,
    gamma = gamma,
    t_end = 500,  # 0.8,
    binary_params = binary_params
)

registered_variables = get_registered_variables(config)


from jf1uids.fluid_equations.fluid import construct_primitive_state3D

dx = config.box_size / (config.num_cells - 1)

# initialize density field
#########################
# uniform density field
# disc_mass = mass_source / 500
# rho = jnp.ones_like(helper_data.r) * disc_mass / (config.box_size ** 3)
# total_injected_mass = jnp.sum(rho) * dx ** 3
# print(f"Injected mass: {total_injected_mass}")


########### FROM PAPER https://arxiv.org/abs/2503.09713 ######################
# === Circumbinary disc initial conditions (from §4.7) ===

# Physical parameters
G = 1.0                
Mtot = jnp.sum(masses)
disc_mass = Mtot / 5000

R_cav = 2.5         # cavity radius [au]
delta_0    = 0.5e-7  #1e-5        # floor parameter
sigma_0    = 1e-5    # choose normalization for Σ(r)
rsafe = 1e-6

# Compute surface density 
sigma = sigma_0 * ((1.0 - delta_0) * jnp.exp(- (R_cav / (r_cyl))**12) + delta_0) 
# sigma = jnp.ones_like(r_cyl) * 1e-5  # uniform surface density
# Disc scale height H(r) = 0.1 r
H = 0.1 * (r_cyl)  # :contentReference[oaicite:1]{index=1}

# 3D density: vertical Gaussian
R_o = 2.0
R_i = 0.2
####paper disc density
# rho = jnp.clip(sigma / (H * jnp.sqrt(2 * jnp.pi)) * jnp.exp(-0.5 * (z_cyl / H)**2),a_min=1e-9)

##uniform in sphere
# rho = jnp.where(helper_data.r <= R_o, disc_mass / (4/3 * jnp.pi * R_o ** 3), 1e-5)
# inner_mass = jnp.sum(where=helper_data.r <= R_o,a=rho)
# print("inner sphere mass: ", inner_mass, "inner/Mtot: ", inner_mass/Mtot)

##uniform everywhere
gas_mass = Mtot / 500
rho = jnp.ones_like(helper_data.r)*disc_mass / (box_size ** 3)

####plummer
b=box_size/4
rho = jnp.ones_like(helper_data.r)*3*gas_mass/4/jnp.pi/b**3*(1+helper_data.r**2/b**2)**(-5/2)


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
v_x=jnp.zeros_like(rho)
v_y=jnp.zeros_like(rho)
v_z=jnp.zeros_like(rho)

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

config = finalize_config(config, initial_state.shape)

from matplotlib.colors import LogNorm

a = num_cells // 2 - 30
b = num_cells // 2 + 30

c = num_cells // 2 + 20
d = num_cells // 2 + 50

save_path = os.path.join('nbody')
plt.imshow(jnp.abs(initial_state[registered_variables.density_index, :, :, num_cells // 2].T), cmap = "jet", origin = "lower", extent=[0, box_size, 0, box_size], norm = LogNorm())
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.savefig(save_path+"/initial_xy.png")
plt.close()

plt.imshow(jnp.abs(initial_state[registered_variables.density_index, num_cells // 2, :, :].T), cmap = "jet", origin = "lower", extent=[0, box_size, 0, box_size], norm = LogNorm())
plt.colorbar()
plt.xlabel("x")
plt.ylabel("z")
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
ax2.set_ylim(0,0.5)
ax2.set_xlabel("r")
ax2.set_ylabel("Velocity")

# plot P / rho^gamma

ax3.scatter(r_cyl.flatten(), initial_state[registered_variables.pressure_index].flatten(), label="P / rho^gamma", s = 1)
ax3.set_ylim(0,1e-7)
ax3.set_xlabel("r")
ax3.set_ylabel("Pressure")
# ax3.set_xscale("log")

# fig.suptitle("Binary")
plt.tight_layout()
fig.savefig(save_path+"/initial_variables.png")
plt.close()

result = time_integration(initial_state, config, params, helper_data, registered_variables)
final_state = result.states[-1]

#results_plotting(result, registered_variables, save_path)

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

# fig.suptitle("Binary")

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
    ax.set_xlabel("r")
    ax.set_ylabel("Density")
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
    ax.set_xlabel("r")
    ax.set_ylabel("velocity")
    ax.set_title(f"Velocity at time {time[i]:.2f}")
    return [im]

ani = animation.FuncAnimation(fig, animate, frames=len(result.states), interval=100)
ani.save(save_path+"/ani_vel.gif")
plt.close(fig)

DENSITY_IDX = registered_variables.density_index   # change if your index is different
projections = []
for state in result.states:
    dens = state[DENSITY_IDX, :, :, :]
    proj = jnp.array(jnp.sum(dens, axis=2))   # shape (nx, ny) or (ny, nx) depending on ordering
    projections.append(proj)

# Determine consistent color scale across frames
projections = [p * cell_width for p in projections]
all_min = float(min(jnp.min(p) for p in projections))
all_max = float(max(jnp.max(p) for p in projections))

fig, ax = plt.subplots(figsize=(6,6))

im = ax.imshow(projections[0], origin='lower', vmax=all_max, vmin=all_min, aspect='equal', cmap='inferno') #norm=LogNorm()
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Column density (code units)")

time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, color="white", fontsize=10,
                    bbox=dict(facecolor='black', alpha=0.4, boxstyle='round'))

ax.set_xlabel("grid x")
ax.set_ylabel("grid y")

def animate(i):
    im.set_data(projections[i])
    try:
        tstr = f"t = {time[i]:.2f}"
    except Exception:
        tstr = f"frame = {i}"
    time_text.set_text(tstr)
    return [im, time_text]

ani = animation.FuncAnimation(fig, animate, frames=len(projections), interval=100)
ani.save(save_path+"/ani_column_density.gif")
plt.close(fig)


import numpy as np
import jax.numpy as jnp
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from pathlib import Path

def plot_isosurface_frame(
    frame_index,
    states,
    density_idx,
    out_png,
    downsample=2,
    level_mode="percentile",   # "percentile" or "fraction" or "value"
    percentile=95.0,           # used when level_mode=="percentile"
    fraction=0.5,              # used when level_mode=="fraction" (fraction of range: min + fraction*(max-min))
    fixed_value=None,          # used when level_mode=="value"
    facecolor=(0.8, 0.3, 0.2),
    alpha=0.7,
    dpi=150,
    elev=25, azim=130,
    verbose=True
):
    """
    Render a single isosurface from states[frame_index] and save as PNG.
    - states: iterable of state arrays indexed like state[variable_index, ix, iy, iz].
    - density_idx: index (e.g. registered_variables.density_index).
    - out_png: path to save PNG (string or Path).
    """
    out_png = Path(out_png)
    # get density array (convert jax -> numpy)
    state = states[frame_index]
    dens = np.array(jnp.asarray(state[density_idx, :, :, :]))
    if not np.isfinite(dens).all():
        if verbose:
            print(f"[frame {frame_index}] contains non-finite values — aborting.")
        # produce placeholder image
        fig = plt.figure(figsize=(6,4), dpi=dpi)
        plt.text(0.5, 0.5, "invalid data", ha='center', va='center')
        fig.savefig(out_png, bbox_inches='tight')
        plt.close(fig)
        return str(out_png)

    # optionally downsample for speed
    if downsample > 1:
        dens_ds = dens[::downsample, ::downsample, ::downsample]
    else:
        dens_ds = dens

    fmin = float(dens_ds.min())
    fmax = float(dens_ds.max())
    if fmin == fmax:
        if verbose:
            print(f"[frame {frame_index}] constant field (min == max == {fmin}). No isosurface.")
        fig = plt.figure(figsize=(6,4), dpi=dpi)
        plt.text(0.5, 0.5, "no surface (constant field)", ha='center', va='center')
        fig.savefig(out_png, bbox_inches='tight')
        plt.close(fig)
        return str(out_png)

    # choose level safely inside (fmin, fmax)
    eps = max(1e-12, 1e-6*(fmax - fmin))
    if level_mode == "percentile":
        level = float(np.percentile(dens_ds, percentile))
    elif level_mode == "fraction":
        level = fmin + fraction * (fmax - fmin)
    elif level_mode == "value":
        if fixed_value is None:
            raise ValueError("fixed_value must be set when level_mode == 'value'")
        level = float(fixed_value)
    else:
        raise ValueError("level_mode must be 'percentile', 'fraction', or 'value'")

    # clamp level into (fmin+eps, fmax-eps)
    level = float(np.clip(level, fmin + eps, fmax - eps))

    # run marching cubes (fast for single frame; still may be heavy for big grids)
    try:
        verts, faces, normals, values = measure.marching_cubes(dens_ds, level=level)
    except Exception as e:
        if verbose:
            print(f"[frame {frame_index}] marching_cubes failed: {e}")
        fig = plt.figure(figsize=(6,4), dpi=dpi)
        plt.text(0.5, 0.5, "marching_cubes failed", ha='center', va='center')
        fig.savefig(out_png, bbox_inches='tight')
        plt.close(fig)
        return str(out_png)

    if len(verts) == 0 or len(faces) == 0:
        if verbose:
            print(f"[frame {frame_index}] no geometry extracted at level {level:.4g}.")
        fig = plt.figure(figsize=(6,4), dpi=dpi)
        plt.text(0.5, 0.5, "no surface at chosen level", ha='center', va='center')
        fig.savefig(out_png, bbox_inches='tight')
        plt.close(fig)
        return str(out_png)

    # scale verts back to original coords if downsampled
    if downsample > 1:
        verts = verts * downsample

    # make triangles and plot with mpl 3D
    triangles = verts[faces]  # (n_faces, 3, 3)
    fig = plt.figure(figsize=(6,6), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(triangles, alpha=alpha)
    mesh.set_facecolor(facecolor)
    mesh.set_edgecolor('none')
    ax.add_collection3d(mesh)

    # axis limits - use original volume shape
    nx, ny, nz = dens.shape
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_zlim(0, nz)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    try:
        tlabel = f"t = {time[frame_index]:.2f}"
    except Exception:
        tlabel = f"frame = {frame_index}"
    ax.set_title(f"Isosurface (level={level:.3g}) — {tlabel}")

    ax.view_init(elev=elev, azim=azim)

    # save PNG
    fig.savefig(out_png, bbox_inches='tight')
    plt.close(fig)
    if verbose:
        print(f"Saved isosurface PNG to: {out_png}")
    return str(out_png)

frame_to_plot = int(config.num_snapshots // 2)  # change to whichever frame you want
out_file = save_path + f"/isosurf_frame_{frame_to_plot:04d}.png"
plot_isosurface_frame(
    frame_index=frame_to_plot,
    states=result.states,
    density_idx=registered_variables.density_index,
    out_png=out_file,
    downsample=2,          # increase to 4 for faster preview (lower quality)
    level_mode="percentile",
    percentile=95.0
)

"""
Volume-render animation (PyVista preferred, fallback alpha-composite).
Drop into your environment where `result.states`, `registered_variables.density_index`,
and `save_path` are defined. Optionally `time` can be present for timestamps.
"""

import os
import matplotlib as mpl
from pathlib import Path
import tempfile
import shutil
from tqdm import tqdm

# Use imageio.v2 to avoid deprecation warnings
import imageio.v2 as imageio
from PIL import Image

# Try to import pyvista but keep script robust if it's missing or fails
try:
    import pyvista as pv
    PV_OK = True
except Exception:
    pv = None
    PV_OK = False

# ----------------- USER CONFIG -----------------
DENSITY_IDX = registered_variables.density_index   # change if needed
OUT_GIF = Path(save_path) / "volume_render_final.gif"   # where final will be written
FPS = 10
frame_step = 1            # sample every nth frame
downsample = 2            # downsample per axis for speed; 1 = full resolution
use_pyvista_first = True  # try PyVista pipeline first if available
window_size = (640, 640)  # px for pyvista screenshots
mpl_figsize = (6, 6)      # inches for matplotlib frames
mpl_dpi = 100             # dpi -> mpl_figsize * dpi = pixel size (6*100=600 px)
cmap_name = "inferno"
opacity_tf = [0.0, 0.02, 0.08, 0.25, 1.0]  # transfer function for PyVista
max_slice_samples = 64    # fallback compositing: max number of slices to composite
# ------------------------------------------------

# Derived settings: ensure consistent pixel size
target_pixels = (int(mpl_figsize[0] * mpl_dpi), int(mpl_figsize[1] * mpl_dpi))
print(f"Target frame pixel size (WxH): {target_pixels}")

# Temporary directory for frames
tmpdir = Path(tempfile.mkdtemp(prefix="vol_render_frames_"))
print("Saving intermediate frames to:", tmpdir)

# frame indices to render
n_frames = len(result.states)
indices = list(range(0, n_frames, frame_step))

frame_files = []

# Helper: save an RGBA (uint8) numpy array to PNG using Matplotlib with fixed size
def save_rgba_with_matplotlib(img_arr_uint8, out_path, title_text=None,
                             figsize=mpl_figsize, dpi=mpl_dpi, cmap=None):
    # img_arr_uint8: HxWx3 or HxWx4 uint8 array
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # fill figure to avoid variable margins
    ax.imshow(img_arr_uint8, origin='lower', aspect='equal')
    ax.axis('off')
    if title_text:
        # draw a small title in axes coords (white text)
        ax.text(0.02, 0.96, title_text, transform=ax.transAxes, color='white', fontsize=12,
                bbox=dict(facecolor='black', alpha=0.4, pad=2))
    # IMPORTANT: do NOT use bbox_inches='tight' or pad_inches -> will change pixel size
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

# Pure-Python fallback: alpha-blend a subset of slices to fake volume render
def composite_volume_to_image(dens, downsample_local, max_samples=max_slice_samples, cmap_name_local=cmap_name):
    # dens: 3D numpy array (nx, ny, nz)
    if downsample_local > 1:
        dens = dens[::downsample_local, ::downsample_local, ::downsample_local]

    fmin, fmax = float(np.nanmin(dens)), float(np.nanmax(dens))
    if not np.isfinite(fmin) or fmin == fmax:
        # return a placeholder image with text
        w, h = target_pixels
        placeholder = np.zeros((h, w, 3), dtype=np.uint8)
        return placeholder

    nx, ny, nz = dens.shape
    cmap_mpl = mpl.colormaps[cmap_name_local] if hasattr(mpl, "colormaps") else mpl.cm.get_cmap(cmap_name_local)

    # choose slice indices to sample: up to max_samples evenly spaced
    if nz <= max_samples:
        z_indices = list(range(nz))
    else:
        z_indices = np.linspace(0, nz - 1, max_samples, dtype=int)

    # accumulate front-to-back or back-to-front as desired; here back->front
    acc_rgb = np.zeros((nx, ny, 3), dtype=np.float32)
    acc_alpha = np.zeros((nx, ny), dtype=np.float32)

    # per-slice alpha tuned relative to sample count
    slice_alpha = min(0.06, 4.0 / max(1, len(z_indices)))  # heuristic

    for z in z_indices:
        slice_ = dens[:, :, z]
        norm = (slice_ - fmin) / (fmax - fmin + 1e-20)
        norm = np.clip(norm, 0.0, 1.0)
        rgba = cmap_mpl(norm)  # returns HxW x4 floats
        rgb = rgba[..., :3]
        alpha = slice_alpha * rgba[..., 3]
        weight = (1.0 - acc_alpha)[..., None] * alpha[..., None]
        acc_rgb = acc_rgb + weight * rgb
        acc_alpha = acc_alpha + (1.0 - acc_alpha) * alpha
        if np.all(acc_alpha >= 0.995):
            break

    out_img = np.clip(acc_rgb, 0.0, 1.0)
    out_img_uint8 = (out_img * 255).astype(np.uint8)
    # Matplotlib expects HxW x3 with uint8
    # We need to ensure the output image has the target pixel dimensions (WxH)
    # Current out_img_uint8 shape is (nx, ny, 3) -- where nx=rows(height), ny=cols(width)
    # Map to (height, width) = (nx, ny)
    # Resize via PIL to exact target_pixels (WxH)
    img_pil = Image.fromarray(out_img_uint8)
    # Note: target_pixels is (width, height)
    img_pil = img_pil.resize(target_pixels, resample=Image.LANCZOS)
    return np.array(img_pil)

# Try PyVista pipeline first (if available and requested)
if PV_OK and use_pyvista_first:
    print("PyVista available: attempting PyVista volume rendering pipeline...")
    try:
        for i in tqdm(indices, desc="Rendering frames (pyvista)"):
            state = result.states[i]
            dens = np.array(jnp.asarray(state[DENSITY_IDX]))
            if not np.isfinite(dens).all():
                print(f"[frame {i}] non-finite -> fallback composite")
                out_path = tmpdir / f"frame_{i:06d}.png"
                img_uint8 = composite_volume_to_image(dens, downsample, max_samples=max_slice_samples)
                title = f"t = {time[i]:.2f}" if 'time' in globals() else f"frame = {i}"
                save_rgba_with_matplotlib(img_uint8, out_path, title_text=title)
                frame_files.append(str(out_path))
                continue

            # downsample for speed if requested
            if downsample > 1:
                dens_ds = dens[::downsample, ::downsample, ::downsample]
            else:
                dens_ds = dens

            fmin, fmax = float(np.nanmin(dens_ds)), float(np.nanmax(dens_ds))
            if not np.isfinite(fmin) or fmin == fmax:
                print(f"[frame {i}] constant or invalid -> fallback composite")
                out_path = tmpdir / f"frame_{i:06d}.png"
                img_uint8 = composite_volume_to_image(dens, downsample, max_samples=max_slice_samples)
                title = f"t = {time[i]:.2f}" if 'time' in globals() else f"frame = {i}"
                save_rgba_with_matplotlib(img_uint8, out_path, title_text=title)
                frame_files.append(str(out_path))
                continue

            # Build a VTK/PyVista grid from numpy using pv.wrap (robust across versions)
            grid = pv.wrap(dens_ds)  # wraps as UniformGrid/ImageData or similar
            # set spacing so that images reflect downsampling scale
            try:
                grid.spacing = (downsample, downsample, downsample)
            except Exception:
                # some wrapped types may not allow assigning spacing; ignore safely
                pass

            # pick scalar name
            array_names = grid.array_names
            if len(array_names) == 0:
                grid["density"] = dens_ds.ravel(order='F')
                scalar_name = "density"
            else:
                scalar_name = array_names[0]

            # Off-screen plotter per-frame
            pl = pv.Plotter(off_screen=True, window_size=window_size)
            try:
                pl.add_volume(grid, scalars=scalar_name, cmap=cmap_name, clim=[fmin, fmax],
                              opacity=opacity_tf, shade=True, blending='composite')
            except Exception as e:
                # if PyVista can't add volume for this grid, fallback to composite for this frame
                print(f"[frame {i}] pv.add_volume error -> fallback composite (error: {e})")
                pl.close()
                out_path = tmpdir / f"frame_{i:06d}.png"
                img_uint8 = composite_volume_to_image(dens, downsample, max_samples=max_slice_samples)
                title = f"t = {time[i]:.2f}" if 'time' in globals() else f"frame = {i}"
                save_rgba_with_matplotlib(img_uint8, out_path, title_text=title)
                frame_files.append(str(out_path))
                continue

            # Text label
            pl.add_text(f"t = {time[i]:.2f}" if 'time' in globals() else f"frame = {i}", color='white', font_size=12)

            # Screenshot path
            frame_path = tmpdir / f"frame_{i:06d}.png"
            # pl.show(screenshot=str(frame_path)) -> prefer this to ensure fixed pixel size
            try:
                pl.show(screenshot=str(frame_path))
            except Exception as e_show:
                # In case show(screenshot=...) fails, fallback to composite
                print(f"[frame {i}] pv.show screenshot failed -> fallback composite ({e_show})")
                pl.close()
                out_path = tmpdir / f"frame_{i:06d}.png"
                img_uint8 = composite_volume_to_image(dens, downsample, max_samples=max_slice_samples)
                title = f"t = {time[i]:.2f}" if 'time' in globals() else f"frame = {i}"
                save_rgba_with_matplotlib(img_uint8, out_path, title_text=title)
                frame_files.append(str(out_path))
                continue

            pl.close()
            # Ensure the saved frame has exact pixel size: if PyVista produced screenshot with different size,
            # we will normalize later when composing the GIF.
            frame_files.append(str(frame_path))

        # After frame generation via PyVista, attempt to compose GIF.
        # If PyVista produced any frames, proceed; otherwise fallback below.
        if len(frame_files) == 0:
            raise RuntimeError("PyVista produced no frames; will run fallback-only pipeline.")
    except Exception as e_pv:
        # If anything unexpected fails, fall back to the pure-Python compositing for all frames
        print("PyVista pipeline failed with exception:", e_pv)
        print("Switching to fallback compositing for all frames.")
        # clear any partial frames
        for p in tmpdir.iterdir():
            try:
                p.unlink()
            except Exception:
                pass
        frame_files = []
        for i in tqdm(indices, desc="Rendering frames (fallback)"):
            state = result.states[i]
            dens = np.array(jnp.asarray(state[DENSITY_IDX]))
            out_path = tmpdir / f"frame_{i:06d}.png"
            img_uint8 = composite_volume_to_image(dens, downsample, max_samples=max_slice_samples)
            title = f"t = {time[i]:.2f}" if 'time' in globals() else f"frame = {i}"
            save_rgba_with_matplotlib(img_uint8, out_path, title_text=title)
            frame_files.append(str(out_path))

else:
    # PyVista not available -> use fallback compositing for all frames
    print("PyVista not available: running pure-Python fallback compositing for all frames.")
    for i in tqdm(indices, desc="Rendering frames (fallback)"):
        state = result.states[i]
        dens = np.array(jnp.asarray(state[DENSITY_IDX]))
        out_path = tmpdir / f"frame_{i:06d}.png"
        img_uint8 = composite_volume_to_image(dens, downsample, max_samples=max_slice_samples)
        title = f"t = {time[i]:.2f}" if 'time' in globals() else f"frame = {i}"
        save_rgba_with_matplotlib(img_uint8, out_path, title_text=title)
        frame_files.append(str(out_path))

# ---------------- Compose GIF robustly ----------------
if len(frame_files) == 0:
    print("No frames created; aborting GIF composition.")
else:
    print(f"Composing GIF from {len(frame_files)} frames...")

    # Determine target size by reading first frame
    with Image.open(frame_files[0]) as im0:
        target_size = im0.size  # (width, height)
    print("Initial target size (from first frame):", target_size)

    # If any other frame differs in size, we'll resize them to target_size.
    def load_and_resize_to_target(path, target_size_local):
        with Image.open(path) as im:
            # ensure RGBA or RGB
            if im.mode not in ("RGBA", "RGB"):
                im = im.convert("RGBA")
            if im.size != target_size_local:
                im = im.resize(target_size_local, resample=Image.LANCZOS)
            # convert to numpy array suitable for imageio
            arr = np.array(im)
            return arr

    # Stream frames to writer to avoid large memory usage
    try:
        with imageio.get_writer(str(OUT_GIF), mode='I', fps=FPS) as writer:
            for p in tqdm(frame_files, desc="Writing GIF"):
                arr = load_and_resize_to_target(p, target_size)
                writer.append_data(arr)
        print("Saved GIF to:", OUT_GIF)
    except Exception as e_write:
        print("Failed to write GIF directly due to:", e_write)
        # As a fallback try composing with imageio.mimsave using a list (may use more memory)
        try:
            imgs = [load_and_resize_to_target(p, target_size) for p in frame_files]
            imageio.mimsave(str(OUT_GIF), imgs, fps=FPS)
            print("Saved GIF (fallback) to:", OUT_GIF)
        except Exception as e2:
            print("Failed to create GIF:", e2)
            raise

# Cleanup temporary frames
try:
    shutil.rmtree(tmpdir)
    print("Cleaned up temporary frames.")
except Exception as e:
    print("Failed to remove temporary directory:", e)

print("Done.")


