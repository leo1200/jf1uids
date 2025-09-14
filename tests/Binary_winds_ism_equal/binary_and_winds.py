import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
from jf1uids._physics_modules._binary._binary_options import NGP, CIC, TSC
from jf1uids._physics_modules._binary._binary_options import BinaryParams
from jf1uids import WindParams

from jf1uids.option_classes.simulation_config import BoundarySettings, BoundarySettings1D

# jf1uids functions
from jf1uids import get_helper_data
from jf1uids import time_integration
from jf1uids.fluid_equations.fluid import construct_primitive_state
from jf1uids.option_classes.simulation_config import finalize_config
from jf1uids import get_registered_variables
from jf1uids.option_classes.simulation_config import BinaryConfig
from jf1uids.option_classes import WindConfig

# units
from jf1uids import CodeUnits
from astropy import units as u
import astropy.constants as c
from astropy.constants import m_p

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
mass_source = 1e-3
P = 12
v_phi0 = (jnp.pi*mass_source/(2*P))**(1/3) #jnp.sqrt(mass_source/2/delta_x)   #for equal mass binary
R_forced = (mass_source*P**2/(2*jnp.pi**2)) ** (1/3)   #forcing radius)
box_size = float(2.5 * R_forced)
num_cells = 100  #60  #128

fixed_timestep = False
dt_max = 0.001

self_gravity_version = SIMPLE_SOURCE_TERM

# self_gravity_version = CONSERVATIVE_SOURCE_TERM
# boundary = REFLECTIVE_BOUNDARY
boundary = OPEN_BOUNDARY

# setup simulation config
config = SimulationConfig(
    runtime_debugging = False,
    progress_bar = True,
    binary_config = BinaryConfig(
        binary = True,
        deposit_particles = TSC,  # Options: "ngp", "cic", "tsc"
        central_object_only = False
    ),
    self_gravity = True,
    mhd=False,
    self_gravity_version = self_gravity_version,
    dimensionality = 3,
    box_size = box_size,
    wind_config = WindConfig(
        stellar_wind = True,
        num_injection_cells = 4,
        trace_wind_density = False,
    ),
    split = UNSPLIT, 
    num_cells = num_cells,
    fixed_timestep = fixed_timestep,
    first_order_fallback= True,
    differentiation_mode = FORWARDS,
    limiter = MINMOD,
    time_integrator = RK2_SSP,
    # time_integrator = MUSCL,
    riemann_solver = HLL,
    return_snapshots = True,
    num_snapshots = 100,
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
# mass_source = 1e-3  #1e-3 works 
rho_source = mass_source / (cell_width ** 3)
print("mass both stars:", 2*mass_source, "density star: ", rho_source)
masses = jnp.array([mass_source, mass_source])  

x1=0.05
x2=-0.05
delta_x = jnp.abs(x1-x2)
# v_phi0 = 0.5 #jnp.sqrt(mass_source/2/delta_x)   #for equal mass binary
print("R_forced / 2: ", R_forced / 2)
print("Azimuthal vel of each star: ", v_phi0)
txv1 = jnp.array([0.0, R_forced/2, 0.0, 0.0, 0.0, -v_phi0, 0.0])    #this is clockwise rotation in the xy plane
txv2 = jnp.array([0.0, -R_forced/2, 0.0, 0.0, 0.0, v_phi0, 0.0])
binary_state = jnp.concatenate([txv1, txv2])

binary_params = BinaryParams(
    masses = masses,
    # masses = mass_source,
    binary_state = binary_state
    # binary_state=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
)
T_orb = 2*jnp.pi*(R_forced) / 2 / v_phi0   #T_orb at r=0.5
print("T_orb_binary: ", T_orb)

# M_star = 40 * u.M_sun
wind_vel_inf = 50.0
# wind_mass_loss_rates = jnp.array([2.965e-3 / 1e6]) /  u.yr * M_star
mass_loss_rate = 1e-10 / 1.309 #2.965e-3 / 1e4 / T_orb * mass_source     #2.965e-3 / 1e6 / T_orb * mass_source
print("mass loss rate: ", mass_loss_rate)

#mass loss rates [0.00347898]
# vel_param = 2000
wind_params = WindParams(
    wind_mass_loss_rates =  jnp.array([mass_loss_rate, mass_loss_rate]),
    wind_final_velocities = jnp.array([wind_vel_inf, wind_vel_inf]) #jnp.array([vel_param, vel_param]) * v_phi0, 
    # wind_injection_positions = jnp.array([[-0.02,0.0,0.0],[0.02,0.0,0.0]])
)
print("mass loss, wind_vel/phi0: ", wind_params.wind_mass_loss_rates, wind_params.wind_final_velocities/v_phi0)
code_length = 5 / R_forced * u.au   
code_mass = 40 / mass_source * u.M_sun
code_velocity = 2000 / wind_vel_inf * u.km / u.s
code_units = CodeUnits(code_length, code_mass, code_velocity)

# mass = code_mass.to(code_units.code_mass).value
# print(mass)

t_end = 2.*T_orb
print("t_end: ", t_end)

print("Physical vel: ", wind_vel_inf * code_velocity)
print("Physical mass: ", mass_source * code_mass)
print("Physical orb sep: ", R_forced * code_length)
print("Physical mass loss rate: ", (mass_loss_rate * code_mass / (code_units.code_time)).to(u.M_sun / u.yr))
rho_0 = 10**2 * c.m_p / u.cm**3 #2 * c.m_p / u.cm**3 
p_0 = 4e6 * u.K / u.cm**3 * c.k_B  #3e4 * u.K / u.cm**3 * c.k_B

P_0_test = p_0.to(code_units.code_pressure).value
print("p0",P_0_test)
rho_0_test = rho_0.to(code_units.code_density).value
print("rho0", rho_0_test)

params = SimulationParams(
    C_cfl = 0.4,
    dt_max = dt_max,
    gamma = gamma,
    t_end = t_end,  
    binary_params = binary_params,
    wind_params = wind_params
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
Mtot = jnp.sum(masses)

R_cav = 2.5          # cavity radius [au]
delta_0    = 5e-7  #1e-5        # floor parameter
sigma_0    = 8e-5   # choose normalization for Σ(r)
rsafe = 1e-5

# Compute surface density 
sigma = sigma_0 * ((1.0 - delta_0) * jnp.exp(- (R_cav / (r_cyl))**12) + delta_0) 
# sigma = jnp.ones_like(r_cyl) * 1e-5  # uniform surface density
# Disc scale height H(r) = 0.1 r
H = 0.1 * (r_cyl + rsafe)  # :contentReference[oaicite:1]{index=1}

# 3D density: vertical Gaussian
R_o = 2.5
R_i = 0.2
#mask = True   #(r_cyl <= R_o) & (r_cyl >= R_i) & (z_cyl >= -0.3) & (z_cyl <= 0.3),
floor = 1e-8  #1e-8
rho = jnp.clip(sigma / (H * jnp.sqrt(2 * jnp.pi)) * jnp.exp(-0.5 * (z_cyl / H)**2), a_min=floor)

#### UNIFORM DENSITY
M = (2 * mass_source) / 100 * 5     # 5 percent of the binary mass
rho = jnp.ones_like(helper_data.r) * M / (4/3*jnp.pi*(1.2*R_forced)**3)            #/ (config.box_size ** 3)
rho = jnp.ones_like(helper_data.r) * rho_0_test

# Sound speed (locally isothermal)
d_1 = jnp.sqrt((x_cyl - x1) ** 2 + y_cyl ** 2 + z_cyl ** 2)
d_2 = jnp.sqrt((x_cyl - x2) ** 2 + y_cyl ** 2 + z_cyl ** 2)
cs = 0.1 * jnp.sqrt(G * ((masses[0] / (d_1 + rsafe)) + (masses[1] / (d_2 + rsafe))))

# Gas pressure 
P = jnp.clip(rho * cs**2, a_min=floor*0.1**2)


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
# v_x = -v_phi *  y_cyl / (r_cyl + rsafe)   ## this is clockwise rotation in the xy plane
# v_y =  v_phi *  x_cyl / (r_cyl + rsafe)
# v_z = jnp.zeros_like(v_x)

# zero velocities
v_x=jnp.zeros_like(rho)
v_y=jnp.zeros_like(rho)
v_z=jnp.zeros_like(rho)

######from STELLAR WINDS################
p0 = 0.016523955885709894
rho0 = 1.3345574378679779
# rho = jnp.ones_like(v_x)*rho0
####uniform pressure
K0 = 0.1
P = K0 * rho      #(polytrope n->inf, isothermal self gravitationg sphere)
P = jnp.ones_like(v_x)*P_0_test

disc_mass=jnp.sum(rho) * cell_width**3     ## use the cell_width definition  (WHY dx???)
disc_inner_mass = jnp.sum(a = rho, where = r_cyl <= 1.2*R_forced) * cell_width ** 3
# print(f"Total disc mass: {total_mass:.3e}")
print(f"Total disc mass: {disc_mass:.3e}")
print("M_disc/M_star: ", disc_mass/Mtot)
print("M_disc_inner/M_star: ", disc_inner_mass/Mtot)
print("vx_max: ", jnp.max(v_x), "vy_max: ", jnp.max(v_y))

# =========================================

# initial thermal energy per unit mass = 0.05
e = 0.05
p = (gamma - 1) * rho * e

###magnetic field
B_0=1e-4
B_x= jnp.ones_like(v_x)*B_0
B_x= jnp.zeros_like(v_x)*B_0         ###ZERO
B_y= jnp.zeros_like(v_x)*B_0
B_z= jnp.zeros_like(v_x)*B_0

# Construct the initial primitive state for the 3D simulation.
initial_state = construct_primitive_state(
    config = config,
    registered_variables = registered_variables,
    density = rho,
    velocity_x = v_y,
    velocity_y = v_x,
    velocity_z = v_z,
    magnetic_field_x = B_x,
    magnetic_field_y = B_y,
    magnetic_field_z = B_z,
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

save_path = os.path.join('Binary_winds5')
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
import numpy as np

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
"""
DENSITY_IDX = registered_variables.density_index
out_gif = save_path + "/ani_rho_radial.gif"
n_radial_bins = num_cells
log_y = True

r_arr = np.array(jnp.asarray(helper_data.r))
r_flat = r_arr.ravel()
r_max = float(r_flat.max())
bin_edges = np.linspace(0.0, r_max, n_radial_bins + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
bin_idx = np.digitize(r_flat, bins=bin_edges) - 1
bin_idx = np.clip(bin_idx, 0, n_radial_bins - 1)
counts = np.bincount(bin_idx, minlength=n_radial_bins).astype(np.int64)
valid = counts > 0

def radial_profile_from_state(state):
    dens = np.array(jnp.asarray(state[DENSITY_IDX]))  # (nx,ny,nz)
    dens_flat = dens.ravel()
    sum_in_bin = np.bincount(bin_idx, weights=dens_flat, minlength=n_radial_bins)
    rho_shell = np.full(n_radial_bins, np.nan, dtype=np.float64)
    rho_shell[valid] = sum_in_bin[valid] / counts[valid]
    return bin_centers, rho_shell

fig, ax = plt.subplots()
line, = ax.plot([], [], '-k', lw=1)
ax.set_xlabel("r")
ax.set_ylabel("Density")
if log_y:
    ax.set_yscale("log")
ax.set_xlim(0.0, r_max)

def init():
    line.set_data([], [])
    return (line,)

def animate_frame(i):
    state = result.states[i]
    r_vals, rho_vals = radial_profile_from_state(state)
    # adjust y-limits based on finite values (optional: set global limits instead)
    finite = np.isfinite(rho_vals)
    if finite.any():
        if log_y:
            ymin = max(rho_vals[finite].min() * 0.5, 1e-16)
            ymax = rho_vals[finite].max() * 2.0
            ax.set_ylim(ymin, ymax)
        else:
            ax.set_ylim(np.nanmin(rho_vals[finite]) * 0.9, np.nanmax(rho_vals[finite]) * 1.1)
    line.set_data(r_vals, rho_vals)
    ax.set_title(f"Spherically-averaged density — t = {time[i]:.2f}" if 'time' in globals() else f"frame {i}")
    return (line,)

ani = animation.FuncAnimation(fig, animate_frame, frames=len(result.states),
                              init_func=init, interval=100, blit=False)
ani.save(out_gif)
plt.close(fig)
"""

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

step = 1
DENSITY_IDX = registered_variables.density_index
indices = list(range(0, len(result.states), step)) 
projections = []
times = []
for idx in indices:
    state = result.states[idx]
    dens = state[DENSITY_IDX, :, :, :]
    proj = jnp.array(jnp.sum(dens, axis=2))   # shape (nx, ny) or (ny, nx) depending on ordering
    projections.append(proj)
    try:
        times.append(time[idx])
    except:
        times = None

# Determine consistent color scale across frames
projections = [p * cell_width for p in projections]
all_min = float(min(jnp.min(p) for p in projections))
all_max = float(max(jnp.max(p) for p in projections))

fig, ax = plt.subplots(figsize=(6,6))

im = ax.imshow(projections[0], origin='lower', aspect='equal', cmap='inferno', norm=LogNorm(vmax=all_max,vmin=all_min)) #,vmax=all_max, vmin=all_min,
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Column density (code units)")

time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, color="white", fontsize=10,
                    bbox=dict(facecolor='black', alpha=0.4, boxstyle='round'))

ax.set_xlabel("grid x")
ax.set_ylabel("grid y")

def animate(i):
    im.set_data(projections[i])
    try:
        tstr = f"t = {times[i]:.2f}"
    except Exception:
        tstr = f"frame = {indices[i]}"
    time_text.set_text(tstr)
    return [im, time_text]

n_frames = len(projections)
ani = animation.FuncAnimation(fig, animate, frames=n_frames, interval=100)
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
    # Render a single isosurface from states[frame_index] and save as PNG.
    # - states: iterable of state arrays indexed like state[variable_index, ix, iy, iz].
    # - density_idx: index (e.g. registered_variables.density_index).
    # - out_png: path to save PNG (string or Path).
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
# Volume-render animation (PyVista preferred, fallback alpha-composite).
# Drop into your environment where `result.states`, `registered_variables.density_index`,
# and `save_path` are defined. Optionally `time` can be present for timestamps.
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

# mask = np.array(jnp.asarray(helper_data.r < 1.8))
mask = True
# Try PyVista pipeline first (if available and requested)
if PV_OK and use_pyvista_first:
    print("PyVista available: attempting PyVista volume rendering pipeline...")
    try:
        for i in tqdm(indices, desc="Rendering frames (pyvista)"):
            state = result.states[i]
            dens = np.array(jnp.asarray(state[DENSITY_IDX]))
            dens = np.where(mask, dens, 0)
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
            dens = np.where(mask, dens, 0)
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
        dens = np.where(mask, dens, 0)
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

