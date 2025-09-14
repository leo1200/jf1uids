import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

# numerics
import jax
import jax.numpy as jnp
# # for now using CPU as of outdated NVIDIA Driver
# jax.config.update('jax_platform_name', 'cpu')
# # jax.config.update('jax_disable_jit', True)
# # 64-bit precision
# jax.config.update("jax_enable_x64", True)

# debug nans
# jax.config.update("jax_debug_nans", True)

# timing
from timeit import default_timer as timer

# plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# fluids
from jf1uids import WindParams
from jf1uids import SimulationConfig
from jf1uids import get_helper_data
from jf1uids import SimulationParams
from jf1uids import time_integration
from jf1uids.fluid_equations.fluid import construct_primitive_state
from jf1uids._physics_modules._stellar_wind.stellar_wind_functions import get_wind_parameters
from jf1uids.option_classes.simulation_config import DOUBLE_MINMOD, LAX_FRIEDRICHS, MUSCL, RK2_SSP, SIMPLE_SOURCE_TERM, SPLIT, UNSPLIT, BoundarySettings, BoundarySettings1D, DONOR_ACCOUNTING, RIEMANN_SPLIT, RIEMANN_SPLIT_UNSTABLE

from jf1uids import get_registered_variables
from jf1uids.option_classes import WindConfig

from jf1uids.option_classes.simulation_config import BACKWARDS, OSHER

# units
from jf1uids import CodeUnits
from astropy import units as u
import astropy.constants as c
from astropy.constants import m_p

# wind-specific
from jf1uids._physics_modules._stellar_wind.weaver import Weaver

# turbulence
from jf1uids.initial_condition_generation.turb import create_turb_field
from jf1uids.option_classes.simulation_config import FORWARDS
from jf1uids.option_classes.simulation_config import (
    BACKWARDS, FORWARDS, HLL, HLLC, MINMOD, OSHER, 
    PERIODIC_BOUNDARY, OPEN_BOUNDARY, REFLECTIVE_BOUNDARY, 
    BoundarySettings, BoundarySettings1D
)

print("👷 Setting up simulation...")

# simulation settings
gamma = 5/3

# spatial domain
box_size = 2.0
num_cells = 180

# activate stellar wind
stellar_wind = True

# turbulence
turbulence = False
wanted_rms = 5 * u.km / u.s

fixed_timestep = False
scale_time = False
dt_max = 0.1
num_timesteps = 1600
boundary=OPEN_BOUNDARY
# setup simulation config
config = SimulationConfig(
    runtime_debugging = False,
    first_order_fallback = True,
    progress_bar = True,
    dimensionality = 3,
    self_gravity_version = SIMPLE_SOURCE_TERM,
    num_ghost_cells = 2,
    box_size = box_size, 
    num_cells = num_cells,
    split = UNSPLIT,
    limiter = MINMOD,
    time_integrator = RK2_SSP,
    wind_config = WindConfig(
        stellar_wind = stellar_wind,
        num_injection_cells = 2,
        trace_wind_density = False,
        real_wind_params = True,

    ),
    fixed_timestep = fixed_timestep,
    differentiation_mode = FORWARDS,
    num_timesteps = num_timesteps,
    return_snapshots = True,
    num_snapshots = 200,
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

helper_data = get_helper_data(config)

registered_variables = get_registered_variables(config)



from jf1uids.initial_condition_generation.turb import create_incompressible_turb_field
from jf1uids.option_classes.simulation_config import finalize_config

# code_length = 3 * u.parsec
# code_mass = 1 * u.M_sun
# code_velocity = 100 * u.km / u.s
# code_units = CodeUnits(code_length, code_mass, code_velocity)
# time domain
C_CFL = 0.4
# 2.5

length_temp = 30  # 1 parsec in au
mass_temp = 1
velocity_temp = 10
# time_temp = length_temp / velocity_temp  #make time_temp as high as possible to have small code time units

code_length = length_temp * u.parsec 
code_mass = mass_temp * u.M_sun
code_velocity = velocity_temp * u.km / u.s
# code_time = code_length / code_velocity
code_units = CodeUnits(code_length, code_mass, code_velocity)

# t_final = 1.2 * 1e6 * u.yr       #1.0 * 1e3 * u.yr
t_final = 6275065 * u.yr 
t_end = t_final.to(code_units.code_time).value
# t_end = t_final / time_temp

print("t_end_code: ", t_end)


# wind parameters
# M_star = 40 * u.M_sun
# wind_final_velocities = jnp.array([2000, 2000])* u.km / u.s
# wind_mass_loss_rates = jnp.array([2.965e-3 / 1e6, 3e-3 / 1e6]) /  u.yr * M_star
import numpy as np
masses_in_Msun = np.array([40.0, 20.0])  

####TODO: two different masses

t_yr, mass_rates_value, vel_scales_value = get_wind_parameters(masses_in_Msun)
##convert to code units
# t_val = t_yr / time_temp
# mass_rates_value = 10**mass_rates_value / mass_temp * time_temp
# vel_scales_value = vel_scales_value / velocity_temp
t_val = (t_yr * u.yr).to(code_units.code_time).value
mass_rates_value = (10**mass_rates_value * u.M_sun / u.yr).to(code_units.code_mass / code_units.code_time).value
vel_scales_value = (vel_scales_value * u.km / u.s).to(code_units.code_velocity).value

###only last n elements
n=0
t_val = t_val[:, -n:] #- (5e6 * u.yr).to(code_units.code_time).value   
mass_rates_value = mass_rates_value[:, -n:]  
vel_scales_value = vel_scales_value[:, -n:]


wind_params = WindParams(
    # wind_mass_loss_rates =  jnp.array([mass_loss_rate, mass_loss_rate2]),
    # wind_final_velocities = jnp.array([wind_vel_inf, wind_vel_inf]), #jnp.array([vel_param, vel_param]) * v_phi0, 
    wind_injection_positions = jnp.array([[-0.1, 0.0, 0.0],[0.1, 0.0, 0.0]]),#jnp.array([[-0.02,0.0,0.0],[0.02,0.0,0.0]])
    real_params = jnp.array([t_val, mass_rates_value, vel_scales_value])
)
print("wind_params.real_params: ", wind_params.real_params)
# if config.wind_config.real_wind_params:
#     print("phys time points: ", t_val * code_time)
#     print("Physical vel: ", vel_scales_value * code_velocity)
#     print("Physical mass loss rate: ", (mass_rates_value * code_mass / code_time))

params = SimulationParams(
    C_cfl = C_CFL,
    dt_max = dt_max,
    gamma = gamma,
    t_end = t_end,
    wind_params = wind_params
)

# homogeneous initial state
rho_0 = 2 * c.m_p / u.cm**3
p_0 = 3e4 * u.K / u.cm**3 * c.k_B

print("p0",p_0.to(code_units.code_pressure).value)

rho = jnp.ones((config.num_cells, config.num_cells, config.num_cells)) * rho_0.to(code_units.code_density).value

print("rho0",rho_0.to(code_units.code_density).value)

u_x = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))
u_y = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))
u_z = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))

x = jnp.linspace(0, config.box_size, config.num_cells)
y = jnp.linspace(0, config.box_size, config.num_cells)
z = jnp.linspace(0, config.box_size, config.num_cells)

turbulence_slope = -2
kmin = 2
kmax = 64

if turbulence:
    ux = create_turb_field(config.num_cells, 1, turbulence_slope, kmin, kmax, seed = 1)
    uy = create_turb_field(config.num_cells, 1, turbulence_slope, kmin, kmax, seed = 2)
    uz = create_turb_field(config.num_cells, 1, turbulence_slope, kmin, kmax, seed = 3)

    # ux, uy, uz = create_incompressible_turb_field(config.num_cells, 1, turbulence_slope, kmin, kmax, seed = 1)

    a = num_cells // 2 - 10
    b = num_cells // 2 + 10

    u_x = ux# .at[a:b, a:b, a:b].set(ux[a:b, a:b, a:b])
    u_y = uy# .at[a:b, a:b, a:b].set(uy[a:b, a:b, a:b])
    u_z = uz# .at[a:b, a:b, a:b].set(uz[a:b, a:b, a:b])

    rms_vel = jnp.sqrt(jnp.mean(ux**2 + uy**2 + uz**2))

    u_x = u_x / rms_vel * wanted_rms.to(code_units.code_velocity).value
    u_y = u_y / rms_vel * wanted_rms.to(code_units.code_velocity).value
    u_z = u_z / rms_vel * wanted_rms.to(code_units.code_velocity).value

p = jnp.ones((config.num_cells, config.num_cells, config.num_cells)) * p_0.to(code_units.code_pressure).value

# construct primitive state

initial_state = construct_primitive_state(
    config = config,
    registered_variables=registered_variables,
    density = rho,
    velocity_x = u_x,
    velocity_y = u_y,
    velocity_z = u_z,
    gas_pressure = p
)

config = finalize_config(config, initial_state.shape)

# velll = jnp.sqrt(u_x**2 + u_y**2 + u_z**2)
# print(f"max velocity: {jnp.max(velll)}")
# print(f"wind velocity: {wind_final_velocities.to(code_units.code_velocity).value}")
# print(f"mass loss rates: {wind_mass_loss_rates.to(code_units.code_mass / code_units.code_time).value}")

result = time_integration(initial_state, config, params, helper_data, registered_variables)
final_state = result.states[-1]

save_path = os.path.join('wolf_rayet_diff2')
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

print(jnp.min(final_state[registered_variables.pressure_index]))


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

ax1.scatter(helper_data.r[:, :, num_cells // 2].flatten(), final_state[registered_variables.density_index, :, :, num_cells // 2].flatten(), label="Final Density", s = 1)
# ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("r")
ax1.set_ylabel("Density")

# velocity profile
# v_r = jnp.sqrt(final_state[registered_variables.velocity_index.x] ** 2 + final_state[registered_variables.velocity_index.y] ** 2 + final_state[registered_variables.velocity_index.z] ** 2)
v_r = jnp.sqrt(final_state[registered_variables.velocity_index.x, :, :, num_cells // 2] ** 2 + final_state[registered_variables.velocity_index.y, :, :, num_cells // 2] ** 2 + final_state[registered_variables.velocity_index.z, :, :, num_cells // 2] ** 2)

ax2.scatter(helper_data.r[:, :, num_cells // 2].flatten(), v_r.flatten(), label="Radial Velocity", s = 1)
ax2.set_xlabel("r")
ax2.set_ylabel("Velocity")

# plot P / rho^gamma

ax3.scatter(helper_data.r[:, :, num_cells // 2].flatten(), final_state[registered_variables.pressure_index, :, :, num_cells // 2].flatten(), label="P / rho^gamma", s = 1)
ax3.set_xlabel("r")
ax3.set_ylabel("Pressure")

# fig.suptitle("Binary")

plt.tight_layout()
fig.savefig(save_path+"/var_finals.png")
s = 45

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# equal aspect ratio
ax1.set_aspect('equal', 'box')
ax2.set_aspect('equal', 'box')
ax3.set_aspect('equal', 'box')

z_level = num_cells // 2

xm, ym = jnp.meshgrid(x, y)

# # on the first axis plot the density
# # log scaler
# norm_rho = LogNorm(vmin = jnp.min(final_state[0, :, :, z_level]), vmax = jnp.max(final_state[0, :, :, z_level]), clip = True)
# norm_p = LogNorm(vmin = jnp.min(final_state[4, :, :, z_level]), vmax = jnp.max(final_state[4, :, :, z_level]), clip = True)



# ax1.scatter(xm.flatten() * code_units.code_length, ym.flatten() * code_units.code_length, c = final_state[0, :, :, z_level].flatten(), s = s, marker = "s")
# ax1.set_title("Density")

# # on the second axis plot the absolute velocity
# abs_vel = jnp.sqrt(final_state[1, :, :, z_level]**2 + final_state[2, :, :, z_level]**2 + final_state[3, :, :, z_level]**2)

# vel_norm = LogNorm(vmin = jnp.min(abs_vel), vmax = jnp.max(abs_vel), clip = True)

# ax2.scatter(xm.flatten() * code_units.code_length, ym.flatten() * code_units.code_length, c = abs_vel.flatten(), s = s, marker = "s")
# ax2.set_title("Velocity")

# # on the third axis plot the pressure
# ax3.scatter(xm.flatten() * code_units.code_length, ym.flatten() * code_units.code_length, c = final_state[4, :, :, z_level].flatten(), s = s, marker = "s") # , norm = norm_p)
# ax3.set_title("Pressure")

ax1.imshow(final_state[0, :, :, z_level].T, origin = "lower", extent = [0, 1, 0, 1], norm = LogNorm())
ax1.set_title("Density")

ax2.imshow(jnp.sqrt(final_state[1, :, :, z_level]**2 + final_state[2, :, :, z_level]**2).T, origin = "lower", extent = [0, 1, 0, 1])
ax2.set_title("Velocity")

ax3.imshow(final_state[4, :, :, z_level].T, origin = "lower", extent = [0, 1, 0, 1], norm = LogNorm())
ax3.set_title("Pressure")

fig.savefig(save_path+'/3d_wind.png')


print(jnp.min(final_state[1, :, :, z_level]), jnp.max(final_state[1, :, :, z_level]))

# print the minimum 
print(jnp.min(final_state[registered_variables.pressure_index]))

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
    im = ax.scatter(helper_data.r[:, :, num_cells // 2].flatten(), state[0, :, :, num_cells // 2].flatten(), s = 1)
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
    im = ax.scatter(helper_data.r[:, :, num_cells // 2].flatten(), jnp.sqrt(state[registered_variables.velocity_index.x, :, :, num_cells // 2] ** 2 + state[registered_variables.velocity_index.y, :, :, num_cells // 2] ** 2 + state[registered_variables.velocity_index.z, :, :, num_cells // 2] ** 2).flatten(), s = 1)
    ax.set_xlabel("r")
    ax.set_ylabel("velocity")
    ax.set_title(f"Velocity at time {time[i]:.2f}")
    return [im]

ani = animation.FuncAnimation(fig, animate, frames=len(result.states), interval=100)
ani.save(save_path+"/ani_vel.gif")
plt.close(fig)

step = 3
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
cell_width = config.box_size / config.num_cells
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
