# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus = 1)
# =======================

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# units
from jf1uids import CodeUnits
from astropy import units as u
import astropy.constants as c
from astropy.constants import m_p

# jf1uids data structures
from jf1uids import SimulationConfig
from jf1uids import SimulationParams

# jf1uids constants
from jf1uids.option_classes.simulation_config import CARTESIAN, HLLC, SPHERICAL, HLL, MINMOD


# jf1uids functions
from jf1uids import get_helper_data
from jf1uids.fluid_equations.fluid import construct_primitive_state
from jf1uids import get_registered_variables
from jf1uids.option_classes.simulation_config import finalize_config

from jf1uids import time_integration

# --- Additional imports for analysis and plotting ---
# For radially averaging the simulation data
from scipy.stats import binned_statistic
# For the exact Sedov-Taylor solution
from exactpack.solvers.sedov.sedov import Sedov


config = SimulationConfig(

    
    geometry = CARTESIAN,
    progress_bar = True,
    runtime_debugging = False,

    riemann_solver = HLLC,
    
    dimensionality = 3,

    exact_end_time = True,

    # ====== RESOLUTION =======
    num_cells = 256,
    # =========================

    # ==== snapshotting ====
    return_snapshots = False,
    # ======================

)


params = SimulationParams(
    t_end = 0.1
)

helper_data = get_helper_data(config)
registered_variables = get_registered_variables(config)

# total explosion energy
E_explosion = 1.0

E_gas = E_explosion

# Ambient (background) physical conditions (adjust as needed)
rho_ambient  = 1.0         # typical ISM density
p_ambient    = 1e-4          # low gas pressure

# Pressures in code units
p_ambient = p_ambient

# --- Set Up the Explosion Injection Region ---

rho = jnp.ones((config.num_cells, config.num_cells, config.num_cells)) * rho_ambient

u_x = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))
u_y = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))
u_z = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))

# currently, we take 10 injection cells
r_explosion = 0.02

# Compute the injection volume (spherical volume in code units)
injection_volume = (4/3) * jnp.pi * r_explosion**3

# Adiabatic indices:
gamma_gas = params.gamma   # for the thermal gas
gamma_cr  = 4/3   # for cosmic rays

# The energy contained in a uniform pressure region is related by:
#   E = p * V / (gamma - 1)
# Hence, the effective explosion pressure in the injection region (in code units)
p_explosion_gas = E_gas * (gamma_gas - 1) / injection_volume

# Convert to code units
p_explosion_gas = p_explosion_gas

# --- Define the Radial Profiles ---
# Get the radial coordinate array (assumed already available)
r = helper_data.r

# Gas pressure: high within the explosion region, ambient elsewhere
p_gas = jnp.where(r < r_explosion, p_explosion_gas, p_ambient)

# construct primitive state
initial_state = construct_primitive_state(
    config = config,
    registered_variables=registered_variables,
    density = rho,
    velocity_x = u_x,
    velocity_y = u_y,
    velocity_z = u_z,
    gas_pressure = p_gas
)

config = finalize_config(config, initial_state.shape)

# Run the simulation
result = time_integration(initial_state, config, params, helper_data, registered_variables)


# =========================================================================
# === Analysis and Plotting ===============================================
# =========================================================================

print("Simulation finished. Starting analysis and plotting...")

# --- Prepare Simulation Data for Plotting ---
# Convert JAX arrays to NumPy arrays for processing
r_flat = np.array(helper_data.r.flatten())
rho_sim = np.array(result[registered_variables.density_index].flatten())
p_sim = np.array(result[registered_variables.pressure_index].flatten())

# Calculate absolute velocity magnitude
vx = result[registered_variables.velocity_index.x]
vy = result[registered_variables.velocity_index.y]
vz = result[registered_variables.velocity_index.z]
v_abs_sim = np.array(jnp.sqrt(vx**2 + vy**2 + vz**2).flatten())


# --- Calculate Radially Averaged Profiles ---
# Create bins for averaging based on radius
num_bins = 100
domain_max_r = np.max(r_flat)
bins = np.linspace(0, domain_max_r, num_bins + 1)
bin_centers = (bins[:-1] + bins[1:]) / 2

# Use binned_statistic to average data in each radial bin
mean_rho, _, _ = binned_statistic(r_flat, rho_sim, statistic='mean', bins=bins)
mean_v_abs, _, _ = binned_statistic(r_flat, v_abs_sim, statistic='mean', bins=bins)
mean_p, _, _ = binned_statistic(r_flat, p_sim, statistic='mean', bins=bins)


# --- Generate the Exact Sedov-Taylor Solution ---
# The Sedov solver needs parameters consistent with the simulation.
# The `eblast` parameter for the solver corresponds to E_explosion / rho_ambient.
eblast_exact = E_explosion / rho_ambient

# Set up the solver for a spherical (geometry=3), uniform ambient medium (omega=0) explosion
sedov_solver = Sedov(geometry=3, eblast=eblast_exact,
                     gamma=params.gamma, omega=0.)

# Generate the solution at the final time of the simulation on a fine grid
r_exact = np.linspace(0.0, domain_max_r, 500)
solution_exact = sedov_solver(r=r_exact, t=params.t_end)



# --- Create the Comparison Plots ---
fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=120)
fig.suptitle(f'Sedov Blast Wave Comparison at t = {params.t_end:.2f} (3D Cartesian Grid, {config.num_cells}³ grid)', fontsize=16)

# --- 1. Density Plot ---
ax = axes[0]
ax.scatter(r_flat, rho_sim, color='lightgray', alpha=0.5, label='simulation (raw data)', rasterized=True)
ax.plot(bin_centers, mean_rho, 'o', color='royalblue', markersize=4, label='simulation (radial avg.)')

# FIXED: Plot exact solution data directly using ax.plot
ax.plot(r_exact, solution_exact['density'], ls='--', lw=2, c='red', label='exact solution')
ax.set_title('Density')
ax.set_xlabel('Radius')
ax.set_ylabel('Density')
ax.grid(True, linestyle=':', alpha=0.7)
ax.set_xlim(0, domain_max_r)


# --- 2. Absolute Velocity Plot ---
ax = axes[1]
# Plot raw simulation data for absolute velocity
ax.scatter(r_flat, v_abs_sim, color='lightgray', alpha=0.5, label='simulation (raw data)', rasterized=True)
ax.plot(bin_centers, mean_v_abs, 'o', color='royalblue', markersize=4, label='simulation (radial avg.)')

# FIXED: Plot exact solution data directly using ax.plot
ax.plot(r_exact, solution_exact['velocity'], ls='--', lw=2, c='red', label='exact solution')
ax.set_title('Absolute Velocity')
ax.set_xlabel('Radius')
ax.set_ylabel(r'$|v|$')
ax.grid(True, linestyle=':', alpha=0.7)
ax.set_xlim(0, domain_max_r)
ax.set_ylim(0, None)


# --- 3. Pressure Plot ---
ax = axes[2]
# FIXED: Plot exact solution data directly using ax.plot
ax.scatter(r_flat, p_sim, color='lightgray', alpha=0.5, label='simulation (raw data)', rasterized=True)
ax.plot(bin_centers, mean_p, 'o', color='royalblue', markersize=4, label='simulation (radial avg.)')

ax.plot(r_exact, solution_exact['pressure'], ls='--', lw=2, c='red', label='exact solution')
ax.set_title('Pressure')
ax.set_xlabel('Radius')
ax.set_ylabel('Pressure')
ax.grid(True, linestyle=':', alpha=0.7)
ax.set_yscale('log') # Pressure spans orders of magnitude, so a log scale is better
ax.set_xlim(0, domain_max_r)
ax.set_ylim(bottom=p_ambient/2)


# --- Create a common legend below the plots ---
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.02))

# Adjust layout and save the figure
plt.tight_layout(rect=[0, 0.1, 1, 0.96]) # Adjust for suptitle and common legend
plt.savefig('figures/sedovHLLC.pdf', dpi = 300, bbox_inches='tight')
print("Comparison plot 'figures/sedov.pdf' has been saved.")