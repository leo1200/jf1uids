# IMPORTANT: check gpustat --watch before
# running scripts on the cluster
# BETTER NOT USE NOTEBOOKS AT ALL,
# THEY BLOCK THE GPU MEMORY IF NOT
# RESET PROPERLY
import yaml

config_file = yaml.safe_load(
    open("home/jalegria/Thesis/jf1uids/corrector_src/config.yaml", "r")
)
import os

# # ==== GPU selection ====
from autocvd import autocvd

autocvd(num_gpus=1)
# # =======================

# numerics
import jax.numpy as jnp
import numpy as np

# jf1uids data structures
from jf1uids import SimulationConfig
from jf1uids import SimulationParams
from jf1uids.option_classes.simulation_config import FORWARDS

# jf1uids setup functions
from jf1uids import get_helper_data
from jf1uids.fluid_equations.fluid import construct_primitive_state
from jf1uids import get_registered_variables
from jf1uids.option_classes.simulation_config import finalize_config

# turbulent ic setup
from jf1uids.initial_condition_generation.turb import create_turb_field

# main simulation function
from jf1uids import time_integration

# units
from jf1uids import CodeUnits
from astropy import units as u
import astropy.constants as c

import h5py

from scipy.ndimage import convolve

from jf1uids.corrector_src.utils.downaverage import downaverage_state

dimensionality = 2


def convolve_lr(
    images: np.array, stride: int = 4, kernel: np.array = np.ones((3, 3, 3)) / 27
) -> np.array:
    n_images = images.shape[0]
    n_channels = images.shape[1]
    convolved_size = images.shape[2] // stride
    convolved_images = np.empty(
        (n_images, n_channels, convolved_size, convolved_size, convolved_size)
    )
    for batch_n in range(n_images):
        for channel in range(n_channels):
            convolved_images[batch_n, channel] = convolve(
                images[batch_n, channel], kernel, mode="reflect", cval=0.0
            )[::stride, ::stride, ::stride]
    return convolved_images


# ===================================================
# ============== ↓ Set up simulation ↓ =============
# ===================================================

base_config = SimulationConfig(
    runtime_debugging=config_file["turbulent_sim"]["runtime_debug"],
    first_order_fallback=config_file["turbulent_sim"]["first_order_fb"],
    progress_bar=config_file["turbulent_sim"]["progress_bar"],
    dimensionality=dimensionality,
    num_ghost_cells=config_file["turbulent_sim"]["num_ghost_cells"],
    box_size=config_file["turbulent_sim"]["box_size"],
    num_cells=config_file["turbulent_sim"]["num_cells"],
    fixed_timestep=config_file["turbulent_sim"][
        "fixed_timestep"
    ],  # to compare intermidiate states from low_res to high_res we need the fixed timestep
    num_timesteps=config_file["turbulent_sim"]["num_snapshots"],
    differentiation_mode=FORWARDS,
    return_snapshots=config_file["turbulent_sim"]["return_snapshots"],
    num_snapshots=config_file["turbulent_sim"]["num_snapshots"],
)

registered_variables = get_registered_variables(base_config)

config_high_res = base_config._replace(
    num_cells=config_file["turbulent_stim"]["num_cells"]
)
config_low_res = base_config._replace(
    num_cells=config_file["turbulent_stim"]["num_cells"]
    // config_file["data"]["upsample_factor"]
)

helper_data_high_res = get_helper_data(config_high_res)
helper_data_low_res = get_helper_data(config_low_res)

# setup the unit system
code_length = 3 * u.parsec
code_mass = 1 * u.M_sun
code_velocity = 100 * u.km / u.s
code_units = CodeUnits(code_length, code_mass, code_velocity)

# time domain
C_CFL = 0.4

# set the final time of the simulation
t_final = 1.0 * 1e4 * u.yr
t_end = t_final.to(code_units.code_time).value

# simulation settings
gamma = 5 / 3

# turbulence
wanted_rms = 50 * u.km / u.s
dt_max = 0.1

# set the simulation parameters
params = SimulationParams(
    C_cfl=C_CFL,
    dt_max=dt_max,
    gamma=gamma,
    t_end=t_end,
)

# homogeneous initial state
rho_0 = 2 * c.m_p / u.cm**3
p_0 = 3e4 * u.K / u.cm**3 * c.k_B

rho = (
    jnp.ones(
        (
            config_high_res.num_cells,
            config_high_res.num_cells,
        )
    )
    * rho_0.to(code_units.code_density).value
)

# turbulence parameters
turbulence_slope = config_file["turbulent_sim"]["turbulence_slope"]
kmin = config_file["turbulent_sim"]["kmin"]
kmax = config_file["turbulent_sim"]["kmax"]

i = 0
a = config_high_res.num_cells // 2 - 10
b = config_high_res.num_cells // 2 + 10
p = (
    jnp.ones(
        (
            config_high_res.num_cells,
            config_high_res.num_cells,
        )
    )
    * p_0.to(code_units.code_pressure).value
)

# ===================================================
# ============== ↓ File setup ↓ =============
# ===================================================


save_path = "./data/jalegria/turbulence"
os.makedirs(save_path, exist_ok=True)

h5_path = os.path.join(save_path, "2d_data.h5")
h5f = h5py.File(h5_path, "w")

max_sims = config_file["turbulent_sim"]["max_sims"]
hr_shape = (
    max_sims * base_config.num_snapshots,
    base_config.dimensionality + 2,
    config_high_res.num_cells,
    config_high_res.num_cells,
)
lr_shape = (
    max_sims * base_config.num_snapshots,
    base_config.dimensionality + 2,
    config_low_res.num_cells,
    config_low_res.num_cells,
)

hr_states_dataset = h5f.create_dataset("hr_states", shape=hr_shape, dtype="float32")
lr_states_dataset = h5f.create_dataset("lr_states", shape=lr_shape, dtype="float32")
energy_dataset = h5f.create_dataset(
    "first_snapshot_energy",
    shape=(max_sims * base_config.num_snapshots),
    dtype="float32",
)
mass_dataset = h5f.create_dataset(
    "first_snapshot_mass", shape=(max_sims * base_config.num_snapshots), dtype="float32"
)

# ===================================================
# ============== ↓ Data creation loop ↓ =============
# ===================================================

snapshot_counter = 0
while i < max_sims:
    u_x = create_turb_field(config_high_res.num_cells, 1, turbulence_slope, kmin, kmax)
    u_y = create_turb_field(config_high_res.num_cells, 1, turbulence_slope, kmin, kmax)

    # scale the turbulence to the desired rms velocity
    rms_vel = jnp.sqrt(jnp.mean(u_x**2 + u_y**2))
    if not jnp.isfinite(rms_vel) or rms_vel == 0.0:
        print("Skipping iteration due to bad rms_vel:", rms_vel)
        continue
    u_x = u_x / rms_vel * wanted_rms.to(code_units.code_velocity).value
    u_y = u_y / rms_vel * wanted_rms.to(code_units.code_velocity).value

    # construct primitive state
    initial_state_high_res = construct_primitive_state(
        config=config_high_res,
        registered_variables=registered_variables,
        density=rho,
        velocity_x=u_x,
        velocity_y=u_y,
        gas_pressure=p,
    )

    config_high_res = finalize_config(
        initial_state_high_res,
        config_high_res,
        params,
        helper_data_high_res,
        registered_variables,
    )
    result_high_res = time_integration(
        initial_state_high_res,
        config_high_res,
        params,
        helper_data_high_res,
        registered_variables,
    )

    if np.all(result_high_res.states[-1] == 0):
        continue

    initial_state_low_res = downaverage_state(
        initial_state_high_res, downsample_factor=config_file["data"]["upsample_factor"]
    )

    result_low_res = time_integration(
        initial_state_low_res,
        config_low_res,
        params,
        helper_data_low_res,
        registered_variables,
    )
    for snapshot_idx, snapshot in enumerate(result_high_res.states):
        hr_states_dataset[snapshot_counter] = np.array(snapshot, dtype="float32")
        lr_states_dataset[snapshot_counter] = np.array(
            result_low_res.states[snapshot_idx], dtype="float32"
        )
        energy_dataset[snapshot_counter] = result_high_res.total_energy[0]
        mass_dataset[snapshot_counter] = result_high_res.total_mass[0]
        snapshot_counter += 1
    i += 1
    if i % 10 == 0:
        print(i)
    del result_high_res
    del result_low_res
    del initial_state_high_res
    del initial_state_low_res
h5f.close()
