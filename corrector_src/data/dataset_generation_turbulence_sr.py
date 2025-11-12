# IMPORTANT: check gpustat --watch before
# running scripts on the cluster
# BETTER NOT USE NOTEBOOKS AT ALL,
# THEY BLOCK THE GPU MEMORY IF NOT
# RESET PROPERLY
# import yaml

# config_file = yaml.safe_load(
#     open("home/jalegria/Thesis/turbulence_sr/config.yaml", "r")
# )
"""
Code to generate h5py dataset with the final state of a given high resolution and downscaled by a given factor
Only works with 3d!!
"""

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

from corrector_src.utils.downaverage import downaverage as downaverage_state
from hydra import initialize, compose
from tqdm import tqdm
import time
import jax

with initialize(config_path="../../configs", version_base="1.2"):
    cfg = compose(
        config_name="config",
        overrides=["data=turbulence", "training=turbulence_optuna"],
    )

dimensionality = 3
max_sims = 2500
upsample_factor = 4
file_name = (
    f"final_state_turb_{cfg.data.hr_res}_{cfg.data.hr_res // upsample_factor}.h5"
)


# setup simulation config
base_config = SimulationConfig(
    runtime_debugging=False,
    first_order_fallback=False,
    progress_bar=False,
    dimensionality=dimensionality,
    num_ghost_cells=2,
    box_size=1.0,
    num_cells=cfg.data.hr_res,
    fixed_timestep=False,  # to compare intermidiate states from low_res to high_res we need the fixed timestep
    differentiation_mode=FORWARDS,
    return_snapshots=False,
    active_nan_checker=True,
)

registered_variables = get_registered_variables(base_config)

config_high_res = base_config._replace(num_cells=cfg.data.hr_res)
config_low_res = base_config._replace(num_cells=cfg.data.hr_res // upsample_factor)

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
    # t_end=cfg.data.t_end,
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
            config_high_res.num_cells,
        )
    )
    * rho_0.to(code_units.code_density).value
)

# turbulence parameters
turbulence_slope = -2
kmin = 2
kmax = 64

a = config_high_res.num_cells // 2 - 10
b = config_high_res.num_cells // 2 + 10
p = (
    jnp.ones(
        (
            config_high_res.num_cells,
            config_high_res.num_cells,
            config_high_res.num_cells,
        )
    )
    * p_0.to(code_units.code_pressure).value
)


def create_initial_state(config):
    infinite_vel = True
    while infinite_vel:
        rng_seed = int(time.time() * 1e6) % (2**32 - 1)

        key = jax.random.key(rng_seed)

        keys = jax.random.split(key, 3)

        u_x = create_turb_field(
            config.num_cells, 1, turbulence_slope, kmin, kmax, key=keys[0]
        )
        u_y = create_turb_field(
            config.num_cells, 1, turbulence_slope, kmin, kmax, key=keys[1]
        )
        u_z = create_turb_field(
            config.num_cells, 1, turbulence_slope, kmin, kmax, key=keys[2]
        )
        # scale the turbulence to the desired rms velocity
        rms_vel = jnp.sqrt(jnp.mean(u_x**2 + u_y**2 + u_z**2))
        if not jnp.isfinite(rms_vel) or rms_vel == 0.0:
            print("Skipping iteration due to bad rms_vel:", rms_vel)
            continue
        infinite_vel = False
        u_x = u_x / rms_vel * wanted_rms.to(code_units.code_velocity).value
        u_y = u_y / rms_vel * wanted_rms.to(code_units.code_velocity).value
        u_z = u_z / rms_vel * wanted_rms.to(code_units.code_velocity).value

        # construct primitive state
        initial_state = construct_primitive_state(
            config=config,
            registered_variables=registered_variables,
            density=rho,
            velocity_x=u_x,
            velocity_y=u_y,
            velocity_z=u_z,
            gas_pressure=p,
        )

        return initial_state


# ===================================================
# ============== ↓ File setup ↓ =============
# ===================================================


save_path = "./data/jalegria/turbulence"
os.makedirs(save_path, exist_ok=True)

h5_path = os.path.join(save_path, file_name)
h5f = h5py.File(h5_path, "w")

hr_shape = (
    max_sims,
    dimensionality + 2,
    config_high_res.num_cells,
    config_high_res.num_cells,
    config_high_res.num_cells,
)
lr_shape = (
    max_sims,
    dimensionality + 2,
    config_low_res.num_cells,
    config_low_res.num_cells,
    config_low_res.num_cells,
)

hr_states_dataset = h5f.create_dataset("hr_states", shape=hr_shape, dtype="float32")
lr_states_dataset = h5f.create_dataset("lr_states", shape=lr_shape, dtype="float32")
# energy_dataset = h5f.create_dataset(
#     "first_snapshot_energy",
#     shape=(max_sims),
#     dtype="float32",
# )
# mass_dataset = h5f.create_dataset(
#     "first_snapshot_mass", shape=(max_sims), dtype="float32"
# )

# ===================================================
# ============== ↓ Data creation loop ↓ =============
# ===================================================

i = 0

for i in tqdm(range(max_sims)):
    is_nan_data = True
    while is_nan_data:
        initial_state_high_res = create_initial_state(config_high_res)
        if i == 0:
            config_high_res = finalize_config(
                config_high_res, initial_state_high_res.shape
            )

        is_nan_data, result_high_res = time_integration(
            initial_state_high_res,
            config_high_res,
            params,
            helper_data_high_res,
            registered_variables,
        )

        if is_nan_data:
            continue

        initial_state_low_res = downaverage_state(
            initial_state_high_res, downscale_factor=upsample_factor
        )

        if i == 0:
            config_low_res = finalize_config(
                config_low_res, initial_state_low_res.shape
            )

        is_nan_data, result_low_res = time_integration(
            initial_state_low_res,
            config_low_res,
            params,
            helper_data_low_res,
            registered_variables,
        )
        if is_nan_data:
            continue

    hr_states_dataset[i] = np.array(result_high_res, dtype="float32")
    lr_states_dataset[i] = np.array(result_low_res, dtype="float32")

    del result_high_res
    del result_low_res
    del initial_state_high_res
    del initial_state_low_res
h5f.close()
