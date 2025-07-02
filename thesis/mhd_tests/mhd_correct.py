# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus = 1, interval = 10)
# =======================

# numerics
import jax
import jax.numpy as jnp

# jf1uids
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

# equinox
import equinox as eqx

# optax for optimization
import optax

# CNN stuff
from jf1uids._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector import CorrectorCNN
from jf1uids._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector_options import CNNMHDParams, CNNMHDconfig

# plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# timing
from timeit import default_timer as timer
from tqdm import tqdm

# io
import numpy as np
import os

# ===================================================
# =============== ↓ simulation setup ↓ ==============
# ===================================================

# baseline config
baseline_config = SimulationConfig(
    progress_bar = True,
    mhd = True,
    dimensionality = 2,
    box_size = 1.0, 
    num_cells = 512,
    differentiation_mode = BACKWARDS,
    riemann_solver = HLL,
    exact_end_time = True,
    return_snapshots = False,
    use_specific_snapshot_timepoints = False,
    num_snapshots = 3
)

# common variable registry
registered_variables = get_registered_variables(baseline_config)

# baseline simulation parameters
params = SimulationParams(
    t_end = 0.2,
    C_cfl = 0.1,
    snapshot_timepoints = jnp.array([0.0, 0.1, 0.2])
)

def get_blast_setup(num_cells):

    # dummy config for setup
    dummy_config = baseline_config._replace(
        num_cells = num_cells,
    )

    helper_data = get_helper_data(dummy_config)

    # Grid size and configuration
    num_cells = dummy_config.num_cells
    # --- Initial Conditions ---
    grid_spacing = dummy_config.box_size / dummy_config.num_cells
    x = jnp.linspace(grid_spacing / 2, dummy_config.box_size - grid_spacing / 2, dummy_config.num_cells)
    y = jnp.linspace(grid_spacing / 2, dummy_config.box_size - grid_spacing / 2, dummy_config.num_cells)
    X, Y = jnp.meshgrid(x, y, indexing="ij")

    r = helper_data.r

    # Initialize state
    rho = jnp.ones_like(X)
    P = jnp.ones_like(X) * 0.1
    r_inj = 0.1 * dummy_config.box_size
    p_inj = 10.0
    P = jnp.where(r**2 < r_inj**2, p_inj, P)

    V_x = jnp.zeros_like(X)
    V_y = jnp.zeros_like(X)

    B_0 = 1 / jnp.sqrt(2)
    B_x = B_0 * jnp.ones_like(X)
    B_y = B_0 * jnp.ones_like(X)
    B_z = jnp.zeros_like(X)

    initial_state = construct_primitive_state(
        config = dummy_config,
        registered_variables = registered_variables,
        density = rho,
        velocity_x = V_x,
        velocity_y = V_y,
        magnetic_field_x = B_x,
        magnetic_field_y = B_y,
        magnetic_field_z = B_z,
        gas_pressure = P
    )

    return initial_state, helper_data

# ===================================================
# =============== ↑ simulation setup ↑ ==============
# ===================================================

# set up and run a high resolution simulation
num_cells = 512
config = baseline_config._replace(
    num_cells = num_cells,
)
initial_state, helper_data = get_blast_setup(num_cells)
# finalize the config
config = finalize_config(config, initial_state.shape)
# run the simulation
result = time_integration(initial_state, config, params, helper_data, registered_variables)
# # plot all snapshots
# fig, axes = plt.subplots(1, len(result.states), figsize=(15, 5))
# for i, state in enumerate(result.states):
#     ax = axes[i]
#     im = ax.imshow(state[registered_variables.density_index, ...], origin='lower', extent=(0, config.box_size, 0, config.box_size))
#     ax.set_title(f'Time: {result.time_points[i]:.2f}')
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     fig.colorbar(im, cax=cax)
# plt.tight_layout()
# plt.savefig('blasts.png', dpi=300)