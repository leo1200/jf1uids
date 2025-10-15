# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus = 1)
# =======================

# numerics
import jax
import jax.numpy as jnp

# plotting
import matplotlib.pyplot as plt

# jf1uids classes
from jf1uids import SimulationConfig
from jf1uids import SimulationParams
from jf1uids.option_classes.simulation_config import (
    DONOR_ACCOUNTING,
    HLLC_LM,
    RIEMANN_SPLIT,
    RIEMANN_SPLIT_UNSTABLE,
    BoundarySettings,
    BoundarySettings1D
)

# jf1uids functions
from jf1uids import get_helper_data
from jf1uids import time_integration
from jf1uids.initial_condition_generation.construct_primitive_state import construct_primitive_state
from jf1uids.option_classes.simulation_config import finalize_config
from jf1uids import get_registered_variables

# jf1uids constants
from jf1uids.option_classes.simulation_config import (
    BACKWARDS, FORWARDS, HLL, HLLC, MINMOD, OSHER, 
    PERIODIC_BOUNDARY, REFLECTIVE_BOUNDARY, 
    BoundarySettings, BoundarySettings1D,
    DOUBLE_MINMOD,
    LAX_FRIEDRICHS,
    MUSCL,
    RK2_SSP,
    SIMPLE_SOURCE_TERM,
    SPLIT,
    UNSPLIT,
    DOUBLE_MINMOD,
    LAX_FRIEDRICHS,
    MUSCL,
    RK2_SSP,
    SIMPLE_SOURCE_TERM,
    SPLIT,
    UNSPLIT,
)

self_gravity_version = RIEMANN_SPLIT_UNSTABLE

# simulation settings
gamma = 5/3

# spatial domain
box_size = 4.0

baseline_config = SimulationConfig(
    runtime_debugging = False,
    progress_bar = True,
    self_gravity = True,
    self_gravity_version = self_gravity_version,
    first_order_fallback = False,
    dimensionality = 3,
    box_size = box_size,
    split = SPLIT,
    differentiation_mode = FORWARDS,
    limiter = MINMOD,
    time_integrator = MUSCL,
    riemann_solver = HLLC,
    boundary_settings = BoundarySettings(
        BoundarySettings1D(
            left_boundary = REFLECTIVE_BOUNDARY,
            right_boundary = REFLECTIVE_BOUNDARY
        ),
        BoundarySettings1D(
            left_boundary = REFLECTIVE_BOUNDARY,
            right_boundary = REFLECTIVE_BOUNDARY
        ),
        BoundarySettings1D(
            left_boundary = REFLECTIVE_BOUNDARY,
            right_boundary = REFLECTIVE_BOUNDARY
        )
    ),
    return_snapshots = False,
    num_snapshots = 60
)

# -------------------------------------------------------------
# =================== ↓ Evrard's Collapse ↓ ===================
# -------------------------------------------------------------

def simulate_collapse(num_cells, t_end = 3.0, return_snapshots = True):

    print("👷 Setting up simulation...")
    # setup simulation config
    config = baseline_config._replace(
        num_cells = num_cells,
        return_snapshots = return_snapshots,
    )

    helper_data = get_helper_data(config)

    params = SimulationParams(
        t_end = t_end,
        C_cfl = 0.4,
    )

    registered_variables = get_registered_variables(config)
  
    R = 1.0
    M = 1.0

    dx = config.box_size / (config.num_cells - 1)

    # initialize density field
    rho = jnp.where(helper_data.r <= R, M / (2 * jnp.pi * R ** 2 * helper_data.r), 1e-4)

    total_injected_mass = jnp.sum(jnp.where(helper_data.r <= R, rho, 0)) * dx ** 3
    print(f"Injected mass: {total_injected_mass}")

    # better ball edges
    # overlap_weights = (R + dx / 2 - helper_data.r) / dx
    # rho = jnp.where((helper_data.r > R - dx / 2) & (helper_data.r < R + dx / 2), rho * overlap_weights, rho)

    # Initialize velocity fields to zero
    v_x = jnp.zeros_like(rho)
    v_y = jnp.zeros_like(rho)
    v_z = jnp.zeros_like(rho)

    # initial thermal energy per unit mass = 0.05
    e = 0.05
    p = (gamma - 1) * rho * e

    # Construct the initial primitive state for the 3D simulation.
    initial_state = construct_primitive_state(
        config = config,
        registered_variables = registered_variables,
        density = rho,
        velocity_x = v_x,
        velocity_y = v_y,
        velocity_z = v_z,
        gas_pressure = p
    )

    config = finalize_config(config, initial_state.shape)

    return jax.block_until_ready(
        time_integration(initial_state, config, params, helper_data, registered_variables)
    ), config, params, helper_data, registered_variables

def resolution_study_collapse():

    num_cells_list = [64, 128]
    line_styles = ['-', '--', '-.', ':']

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    for num_cells, line_style in zip(num_cells_list, line_styles):
        print(f"Running simulation for {num_cells} cells...")
        snapshots, _, _, _, _ = simulate_collapse(num_cells)
        total_energy = snapshots.total_energy
        internal_energy = snapshots.internal_energy
        kinetic_energy = snapshots.kinetic_energy
        gravitational_energy = snapshots.gravitational_energy
        time = snapshots.time_points
        ax.plot(time, total_energy, label="Total Energy, N = " + str(num_cells), color = 'black', linestyle = line_style)
        ax.plot(time, internal_energy, label="Internal Energy, N = " + str(num_cells), color = 'green', linestyle = line_style)
        ax.plot(time, kinetic_energy, label="Kinetic Energy, N = " + str(num_cells), color = 'red', linestyle = line_style)
        ax.plot(time, gravitational_energy, label="Gravitational Energy, N = " + str(num_cells), color = 'blue', linestyle = line_style)
        ax.set_xlabel("Time")
        ax.set_ylabel("Energy")

    ax.set_ylim(-2.5, 2.5)

    ax.legend(fontsize="x-small", ncol=len(num_cells_list))
    ax.set_title("Resolution Study for Evrard's Collapse")

    plt.savefig(f"collapse_resolution_study_{'simple' if self_gravity_version == SIMPLE_SOURCE_TERM else 'conservative'}_source_term.svg")

def radial_profile_study():

    num_cells_list = [64, 128]

    for num_cells in num_cells_list:

        print(f"Running radial profile simulation for {num_cells} cells...")

        final_state, _, params, helper_data, registered_variables = simulate_collapse(num_cells, t_end = 0.8, return_snapshots = False)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

        ax1.scatter(helper_data.r.flatten(), final_state[registered_variables.density_index].flatten(), label="Final Density", s = 1)
        # x and y log scale
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xlim(1e-2, 6e-1)
        ax1.set_ylim(1e-2, 1e3)
        ax1.set_xlabel("r")
        ax1.set_ylabel("Density")

        # velocity profile
        v_r = -jnp.sqrt(final_state[registered_variables.velocity_index.x] ** 2 + final_state[registered_variables.velocity_index.y] ** 2 + final_state[registered_variables.velocity_index.z] ** 2)

        ax2.scatter(helper_data.r.flatten(), v_r.flatten(), label="Radial Velocity", s = 1)
        # log x scale
        ax2.set_xscale("log")
        ax2.set_xlim(1e-2, 6e-1)
        ax2.set_xlabel("r")
        ax2.set_ylabel("Velocity")

        # plot P / rho^gamma
        ax3.scatter(helper_data.r.flatten(), final_state[registered_variables.pressure_index].flatten() / final_state[registered_variables.density_index].flatten() ** params.gamma, label="P / rho^gamma", s = 1)
        ax3.set_xlim(4.0 / num_cells, 6e-1)
        ax3.set_ylim(0, 0.2)
        ax3.set_xlabel("r")
        ax3.set_ylabel("P / rho^gamma")
        ax3.set_xscale("log")

        fig.suptitle("3D Collapse Test")

        plt.tight_layout()

        plt.savefig(f"collapse_radial_profile_{num_cells}.png")


resolution_study_collapse()
radial_profile_study()


# -------------------------------------------------------------
# =================== ↑ Evrard's Collapse ↑ ===================
# -------------------------------------------------------------