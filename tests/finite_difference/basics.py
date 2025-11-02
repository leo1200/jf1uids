# numerics
import jax
import jax.numpy as jnp

# plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# fluids
from jf1uids import SimulationConfig
from jf1uids import get_helper_data
from jf1uids import SimulationParams
from jf1uids import time_integration
from jf1uids._finite_difference._fluid_equations._eigen import _eigen_x
from jf1uids._finite_difference._fluid_equations._equations import conserved_state_from_primitive_mhd, primitive_state_from_conserved_mhd
from jf1uids.initial_condition_generation.construct_primitive_state import construct_primitive_state
from jf1uids import get_registered_variables
from jf1uids.option_classes.simulation_config import DOUBLE_MINMOD, HLLC_LM, LAX_FRIEDRICHS, VAN_ALBADA_PP, finalize_config
import numpy as np
from matplotlib.colors import LogNorm

from jf1uids._finite_volume._magnetic_update._magnetic_field_update import magnetic_update

from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

# jf1uids constants
from jf1uids.option_classes.simulation_config import (
    BACKWARDS, FORWARDS, HLL, HLLC, MINMOD,
    OSHER, PERIODIC_BOUNDARY, BoundarySettings, BoundarySettings1D
)

def setup_blast_simulation(num_cells, B0, theta, phi):

    # spatial domain
    box_size = 1.0

    # setup simulation config
    config = SimulationConfig(
        runtime_debugging = False,
        progress_bar = True,
        mhd = True,
        dimensionality = 3,
        box_size = box_size, 
        num_cells = num_cells,
        limiter = MINMOD,
        riemann_solver = HLL,
        exact_end_time = True,
        boundary_settings = BoundarySettings(
            BoundarySettings1D(
                left_boundary = PERIODIC_BOUNDARY,
                right_boundary = PERIODIC_BOUNDARY
            ),
            BoundarySettings1D(
                left_boundary = PERIODIC_BOUNDARY,
                right_boundary = PERIODIC_BOUNDARY
            ),
            BoundarySettings1D(
                left_boundary = PERIODIC_BOUNDARY,
                right_boundary = PERIODIC_BOUNDARY
            )
        ),
    )

    helper_data = get_helper_data(config)

    params = SimulationParams(
        t_end = 0.01,
        C_cfl = 0.4,
        gamma = 1.4
    )

    registered_variables = get_registered_variables(config)

    r = helper_data.r

    rho = jnp.ones_like(r)
    P = jnp.ones_like(r) * 0.1
    r_inj = 0.1 * box_size
    p_inj = 1000
    P = jnp.where(r**2 < r_inj**2, p_inj, P)

    V_x = jnp.zeros_like(r)
    V_y = jnp.zeros_like(r)
    V_z = jnp.zeros_like(r)

    B_x = B0 * jnp.sin(theta) * jnp.cos(phi)
    B_y = B0 * jnp.sin(theta) * jnp.sin(phi)
    B_z = B0 * jnp.cos(theta)

    print(f"Magnetic field: Bx={B_x}, By={B_y}, Bz={B_z}")

    B_x = jnp.ones_like(r) * B_x
    B_y = jnp.ones_like(r) * B_y
    B_z = jnp.ones_like(r) * B_z

    initial_state = construct_primitive_state(
        config = config,
        registered_variables = registered_variables,
        density = rho,
        velocity_x = V_x,
        velocity_y = V_y,
        velocity_z = V_z,
        magnetic_field_x = B_x,
        magnetic_field_y = B_y,
        magnetic_field_z = B_z,
        gas_pressure = P
    )

    config = finalize_config(config, initial_state.shape)

    return initial_state, config, registered_variables, params, helper_data

num_cells = 32
B0 = 100 / jnp.sqrt(4 * jnp.pi)
theta = jnp.pi / 2
phi = jnp.pi / 4

initial_state, config, registered_variables, params, helper_data = setup_blast_simulation(num_cells, B0, theta, phi)

conserved_state = conserved_state_from_primitive_mhd(
    primitive_state = initial_state,
    gamma = params.gamma,
    config = config,
    registered_variables = registered_variables,
)

primitive_state = primitive_state_from_conserved_mhd(
    conserved_state = conserved_state,
    gamma = params.gamma,
    config = config,
    registered_variables = registered_variables,
)

# check that initial state and primitive state from conserved match
assert jnp.allclose(initial_state, primitive_state), "Initial state and primitive state from conserved do not match!"

lamb, R, L = _eigen_x(
    conserved_state,
    5/3,
    registered_variables,
)