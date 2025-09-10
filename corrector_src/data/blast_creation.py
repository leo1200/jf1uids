# ==== GPU selection ====
# from autocvd import autocvd

# autocvd(num_gpus=1)
# =======================

# numerics
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# fluids
from jf1uids import SimulationConfig
from jf1uids import get_helper_data
from jf1uids import SimulationParams
from jf1uids import time_integration
from jf1uids.fluid_equations.fluid import construct_primitive_state
from jf1uids import get_registered_variables
from jf1uids.option_classes.simulation_config import finalize_config

# jf1uids constants
from jf1uids.option_classes.simulation_config import (
    BACKWARDS,
    FORWARDS,
    HLL,
    HLLC,
    MINMOD,
    OSHER,
    PERIODIC_BOUNDARY,
    BoundarySettings,
    BoundarySettings1D,
)

# units
from jf1uids import CodeUnits
from astropy import units as u

import random

import yaml

config_file = yaml.safe_load(open("config.yaml", "r"))


def initial_blast_state(num_cells):
    adiabatic_index = 5 / 3
    box_size = 1.0
    fixed_timestep = True
    dt_max = 0.1
    mhd = True

    # setup simulation config
    config = SimulationConfig(
        runtime_debugging=False,
        first_order_fallback=False,
        progress_bar=True,
        dimensionality=3,
        num_ghost_cells=2,
        box_size=box_size,
        num_cells=num_cells,
        mhd=mhd,
        fixed_timestep=fixed_timestep,
        differentiation_mode=BACKWARDS,
        riemann_solver=HLL,
        limiter=0,
        return_snapshots=True,
        num_snapshots=80,
        # boundary_settings=BoundarySettings(
        #    x=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        #    y=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        #    z=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        # ),
    )

    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)

    # setup the unit system
    code_length = 3 * u.parsec
    code_mass = 1 * u.M_sun
    code_velocity = 100 * u.km / u.s
    code_units = CodeUnits(code_length, code_mass, code_velocity)

    # time domain
    C_CFL = 0.4  # Courant-Friedrichs-Lewy number
    t_final = 1.0 * 1e4 * u.yr
    t_end = t_final.to(code_units.code_time).value

    # set the simulation parameters
    params = SimulationParams(
        C_cfl=C_CFL,
        dt_max=dt_max,
        gamma=adiabatic_index,
        t_end=t_end,
    )

    grid_spacing = config.box_size / config.num_cells
    x = jnp.linspace(
        grid_spacing / 2, config.box_size - grid_spacing / 2, config.num_cells
    )
    y = jnp.linspace(
        grid_spacing / 2, config.box_size - grid_spacing / 2, config.num_cells
    )
    z = jnp.linspace(
        grid_spacing / 2, config.box_size - grid_spacing / 2, config.num_cells
    )

    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")

    r = helper_data.r

    # Initialize state
    rho = jnp.ones_like(X)
    P = jnp.ones_like(X) * 0.1
    r_inj = 0.1 * box_size
    p_inj = 10.0
    P = jnp.where(r**2 < r_inj**2, p_inj, P)

    u_x = jnp.zeros_like(X)
    u_y = jnp.zeros_like(X)
    u_z = jnp.zeros_like(X)

    B_0 = 1 / np.sqrt(2)
    B_x = B_0 * jnp.ones_like(X)
    B_y = B_0 * jnp.ones_like(X)
    B_z = jnp.zeros_like(X)

    initial_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=rho,
        velocity_x=u_x,
        velocity_y=u_y,
        velocity_z=u_z,
        gas_pressure=P,
        magnetic_field_x=B_x,
        magnetic_field_y=B_y,
        magnetic_field_z=B_z,
    )
    return initial_state, config, params, helper_data, registered_variables


def randomized_initial_blast_state(num_cells, randomizers=None):
    adiabatic_index = 5 / 3
    box_size = 1.0
    fixed_timestep = True
    dt_max = 0.1
    mhd = True

    # setup simulation config
    config = SimulationConfig(
        runtime_debugging=False,
        first_order_fallback=False,
        progress_bar=False,
        dimensionality=3,
        num_ghost_cells=2,
        box_size=box_size,
        num_cells=num_cells,
        mhd=mhd,
        fixed_timestep=fixed_timestep,
        differentiation_mode=BACKWARDS,
        riemann_solver=HLL,
        limiter=0,
        return_snapshots=True,
        num_snapshots=80,
        boundary_settings=BoundarySettings(),
        # boundary_settings=BoundarySettings(
        #    x=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        #    y=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        #    z=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        # ),
    )

    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)

    # setup the unit system
    code_length = 3 * u.parsec
    code_mass = 1 * u.M_sun
    code_velocity = 100 * u.km / u.s
    code_units = CodeUnits(code_length, code_mass, code_velocity)

    # time domain
    C_CFL = 0.4  # Courant-Friedrichs-Lewy number
    t_final = 1.0 * 1e4 * u.yr
    t_end = t_final.to(code_units.code_time).value

    # set the simulation parameters
    params = SimulationParams(
        C_cfl=C_CFL,
        dt_max=dt_max,
        gamma=adiabatic_index,
        t_end=t_end,
    )

    grid_spacing = config.box_size / config.num_cells
    x = jnp.linspace(
        grid_spacing / 2, config.box_size - grid_spacing / 2, config.num_cells
    )
    y = jnp.linspace(
        grid_spacing / 2, config.box_size - grid_spacing / 2, config.num_cells
    )
    z = jnp.linspace(
        grid_spacing / 2, config.box_size - grid_spacing / 2, config.num_cells
    )

    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")

    r = helper_data.r
    if randomizers is None:
        randomizers = [
            random.uniform(
                config_file["blast_creation"]["randomizer_1"][0],
                config_file["blast_creation"]["randomizer_1"][0],
            ),
            random.uniform(
                config_file["blast_creation"]["randomizer_2"][0],
                config_file["blast_creation"]["randomizer_2"][1],
            ),
            random.uniform(
                config_file["blast_creation"]["randomizer_3"][0],
                config_file["blast_creation"]["randomizer_3"][1],
            ),
        ]
    # Initialize state
    rho = jnp.ones_like(X)
    P = jnp.ones_like(X) * 0.1
    r_inj = 0.1 * box_size * randomizers[0]
    p_inj = 10.0 * randomizers[1]
    P = jnp.where(r**2 < r_inj**2, p_inj, P)

    u_x = jnp.zeros_like(X)
    u_y = jnp.zeros_like(X)
    u_z = jnp.zeros_like(X)

    B_0 = 1 / np.sqrt(2) * randomizers[2]
    B_x = B_0 * jnp.ones_like(X)
    B_y = B_0 * jnp.ones_like(X)
    B_z = jnp.zeros_like(X)

    initial_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=rho,
        velocity_x=u_x,
        velocity_y=u_y,
        velocity_z=u_z,
        gas_pressure=P,
        magnetic_field_x=B_x,
        magnetic_field_y=B_y,
        magnetic_field_z=B_z,
    )
    return initial_state, config, params, helper_data, registered_variables, randomizers


def run_blast_simulation(num_cells):
    initial_state, config, params, helper_data, registered_variables = (
        initial_blast_state(num_cells)
    )
    config = finalize_config(config, initial_state.shape)

    final_state = time_integration(
        initial_state, config, params, helper_data, registered_variables
    )

    return final_state
