"""
didnt manage it to get it working, assumed jf1uids had an energy conserving scheme
"""

from autocvd import autocvd

num_gpus = 1
num_snapshots = 40
resolution = 128
multi_gpu = num_gpus > 1
split_turb = (2, 2, 1)
split_training = (1, 2, 2, 1)
assert sum(x for x in split_turb if x > 1) == num_gpus or num_gpus == 1, (
    f"Sum of splits {sum(x for x in split_turb if x > 1)} != num_gpus ({num_gpus})"
)
assert sum(x for x in split_training if x > 1) == num_gpus or num_gpus == 1, (
    f"Sum of splits {sum(x for x in split_training if x > 1)} != num_gpus ({num_gpus})"
)
return_states = True
sgs_turb_energy = True
runtime_debugging = False
progress_bar = True
autocvd(num_gpus=num_gpus)

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# os.environ["JAX_TRACEBACK_FILTERING"] = "off"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.2"

import numpy as np

import jax.numpy as jnp
import jax
from jax.sharding import PartitionSpec as P


import matplotlib.pyplot as plt
import time
import hydra
from jf1uids import SimulationConfig
from jf1uids import get_helper_data
from jf1uids import SimulationParams
from corrector_src.turbulence_integration.time_integration import time_integration
from jf1uids.data_classes.simulation_helper_data import HelperData
from jf1uids.fluid_equations.registered_variables import RegisteredVariables
from jf1uids.fluid_equations.fluid import construct_primitive_state
from jf1uids import get_registered_variables
from jf1uids.option_classes.simulation_config import SnapshotSettings, finalize_config
from jf1uids.initial_condition_generation.turb import create_turb_field
from jf1uids._physics_modules._cooling.cooling_options import CoolingConfig
from jf1uids.option_classes.simulation_config import (
    FORWARDS,
    HLL,
    VARAXIS,
    XAXIS,
    YAXIS,
    ZAXIS,
)


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
import astropy.constants as c
from corrector_src.turbulence_integration.turbulent_energy import initialize_e_turb

import matplotlib.pyplot as plt
from matplotlib import animation
import jax.numpy as jnp


def animate_simulation(final_states, z_level, gif_name="sgs_turb"):
    """
    Animate 3D simulation data.

    final_states shape: (time, channels, x, y, z)
    Channels:
      0 - density
      1-3 - velocity components
      4 - pressure
      5-7 - magnetic field components
      8 - turbulent energy (new)
    """
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    # Density
    cax0 = axs[0].imshow(
        final_states[0, 0, :, :, z_level].T,
        origin="lower",
        norm=plt.Normalize(vmin=0, vmax=1),
    )
    fig.colorbar(cax0, ax=axs[0])
    axs[0].set_title("Density")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")

    # Velocity magnitude
    cax1 = axs[1].imshow(
        jnp.sqrt(
            final_states[0, 1, :, :, z_level] ** 2
            + final_states[0, 2, :, :, z_level] ** 2
            + final_states[0, 3, :, :, z_level] ** 2
        ).T,
        origin="lower",
        norm=plt.Normalize(vmin=0, vmax=1),
    )
    fig.colorbar(cax1, ax=axs[1])
    axs[1].set_title("Velocity Magnitude")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")

    # Pressure
    cax2 = axs[2].imshow(
        final_states[0, 4, :, :, z_level].T,
        origin="lower",
        norm=plt.Normalize(vmin=0, vmax=1),
    )
    fig.colorbar(cax2, ax=axs[2])
    axs[2].set_title("Pressure")
    axs[2].set_xlabel("x")
    axs[2].set_ylabel("y")

    # Turbulent energy (new)
    cax3 = axs[3].imshow(
        final_states[0, 8, :, :, z_level].T,
        origin="lower",
        norm=plt.Normalize(vmin=0, vmax=1),
    )
    fig.colorbar(cax3, ax=axs[3])
    axs[3].set_title("Turbulent Energy")
    axs[3].set_xlabel("x")
    axs[3].set_ylabel("y")

    # Update function for animation
    def animate_frame(i):
        cax0.set_array(final_states[i, 0, :, :, z_level].T)
        cax1.set_array(
            jnp.sqrt(
                final_states[i, 1, :, :, z_level] ** 2
                + final_states[i, 2, :, :, z_level] ** 2
                + final_states[i, 3, :, :, z_level] ** 2
            ).T
        )
        cax2.set_array(final_states[i, 4, :, :, z_level].T)
        cax3.set_array(final_states[i, 8, :, :, z_level].T)
        return cax0, cax1, cax2, cax3

    ani = animation.FuncAnimation(
        fig, animate_frame, frames=final_states.shape[0], interval=50
    )
    ani.save("turbulence/figures/" + gif_name + ".gif")
    plt.show()


def randomized_turbulent_initial_state(
    num_cells: int,
    mhd: bool,
):
    "Creates a turbulent initial state with mhd"
    adiabatic_index = 5 / 3
    box_size = 1.0
    dt_max = 0.1
    cooling_config = CoolingConfig(cooling=True)
    snapshot_settings = SnapshotSettings(return_states=return_states)
    # setup simulation config
    config = SimulationConfig(
        runtime_debugging=runtime_debugging,
        first_order_fallback=False,
        progress_bar=progress_bar,
        dimensionality=3,
        num_ghost_cells=2,
        box_size=box_size,
        num_cells=num_cells,
        mhd=mhd,
        fixed_timestep=False,
        differentiation_mode=FORWARDS,
        riemann_solver=HLL,
        limiter=0,
        return_snapshots=True,
        num_snapshots=num_snapshots,
        boundary_settings=BoundarySettings(),
        cooling_config=cooling_config,
        memory_analysis=False,
        snapshot_settings=snapshot_settings,
        sgs_turb_energy=sgs_turb_energy,
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
    wanted_rms = 50 * u.km / u.s
    dt_max = 0.1

    # set the simulation parameters
    params = SimulationParams(
        C_cfl=C_CFL,
        dt_max=dt_max,
        gamma=adiabatic_index,
        t_end=t_end,
    )

    # homogeneous initial state
    rho_0 = 2 * c.m_p / u.cm**3
    p_0 = 3e4 * u.K / u.cm**3 * c.k_B

    density = (
        jnp.ones((num_cells, num_cells, num_cells))
        * rho_0.to(code_units.code_density).value
    )

    # turbulence parameters
    turbulence_slope = -2
    kmin = 2
    kmax = 64

    p = (
        jnp.ones((num_cells, num_cells, num_cells))
        * p_0.to(code_units.code_pressure).value
    )
    rng_seed = int(time.time() * 1e6) % (2**32 - 1)
    key = jax.random.key(rng_seed)

    keys = jax.random.split(key, 3)
    if multi_gpu:
        sharding_mesh_no_var = jax.make_mesh(split_turb, (XAXIS, YAXIS, ZAXIS))
        named_sharding_no_var = jax.NamedSharding(
            sharding_mesh_no_var, P(XAXIS, YAXIS, ZAXIS)
        )

    u_x = create_turb_field(
        config.num_cells,
        1,
        turbulence_slope,
        kmin,
        kmax,
        key=keys[0],
        sharding=named_sharding_no_var if multi_gpu else None,
    )

    u_y = create_turb_field(
        config.num_cells,
        1,
        turbulence_slope,
        kmin,
        kmax,
        key=keys[1],
        sharding=named_sharding_no_var if multi_gpu else None,
    )

    u_z = create_turb_field(
        config.num_cells,
        1,
        turbulence_slope,
        kmin,
        kmax,
        key=keys[2],
        sharding=named_sharding_no_var if multi_gpu else None,
    )

    # scale the turbulence to the desired rms velocity
    rms_vel = jnp.sqrt(jnp.mean(u_x**2 + u_y**2 + u_z**2))

    u_x = u_x / rms_vel * wanted_rms.to(code_units.code_velocity).value
    u_y = u_y / rms_vel * wanted_rms.to(code_units.code_velocity).value
    u_z = u_z / rms_vel * wanted_rms.to(code_units.code_velocity).value
    # construct primitive state

    if mhd:
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

        B_0 = 1 / np.sqrt(2)
        B_x = jnp.zeros_like(X)
        B_y = jnp.zeros_like(X)
        B_z = B_0 * jnp.ones_like(X)

        initial_state = construct_primitive_state(
            config=config,
            registered_variables=registered_variables,
            density=density,
            velocity_x=u_x,
            velocity_y=u_y,
            velocity_z=u_z,
            gas_pressure=p,
            magnetic_field_x=B_x,
            magnetic_field_y=B_y,
            magnetic_field_z=B_z,
        )
    else:
        initial_state = construct_primitive_state(
            config=config,
            registered_variables=registered_variables,
            density=density,
            velocity_x=u_x,
            velocity_y=u_y,
            velocity_z=u_z,
            gas_pressure=p,
        )
    config = finalize_config(config, initial_state.shape)
    initial_state = initialize_e_turb(
        state=initial_state, registered_variables=registered_variables, fraction=0.01
    )
    return (
        initial_state,
        config,
        params,
        helper_data,
        registered_variables,
        rng_seed,
    )


(
    initial_state,
    config,
    params,
    helper_data,
    registered_variables,
    rng_seed,
) = randomized_turbulent_initial_state(num_cells=resolution, mhd=False)


# def save_snapshot(time, state, registered_variables):

#     jnp.save(f"turbulence/states_{time}.npy", state)
if multi_gpu:
    sharding_mesh = jax.make_mesh(split_training, (VARAXIS, XAXIS, YAXIS, ZAXIS))
    named_sharding = jax.NamedSharding(sharding_mesh, P(VARAXIS, XAXIS, YAXIS, ZAXIS))
    primitive_state = jax.device_put(initial_state, named_sharding)

    jax.debug.visualize_array_sharding(primitive_state[0, :, :, 0])
    #
    helper_data = get_helper_data(config, named_sharding)

fig, ax = plt.subplots(1, 1)
im = ax.imshow(initial_state[8, :, :, resolution // 2])

ax.set_title("initial turb_energy")
fig.colorbar(im, ax=ax)
fig.savefig("turbulence/figures/init_turb_energy.png")
plt.show()


sim_snapshots = time_integration(
    initial_state, config, params, helper_data, registered_variables
)

animate_simulation(sim_snapshots.states, resolution // 2)

fig, ax = plt.subplots(1, 1)

ax.plot(
    sim_snapshots.time_points,
    jnp.sum(sim_snapshots.states[:, 8, :, :, :], axis=(2, 3, 4)),
)

ax.set_title("Total Turbulent Energy vs Time")
ax.set_xlabel("Time")
ax.set_ylabel("Total Turbulent Energy")

fig.savefig("turbulence/figures/turbulent_energy_vs_time.png")
plt.show()
