# ==== GPU selection ====
from autocvd import autocvd

autocvd(num_gpus=1)
# =======================

# numerics
import jax
import jax.numpy as jnp

# timing
from timeit import default_timer as timer

# plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
from matplotlib import animation

# fluids
from jf1uids import WindParams
from jf1uids import SimulationConfig
from jf1uids import get_helper_data
from jf1uids import SimulationParams
from jf1uids import time_integration
from jf1uids.fluid_equations.fluid import (
    construct_primitive_state,
    get_absolute_velocity,
    total_energy_from_primitives,
)

from jf1uids import get_registered_variables
from jf1uids.option_classes import WindConfig

from jf1uids.option_classes.simulation_config import (
    BACKWARDS,
    HLL,
    HYBRID_HLLC,
    OSHER,
    PERIODIC_BOUNDARY,
    BoundarySettings,
    BoundarySettings1D,
)

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

from jf1uids.option_classes.simulation_config import finalize_config


# power spectra
import Pk_library as PKL


import numpy as np

resolutions = [256, 180, 120, 60]
t_final = 6e4
snapshots = 40
z_level = [r // 2 for r in resolutions]


def run_turbulent_simulation(
    num_cells,
    num_snapshots,
    stellar_wind=True,
    turbulence=True,
    t_final=2e5 * u.yr,
    initial_state_given=None,
):
    # simulation settings
    gamma = 5 / 3

    # spatial domain
    box_size = 3.0
    num_cells = num_cells

    wanted_rms = 50 * u.km / u.s

    # setup simulation config
    config = SimulationConfig(
        runtime_debugging=False,
        first_order_fallback=True,
        progress_bar=True,
        dimensionality=3,
        num_ghost_cells=2,
        box_size=box_size,
        num_cells=num_cells,
        wind_config=WindConfig(
            stellar_wind=stellar_wind,
            num_injection_cells=12,
            trace_wind_density=False,
        ),
        differentiation_mode=FORWARDS,
        boundary_settings=BoundarySettings(
            BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
            ),
            BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
            ),
            BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
            ),
        ),
        riemann_solver=HYBRID_HLLC,
        return_snapshots=True,
        num_snapshots=num_snapshots,
    )

    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)

    code_length = 3 * u.parsec
    code_mass = 1 * u.M_sun
    code_velocity = 100 * u.km / u.s
    code_units = CodeUnits(code_length, code_mass, code_velocity)

    # time domain
    C_CFL = 0.4

    t_end = t_final.to(code_units.code_time).value

    # wind parameters
    M_star = 40 * u.M_sun
    wind_final_velocity = 2000 * u.km / u.s
    wind_mass_loss_rate = 2.965e-3 / (1e6 * u.yr) * M_star

    wind_params = WindParams(
        wind_mass_loss_rate=wind_mass_loss_rate.to(
            code_units.code_mass / code_units.code_time
        ).value,
        wind_final_velocity=wind_final_velocity.to(code_units.code_velocity).value,
    )

    params = SimulationParams(
        C_cfl=C_CFL, gamma=gamma, t_end=t_end, wind_params=wind_params
    )

    # homogeneous initial state
    rho_0 = 2 * c.m_p / u.cm**3
    p_0 = 3e4 * u.K / u.cm**3 * c.k_B

    rho = (
        jnp.ones((config.num_cells, config.num_cells, config.num_cells))
        * rho_0.to(code_units.code_density).value
    )

    u_x = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))
    u_y = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))
    u_z = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))

    turbulence_slope = -2.0
    kmin = 2
    kmax = int(0.6 * (num_cells // 2))

    if turbulence:
        key = jax.random.PRNGKey(42)
        key, sk1, sk2, sk3 = jax.random.split(key, 4)

        ux = create_turb_field(
            config.num_cells, 1, turbulence_slope, kmin, kmax, key=sk1
        )
        uy = create_turb_field(
            config.num_cells, 1, turbulence_slope, kmin, kmax, key=sk2
        )
        uz = create_turb_field(
            config.num_cells, 1, turbulence_slope, kmin, kmax, key=sk3
        )

        rms_vel = jnp.sqrt(jnp.mean(ux**2 + uy**2 + uz**2))

        u_x = ux / rms_vel * wanted_rms.to(code_units.code_velocity).value
        u_y = uy / rms_vel * wanted_rms.to(code_units.code_velocity).value
        u_z = uz / rms_vel * wanted_rms.to(code_units.code_velocity).value

        # print the maximum velocity
        print(
            f"Maximum velocity: {(jnp.max(jnp.sqrt(u_x**2 + u_y**2 + uz**2)) * code_units.code_velocity).to(u.km / u.s).value:.2f} km/s"
        )

    p = (
        jnp.ones((config.num_cells, config.num_cells, config.num_cells))
        * p_0.to(code_units.code_pressure).value
    )

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

    if initial_state_given is not None:
        # use the given initial state
        initial_state = initial_state_given

    config = finalize_config(config, initial_state.shape)

    return (
        initial_state,
        time_integration(
            initial_state, config, params, helper_data, registered_variables
        ),
        config,
        registered_variables,
        params,
    )


def get_energy(primitive_state, config, registered_variables, params):
    """Calculate the total energy from the primitive state."""
    rho = primitive_state[registered_variables.density_index]
    u = get_absolute_velocity(primitive_state, config, registered_variables)
    p = primitive_state[registered_variables.pressure_index]
    return total_energy_from_primitives(rho, u, p, params.gamma)


def get_power_spectra(num_cells, snapshots, t_final, z_level):
    # turbulence only simulation
    (initial_state_turb, snapshot_data, config, registered_variables, params) = (
        run_turbulent_simulation(
            num_cells=num_cells,
            num_snapshots=snapshots,
            stellar_wind=False,
            turbulence=True,
            t_final=t_final * u.yr,
        )
    )

    num_cells = initial_state_turb.shape[-1]
    energies = []
    spectrums = []
    ks = []

    for state in snapshot_data.states:
        energy = np.array(
            get_energy(state, config, registered_variables, params), dtype=np.float32
        )
        energies.append(energy)
        pk_energy = PKL.Pk(
            delta=energy, BoxSize=1, axis=0, MAS="None", threads=6, verbose=False
        )
        spectrums.append(pk_energy.Pk1D)
        ks.append(pk_energy.k1D)

    return (
        spectrums,
        ks,
        snapshot_data.time_points,
        jnp.sqrt(
            snapshot_data.states[:, 1, :, :, z_level] ** 2
            + snapshot_data.states[:, 2, :, :, z_level] ** 2
            + snapshot_data.states[:, 3, :, :, z_level] ** 2
        ),
    )


fig, axs = plt.subplots(2, len(resolutions), figsize=(15, 8))
spectrums = []

# --- Precompute all data ---
for i, resolution in enumerate(resolutions):
    spectrum, k, time_points, vel_states = get_power_spectra(
        num_cells=resolution, snapshots=snapshots, t_final=t_final, z_level=z_level[i]
    )

    # --- Top row: velocity field ---
    ax_vel = axs[0, i]
    im = ax_vel.imshow(
        vel_states[0],
        cmap="plasma",
        origin="lower",
        vmin=0,
        vmax=1,
    )
    ax_vel.set_title(f"Velocity field (z={z_level[i]})")
    ax_vel.axis("off")

    # --- Bottom row: power spectrum ---
    ax_spec = axs[1, i]
    ax_spec.plot(k[0], k[0] ** -2, label=r"$k^{-2}$ reference", linestyle="--")

    (line,) = ax_spec.plot([], [], lw=2, label=r"$P_{1D}(k_\parallel)$")

    ax_spec.set_xscale("log")
    ax_spec.set_yscale("log")
    ax_spec.set_xlabel("k")
    ax_spec.set_ylabel("P(k)")
    ax_spec.set_ylim(1e-6, 1e-1)
    ax_spec.set_xlim(min(k[0]), max(k[0]))
    ax_spec.set_title(f"Power spectrum {resolution}")
    ax_spec.legend()

    spectrums.append(
        {
            "line": line,
            "im": im,
            "data": spectrum,
            "k": k,
            "vel": vel_states,
            "time_points": time_points,
        }
    )

title = fig.suptitle(f"t = {spectrums[0]['time_points'][0]:.2f}")


# --- Animation setup ---
def init():
    for spec in spectrums:
        spec["line"].set_data([], [])
        spec["im"].set_array(spec["vel"][0])
    return [s["line"] for s in spectrums] + [s["im"] for s in spectrums] + [title]


def animate(j):
    for spec in spectrums:
        spec["line"].set_data(spec["k"][j], spec["data"][j])
        spec["im"].set_array(spec["vel"][j])
    title.set_text(f"t = {spectrums[0]['time_points'][j]:.2f}")
    return [s["line"] for s in spectrums] + [s["im"] for s in spectrums] + [title]


ani = animation.FuncAnimation(
    fig,
    animate,
    init_func=init,
    frames=len(spectrums[0]["time_points"]),
    interval=100,
    blit=False,
)

ani.save("turb_power_spectra.mp4", writer=animation.FFMpegWriter(fps=10, bitrate=1800))
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
