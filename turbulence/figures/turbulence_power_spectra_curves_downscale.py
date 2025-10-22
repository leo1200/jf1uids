# ==== GPU selection ====
from autocvd import autocvd

num_gpus = 4
multi_gpu = num_gpus > 1
split_turb = (2, 2, 1)
split_training = (1, 2, 2, 1)

# assert sum(x for x in split_turb if x > 1) == num_gpus or num_gpus == 1, (
#     f"Sum of splits {sum(x for x in split_turb if x > 1)} != num_gpus ({num_gpus})"
# )
# assert sum(x for x in split_training if x > 1) == num_gpus or num_gpus == 1, (
#     f"Sum of splits {sum(x for x in split_training if x > 1)} != num_gpus ({num_gpus})"
# )

autocvd(num_gpus=num_gpus)
# =======================

# numerics
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

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
from jf1uids._physics_modules._cooling.cooling_options import CoolingConfig
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
    VARAXIS,
    XAXIS,
    YAXIS,
    ZAXIS,
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
from corrector_src.utils.downaverage import downaverage_state

resolutions = [240, 120, 60]
animation_name = "turbulence/figures/combined_spectra_with_ref"

downscale = True
cooling = True
stellar_wind = False
turbulence = True
if cooling:
    animation_name += "_cooling"


if downscale:
    downscale_ratio = [r / resolutions[-1] for r in resolutions]
    assert all(r.is_integer() and r > 0 for r in downscale_ratio), (
        "Not all downscale ratios are natural numbers"
    )
    print(downscale_ratio)
    z_level = [resolutions[-1] // 2 for r in resolutions]
    animation_name += "_downscaled"

else:
    z_level = [r // 2 for r in resolutions]
t_final = 6e4
snapshots = 30


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
    box_size = 1.0
    num_cells = num_cells

    wanted_rms = 50 * u.km / u.s
    cooling_config = CoolingConfig(cooling=cooling)
    # setup simulation config
    config = SimulationConfig(
        runtime_debugging=False,
        first_order_fallback=False,
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
        boundary_settings=BoundarySettings(),
        riemann_solver=HLL,
        return_snapshots=True,
        num_snapshots=num_snapshots,
        cooling_config=cooling_config,
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
    kmax = 64  # int(0.6 * (num_cells // 2))

    if turbulence:
        if multi_gpu:
            sharding_mesh_no_var = jax.make_mesh(split_turb, (XAXIS, YAXIS, ZAXIS))
            named_sharding_no_var = jax.NamedSharding(
                sharding_mesh_no_var, P(XAXIS, YAXIS, ZAXIS)
            )

        key = jax.random.PRNGKey(42)
        key, sk1, sk2, sk3 = jax.random.split(key, 4)

        ux = create_turb_field(
            config.num_cells,
            1,
            turbulence_slope,
            kmin,
            kmax,
            key=sk1,
            sharding=named_sharding_no_var if multi_gpu else None,
        )
        uy = create_turb_field(
            config.num_cells,
            1,
            turbulence_slope,
            kmin,
            kmax,
            key=sk2,
            sharding=named_sharding_no_var if multi_gpu else None,
        )
        uz = create_turb_field(
            config.num_cells,
            1,
            turbulence_slope,
            kmin,
            kmax,
            key=sk3,
            sharding=named_sharding_no_var if multi_gpu else None,
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
    if multi_gpu:
        sharding_mesh = jax.make_mesh(split_training, (VARAXIS, XAXIS, YAXIS, ZAXIS))
        named_sharding = jax.NamedSharding(
            sharding_mesh, P(VARAXIS, XAXIS, YAXIS, ZAXIS)
        )
        primitive_state = jax.device_put(initial_state, named_sharding)

        jax.debug.visualize_array_sharding(primitive_state[0, :, :, 0])
        #
        helper_data = get_helper_data(config, named_sharding)

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


def get_power_spectra(num_cells, snapshots, t_final, z_level, downsample_factor=None):
    # turbulence only simulation
    (initial_state_turb, snapshot_data, config, registered_variables, params) = (
        run_turbulent_simulation(
            num_cells=num_cells,
            num_snapshots=snapshots,
            stellar_wind=stellar_wind,
            turbulence=turbulence,
            t_final=t_final * u.yr,
        )
    )

    energies = []
    spectrums = []
    ks = []
    for state in snapshot_data.states:
        energy = np.array(
            get_energy(state, config, registered_variables, params), dtype=np.float32
        )
        if downscale:
            energy = downaverage_state(
                energy[None, ...], downsample_factor=downsample_factor
            )[0]

        energies.append(energy)
        pk_energy = PKL.Pk(
            delta=energy, BoxSize=1, axis=0, MAS="None", threads=6, verbose=False
        )
        spectrums.append(pk_energy.Pk1D)
        ks.append(pk_energy.k1D)

    return (
        initial_state_turb,
        spectrums,
        ks,
        snapshot_data.time_points,
        jnp.sqrt(
            snapshot_data.states[:, 1, :, :, z_level] ** 2
            + snapshot_data.states[:, 2, :, :, z_level] ** 2
            + snapshot_data.states[:, 3, :, :, z_level] ** 2
        ),
    )


# fig, axs = plt.subplots(2, len(resolutions), figsize=(15, 8))
# spectrums = []

# # --- Precompute all data ---
# for i, resolution in enumerate(resolutions):
#     spectrum, k, time_points, vel_states = get_power_spectra(
#         num_cells=resolution, snapshots=snapshots, t_final=t_final, z_level=z_level[i]
#     )

#     # --- Top row: velocity field ---
#     ax_vel = axs[0, i]
#     im = ax_vel.imshow(
#         vel_states[0],
#         cmap="plasma",
#         origin="lower",
#         vmin=0,
#         vmax=1,
#     )
#     ax_vel.set_title(f"Velocity field (z={z_level[i]})")
#     ax_vel.axis("off")

#     # --- Bottom row: power spectrum ---
#     ax_spec = axs[1, i]
#     ax_spec.plot(k[0], k[0] ** -2, label=r"$k^{-2}$ reference", linestyle="--")

#     (line,) = ax_spec.plot([], [], lw=2, label=r"$P_{1D}(k_\parallel)$")

#     ax_spec.set_xscale("log")
#     ax_spec.set_yscale("log")
#     ax_spec.set_xlabel("k")
#     ax_spec.set_ylabel("P(k)")
#     ax_spec.set_ylim(1e-6, 1e-1)
#     ax_spec.set_xlim(min(k[0]), max(k[0]))
#     ax_spec.set_title(f"Power spectrum {resolution}")
#     ax_spec.legend()

#     spectrums.append(
#         {
#             "line": line,
#             "im": im,
#             "data": spectrum,
#             "k": k,
#             "vel": vel_states,
#             "time_points": time_points,
#         }
#     )

# title = fig.suptitle(f"t = {spectrums[0]['time_points'][0]:.2f}")


# # --- Animation setup ---
# def init():
#     for spec in spectrums:
#         spec["line"].set_data([], [])
#         spec["im"].set_array(spec["vel"][0])
#     return [s["line"] for s in spectrums] + [s["im"] for s in spectrums] + [title]


# def animate(j):
#     for spec in spectrums:
#         spec["line"].set_data(spec["k"][j], spec["data"][j])
#         spec["im"].set_array(spec["vel"][j])
#     title.set_text(f"t = {spectrums[0]['time_points'][j]:.2f}")
#     return [s["line"] for s in spectrums] + [s["im"] for s in spectrums] + [title]


# ani = animation.FuncAnimation(
#     fig,
#     animate,
#     init_func=init,
#     frames=len(spectrums[0]["time_points"]),
#     interval=100,
#     blit=False,
# )

# ani.save("turb_power_spectra.mp4", writer=animation.FFMpegWriter(fps=10, bitrate=1800))
# plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.show()
spectrums = []
initial_state_hr = None

for i, resolution in enumerate(resolutions):
    print(f"\n=== Running resolution {resolution}³ ===")

    if i == 0:
        # --- High-resolution reference run ---
        initial_state_hr, spectrum, k, time_points, vel_states = get_power_spectra(
            num_cells=resolution,
            snapshots=snapshots,
            t_final=t_final,
            z_level=z_level[i],
            downsample_factor=downscale_ratio[i],
        )
    else:
        # --- Downscale high-resolution initial state ---
        downsample_factor = int(resolutions[0] / resolutions[i])
        print(f"→ Downscaling high-res initial state by factor {downsample_factor}")

        new_shape = (
            initial_state_hr.shape[0],
            int(resolutions[0] / downsample_factor),
            int(resolutions[0] / downsample_factor),
            int(resolutions[0] / downsample_factor),
        )
        initial_state_lr = jax.image.resize(
            initial_state_hr, new_shape, method="linear"
        )
        print(initial_state_lr.shape, resolution)
        # --- Run low-res simulation using downscaled initial state ---
        (
            _,
            snapshot_data,
            config,
            registered_variables,
            params,
        ) = run_turbulent_simulation(
            num_cells=resolution,
            num_snapshots=snapshots,
            stellar_wind=stellar_wind,
            turbulence=turbulence,
            t_final=t_final * u.yr,
            initial_state_given=initial_state_lr,
        )

        # Compute power spectra from the results
        energies = []
        spectrums_i = []
        ks = []
        for state in snapshot_data.states:
            energy = np.array(
                get_energy(state, config, registered_variables, params),
                dtype=np.float32,
            )
            if downscale:
                energy = downaverage_state(
                    energy[None, ...], downsample_factor=downscale_ratio[i]
                )[0]

            pk_energy = PKL.Pk(
                delta=energy, BoxSize=1, axis=0, MAS="None", threads=6, verbose=False
            )
            spectrums_i.append(pk_energy.Pk1D)
            ks.append(pk_energy.k1D)

        spectrum = spectrums_i
        k = ks
        time_points = snapshot_data.time_points

    spectrums.append(
        {
            "data": spectrum,
            "k": k,
            "label": f"{resolution}³ cells",
            "time_points": time_points,
        }
    )


# --- Single plot for all spectra ---
fig, ax = plt.subplots(figsize=(8, 6))

lines = []
for spec in spectrums:
    (line,) = ax.plot([], [], lw=2, label=spec["label"])
    lines.append(line)

# --- k^-2 reference line ---
k_ref = spectrums[0]["k"][0]  # use k from first resolution
P_ref = k_ref**-2
(ref_line,) = ax.plot(k_ref, P_ref, "k--", label=r"$k^{-2}$ reference")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("k")
ax.set_ylabel("P(k)")
ax.set_ylim(1e-6, 1e-1)
ax.set_title("Power spectra for all resolutions")
ax.legend()
title = ax.text(0.5, 1.05, "", transform=ax.transAxes, ha="center")


# --- Animation setup ---
def init():
    for line in lines:
        line.set_data([], [])
    title.set_text(f"t = {spectrums[0]['time_points'][0]:.2f}")
    return lines + [title]


def animate(j):
    for line, spec in zip(lines, spectrums):
        line.set_data(spec["k"][j], spec["data"][j])
    title.set_text(f"t = {spectrums[0]['time_points'][j]:.2f}")
    return lines + [title]


ani = animation.FuncAnimation(
    fig,
    animate,
    init_func=init,
    frames=len(spectrums[0]["time_points"]),
    interval=100,
    blit=False,
)

ani.save(animation_name + ".mp4", writer=animation.FFMpegWriter(fps=10, bitrate=1800))
plt.show()
