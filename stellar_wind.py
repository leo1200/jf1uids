print("🚀 stellar wind simulation")

# ================== Imports ===================

# numerics
import jax
import jax.numpy as jnp
import optax

# timing
from timeit import default_timer as timer

# plotting
import matplotlib.pyplot as plt

# fluids
from jf1uids.boundaries import OPEN_BOUNDARY, REFLECTIVE_BOUNDARY
from jf1uids.geometry import SPHERICAL
from jf1uids.physics_modules.stellar_wind import WindParams
from jf1uids.postprocessing import shock_sensor, strongest_shock_radius
from jf1uids.simulation_config import SimulationConfig
from jf1uids.simulation_helper_data import get_helper_data
from jf1uids.simulation_params import SimulationParams
from jf1uids.time_integration import time_integration
from jf1uids.fluid import primitive_state

# units
from jf1uids.unit_helpers import CodeUnits
from astropy import units as u
import astropy.constants as c
from astropy.constants import m_p

# wind-specific
from weaver import Weaver
from jax import grad

# # for now using CPU as of outdated NVIDIA Driver
jax.config.update('jax_platform_name', 'cpu')

# # 64-bit precision
jax.config.update("jax_enable_x64", True)

# =================================================

# ============== SIMULATION SETTINGS ==============

print("👷 Setting up simulation...")

# simulation settings
gamma = 5/3

# spatial domain
alpha = SPHERICAL
box_size = 1.0
num_cells = 2000
dx = box_size / num_cells

fixed_timestep = False
num_timesteps = 8000

# introduce constants to 
# make this more readable
left_boundary = REFLECTIVE_BOUNDARY
right_boundary = OPEN_BOUNDARY

# activate stellar wind
stellar_wind = True

# setup simulation config
config = SimulationConfig(
    alpha_geom = alpha,
    box_size = box_size, 
    num_cells = num_cells,
    left_boundary = left_boundary, 
    right_boundary = right_boundary, 
    stellar_wind = stellar_wind,
    fixed_timestep = fixed_timestep,
    num_timesteps = num_timesteps
)

helper_data = get_helper_data(config)

# =====================================================

# =============== SIMULATION PARAMETERS ===============

# code units
code_length = 3 * u.parsec
code_mass = 1e-4 * u.M_sun
code_velocity = 1 * u.km / u.s
code_units = CodeUnits(code_length, code_mass, code_velocity)

# time domain
C_CFL = 0.8
t_final = 2.5 * 1e4 * u.yr
t_end = t_final.to(code_units.code_time).value
dt_max = 0.0001 * t_end

# wind parameters
M_star = 32 * u.M_sun
wind_final_velocity = 1500 * u.km / u.s
wind_mass_loss_rate = 2.965e-3 / (1e6 * u.yr) * M_star

wind_params = WindParams(
    wind_mass_loss_rate = wind_mass_loss_rate.to(code_units.code_mass / code_units.code_time).value,
    wind_final_velocity = wind_final_velocity.to(code_units.code_velocity).value
)

params = SimulationParams(
    C_cfl = C_CFL,
    dt_max = dt_max,
    dx = dx,
    gamma = gamma,
    t_end = t_end,
    wind_params=wind_params
)

# ======================================================

# =========== SETTING UP THE INITIAL STATE =============

# homogeneous initial state
rho_0 = 2 * c.m_p / u.cm**3
p_0 = 3e4 * u.K / u.cm**3 * c.k_B

rho_init = jnp.ones(num_cells) * rho_0.to(code_units.code_density).value
u_init = jnp.zeros(num_cells)
p_init = jnp.ones(num_cells) * p_0.to(code_units.code_pressure).value

# get initial state
initial_state = primitive_state(rho_init, u_init, p_init)

print("✅ Simulation setup complete.")

# ======================================================

# ============= RUNNING THE SIMULATION =================

print("👷 Running simulation...")

start = timer()
final_state = time_integration(initial_state, config, params, helper_data)

end = timer()

print(f"✅ Simulation finished in {end - start} seconds")


# ======================================================


# ============= SENSITIVITY ANALYSIS ===================

# print("👷 Gradient analysis ")

# # calculate the derivative of the final state with respect to params.wind_params.final_velocity
# # this is the sensitivity of the final state to the final velocity

# # vel_sens = jax.jacfwd(lambda velocity: time_integration(initial_state, config, SimulationParams(
# #     C_cfl=params.C_cfl,
# #     dt_max=params.dt_max,
# #     dx=params.dx,
# #     gamma=params.gamma,
# #     t_end=params.t_end,
# #     wind_params=WindParams(
# #         wind_mass_loss_rate=params.wind_params.wind_mass_loss_rate,
# #         wind_final_velocity=velocity
# #     )
# # ), helper_data))(params.wind_params.wind_final_velocity)

# sample_simulation = lambda velocity, mass_loss_rate: time_integration(initial_state, config, SimulationParams(
#     C_cfl=params.C_cfl,
#     dt_max=params.dt_max,
#     dx=params.dx,
#     gamma=params.gamma,
#     t_end=params.t_end,
#     wind_params=WindParams(
#         wind_mass_loss_rate=mass_loss_rate,
#         wind_final_velocity=velocity
#     )
# ), helper_data)

# # generate a reference simulation
# M_star = 40 * u.M_sun
# wind_final_velocity = 2000 * u.km / u.s
# wind_mass_loss_rate = 2.965e-3 / (1e6 * u.yr) * M_star

# reference_params = WindParams(
#     wind_mass_loss_rate = wind_mass_loss_rate.to(code_units.code_mass / code_units.code_time).value,
#     wind_final_velocity = wind_final_velocity.to(code_units.code_velocity).value
# )

# reference_simulation = sample_simulation(
#     reference_params.wind_final_velocity,
#     reference_params.wind_mass_loss_rate
# )

# def density_loss(velocity, mass_loss_rate):
#     final_state = sample_simulation(velocity, mass_loss_rate)
#     return jnp.sum(jnp.abs(final_state - reference_simulation))

# reference_shock_radius = strongest_shock_radius(reference_simulation, helper_data, 10, 5)

# def shock_radius_loss(velocity, mass_loss_rate):
#     final_state = sample_simulation(velocity, mass_loss_rate)
#     return jnp.abs(strongest_shock_radius(final_state, helper_data, 10, 5) - reference_shock_radius)

# # try to find the original wind parameters via gradient descent
# M_star_optim = 30 * u.M_sun
# wind_final_velocity_optim = (1500 * u.km / u.s).to(code_units.code_velocity).value
# wind_mass_loss_rate_optim = (2.965e-3 / (1e6 * u.yr) * M_star_optim).to(code_units.code_mass / code_units.code_time).value

# print(f"Optimal wind final velocity: {(wind_final_velocity_optim * code_units.code_velocity).to(u.km / u.s)}")
# print(f"Optimal wind mass loss rate: {(wind_mass_loss_rate_optim * code_units.code_mass / code_units.code_time).to(2.965e-3 / (1e6 * u.yr) * u.M_sun)}")

# print(wind_final_velocity_optim, wind_mass_loss_rate_optim)

# learning_rate = 10000
# # gradient descent
# for i in range(3000):
#     grad_loss = grad(shock_radius_loss, argnums=(0, 1))(wind_final_velocity_optim, wind_mass_loss_rate_optim)
#     print(grad_loss)
#     wind_final_velocity_optim -= learning_rate * grad_loss[0]
#     wind_mass_loss_rate_optim -= learning_rate * grad_loss[1]

#     if i % 100 == 0:
#         print(f"Loss: {shock_radius_loss(wind_final_velocity_optim, wind_mass_loss_rate_optim)}")

# # solver = optax.noisy_sgd(learning_rate=100)
# # state = solver.init((wind_final_velocity_optim, wind_mass_loss_rate_optim))

# # for i in range(1000):
# #     grad_loss = grad(density_loss, argnums=(0, 1))(wind_final_velocity_optim, wind_mass_loss_rate_optim)
# #     updates, state = solver.update(grad_loss, state)
# #     wind_final_velocity_optim, wind_mass_loss_rate_optim = optax.apply_updates((wind_final_velocity_optim, wind_mass_loss_rate_optim), updates)

# #     if i % 100 == 0:
# #         print(f"Loss: {density_loss(wind_final_velocity_optim, wind_mass_loss_rate_optim)}")

#     print(f"Optimal wind final velocity: {(wind_final_velocity_optim * code_units.code_velocity).to(u.km / u.s)}")
#     print(f"Optimal wind mass loss rate: {(wind_mass_loss_rate_optim * code_units.code_mass / code_units.code_time).to(2.965e-3 / (1e6 * u.yr) * u.M_sun)}")

# print("✅ Sensitivity calculation complete.")

# ======================================================

# ==================== PLOTTING =========================

def plot_weaver_comparison(final_state, params, helper_data, code_units, rho_0, p_0):
    print("👷 generating plots")

    rho, vel, p = final_state
    internal_energy = p / ((gamma - 1) * rho)

    rho = rho * code_units.code_density
    vel = vel * code_units.code_velocity
    p = p * code_units.code_pressure

    r = helper_data.geometric_centers * code_units.code_length

    # get weaver solution
    weaver = Weaver(
        params.wind_params.wind_final_velocity * code_units.code_velocity,
        params.wind_params.wind_mass_loss_rate * code_units.code_mass / code_units.code_time,
        rho_0,
        p_0
    )
    current_time = params.t_end * code_units.code_time

    # density
    r_density_weaver, density_weaver = weaver.get_density_profile(0.01 * u.parsec, 3.5 * u.parsec, current_time)
    r_density_weaver = r_density_weaver.to(u.parsec)
    density_weaver = (density_weaver / m_p).to(u.cm**-3)

    # velocity
    r_velocity_weaver, velocity_weaver = weaver.get_velocity_profile(0.01 * u.parsec, 3.5 * u.parsec, current_time)
    r_velocity_weaver = r_velocity_weaver.to(u.parsec)
    velocity_weaver = velocity_weaver.to(u.km / u.s)

    # pressure
    r_pressure_weaver, pressure_weaver = weaver.get_pressure_profile(0.01 * u.parsec, 3.5 * u.parsec, current_time)
    r_pressure_weaver = r_pressure_weaver.to(u.parsec)
    pressure_weaver = (pressure_weaver / c.k_B).to(u.cm**-3 * u.K)

    fig, axs = plt.subplots(4, 1, figsize=(5, 20))

    axs[0].set_yscale("log")
    axs[0].plot(r.to(u.parsec), (rho / m_p).to(u.cm**-3), label="radial 1d simulation")
    axs[0].plot(r_density_weaver, density_weaver, "--", label="Weaver solution")
    axs[0].set_title("Density")
    axs[0].set_xlim(0, 3)
    # xlim 0 to 0.3
    # show legend
    axs[0].legend()

    axs[1].set_yscale("log")
    axs[1].plot(r.to(u.parsec), (p / c.k_B).to(u.K / u.cm**3), label="radial 1d simulation")
    axs[1].plot(r_pressure_weaver, pressure_weaver, "--", label="Weaver solution")
    axs[1].set_title("Pressure")
    axs[1].set_xlim(0, 3)
    # show legend
    axs[1].legend()

    axs[2].set_yscale("log")
    axs[2].plot(r.to(u.parsec), vel.to(u.km / u.s), label="radial 1d simulation")
    axs[2].plot(r_velocity_weaver, velocity_weaver, "--", label="Weaver solution")
    axs[2].set_title("Velocity")
    # ylim 1 to 1e4 km/s
    axs[2].set_ylim(1, 1e4)
    axs[2].set_xlim(0, 3)
    # show legend
    axs[2].legend()
    # tight layout
    plt.tight_layout()

    # plt.savefig("figures/stellar_wind.png")

    print("✅ plotting done")

# def plot_shock_sensitivity(final_state, helper_data, code_units):
    sensitivity = shock_sensor(final_state) * u.dimensionless_unscaled
    r = helper_data.geometric_centers[1:-1] * code_units.code_length

    r_maxx = strongest_shock_radius(final_state, helper_data, 60, 60) * code_units.code_length

    axs[3].plot(r.to(u.parsec)[50:-50], sensitivity[50:-50], label="Shock sensor")

    axs[3].axvline(r_maxx.to(u.parsec).value, color="red", label="Strongest shock")

    axs[3].set_title("Shock sensor")
    axs[3].set_xlim(0, 3)
    # axs[3].set_yscale("log")

    plt.savefig("figures/shock_sensor.png")

plot_weaver_comparison(final_state, params, helper_data, code_units, rho_0, p_0)

# plot_shock_sensitivity(final_state, helper_data, code_units)

# =======================================================