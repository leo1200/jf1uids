# =============== Imports ===============

# numerics
import jax
import jax.numpy as jnp

# plotting
import matplotlib.pyplot as plt

# fluids
from jf1uids.boundaries import OPEN_BOUNDARY, REFLECTIVE_BOUNDARY
from jf1uids.geometry import SPHERICAL, center_of_volume, r_hat_alpha
from jf1uids.physics_modules.stellar_wind import WindParams
from jf1uids.simulation_config import SimulationConfig
from jf1uids.simulation_helper_data import SimulationHelperData
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

# for now using CPU as of outdated NVIDIA Driver
jax.config.update('jax_platform_name', 'cpu')

# 64-bit precision
jax.config.update("jax_enable_x64", True)

# ========================================

# simulation settings
gamma = 5/3

# code units
code_length = 3 * u.parsec
code_mass = 1e5 * u.M_sun
code_velocity = 1 * u.km / u.s
code_units = CodeUnits(code_length, code_mass, code_velocity)

# spatial domain
alpha = SPHERICAL
L = 1.0
N_grid = 4000
dx = L / N_grid
r = jnp.linspace(dx/2, L + dx / 2, N_grid)

# print the shape of r
# print(r.shape)


rv = center_of_volume(r, dx, alpha)
r_hat = r_hat_alpha(r, dx, alpha)

# introduce constants to 
# make this more readable
left_boundary = REFLECTIVE_BOUNDARY
right_boundary = OPEN_BOUNDARY

stellar_wind = True

config = SimulationConfig(alpha_geom = alpha, left_boundary = left_boundary, right_boundary = right_boundary, stellar_wind = stellar_wind)
helper_data = SimulationHelperData(r_hat_alpha = r_hat, volumetric_centers = rv, geometric_centers = r)

# time domain
C_CFL = 0.8
t_final = 1.5 * 1e4 * u.yr
t_end = t_final.to(code_units.code_time).value
dt_max = 0.0001 * t_end

# ISM density
rho_0 = 2 * c.m_p / u.cm**3
# ISM pressure
p_0 = 3e4 * u.K / u.cm**3 * c.k_B

rho_init = jnp.ones(N_grid) * rho_0.to(code_units.code_density).value
u_init = jnp.zeros(N_grid)
p_init = jnp.ones(N_grid) * p_0.to(code_units.code_pressure).value

# get initial state
initial_state = primitive_state(rho_init, u_init, p_init)

# wind parameters
M_star = 40 * u.M_sun
wind_final_velocity = 2000 * u.km / u.s
wind_mass_loss_rate = 2.965e-3 / (1e6 * u.yr) * M_star

wind_params = WindParams(wind_mass_loss_rate = wind_mass_loss_rate.to(code_units.code_mass / code_units.code_time).value, wind_final_velocity = wind_final_velocity.to(code_units.code_velocity).value)

params = SimulationParams(C_cfl = C_CFL, dt_max = dt_max, dx = dx, gamma = gamma, t_end = t_end, wind_params=wind_params)

# run the simulation
final_state = time_integration(initial_state, config, params, helper_data)
rho, vel, p = final_state
internal_energy = p / ((gamma - 1) * rho)

# print the shape of rho
# print(rho.shape)

rho = rho * code_units.code_density
vel = vel * code_units.code_velocity
p = p * code_units.code_pressure

r = r * code_units.code_length

# print the shape of r
# print(r.shape)

# get weaver solution
weaver = Weaver(wind_final_velocity, wind_mass_loss_rate, rho_0, p_0)
current_time = t_final

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

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].set_yscale("log")
axs[0].plot(r.to(u.parsec), (rho / m_p).to(u.cm**-3), label="radial 1d simulation")
axs[0].plot(r_density_weaver, density_weaver, "--", label="Weaver solution")
axs[0].set_title("Density")
# xlim 0 to 0.3
# show legend
axs[0].legend()

axs[1].set_yscale("log")
axs[1].plot(r.to(u.parsec), (p / c.k_B).to(u.K / u.cm**3), label="radial 1d simulation")
axs[1].plot(r_pressure_weaver, pressure_weaver, "--", label="Weaver solution")
axs[1].set_title("Pressure")
# xlim 0 to 0.3
# show legend
axs[1].legend()

axs[2].set_yscale("log")
axs[2].plot(r.to(u.parsec), vel.to(u.km / u.s), label="radial 1d simulation")
axs[2].plot(r_velocity_weaver, velocity_weaver, "--", label="Weaver solution")
axs[2].set_title("Velocity")
# ylim 1 to 1e4 km/s
axs[2].set_ylim(1, 1e4)
# show legend
axs[2].legend()
# tight layout
plt.tight_layout()

plt.savefig("figures/stellar_wind.png")