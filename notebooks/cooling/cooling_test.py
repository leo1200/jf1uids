# TODO: fix units

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# numerics
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

# timing
from timeit import default_timer as timer

# plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# fluids
from jf1uids import WindParams
from jf1uids import SimulationConfig
from jf1uids import SimulationParams
from jf1uids.option_classes import WindConfig
from jf1uids._physics_modules._cooling.cooling_options import CoolingConfig, CoolingParams

from jf1uids import get_helper_data
from jf1uids.fluid_equations.fluid import construct_primitive_state
from jf1uids import get_registered_variables
from jf1uids.option_classes.simulation_config import finalize_config

from jf1uids import time_integration

# jf1uids constants
from jf1uids.option_classes.simulation_config import OPEN_BOUNDARY, REFLECTIVE_BOUNDARY, SPHERICAL

# units
from jf1uids import CodeUnits
from astropy import units as u
import astropy.constants as c
from astropy.constants import m_p

# wind-specific
from jf1uids._physics_modules._stellar_wind.weaver import Weaver


print("👷 Setting up simulation...")

# simulation settings
gamma = 5/3

# spatial domain
geometry = SPHERICAL
box_size = 1.0
num_cells = 1001

left_boundary = REFLECTIVE_BOUNDARY
right_boundary = OPEN_BOUNDARY

# activate stellar wind
stellar_wind = True

# setup simulation config
config = SimulationConfig(
    runtime_debugging = False,
    progress_bar = False,
    geometry = geometry,
    box_size = box_size, 
    num_cells = num_cells,
    wind_config = WindConfig(
        stellar_wind = stellar_wind,
        num_injection_cells = 30,
        trace_wind_density = False,
    ),
    cooling_config = CoolingConfig(
        cooling = True
    )
)

helper_data = get_helper_data(config)
registered_variables = get_registered_variables(config)

# code units
code_length = 3 * u.parsec
code_mass = 1e-3 * u.M_sun
code_velocity = 1 * u.km / u.s
code_units = CodeUnits(code_length, code_mass, code_velocity)

# cooling params
hydrogen_mass_fraction = 0.76
metal_mass_fraction = 0.02
reference_temperature = (1e8 * u.K * c.k_B / c.m_p).to(code_units.code_energy / code_units.code_mass).value
# without a floor temperature, the simulations crash
floor_temperature = (2e4 * u.K * c.k_B / c.m_p).to(code_units.code_energy / code_units.code_mass).value
factor = (1e-23 * u.erg * u.cm ** 3 / u.s / c.m_p ** 2).to(code_units.code_energy * code_units.code_length ** 3 / (code_units.code_time * code_units.code_mass ** 2)).value
exponent = -0.7

print("reference temperature", reference_temperature)
print("factor", factor)

# time stepping
C_CFL = 0.8
t_final = 2.5 * 1e4 * u.yr
t_end = t_final.to(code_units.code_time).value
dt_max = 0.1 * t_end

# wind parameters
M_star = 40 * u.M_sun
wind_final_velocity = 2000 * u.km / u.s
wind_mass_loss_rate = 2.965e-3 / (1e6 * u.yr) * M_star
wind_params = WindParams(
    wind_mass_loss_rate = wind_mass_loss_rate.to(code_units.code_mass / code_units.code_time).value,
    wind_final_velocity = wind_final_velocity.to(code_units.code_velocity).value
)

# simulation params
params = SimulationParams(
    C_cfl = C_CFL,
    dt_max = dt_max,
    gamma = gamma,
    t_end = t_end,
    wind_params = wind_params,
    cooling_params = CoolingParams(
        hydrogen_mass_fraction = hydrogen_mass_fraction,
        metal_mass_fraction = metal_mass_fraction,
        reference_temperature = reference_temperature,
        floor_temperature = floor_temperature,
        factor = factor,
        exponent = exponent
    )
)

# homogeneous initial state
rho_0 = 2 * c.m_p / u.cm**3
p_0 = 3e4 * u.K / u.cm**3 * c.k_B

rho_init = jnp.ones(num_cells) * rho_0.to(code_units.code_density).value
u_init = jnp.zeros(num_cells)
p_init = jnp.ones(num_cells) * p_0.to(code_units.code_pressure).value

# get initial state
initial_state = construct_primitive_state(
    config = config,
    registered_variables = registered_variables,
    density = rho_init,
    velocity_x = u_init,
    gas_pressure = p_init
)

config = finalize_config(config, initial_state.shape)

# main simulation loop
final_state = time_integration(initial_state, config, params, helper_data, registered_variables)

# compare with weaver solution
def plot_weaver_comparison(axs, final_state, params, helper_data, code_units, rho_0, p_0):
    print("👷 generating plots")

    rho = final_state[registered_variables.density_index]
    vel = final_state[registered_variables.velocity_index]
    p = final_state[registered_variables.pressure_index]

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

    axs[0].set_yscale("log")
    axs[0].plot(r.to(u.parsec), (rho / m_p).to(u.cm**-3), label="jf1uids")
    axs[0].plot(r_density_weaver, density_weaver, "--", label="Weaver solution")
    axs[0].set_title("density")
    axs[0].set_ylabel(r"$\rho$ in m$_p$ cm$^{-3}$")
    axs[0].set_xlim(0, 3)
    axs[0].legend(loc="upper left")
    axs[0].set_xlabel("r in pc")

    axs[1].set_yscale("log")
    axs[1].plot(r.to(u.parsec), (p / c.k_B).to(u.K / u.cm**3), label="jf1uids")
    axs[1].plot(r_pressure_weaver, pressure_weaver, "--", label="Weaver solution")
    axs[1].set_title("pressure")
    axs[1].set_ylabel(r"$p$/k$_b$ in K cm$^{-3}$")
    axs[1].set_xlim(0, 3)
    axs[1].legend(loc="upper left")
    axs[1].set_xlabel("r in pc")


    axs[2].set_yscale("log")
    axs[2].plot(r.to(u.parsec), vel.to(u.km / u.s), label="jf1uids")
    axs[2].plot(r_velocity_weaver, velocity_weaver, "--", label="Weaver solution")
    axs[2].set_title("velocity")
    axs[2].set_ylim(1, 1e4)
    axs[2].set_xlim(0, 3)
    axs[2].set_ylabel("v in km/s")
    axs[2].legend(loc="upper right")
    axs[2].set_xlabel("r in pc")


fig, axs = plt.subplots(1, 3, figsize=(15, 5))
plot_weaver_comparison(axs, final_state, params, helper_data, code_units, rho_0, p_0)
plt.tight_layout()

plt.savefig("cooling_test.svg")