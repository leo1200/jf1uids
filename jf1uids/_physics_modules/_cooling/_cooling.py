# For proper cooling something like Grackle
# might be used. For now we are interested in
# the most simple cooling model.

# dE/dt + ... = Phi(T, rho)
# Phi(T, rho) = n_H * Gamma(T) - n_H ^ 2 * Lambda(T)

# for a simple cooling term (Lambda) see 5.3 in
# https://arxiv.org/pdf/2111.03399

# also see
# https://academic.oup.com/mnras/article/502/3/3179/6081066

# and
# https://iopscience.iop.org/article/10.1088/0067-0049/181/2/391

# where this source is handled with
# Brents method? (or Joung & Mac Low 2006)
# see also
# https://iopscience.iop.org/article/10.3847/1538-4357/abc011

# HERE WE FOLLOW TOWNSEND 2009

# IDEA: find smart way to directly train a neural network for the temporal
# evolution function and its inverse

# for a test case on stellar wind look at https://www.sciencedirect.com/science/article/pii/S0045793010002914

# Tref (usually the highest temperature in the cooling table, though this is not requisite).

# for cooling to work we also need a unit system

from typing import Tuple

import jax

from jf1uids._physics_modules._cooling.cooling_options import CoolingParams
from jf1uids.fluid_equations.registered_variables import RegisteredVariables
from jf1uids.option_classes.simulation_config import FIELD_TYPE, STATE_TYPE

import jax.numpy as jnp

from jf1uids.option_classes.simulation_params import SimulationParams

# NOTE the rescaled units
# \tilde{T} = T * k_B / u
# \tilde{\Lambda} = \lambda / u^2

def get_effective_molecular_weights(
    hydrogen_mass_fraction: float, # X
    metal_mass_fraction: float, # Z
) -> Tuple[float, float, float]:
    """
    Calculate the mean molecular weight (mu) 
    and the effective molecular weights for 
    electrons (mu_e), hydrogen (mu_H)
    """

    # mean molecular weight
    mu = 1.0 / (
        2 * hydrogen_mass_fraction + 3 * (1 - hydrogen_mass_fraction - metal_mass_fraction) / 4 + metal_mass_fraction / 2
    )

    # effective molecular weight for electrons
    mu_e = 2 * 1.0 / (1 + hydrogen_mass_fraction)

    # effective molecular weight for hydrogen
    mu_H = 1.0 / hydrogen_mass_fraction

    return mu, mu_e, mu_H

def get_particle_number_density(
    density: FIELD_TYPE,
    mean_molecular_weight: float
) -> FIELD_TYPE:
    return density / mean_molecular_weight

def get_pressure_from_temperature(
    density: FIELD_TYPE,
    temperature: FIELD_TYPE,
    hydrogen_mass_fraction: float,
    metal_mass_fraction: float,
) -> FIELD_TYPE:
    """
    P = n * \tilde{T}
    """

    # calculate the effective molecular weights
    mu, _, _ = get_effective_molecular_weights(
        hydrogen_mass_fraction,
        metal_mass_fraction,
    )

    # calculate the particle number density
    n = get_particle_number_density(
        density,
        mu
    )

    # calculate the pressure
    return n * temperature

def get_temperature_from_pressure(
    density: FIELD_TYPE,
    pressure: FIELD_TYPE,
    hydrogen_mass_fraction: float,
    metal_mass_fraction: float,
) -> FIELD_TYPE:
    """
    \tilde{T} = P / \tilde{n}
    """

    # calculate the effective molecular weights
    mu, _, _ = get_effective_molecular_weights(
        hydrogen_mass_fraction,
        metal_mass_fraction
    )

    # calculate the particle number density
    n = get_particle_number_density(
        density,
        mu
    )

    # calculate the temperature
    return pressure / n # so the density must never be zero

    
# \Lambda(T)
def cooling_rate_power_law(
    temperature: FIELD_TYPE,
    reference_temperature: float,
    factor: float,
    exponent: float,
):
    return factor * (temperature / reference_temperature) ** exponent

# t_cool
def cooling_time(
    density: FIELD_TYPE,
    temperature: FIELD_TYPE,
    hydrogen_mass_fraction: float,
    metal_mass_fraction: float,
    gamma: float,
    reference_temperature: float,
    factor: float,
    exponent: float,
) -> FIELD_TYPE:
    """
    t_cool = (k * mu_e * mu_H * T) / ((gamma - 1) * rho * mu * Lambda(T))
    """

    # calculate the effective molecular weights
    mu, mu_e, mu_H = get_effective_molecular_weights(
        hydrogen_mass_fraction,
        metal_mass_fraction,
    )

    # calculate the cooling rate
    cooling_rate = cooling_rate_power_law(
        temperature,
        reference_temperature,
        factor,
        exponent
    )

    # calculate the cooling time
    return (mu_e * mu_H * temperature) / (
        (gamma - 1) * density * mu * cooling_rate
    )

# Y(T)
def power_law_temporal_evolution_function(
    temperature: FIELD_TYPE, # T
    reference_temperature: float, # T_ref
    exponent: float, # alpha
) -> FIELD_TYPE:
    """
    1/(1 - alpha) * (1 - (T/T_ref)^(1-alpha)) for alpha != 1
    -log(T/T_ref) for alpha = 1
    """
    return jax.lax.cond(
        exponent != 1,
        lambda: (1 / (1 - exponent)) * (1 - (temperature / reference_temperature) ** (1 - exponent)),
        lambda: -jnp.log(temperature / reference_temperature)
    )

# Y^-1(Y)
def power_law_temporal_evolution_function_inverse(
    temporal_evolution_function: FIELD_TYPE, # Y
    reference_temperature: float, # T_ref
    exponent: float, # alpha
) -> FIELD_TYPE:
    """
    T_ref * (1 - (1 - alpha) * Y)^(1/(1-alpha)) for alpha != 1
    T_ref * exp(-Y) for alpha = 1
    """
    return jax.lax.cond(
        exponent != 1,
        lambda: reference_temperature * (1 - (1 - exponent) * temporal_evolution_function) ** (1 / (1 - exponent)),
        lambda: reference_temperature * jnp.exp(-temporal_evolution_function)
    )

def update_temperature(
    density: FIELD_TYPE,
    temperature: FIELD_TYPE,
    time_step: float,
    hydrogen_mass_fraction: float,
    metal_mass_fraction: float,
    gamma: float,
    reference_temperature: float,
    factor: float,
    exponent: float,
) -> FIELD_TYPE:
    """
    T_new = Y^-1[Y(T) + T / T_ref * \Lambda(T_ref) / \Lambda(T) * delta_t / t_cool]
    """
    # calculate the cooling time
    t_cool = cooling_time(
        density,
        temperature,
        hydrogen_mass_fraction,
        metal_mass_fraction,
        gamma,
        reference_temperature,
        factor,
        exponent
    )

    # calculate the cooling rate
    cooling_rate = cooling_rate_power_law(
        temperature,
        reference_temperature,
        factor,
        exponent
    )

    cooling_rate_reference = cooling_rate_power_law(
        reference_temperature,
        reference_temperature,
        factor,
        exponent
    )

    # calculate the temporal evolution function
    temporal_evolution_function = power_law_temporal_evolution_function(
        temperature,
        reference_temperature,
        exponent
    )

    # calculate the new temperature
    new_temperature = power_law_temporal_evolution_function_inverse(
        temporal_evolution_function + (temperature / reference_temperature) * (cooling_rate_reference / cooling_rate) * time_step / t_cool,
        reference_temperature,
        exponent
    )

    return new_temperature

def update_pressure_by_cooling(
    primitive_state: STATE_TYPE,
    registered_variables: RegisteredVariables,
    simulation_params: SimulationParams,
    time_step: float,
) -> STATE_TYPE:
    
    # get the parameters
    cooling_params = simulation_params.cooling_params
    hydrogen_mass_fraction = cooling_params.hydrogen_mass_fraction
    metal_mass_fraction = cooling_params.metal_mass_fraction
    gamma = simulation_params.gamma

    # get the density and pressure
    density = primitive_state[registered_variables.density_index]
    pressure = primitive_state[registered_variables.pressure_index]

    # get the temperature
    temperature = get_temperature_from_pressure(
        density,
        pressure,
        hydrogen_mass_fraction,
        metal_mass_fraction,
    )

    # update the temperature
    new_temperature = update_temperature(
        density,
        temperature,
        time_step,
        hydrogen_mass_fraction,
        metal_mass_fraction,
        gamma,
        cooling_params.reference_temperature,
        cooling_params.factor,
        cooling_params.exponent
    )

    new_temperature = jnp.where(
        new_temperature > cooling_params.floor_temperature,
        new_temperature,
        temperature
    )

    # new_temperature = temperature

    # update the pressure
    new_pressure = get_pressure_from_temperature(
        density,
        new_temperature,
        hydrogen_mass_fraction,
        metal_mass_fraction,
    )

    # set the new pressure
    primitive_state = primitive_state.at[registered_variables.pressure_index].set(new_pressure)

    # return the updated primitive state
    return primitive_state