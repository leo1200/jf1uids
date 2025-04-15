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

from jf1uids.option_classes.simulation_config import FIELD_TYPE

import jax.numpy as jnp




def get_effective_molecular_weights(
    hydrogen_mass_fraction: float, # X
    metal_mass_fraction: float, # Z
    atomic_mass_unit: float # u
) -> Tuple[float, float, float]:
    """
    Calculate the mean molecular weight (mu) 
    and the effective molecular weights for 
    electrons (mu_e), hydrogen (mu_H)
    """

    # mean molecular weight
    mu = atomic_mass_unit / (
        2 * hydrogen_mass_fraction + 3 * (1 - hydrogen_mass_fraction - metal_mass_fraction) / 4 + metal_mass_fraction / 2
    )

    # effective molecular weight for electrons
    mu_e = 2 * atomic_mass_unit / (1 + metal_mass_fraction)

    # effective molecular weight for hydrogen
    mu_H = atomic_mass_unit / metal_mass_fraction

    return mu, mu_e, mu_H

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
    atomic_mass_unit: float,
    boltzmann_constant: float,
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
        atomic_mass_unit
    )

    # calculate the cooling rate
    cooling_rate = cooling_rate_power_law(
        temperature,
        reference_temperature,
        factor,
        exponent
    )

    # calculate the cooling time
    return (boltzmann_constant * mu_e * mu_H * temperature) / (
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
    if exponent != 1:
        return (1 / (1 - exponent)) * (1 - (temperature / reference_temperature) ** (1 - exponent))
    else:
        return -jnp.log(temperature / reference_temperature)

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
    if exponent != 1:
        return reference_temperature * (1 - (1 - exponent) * temporal_evolution_function) ** (1 / (1 - exponent))
    else:
        return reference_temperature * jnp.exp(-temporal_evolution_function)
    

def update_temperature(
    density: FIELD_TYPE,
    temperature: FIELD_TYPE,
    time_step: float,
    hydrogen_mass_fraction: float,
    metal_mass_fraction: float,
    atomic_mass_unit: float,
    boltzmann_constant: float,
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
        atomic_mass_unit,
        boltzmann_constant,
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

    # floor the temperature to 10^4 Kelvin
    new_temperature = jnp.maximum(new_temperature, 1e4)

    return new_temperature