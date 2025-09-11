# TOWNSEND SCHEME DOES NOT WORK CURRENTLY

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

from functools import partial

import jax

from jf1uids._physics_modules._cooling.cooling_options import COOLING_CURVE_TYPE, PIECEWISE_POWER_LAW, SIMPLE_POWER_LAW, CoolingParams
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
@partial(jax.jit, static_argnames = ("cooling_curve_type",))
def cooling_time(
    density: FIELD_TYPE,
    temperature: FIELD_TYPE,
    hydrogen_mass_fraction: float,
    metal_mass_fraction: float,
    gamma: float,
    cooling_curve_type: int,
    cooling_curve_params: COOLING_CURVE_TYPE
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
    cooling_rate = _cooling_rate(
        temperature,
        cooling_curve_type,
        cooling_curve_params,
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

# piecewise power law
@partial(jax.vmap, in_axes = (0, None, None, None), out_axes = 0)
def _evaluate_piecewise_power_law(
    T_in,
    T_table,
    Lambda_table,
    alpha_table,
):

    def eval_in_range(T_in):
        k = jnp.searchsorted(T_table, T_in) - 1

        # clip k to be in the valid range
        k = jnp.clip(k, 0, len(T_table) - 2)

        alpha_k = alpha_table[k]
        T_k = T_table[k]
        Lambda_k = Lambda_table[k]
        return Lambda_k * (T_in / T_k) ** alpha_k
    
    return jax.lax.cond(
        # check if T_in is in the table range
        (T_in >= T_table[0]) & (T_in <= T_table[-1]),
        eval_in_range,
        lambda _: 0.0, # return 0 if out of range
        T_in
    )

@partial(jax.vmap, in_axes = (0, None, None, None, None), out_axes = 0)
def _piecewise_power_law_temporal_evolution_function(
    T_in,
    T_table,
    Lambda_table,
    alpha_table,
    Y_table
):
    def eval_in_range(T_in):
        k = jnp.searchsorted(T_table, T_in) - 1

        # clip k to be in the valid range
        k = jnp.clip(k, 0, len(T_table) - 2)

        alpha_k = alpha_table[k]
        T_k = T_table[k]
        Lambda_k = Lambda_table[k]
        Y_k = Y_table[k]
        return Y_k + jax.lax.cond(
            alpha_k != 1.0,
            lambda: 1 / (1 - alpha_k) * Lambda_table[-1] / Lambda_k * T_k / T_table[-1] * (1 - (T_k / T_in) ** (alpha_k - 1)),
            lambda: Lambda_table[-1] / Lambda_k * T_k / T_table[-1] * jnp.log(T_k / T_in),
        )
    
    return jax.lax.cond(
        # check if T_in is in the table range
        (T_in >= T_table[0]) & (T_in <= T_table[-1]),
        eval_in_range,
        lambda _: 0.0, # return 0 if out of range
        T_in
    )

@partial(jax.vmap, in_axes = (0, None, None, None, None), out_axes = 0)
def _piecewise_power_law_temporal_evolution_function_inverse(
    Y_in,
    T_table,
    Lambda_table,
    alpha_table,
    Y_table
):
    def eval_in_range(Y_in):
        # k such that Y_k >= Y >= Y_{k+1}
        k = jnp.searchsorted(-Y_table, -Y_in) - 1

        # clip k to be in the valid range
        k = jnp.clip(k, 0, len(Y_table) - 2)

        alpha_k = alpha_table[k]
        T_k = T_table[k]
        Lambda_k = Lambda_table[k]
        Y_k = Y_table[k]
        return jax.lax.cond(
            alpha_k != 1.0,
            lambda: T_k * (1 - (1 - alpha_k) * (Y_in - Y_k) * Lambda_k / Lambda_table[-1] * T_table[-1] / T_k) ** (1 / (1 - alpha_k)),
            lambda: T_k * jnp.exp(-(Y_in - Y_k) * Lambda_k / Lambda_table[-1] * T_table[-1] / T_k),
        )
    
    return jax.lax.cond(
        # check if Y_in is in the table range,
        # Y_table is monotonically decreasing
        (Y_in >= Y_table[-1]) & (Y_in <= Y_table[0]),
        eval_in_range,
        lambda _: jnp.where(Y_in < Y_table[-1], T_table[-1], T_table[0]),
        Y_in
    )

@partial(jax.jit, static_argnames = ("cooling_curve_type",))
def _cooling_rate(
    temperature: FIELD_TYPE,
    cooling_curve_type: int,
    cooling_curve_params: COOLING_CURVE_TYPE,
) -> FIELD_TYPE:
    if cooling_curve_type == SIMPLE_POWER_LAW:
        return cooling_rate_power_law(
            temperature,
            cooling_curve_params.reference_temperature,
            cooling_curve_params.factor,
            cooling_curve_params.exponent
        )
    elif cooling_curve_type == PIECEWISE_POWER_LAW:
        return _evaluate_piecewise_power_law(
            temperature,
            10**cooling_curve_params.log10_T_table,
            10**cooling_curve_params.log10_Lambda_table,
            cooling_curve_params.alpha_table
        )
    else:
        raise ValueError(f"Unknown cooling curve type: {cooling_curve_type}")
    
@partial(jax.jit, static_argnames = ("cooling_curve_type",))
def _temporal_evolution_function(
    temperature: FIELD_TYPE,
    cooling_curve_type: int,
    cooling_curve_params: COOLING_CURVE_TYPE,
) -> FIELD_TYPE:
    if cooling_curve_type == SIMPLE_POWER_LAW:
        return power_law_temporal_evolution_function(
            temperature,
            cooling_curve_params.reference_temperature,
            cooling_curve_params.exponent
        )
    elif cooling_curve_type == PIECEWISE_POWER_LAW:
        return _piecewise_power_law_temporal_evolution_function(
            temperature,
            10**cooling_curve_params.log10_T_table,
            10**cooling_curve_params.log10_Lambda_table,
            cooling_curve_params.alpha_table,
            cooling_curve_params.Y_table
        )
    else:
        raise ValueError(f"Unknown cooling curve type: {cooling_curve_type}")
    
@partial(jax.jit, static_argnames = ("cooling_curve_type",))
def _temporal_evolution_function_inverse(
    temporal_evolution_function: FIELD_TYPE,
    cooling_curve_type: int,
    cooling_curve_params: COOLING_CURVE_TYPE,
) -> FIELD_TYPE:
    if cooling_curve_type == SIMPLE_POWER_LAW:
        return power_law_temporal_evolution_function_inverse(
            temporal_evolution_function,
            cooling_curve_params.reference_temperature,
            cooling_curve_params.exponent
        )
    elif cooling_curve_type == PIECEWISE_POWER_LAW:
        return _piecewise_power_law_temporal_evolution_function_inverse(
            temporal_evolution_function,
            10**cooling_curve_params.log10_T_table,
            10**cooling_curve_params.log10_Lambda_table,
            cooling_curve_params.alpha_table,
            cooling_curve_params.Y_table
        )
    else:
        raise ValueError(f"Unknown cooling curve type: {cooling_curve_type}")
    

@partial(jax.jit, static_argnames = ("cooling_curve_type",))
def update_temperature(
    density: FIELD_TYPE,
    temperature: FIELD_TYPE,
    time_step: float,
    hydrogen_mass_fraction: float,
    metal_mass_fraction: float,
    gamma: float,
    cooling_curve_type: int,
    cooling_curve_params: COOLING_CURVE_TYPE,
) -> FIELD_TYPE:
    """
    T_new = Y^-1[Y(T) + T / T_ref * \Lambda(T_ref) / \Lambda(T) * delta_t / t_cool]
    """

    reference_temperature = cooling_curve_params.reference_temperature

    # calculate the cooling time
    # not numerically stable, divisiion
    # by the cooling
    # t_cool = cooling_time(
    #     density,
    #     temperature,
    #     hydrogen_mass_fraction,
    #     metal_mass_fraction,
    #     gamma,
    #     cooling_curve_type,
    #     cooling_curve_params
    # )

    # calculate the cooling rate
    cooling_rate = _cooling_rate(
        temperature,
        cooling_curve_type,
        cooling_curve_params
    )

    cooling_rate_reference = _cooling_rate(
        jnp.array([reference_temperature]),
        cooling_curve_type,
        cooling_curve_params
    )

    # calculate the temporal evolution function
    temporal_evolution_function = _temporal_evolution_function(
        temperature,
        cooling_curve_type,
        cooling_curve_params
    )

    # calculate the new temperature
    # new_temperature = _temporal_evolution_function_inverse(
    #     temporal_evolution_function + (temperature / reference_temperature) * (cooling_rate_reference / cooling_rate) * time_step / t_cool,
    #     cooling_curve_type,
    #     cooling_curve_params
    # )

    mu, mu_e, mu_H = get_effective_molecular_weights(
        hydrogen_mass_fraction,
        metal_mass_fraction,
    )

    new_temperature = _temporal_evolution_function_inverse(
        temporal_evolution_function + cooling_rate_reference / reference_temperature * ((gamma - 1) * density * mu) / (mu_e * mu_H) * time_step,
        cooling_curve_type,
        cooling_curve_params
    )

    return new_temperature

@partial(jax.jit, static_argnames = ("cooling_curve_type",))
def update_temperature_explicit(
    density: FIELD_TYPE,
    temperature: FIELD_TYPE,
    time_step: float,
    hydrogen_mass_fraction: float,
    metal_mass_fraction: float,
    gamma: float,
    cooling_curve_type: int,
    cooling_curve_params: COOLING_CURVE_TYPE,
) -> FIELD_TYPE:
    """
    T_new = T - (gamma - 1) * rho * \mu / (mu_e * mu_H * k) * Lambda(T) * delta_t
    (units absorbed in Lambda)
    """

    # calculate the cooling rate
    cooling_rate = _cooling_rate(
        temperature,
        cooling_curve_type,
        cooling_curve_params
    )

    mu, mu_e, mu_H = get_effective_molecular_weights(
        hydrogen_mass_fraction,
        metal_mass_fraction,
    )

    return temperature - (cooling_rate * (gamma - 1) * density * mu) / (mu_e * mu_H) * time_step

@partial(jax.jit, static_argnames = ("cooling_curve_type", "registered_variables"))
def update_pressure_by_cooling(
    primitive_state: STATE_TYPE,
    registered_variables: RegisteredVariables,
    cooling_curve_type: int,
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
    # new_temperature = update_temperature(
    #     density,
    #     temperature,
    #     time_step,
    #     hydrogen_mass_fraction,
    #     metal_mass_fraction,
    #     gamma,
    #     cooling_curve_type,
    #     cooling_params.cooling_curve_params
    # )

    new_temperature = update_temperature_explicit(
        density,
        temperature,
        time_step,
        hydrogen_mass_fraction,
        metal_mass_fraction,
        gamma,
        cooling_curve_type,
        cooling_params.cooling_curve_params
    )

    new_temperature = jnp.where(
        (new_temperature > cooling_params.floor_temperature),
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