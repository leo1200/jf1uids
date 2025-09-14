from typing import NamedTuple
from typing import Union
import jax.numpy as jnp

SIMPLE_POWER_LAW = 1
PIECEWISE_POWER_LAW = 2

class CoolingConfig(NamedTuple):
    cooling: bool = False
    cooling_curve_type: int = SIMPLE_POWER_LAW

class SimplePowerLawParams(NamedTuple):
    factor: float = 1.0
    exponent: float = 1.0
    reference_temperature: float = 1e8

class PiecewisePowerLawParams(NamedTuple):
    log10_T_table: jnp.ndarray = jnp.array([])
    log10_Lambda_table: jnp.ndarray = jnp.array([])
    alpha_table: jnp.ndarray = jnp.array([])
    Y_table: jnp.ndarray = jnp.array([])
    reference_temperature: float = 1e8

COOLING_CURVE_TYPE = Union[SimplePowerLawParams, PiecewisePowerLawParams]

class CoolingParams(NamedTuple):
    # NOTE: CURRENTLY ONLY POWER LAW COOLING
    hydrogen_mass_fraction: float = 0.76
    metal_mass_fraction: float = 0.02

    floor_temperature: float = 1e4

    cooling_curve_params: COOLING_CURVE_TYPE = SimplePowerLawParams()