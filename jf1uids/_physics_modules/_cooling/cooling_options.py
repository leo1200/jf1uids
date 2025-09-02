from typing import NamedTuple

class CoolingConfig(NamedTuple):
    cooling: bool = False

class CoolingParams(NamedTuple):
    # NOTE: CURRENTLY ONLY POWER LAW COOLING
    hydrogen_mass_fraction: float = 0.76
    metal_mass_fraction: float = 0.02
    reference_temperature: float = 1e4
    floor_temperature: float = 1e4
    factor: float = 1.0
    exponent: float = 1.0