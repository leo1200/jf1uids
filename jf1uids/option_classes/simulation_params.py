from typing import NamedTuple

from jf1uids.physics_modules.stellar_wind.stellar_wind import WindParams

# Different from the simulation configuration, the simulation parameters
# do not require recompilation when changed.

# Examples are the CFL number or settings of any physics module
# that might be implemented.

class SimulationParams(NamedTuple):
    C_cfl: float = 0.8
    gamma: float = 5/3

    dt_max: float = 0.001
    t_end: float = 0.2

    # parameters of physics modules
    wind_params: WindParams = WindParams()