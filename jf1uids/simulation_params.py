from typing import NamedTuple

# Different from the simulation configuration, the simulation parameters
# do not require recompilation when changed.

# Examples are the CFL number or settings of any physics module
# that might be implemented.

class SimulationParams(NamedTuple):
    dt_max: float
    C_cfl: float
    dx: float
    gamma: float
    t_end: float