from typing import NamedTuple

# The simulation configuration are parameters defining the simulation
# where changes of necessitate recompilation.

class SimulationConfig(NamedTuple):
    # Simulation parameters
    alpha_geom: int = 0 # 0 -> cartesian, 1 -> cylindrical, 2 -> spherical
    left_boundary: int = 0 # 0 -> open, 1 -> reflective
    right_boundary: int = 0 # 0 -> open, 1 -> reflective

    stellar_wind: bool = False

    # TODO: add more configs