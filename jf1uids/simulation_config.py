from typing import NamedTuple

# The simulation configuration are parameters defining the simulation
# where changes of necessitate recompilation.

class SimulationConfig(NamedTuple):
    # Simulation parameters
    alpha_geom: int = 0 # 0 -> cartesian, 1 -> cylindrical, 2 -> spherical
    box_size: float = 1.0
    num_cells: int = 400

    left_boundary: int = 0 # 0 -> open, 1 -> reflective
    right_boundary: int = 0 # 0 -> open, 1 -> reflective

    # if you want to use a fixed timestep
    # this changes how the simulation is
    # compiled, mind the CFL criterion
    fixed_timestep: bool = False
    num_timesteps: int = 1000

    stellar_wind: bool = False

    # checkpointing
    checkpointing: bool = False
    num_checkpoints: int = 10

    # TODO: add more configs