from typing import NamedTuple

from jf1uids._geometry.boundaries import OPEN_BOUNDARY
from jf1uids._physics_modules._stellar_wind.stellar_wind_options import WindConfig

from jf1uids import CARTESIAN

FORWARDS = 0
BACKWARDS = 1

class SimulationConfig(NamedTuple):
    """Configuration object for the simulation.
    The simulation configuration are parameters defining 
    the simulation where changes necessitate recompilation."""

    # Simulation parameters

    #: The geometry of the simulation.
    geometry: int = CARTESIAN

    #: The size of the simulation box.
    box_size: float = 1.0

    #: The number of cells in the simulation (including ghost cells).
    num_cells: int = 400

    #: The reconstruction order is the number of 
    #: cells on each side of the cell of interest
    #: used to calculate the gradients for the
    #: reconstruction at the interfaces.
    reconstruction_order: int = 1

    # Explanation of the ghost cells
    #                                |---------|
    #                           |---------|
    # stencil              |---------|
    # cells            || 1g | 2g | 3c | 4g | 5g ||
    # reconstructions        |L  R|L  R|L  R|    |
    # fluxes                     -->  -->
    # update                      | 3c'|
    # --> all others are ghost cells

    #: The number of ghost cells.
    num_ghost_cells: int = reconstruction_order + 1

    #: The width of the cells.
    dx: float = box_size / (num_cells - 1)

    #: The left boundary condition.
    left_boundary: int = OPEN_BOUNDARY

    #: The right boundary condition.
    right_boundary: int = OPEN_BOUNDARY

    #: Enables a fixed timestep for the simulation
    #: based on the specified number of timesteps.
    fixed_timestep: bool = False

    #: The number of timesteps for the fixed timestep mode.
    num_timesteps: int = 1000

    #: The differentiation mode one whats to use
    #: the solver in (forwards or backwards).
    differentiation_mode: int = FORWARDS

    #: The number of checkpoints used in the setup
    #: with backwards differetiability and adaptive
    #: time stepping.
    num_checkpoints: int = 100

    #: Return intermediate snapshots of the time evolution
    #: insteat of only the final fluid state.
    return_snapshots: bool = False

    #: The number of snapshots to return.
    num_snapshots: int = 10

    #: Fallback to the first order Godunov scheme.
    first_order_fallback: bool = False

    # physical modules

    #: The configuration for the stellar wind module.
    wind_config: WindConfig = WindConfig()