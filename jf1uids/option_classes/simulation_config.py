from types import NoneType
from typing import NamedTuple, Union

from jf1uids._physics_modules._stellar_wind.stellar_wind_options import WindConfig

from jaxtyping import Array, Float

# differentiation modes
FORWARDS = 0
BACKWARDS = 1

# limiter types
MINMOD = 0
OSHER = 1

# Riemann solvers
HLL = 0
HLLC = 1

OPEN_BOUNDARY = 0
REFLECTIVE_BOUNDARY = 1
PERIODIC_BOUNDARY = 2

CARTESIAN = 0
CYLINDRICAL = 1
SPHERICAL = 2

STATE_TYPE = Union[Float[Array, "num_vars num_cells_x"], Float[Array, "num_vars num_cells_x num_cells_y"], Float[Array, "num_vars num_cells_x num_cells_y num_cells_z"]]
STATE_TYPE_ALTERED = Union[Float[Array, "num_vars num_cells_a"], Float[Array, "num_vars num_cells_a num_cells_b"], Float[Array, "num_vars num_cells_a num_cells_b num_cells_c"]]

class BoundarySettings1D(NamedTuple):
    left_boundary: int = OPEN_BOUNDARY
    right_boundary: int = OPEN_BOUNDARY

class BoundarySettings(NamedTuple):
    x: BoundarySettings1D = BoundarySettings1D()
    y: BoundarySettings1D = BoundarySettings1D()
    z: BoundarySettings1D = BoundarySettings1D()

class SimulationConfig(NamedTuple):
    """Configuration object for the simulation.
    The simulation configuration are parameters defining 
    the simulation where changes necessitate recompilation."""

    # Simulation parameters

    #: Debug runtime errors, throws exceptions
    #: on e.g. negative pressure or density.
    #: Significantly reduces performance.
    runtime_debugging: bool = False

    #: Activate progress bar
    progress_bar: bool = False

    #: The number of dimensions of the simulation.
    dimensionality: int = 1

    #: The geometry of the simulation.
    geometry: int = CARTESIAN

    #: Magnetohydrodynamics switch.
    mhd: bool = False

    #: The size of the simulation box.
    box_size: float = 1.0

    #: The number of cells in the simulation (including ghost cells).
    num_cells: int = 400

    #: The reconstruction order is the number of 
    #: cells on each side of the cell of interest
    #: used to calculate the gradients for the
    #: reconstruction at the interfaces.
    reconstruction_order: int = 1

    #: The limiter for the reconstruction.
    limiter: int = MINMOD

    #: The Riemann solver used
    riemann_solver: int = HLLC

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

    #: Boundary settings for the simulation.
    boundary_settings: Union[NoneType, BoundarySettings1D, BoundarySettings] = None

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
    #: instead of only the final fluid state.
    return_snapshots: bool = False

    #: The number of snapshots to return.
    num_snapshots: int = 10

    #: Fallback to the first order Godunov scheme.
    first_order_fallback: bool = False

    # physical modules

    #: The configuration for the stellar wind module.
    wind_config: WindConfig = WindConfig()

    #: Cosmic rays
    simplified_cosmic_rays: bool = False