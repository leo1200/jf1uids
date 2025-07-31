from types import NoneType
from typing import NamedTuple, Union

from jf1uids._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector_options import CNNMHDconfig
from jf1uids._physics_modules._cooling.cooling_options import CoolingConfig
from jf1uids._physics_modules._cosmic_rays.cosmic_ray_options import CosmicRayConfig
from jf1uids._physics_modules._neural_net_force._neural_net_force_options import NeuralNetForceConfig
from jf1uids._physics_modules._stellar_wind.stellar_wind_options import WindConfig

from jaxtyping import Array, Float

# ===================== constant definition =====================

# differentiation modes
FORWARDS = 0
BACKWARDS = 1

# limiter types
MINMOD = 0
OSHER = 1
DOUBLE_MINMOD = 2
SUPERBEE = 3
VAN_ALBADA = 4
VAN_ALBADA_PP = 5

# splitting modes
UNSPLIT = 0
SPLIT = 1

# Riemann solvers
HLL = 0
HLLC = 1
HLLC_LM = 2
LAX_FRIEDRICHS = 3

# time integrators
RK2_SSP = 0
MUSCL = 1

# boundary conditions
OPEN_BOUNDARY = 0
REFLECTIVE_BOUNDARY = 1
PERIODIC_BOUNDARY = 2
MHD_JET_BOUNDARY = 3

GAS_STATE = 0
VELOCITY_ONLY = 1
MAGNETIC_FIELD_ONLY = 2

# geometry types
CARTESIAN = 0
CYLINDRICAL = 1
SPHERICAL = 2

# axes
VARAXIS = 0
XAXIS = 1
YAXIS = 2
ZAXIS = 3

# self-gravity versions
SIMPLE_SOURCE_TERM = 0
CONSERVATIVE_SOURCE_TERM = 1

# ============================================================

# ===================== type definitions =====================

STATE_TYPE = Union[
    Float[Array, "num_vars num_cells_x"],
    Float[Array, "num_vars num_cells_x num_cells_y"],
    Float[Array, "num_vars num_cells_x num_cells_y num_cells_z"]
]

STATE_TYPE_ALTERED = Union[
    Float[Array, "num_vars num_cells_a"],
    Float[Array, "num_vars num_cells_a num_cells_b"],
    Float[Array, "num_vars num_cells_a num_cells_b num_cells_c"]
]

FIELD_TYPE = Union[
    Float[Array, "num_cells_x"],
    Float[Array, "num_cells_x num_cells_y"],
    Float[Array, "num_cells_x num_cells_y num_cells_z"]
]

# =============================================================

class BoundarySettings1D(NamedTuple):
    left_boundary: int = OPEN_BOUNDARY
    right_boundary: int = OPEN_BOUNDARY

class BoundarySettings(NamedTuple):
    x: BoundarySettings1D = BoundarySettings1D()
    y: BoundarySettings1D = BoundarySettings1D()
    z: BoundarySettings1D = BoundarySettings1D()

class SimulationConfig(NamedTuple):
    """
    Configuration object for the simulation.
    The simulation configuration are parameters defining 
    the simulation where changes necessitate recompilation.
    """

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

    #: Self gravity switch, currently only
    #: for periodic boundaries.
    self_gravity: bool = False
    self_gravity_version: int = CONSERVATIVE_SOURCE_TERM

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
    limiter: int = VAN_ALBADA_PP

    #: The Riemann solver used
    riemann_solver: int = LAX_FRIEDRICHS

    #: Dimensional splitting / unsplit mode.
    split: int = UNSPLIT

    #: Time integration method.
    time_integrator: int = RK2_SSP

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

    #: Grid spacing.
    grid_spacing: float = box_size / num_cells

    #: Boundary settings for the simulation.
    boundary_settings: Union[NoneType, BoundarySettings1D, BoundarySettings] = None

    #: Enables a fixed timestep for the simulation
    #: based on the specified number of timesteps.
    fixed_timestep: bool = False

    #: Exactly reach the end time. In adaptive timestepping,
    #: one might otherwise overshoot.
    exact_end_time: bool = True

    #: Adds the sources with the current timestep to
    #: a hypothetical state to estimate the actual timestep.
    #: Useful for time-dependent sources, but additional
    #: computational overhead.
    source_term_aware_timestep: bool = False

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

    #: Call a user given function on the snapshot data,
    #: e.g. for saving or plotting. Must have signature
    #: callback(time, state, registered_variables).
    activate_snapshot_callback: bool = False

    #: Return snapshots at specific time points.
    use_specific_snapshot_timepoints: bool = False
    specific_snapshot_timepoints: tuple = ()

    #: The number of snapshots to return.
    num_snapshots: int = 10

    #: Fallback to the first order Godunov scheme.
    first_order_fallback: bool = False

    # physical modules

    #: The configuration for the stellar wind module.
    wind_config: WindConfig = WindConfig()

    #: Cosmic rays
    cosmic_ray_config: CosmicRayConfig = CosmicRayConfig()

    #: The configuration for the cooling module.
    cooling_config: CoolingConfig = CoolingConfig()

    #: Configuration of the neural network force module.
    neural_net_force_config: NeuralNetForceConfig = NeuralNetForceConfig()

    #: Configuration of the CNN MHD corrector module.
    cnn_mhd_corrector_config: CNNMHDconfig = CNNMHDconfig()


def finalize_config(config: SimulationConfig, state_shape) -> SimulationConfig:
    """Finalizes the simulation configuration."""

    num_cells = state_shape[-1]
    config = config._replace(num_cells = num_cells)

    if config.dimensionality == 2:
        num_cells_x, num_cells_y = state_shape[-2:]
        if num_cells_x != num_cells_y:
            raise ValueError("The number of cells in x and y must be equal.")
    elif config.dimensionality == 3:
        num_cells_x, num_cells_y, num_cells_z = state_shape[-3:]
        if num_cells_x != num_cells_y or num_cells_x != num_cells_z:
            raise ValueError("The number of cells in x, y and z must be equal.")

    config = config._replace(grid_spacing = config.box_size / config.num_cells)

    if config.geometry == SPHERICAL:

        print("For spherical geometry, only HLL is currently supported. Also, only the unsplit mode has been tested.")
        config = config._replace(grid_spacing = config.box_size / (config.num_cells - 1))

        if config.riemann_solver != HLL:
            print("Setting HLL Riemann solver for spherical geometry.")
            config = config._replace(riemann_solver = HLL)

        if config.split != UNSPLIT:
            print("Setting unsplit mode for spherical geometry")
            config = config._replace(split = UNSPLIT)

        if config.limiter == VAN_ALBADA or config.limiter == VAN_ALBADA_PP:
            print("Setting minmod limiter for spherical geometry")
            config = config._replace(limiter = MINMOD)

        if config.time_integrator != MUSCL:
            print("Setting MUSCL time integrator for spherical geometry")
            config = config._replace(time_integrator = MUSCL)

    if (config.limiter == MINMOD or config.limiter == DOUBLE_MINMOD or config.limiter == SUPERBEE or config.riemann_solver == HLL or config.riemann_solver == HLLC or config.riemann_solver == HLLC_LM) and config.split == UNSPLIT:
        print("Due to different implementation approaches, currently only a limited set of Riemann solver and limiters work in unsplit mode, namely LAX_FRIEDRICHS with VAN_ALBADA limiter.")
        print("Switching to split mode.")
        config = config._replace(split = SPLIT)
        config = config._replace(time_integrator = MUSCL)
        
    # set boundary conditions if not set
    if config.boundary_settings is None:

        if config.geometry == CARTESIAN:
            print("Automatically setting open boundaries for Cartesian geometry.")
            if config.dimensionality == 1:
                config = config._replace(boundary_settings = BoundarySettings1D(left_boundary = OPEN_BOUNDARY, right_boundary = OPEN_BOUNDARY))
            else:
                config = config._replace(boundary_settings = BoundarySettings())
        elif config.geometry == SPHERICAL and config.dimensionality == 1:
            print("Automatically setting reflective left and open right boundary for spherical geometry.")
            config = config._replace(boundary_settings = BoundarySettings1D(left_boundary = REFLECTIVE_BOUNDARY, right_boundary = OPEN_BOUNDARY))

    if config.wind_config.stellar_wind:
        print("For stellar wind simulations, we need source term aware timesteps, turning on.")
        config = config._replace(source_term_aware_timestep = True)
    
    return config