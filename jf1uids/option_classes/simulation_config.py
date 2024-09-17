from typing import NamedTuple

# The simulation configuration are parameters defining the simulation
# where changes of necessitate recompilation.

class SimulationConfig(NamedTuple):
    # Simulation parameters
    alpha_geom: int = 0 # 0 -> cartesian, 1 -> cylindrical, 2 -> spherical
    box_size: float = 1.0
    num_cells: int = 400

    # TODO: set num_ghost_cells automatically
    reconstruction_order: int = 1
    # The reconstruction order is the number of 
    # cells on each side of the cell of interest
    # used to calculate the gradients for the
    # reconstruction at the interfaces.

    #                                |---------|
    #                           |---------|
    # stencil              |---------|
    # cells            || 1g | 2g | 3c | 4g | 5g ||
    # reconstructions        |L  R|L  R|L  R|    |
    # fluxes                     -->  -->
    # update                      | 3c'|
    # --> all others are ghost cells

    num_ghost_cells: int = reconstruction_order + 1

    # HAS TO BE UPDATED MANUALLY
    dx: float = box_size / (num_cells - 1)

    left_boundary: int = 0 # 0 -> open, 1 -> reflective
    right_boundary: int = 0 # 0 -> open, 1 -> reflective

    # if you want to use a fixed timestep
    # this changes how the simulation is
    # compiled, mind the CFL criterion
    fixed_timestep: bool = False
    num_timesteps: int = 1000
    adaptive_timesteps_backwards_differentiable: bool = False
    num_checkpoints: int = 100

    stellar_wind: bool = False

    # intermediate saving of the simulation
    intermediate_saves: bool = False
    num_saves: int = 10

    first_order_fallback: bool = False
    # TODO: add more configs