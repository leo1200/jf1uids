"""
The goal here is to implement (dynamic) adaptive mesh refinement 
(adaptive multiresolution), starting with 1d hydrodynamics.

"""

# basic fluid operations
from fluid import (
    _calculate_average_gradients,
    _calculate_limited_gradients,
    _hll_solver,
    _open_boundaries,
    conserved_state_from_primitive,
    primitive_state_from_conserved,
)

# numerics
import jax
import jax.numpy as jnp

# for type annotations,
# use e.g. beartype for checking
from typing import Tuple
from jaxtyping import Array, Bool, Float, Int

# pytree-objects via eqx.Module
import equinox as eqx

# plotting
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# types compatible with tracing
FloatScalar = float | Float[Array, ""]
IntScalar = int | Int[Array, ""]
BoolScalar = bool | Bool[Array, ""]

# -------------------------------------------------------------
# ================== ↓ Basic List Operations ↓ ================
# -------------------------------------------------------------


@jax.jit
def get_index_range_mask(
    buffer: Float[Array, "num_vars buffer_size"],
    start_index: IntScalar,
    end_index: IntScalar,
) -> Bool[Array, "buffer_size"]:
    """
    Returns a boolean mask that is True for the elements 
    of the buffer that are within the
    specified range.

    Args:
        buffer: The buffer.
        start_index: The start index of the range.
        end_index: The end index of the range.

    Returns:
        The boolean mask.
    """

    indices = jnp.arange(buffer.shape[1])
    mask = (indices >= start_index) & (indices < end_index)
    return mask


# We are going to work with sorted lists, in 1D that is 
# trivial, in higher dimensions we would e.g. sort by the
# Morton index (-> space-filling curves). In 1D having a sorted
# list means that we can apply the same vectorized operations 
# that we would apply for the static discretization
# (neighborhood is trivial). Any search operation will be cheaper
# on a sorted list (-> binary search) but there is memory 
# management overhead.


@jax.jit
def insert_into_buffer(
    list1: Float[Array, "num_vars buffer_size"],
    list2: Float[Array, "num_vars buffer_size"],
    mask: Float[Array, " buffer_size"],
) -> Float[Array, "num_vars buffer_size"]:
    """
    Inserts the masked elements of list2 into list1, to the right 
    of the masked elements of list1. Enough space should be available
    at the end of the list1 buffer to accommodate the
    masked elements (no error checking is performed).

    WARNING: One can currently not insert elements next 
    to the leftmost element of the list1 buffer.

    Args:
        list1: The first list, into which the masked elements 
               of list2 will be inserted.
        list2: The second list, from which the masked 
               elements will be inserted.
        mask: A boolean mask indicating which elements of 
              list2 should be inserted into list1.

    Returns:
        The updated list1 buffer.
    """

    # Let us make the code readable using the integer example
    # list1 = jnp.array([1, 2, 3, 4, 5, 0, 0, 0, 0, 0])
    # mask  = jnp.array([0, 1, 0, 1, 1, 0, 0, 0, 0, 0], dtype=bool)
    # list2 = jnp.array([4, 5, 6, 7, 8, 0, 0, 0, 0, 0])

    # with wanted result
    # insertions            ____     ____  ____
    # res   = jnp.array([1, 2, 5, 3, 4, 7, 5, 8, 0, 0])

    # let us get a list of the indices
    index_list = jnp.arange(mask.shape[0])
    # ind.  = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    # these will be the indices where the refined blocks end
    refinement_indices = (index_list + jnp.cumsum(mask)) * mask
    # r.i.  = jnp.array([0, 2, 0, 5, 7, 0, 0, 0, 0, 0])
    # referring to             _        _     _

    # now we construct an index set where the future
    # fill-in spots filled with duplicates
    spreader = jnp.zeros_like(refinement_indices, dtype=int)
    spreader = spreader.at[refinement_indices].set(1)
    spreader = spreader.at[0].set(0)
    refinement_subs = jnp.cumsum(spreader)
    spreader = index_list - refinement_subs
    # sp.  = jnp.array([0, 1, 1, 2, 3, 3, 4, 4, 5, 6])

    # spread out the lists
    list2 = list2[:, spreader]
    list1 = list1[:, spreader]

    # spread out the mask
    mask = mask[spreader]

    # get a mask on the spots where we want to insert
    refinement_insertion_mask = jnp.zeros_like(mask, dtype=bool)
    refinement_insertion_mask = refinement_insertion_mask.at[1:].set(
        (spreader[1:] - spreader[:-1]) > 0
    )
    refinement_insertion_mask = jnp.logical_xor(refinement_insertion_mask * mask, mask)

    # perform the insertion
    list1 = jnp.where(refinement_insertion_mask, list2, list1)

    return list1

@jax.jit
def remove_from_buffer(
    list: Float[Array, "num_vars buffer_size"],
    mask: Float[Array, " buffer_size"],
) -> Float[Array, "num_vars buffer_size"]:
    """
    Removes the masked elements from the list, shifting values to the
    left to fill the gaps, right-most elements that get freed up are
    set to the same value as the former right-most element.

    Args:
        list: The list from which elements should be removed.
        mask: A boolean mask indicating which elements should be removed.

    Returns:
        The updated list.
    """

    # let us again consider an example
    # list = jnp.array([1, 2, 3, 4, 5, 0, 0, 0, 0, 0])
    # mask = jnp.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
    # with wanted result
    # res  = jnp.array([1, 4, 5, 0, 0, 0, 0, 0, 0, 0])

    index_list = jnp.arange(mask.shape[0])
    # ind. = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    setter = (index_list - jnp.cumsum(mask)) * (1 - mask)
    # set. = jnp.array([0, 0, 0, 1, 2, 3, 4, 5, 6, 7])

    # construct the shifter
    shifter = index_list.at[setter].set(index_list * (1 - mask))
    # shi. = jnp.array([[0, 3, 4, 5, 6, 7, 8, 9, 8, 9]])

    # handle the right-most elements freed up
    max_used_index = mask.shape[0] - jnp.sum(mask)
    max_index = mask.shape[0] - 1
    upper_mask = (index_list < max_used_index) & (index_list < max_index)
    shifter = jnp.where(upper_mask, shifter, max_index)
    # shi. = jnp.array([[0, 3, 4, 5, 6, 7, 8, 9, 9, 9]])

    return list[:, shifter]

class _BufferedList(eqx.Module):
    buffer: Float[Array, "num_vars buffer_size"]
    num_cells: IntScalar

    @jax.jit
    def __getitem__(self, item: IntScalar) -> FloatScalar:
        return self.buffer[item]

    @jax.jit
    def set_elements(
        self,
        elements: Float[Array, "num_vars buffer_size"],
        mask: Bool[Array, "buffer_size"],
    ):
        """
        Sets the masked elements in the buffer.

        Args:
            elements: The elements to set.
            mask: The mask indicating which elements to set.

        Returns:
            The updated buffer.
        """

        buffer = jnp.where(mask, elements, self.buffer)
        num_cells = self.num_cells

        return _BufferedList(buffer, num_cells)

    @jax.jit
    def reset_allocatable_elements(self):
        """
        Resets the allocatable elements in the buffer. This is 
        useful when we operate on the whole buffer 
        (see https://github.com/jax-ml/jax/discussions/19178) 
        but want to reset garbage regions.

        Returns:
            The updated buffer.
        """

        deletion_mask = get_index_range_mask(
            self.buffer, self.num_cells, self.buffer.shape[1]
        )

        buffer = jnp.where(deletion_mask, 1.0, self.buffer)

        num_cells = self.num_cells

        return _BufferedList(buffer, num_cells)

    @jax.jit
    def insert_elements(
        self,
        elements: Float[Array, "num_vars buffer_size"],
        mask: Bool[Array, "buffer_size"],
    ):
        """
        Inserts the masked elements into the buffer.

        Args:
            elements: The elements to insert.
            mask: The mask indicating which elements to insert.

        Returns:
            The updated buffer.
        """

        buffer = insert_into_buffer(self.buffer, elements, mask)
        num_cells = self.num_cells + jnp.sum(mask)

        return _BufferedList(buffer, num_cells)
    
    @jax.jit
    def remove_elements(
        self,
        mask: Bool[Array, "buffer_size"],
    ):
        """
        Removes the masked elements from the buffer.

        Args:
            mask: The mask indicating which elements to remove.

        Returns:
            The updated buffer.
        """

        buffer = remove_from_buffer(self.buffer, mask)
        num_cells = self.num_cells - jnp.sum(mask)

        return _BufferedList(buffer, num_cells)


# -------------------------------------------------------------
# ================== ↑ Basic List Operations ↑ ================
# -------------------------------------------------------------


# -------------------------------------------------------------
# ==================== ↓ AMR Operations ↓ =====================
# -------------------------------------------------------------


@jax.jit
def get_refinement_mask(
    primitive_states: Float[Array, "num_prims buffer_size"],
    center_buffer: Float[Array, "buffer_size"],
    refinement_levels: Float[Array, "buffer_size"],
    maximum_refinement: int,
    refinement_tolerance: float,
    num_cells: int,
) -> Bool[Array, "buffer_size"]:
    """
    Returns a mask indicating which cells should be refined.

    Args:
        primitive_states: The primitive states.
        center_buffer: The cell centers.
        refinement_levels: The refinement levels.
        refinement_tolerance: The derefinement tolerance.
        num_cells: The number of cells

    Returns:
        A mask indicating which cells should be refined.
    """

    # refinement criterion
    refinement_criterion = jnp.zeros(primitive_states.shape[1], dtype=bool)

    # get the limited gradients
    average_gradients = _calculate_average_gradients(primitive_states, center_buffer)

    # calculate the density jump
    density_jump = jnp.abs(average_gradients[0])

    # calculate the pressure jump
    pressure_jump = jnp.abs(average_gradients[2])

    # calculate the velocity jump
    velocity_jump = jnp.abs(average_gradients[1])

    # calculate the density jump criterion
    density_jump_criterion = density_jump > refinement_tolerance

    # calculate the pressure jump criterion
    pressure_jump_criterion = pressure_jump > refinement_tolerance

    # calculate the velocity jump criterion
    velocity_jump_criterion = velocity_jump > refinement_tolerance

    # calculate the refinement criterion
    refinement_criterion_no_pad = jnp.logical_or(
        density_jump_criterion,
        jnp.logical_or(pressure_jump_criterion, velocity_jump_criterion),
    )

    refinement_criterion = refinement_criterion.at[1:-1].set(
        refinement_criterion_no_pad
    )

    # ignore cells where the refinement level is already at the maximum
    refinement_criterion = jnp.logical_and(
        refinement_criterion, refinement_levels < maximum_refinement
    )

    # do not refine the first and last cell
    refinement_criterion = refinement_criterion.at[0].set(False)
    mask = get_index_range_mask(primitive_states, 0, num_cells - 1)
    refinement_criterion = jnp.where(mask, refinement_criterion, False)

    # balance refinement criterion
    refinement_criterion = balance_refinement_criterion(
        refinement_criterion, refinement_levels
    )

    return refinement_criterion


@jax.jit
def balance_refinement_criterion(
    refinement_criterion: Bool[Array, "buffer_size"],
    refinement_levels: Float[Array, "buffer_size"],
) -> Bool[Array, "buffer_size"]:
    """
    For numerical stability reasons, we want the refinement level
    to be 'balanced', neighboring cells should not differ by more
    than one level (2:1 refinement ratio). This function does
    not handle ripple effects (where a balancing refinement
    requires further refinements, which should be fine as we
    only refine one level at a time).

    Args:
        refinement_criterion: Mask indicating which cells should 
                              be refined.
        refinement_levels: The refinement levels of the cells prior 
                           to refining with the current refinement 
                           criterion.

    Returns:
        The refinement criterion after one balancing step. E.g. for
        refinement_criterion = [0, 1, 0, 0] and 
        refinement_levels    = [0, 1, 1, 0] the output would be
        returns ...............[1, 1, 1, 0].
    """

    hypothetical_refinement_levels = refinement_levels + jnp.where(
        refinement_criterion, 1, 0
    )
    # list with jumps 0 -> 1, 1 -> 2, ...
    level_jumps = (
        hypothetical_refinement_levels[1:] - hypothetical_refinement_levels[:-1]
    )
    # level increases
    level_increases = level_jumps > 1
    # level decreases
    level_decreases = level_jumps < -1

    # set the refinement criterion
    refinement_criterion = refinement_criterion.at[0:-1].set(
        jnp.logical_or(refinement_criterion[0:-1], level_increases)
    )
    refinement_criterion = refinement_criterion.at[1:].set(
        jnp.logical_or(refinement_criterion[1:], level_decreases)
    )

    return refinement_criterion


@jax.jit
def calculate_refined_cells(
    fluid_data: _BufferedList,
) -> tuple[_BufferedList, _BufferedList]:
    """
    When a cell is split, the fluid quantities on the
    new cells are calculated by interpolation with 
    limited gradients (retaining conservation properties).

    Args:
        fluid_data: The fluid data.

    Returns:
        The fluid data for the left and right refined cells
        for all cells (so also doing unnecessary work).

    """

    # extract data
    primitive_states = fluid_data.buffer[0:3, :]
    center_buffer = fluid_data.buffer[3, :]
    volume_buffer = fluid_data.buffer[4, :]
    refinement_level_buffer = fluid_data.buffer[5, :]
    num_cells = fluid_data.num_cells

    # calculate the limited gradients
    limited_gradients_no_pad = _calculate_limited_gradients(
        primitive_states, center_buffer
    )
    limited_gradients = jnp.zeros(primitive_states.shape, dtype=primitive_states.dtype)
    limited_gradients = limited_gradients.at[:, 1:-1].set(limited_gradients_no_pad)

    # calculate the refined cell centers, left
    refined_center_left = center_buffer - 0.25 * volume_buffer

    # calculate the refined cell centers, right
    refined_center_right = center_buffer + 0.25 * volume_buffer

    # calculate the refined cell volumes
    refined_volume = 0.5 * volume_buffer

    # calculate the refined cell refinement levels
    refined_refinement_level = refinement_level_buffer + 1

    # calculate the refined cell primitive states
    refined_primitive_states_right = (
        primitive_states + limited_gradients * refined_volume
    )

    # calculate the refined cell primitive states
    refined_primitive_states_left = (
        primitive_states - limited_gradients * refined_volume
    )

    # fluid data left
    fluid_data_left = _BufferedList(
        jnp.vstack(
            [
                refined_primitive_states_left,
                refined_center_left,
                refined_volume,
                refined_refinement_level,
            ]
        ),
        num_cells,
    )

    # fluid data right
    fluid_data_right = _BufferedList(
        jnp.vstack(
            [
                refined_primitive_states_right,
                refined_center_right,
                refined_volume,
                refined_refinement_level,
            ]
        ),
        num_cells,
    )

    return fluid_data_left, fluid_data_right


@jax.jit
def refine(
    fluid_data: _BufferedList, refinement_tolerance: float, maximum_refinement: int
) -> _BufferedList:
    """
    Apply refinement to the fluid data until no further refinement
    is necessary given our refinement criterion.

    Args:
        fluid_data: The fluid data.
        refinement_tolerance: The refinement tolerance.
        maximum_refinement: The maximum refinement level.

    Returns:
        The refined fluid data.
    """

    # condition for continuing refinement
    def refinement_confition(fluid_data: _BufferedList) -> BoolScalar:
        refinement_mask = get_refinement_mask(
            fluid_data.buffer[0:3, :],
            fluid_data.buffer[3, :],
            fluid_data.buffer[5, :],
            maximum_refinement,
            refinement_tolerance,
            fluid_data.num_cells,
        )
        return jnp.any(refinement_mask)

    # refinement function
    def apply_refinement(fluid_data: _BufferedList) -> _BufferedList:
        fluid_data_left, fluid_data_right = calculate_refined_cells(fluid_data)
        refinement_mask = get_refinement_mask(
            fluid_data.buffer[0:3, :],
            fluid_data.buffer[3, :],
            fluid_data.buffer[5, :],
            maximum_refinement,
            refinement_tolerance,
            fluid_data.num_cells,
        )
        fluid_data = fluid_data.set_elements(fluid_data_left.buffer, refinement_mask)
        fluid_data = fluid_data.insert_elements(
            fluid_data_right.buffer, refinement_mask
        )
        fluid_data = fluid_data.reset_allocatable_elements()
        return fluid_data

    # refinement loop
    # for reverse mode AD, equinox.internal._loop.checkpointed.checkpointed_while_loop
    fluid_data = jax.lax.while_loop(refinement_confition, apply_refinement, fluid_data)

    return fluid_data


def get_derefinement_masks(
    primitive_states: Float[Array, "num_prims buffer_size"],
    center_buffer: Float[Array, "buffer_size"],
    refinement_levels: Float[Array, "buffer_size"],
    derefinement_tolerance: float,
    num_cells: int,
) -> Tuple[Bool[Array, "buffer_size"], Bool[Array, "buffer_size"]]:
    """
    Masks for derefinement.

    Args:
        primitive_states: The primitive states.
        center_buffer: The cell centers.
        refinement_levels: The refinement levels.
        derefinement_tolerance: The derefinement tolerance.
        num_cells: The number of cells

    Returns:
        A tuple of a mask indicating what cells should be overwritten and 
        one indicating what cells should be deleted
    """

    # derefinement criterion
    derefinement_criterion = jnp.zeros(primitive_states.shape[1], dtype=bool)

    # get the limited gradients
    average_gradients = _calculate_average_gradients(primitive_states, center_buffer)

    # calculate the density jump
    density_jump = jnp.abs(average_gradients[0])

    # calculate the pressure jump
    pressure_jump = jnp.abs(average_gradients[2])

    # calculate the velocity jump
    velocity_jump = jnp.abs(average_gradients[1])

    # calculate the density jump criterion
    density_jump_criterion = density_jump < derefinement_tolerance

    # calculate the pressure jump criterion
    pressure_jump_criterion = pressure_jump < derefinement_tolerance

    # calculate the velocity jump criterion
    velocity_jump_criterion = velocity_jump < derefinement_tolerance

    # calculate the refinement criterion
    derefinement_criterion_no_pad = jnp.logical_and(
        density_jump_criterion,
        jnp.logical_and(pressure_jump_criterion, velocity_jump_criterion),
    )

    derefinement_criterion = derefinement_criterion.at[1:-1].set(
        derefinement_criterion_no_pad
    )

    # ignore cells where the refinement level is already at the maximum
    derefinement_criterion = jnp.logical_and(
        derefinement_criterion, refinement_levels > 0
    )

    # do not refine the first and last cell
    derefinement_criterion = derefinement_criterion.at[0].set(False)
    mask = get_index_range_mask(primitive_states, 0, num_cells - 1)
    derefinement_criterion = jnp.where(mask, derefinement_criterion, False)

    # TODO: also balance derefinement criterion

    # we can only derefine cells where the neighbor has the same refinement level
    # and both fulfill the derefinement criterion
    derefinement_criterion = derefinement_criterion.at[0:-1].set(derefinement_criterion[1:] & derefinement_criterion[:-1])
    derefinement_criterion = derefinement_criterion.at[0:-1].set((refinement_levels[1:] == refinement_levels[:-1]) & derefinement_criterion[:-1])

    # in the overwriting mask, we want the left neighbors, which in our sorted
    # list will always be on even indices
    overwriting_mask = derefinement_criterion.at[1::2].set(False)

    # the deletion mask are the right neighbors, which we get by rolling the
    # overwriting mask by one
    deletion_mask = jnp.roll(overwriting_mask, 1)

    # print(overwriting_mask)

    return overwriting_mask, deletion_mask

@jax.jit
def calculate_derefined_cells(
    fluid_data: _BufferedList,
) -> _BufferedList:
    """
    Replaces the fluid quantities on every cell with the average
    of the current cell and its right neighbor.

    Args:
        fluid_data: The fluid data.

    Returns:
        The derefined fluid data.
    
    """
    buffer = fluid_data.buffer

    # the primitive state and centers are set to the average
    buffer = buffer.at[0:4, 0:-1].set((buffer[0:4, 0:-1] + buffer[0:4, 1:]) / 2)

    # the volume is set to the sum
    buffer = buffer.at[4, 0:-1].set(buffer[4, 0:-1] + buffer[4, 1:])

    # the refinement levels are subtracted by one
    buffer = buffer.at[5, 0:-1].set(buffer[5, 0:-1] - 1)

    return _BufferedList(buffer, fluid_data.num_cells)

@jax.jit
def derefine(
    fluid_data: _BufferedList, derefinement_tolerance: float
) -> _BufferedList:
    """
    Apply derefinement to the fluid data until no further derefinement
    is necessary given our derefinement criterion.

    Args:
        fluid_data: The fluid data.
        derefinement_tolerance: The derefinement tolerance.

    Returns:
        The derefined fluid data.
    """

    # no loop because we do not want to derefine multiple levels in one
    # timestep

    overwriting_mask, deletion_mask = get_derefinement_masks(
        fluid_data.buffer[0:3, :],
        fluid_data.buffer[3, :],
        fluid_data.buffer[5, :],
        derefinement_tolerance,
        fluid_data.num_cells,
    )

    fluid_data = fluid_data.set_elements(
        calculate_derefined_cells(fluid_data).buffer, overwriting_mask
    )

    fluid_data = fluid_data.remove_elements(deletion_mask)
    fluid_data = fluid_data.reset_allocatable_elements()
    return fluid_data

# -------------------------------------------------------------
# ==================== ↑ AMR Operations ↑ =====================
# -------------------------------------------------------------

# -------------------------------------------------------------
# =================== ↓ Time Integration ↓ ====================
# -------------------------------------------------------------


@jax.jit
def time_integration_fixed_stepsize(
    fluid_data: _BufferedList,
    dt: FloatScalar,
    num_steps: int,
    maximum_refinement: int,
    refinement_tolerance: float,
    derefinement_tolerance: float,
) -> _BufferedList:
    """
    A simple first order Godunov finite volume scheme for
    evolving the fluid. Here with a global fixed time step.

    Args:
        fluid_data: The fluid data at the initial time.
        dt: The time step.
        num_steps: The number of time steps.
        maximum_refinement: The maximum refinement level.
        refinement_tolerance: The refinement tolerance.

    Returns:
        The updated fluid data after num_steps time steps dt.
    """

    def step(_: int, fluid_data: _BufferedList) -> _BufferedList:

        # derefine the mesh
        fluid_data = derefine(fluid_data, derefinement_tolerance)

        # refine the mesh
        fluid_data = refine(fluid_data, refinement_tolerance, maximum_refinement)
        primitive_state = fluid_data.buffer[0:3, :]
        volume_buffer = fluid_data.buffer[4, :]

        # mind the boundaries
        primitive_state = _open_boundaries(primitive_state, fluid_data.num_cells)

        # calculate the fluxes
        fluxes = _hll_solver(primitive_state[:, :-1], primitive_state[:, 1:], 5 / 3)
        conserved_state = conserved_state_from_primitive(primitive_state, 5 / 3)
        conserved_state = conserved_state.at[:, 1:-1].add(
            -dt / volume_buffer[1:-1] * (fluxes[:, 1:] - fluxes[:, :-1])
        )
        primitive_state = primitive_state_from_conserved(conserved_state, 5 / 3)

        # update the fluid data
        fluid_data = _BufferedList(
            jnp.vstack([primitive_state, fluid_data.buffer[3:, :]]),
            fluid_data.num_cells,
        )

        # reset allocatable elements
        fluid_data = fluid_data.reset_allocatable_elements()

        return fluid_data

    fluid_data = jax.lax.fori_loop(0, num_steps, step, fluid_data)


    return fluid_data


# -------------------------------------------------------------
# =================== ↑ Time Integration ↑ ====================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ======================= ↓ Plotting ↓ ========================
# -------------------------------------------------------------


def plot_density(ax, fluid_data: _BufferedList) -> None:
    """
    Plots the fluid density.

    Args:
        ax: The axis.
        fluid_data: The fluid data.
    """

    primitive_state = fluid_data.buffer[0:3, 0 : fluid_data.num_cells]
    center_buffer = fluid_data.buffer[3, 0 : fluid_data.num_cells]
    cell_volumes = fluid_data.buffer[4, 0 : fluid_data.num_cells]

    # plot fluid density
    ax.plot(center_buffer, primitive_state[0], ".-", label="Density")

    # plot the cell boundaries, vlines at cell centers + volume/2
    ax.vlines(
        center_buffer + 0.5 * cell_volumes,
        0,
        1,
        color="black",
        # linestyle="dashed",
        alpha=0.1,
    )

    # labeling
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.set_title("simple adaptive mesh refinement")


def animation(
    fluid_data: _BufferedList,
    dt: FloatScalar,
    num_steps: int,
    maximum_refinement: int,
    refinement_tolerance: float,
    derefinement_tolerance: float,
) -> None:
    """
    A simple animation of the fluid density.

    Args:
        fluid_data: The initial state fluid data.
        dt: The time step.
        num_steps: The number of time steps.
        maximum_refinement: The maximum refinement level.
        refinement_tolerance: The refinement tolerance.
    """
    

    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def update(frame):
        ax.clear()
        nonlocal fluid_data
        fluid_data = time_integration_fixed_stepsize(
            fluid_data, dt, 1, maximum_refinement, refinement_tolerance, derefinement_tolerance
        )
        plot_density(ax, fluid_data)

    ani = FuncAnimation(fig, update, frames=num_steps, repeat=False, interval=1)
    ani.save("amr.gif")


# -------------------------------------------------------------
# ======================= ↑ Plotting ↑ ========================
# -------------------------------------------------------------


# -------------------------------------------------------------
# ================= ↓ Shock Tube Example ↓ ====================
# -------------------------------------------------------------

def shock_tube_example():
    """
    A simple shock tube example.
    """
    # create example initial fluid state
    shock_pos = 0.5
    N = 20
    r = jnp.linspace(0, 1, N)
    dx_coarse = 1 / (N - 1)
    rho = jnp.where(r < shock_pos, 1.0, 0.125)
    u = jnp.ones_like(r) * 0.0
    p = jnp.where(r < shock_pos, 1.0, 0.1)

    # create buffers
    buffer_size = 200
    fluid_buffer = jnp.full((3, buffer_size), 1.0)
    center_buffer = jnp.linspace(1, 2, buffer_size)
    volume_buffer = jnp.full((buffer_size,), 1.0)
    refinement_level_buffer = jnp.full((buffer_size,), 1.0)

    # fill buffers with initial fluid state
    fluid_buffer = fluid_buffer.at[:, 0:N].set(jnp.stack([rho, u, p], axis=0))
    center_buffer = center_buffer.at[0:N].set(r)
    volume_buffer = volume_buffer.at[0:N].set(dx_coarse)
    refinement_level_buffer = refinement_level_buffer.at[0:N].set(0)

    # create fluid data
    fluid_data = _BufferedList(
        jnp.vstack([fluid_buffer, center_buffer, volume_buffer, refinement_level_buffer]), N
    )

    derefinement_tolerance = 0.5
    refinement_tolerance = 5.0

    # run animation
    animation(fluid_data, 0.001, 200, 3, refinement_tolerance, derefinement_tolerance)

if __name__ == "__main__":
    shock_tube_example()

# -------------------------------------------------------------
# ================= ↑ Shock Tube Example ↑ ====================
# -------------------------------------------------------------