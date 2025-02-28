"""
Fourier-based Poisson solver and simple source term handling
of self gravity. To be improved to an energy-conserving scheme.
"""

# general
from functools import partial
import jax.numpy as jnp
import jax

# typing
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker
from typing import Union

# fft, in the future use
# https://github.com/DifferentiableUniverseInitiative/JaxPM
from jax.numpy.fft import fftn, ifftn

# jf1uids data classes
from jf1uids._stencil_operations._stencil_operations import _stencil_add
from jf1uids.data_classes.simulation_helper_data import HelperData
from jf1uids.fluid_equations.registered_variables import RegisteredVariables
from jf1uids.option_classes.simulation_config import STATE_TYPE_ALTERED, SimulationConfig

# jf1uids constants
from jf1uids.option_classes.simulation_config import FIELD_TYPE, HLL, HLLC, OPEN_BOUNDARY, PERIODIC_BOUNDARY, STATE_TYPE

# jf1uids functions
from jf1uids._geometry.boundaries import _boundary_handler
from jf1uids._riemann_solver.hll import _hll_solver, _hllc_solver
from jf1uids._state_evolution.reconstruction import _reconstruct_at_interface
from jf1uids.fluid_equations.euler import _euler_flux
from jf1uids.fluid_equations.fluid import conserved_state_from_primitive, primitive_state_from_conserved


@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['grid_spacing', 'config'])
def _compute_gravitational_potential(
    gas_density: FIELD_TYPE,
    grid_spacing: float,
    config: SimulationConfig,
    G: Union[float, Float[Array, ""]] = 1.0
) -> FIELD_TYPE:
    """
    Compute the gravitational potential using FFT to solve Poisson's equation for
    periodic and open boundaries (via the Hockney & Eastwood method).

    Args:
        gas_density: The gas density field.
        grid_spacing: The grid spacing.
        config: The simulation configuration.
        G: The gravitational constant.

    Returns:
        The gravitational potential.

    """

    # TODO: remove ghost cells in this computations (?)

    dimensionality = config.dimensionality

    gas_density = gas_density - jnp.mean(gas_density)

    # we only use outflow if not all boundaries are periodic
    # SO THERES EITHER ALL PERIODIC OR NONE
    # TODO: improve

    non_periodic_boundaries = False

    if dimensionality == 1:
        if not (
            config.boundary_settings.left_boundary == PERIODIC_BOUNDARY and
            config.boundary_settings.right_boundary == PERIODIC_BOUNDARY
        ):
            non_periodic_boundaries = True
    elif dimensionality == 2:
        if not (
            config.boundary_settings.x.left_boundary == PERIODIC_BOUNDARY and
            config.boundary_settings.x.right_boundary == PERIODIC_BOUNDARY and
            config.boundary_settings.y.left_boundary == PERIODIC_BOUNDARY and
            config.boundary_settings.y.right_boundary == PERIODIC_BOUNDARY
        ):
            non_periodic_boundaries = True
    elif dimensionality == 3:
        if not (
            config.boundary_settings.x.left_boundary == PERIODIC_BOUNDARY and
            config.boundary_settings.x.right_boundary == PERIODIC_BOUNDARY and
            config.boundary_settings.y.left_boundary == PERIODIC_BOUNDARY and
            config.boundary_settings.y.right_boundary == PERIODIC_BOUNDARY and
            config.boundary_settings.z.left_boundary == PERIODIC_BOUNDARY and
            config.boundary_settings.z.right_boundary == PERIODIC_BOUNDARY
        ):
            non_periodic_boundaries = True

    if not non_periodic_boundaries:
        # -----------------------------
        # Periodic boundaries version
        # -----------------------------
        # Compute FFT of the density field.
        density_k = fftn(gas_density)

        # Build the k–vector (note that fftfreq returns cycles/unit; multiply by 2π).
        num_cells = gas_density.shape[0]
        k_base = jnp.fft.fftfreq(num_cells, d=grid_spacing) * 2 * jnp.pi

        if dimensionality == 1:
            k = k_base  # 1D case.
            k_squared = k ** 2
        elif dimensionality == 2:
            kx, ky = jnp.meshgrid(k_base, k_base, indexing="ij")
            k_squared = kx**2 + ky**2
        elif dimensionality == 3:
            kx, ky, kz = jnp.meshgrid(k_base, k_base, k_base, indexing="ij")
            k_squared = kx**2 + ky**2 + kz**2

        # Avoid division by zero (k = 0 mode).
        k_squared = jnp.where(k_squared == 0, 1e-12, k_squared)
        greens_function = jnp.where(k_squared > 1e-12, -4 * jnp.pi * G / k_squared, -1/(4 * jnp.pi))

        # Multiply in Fourier space and invert.
        potential_k = greens_function * density_k
        gravitational_potential = jnp.real(ifftn(potential_k))
        return gravitational_potential  * grid_spacing ** dimensionality

    else:
        # ----------------------------------------------------
        # Open boundaries version via Hockney & Eastwood method
        # ----------------------------------------------------
        #
        # (a) Extend the domain to twice the size in each dimension.
        original_shape = gas_density.shape
        extended_shape = tuple(2 * s for s in original_shape)

        # Embed the original density in the (0,...,0) corner of the extended grid.
        extended_density = jnp.zeros(extended_shape, dtype=gas_density.dtype)
        slices = tuple(slice(0, s) for s in original_shape)
        extended_density = extended_density.at[slices].set(gas_density)

        # (b) Construct the Green's function on the extended grid.
        #
        # The Hockney–Eastwood prescription is to compute, for each dimension,
        #     pos = [0, 1, 2, ..., n-1, 2n - n, ..., 1] * grid_spacing,
        # which is equivalent to:
        #
        #    pos = np.arange(2*n);  pos = where(pos < n, pos, 2*n - pos)
        #
        # This yields the “minimum–image” distances from the source placed at the origin.
        grids = []
        for s in original_shape:
            n = s
            extended_n = 2 * n
            pos = jnp.arange(extended_n)
            pos = jnp.where(pos < n, pos, 2 * n - pos)
            pos = pos * grid_spacing
            grids.append(pos)

        # Now create the distance array r on the extended grid.
        if dimensionality == 1:
            r = grids[0]  # Already nonnegative.
        elif dimensionality == 2:
            x, y = jnp.meshgrid(grids[0], grids[1], indexing="ij")
            r = jnp.sqrt(x**2 + y**2)
        elif dimensionality == 3:
            x, y, z = jnp.meshgrid(grids[0], grids[1], grids[2], indexing="ij")
            r = jnp.sqrt(x**2 + y**2 + z**2)

        # Replace any zero distance with grid_spacing (to avoid singularity).
        r_safe = jnp.where(r == 0, grid_spacing, r)

        # (c) Define the isolated (open–boundary) Green's function.
        if dimensionality == 1:
            # For 1D (solving φ'' = 4πGδ(x)), the solution is φ = -2πG |x|
            kernel = -2 * jnp.pi * G * r
        elif dimensionality == 2:
            # In 2D, φ = -2G log(r)  (up to an additive constant).
            kernel = -2 * G * jnp.log(r_safe)
        elif dimensionality == 3:
            # In 3D, the isolated potential is φ = -G/r.
            kernel = -G / r_safe

        # (d) FFT–convolve: Multiply the FFTs of the extended density and the Green's function.
        density_k_ext = fftn(extended_density)
        kernel_k_ext = fftn(kernel)
        potential_ext = jnp.real(ifftn(density_k_ext * kernel_k_ext))

        # (e) Extract the portion of the potential corresponding to the original grid.
        gravitational_potential = potential_ext[slices]
        return gravitational_potential * grid_spacing ** dimensionality

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['axis', 'grid_spacing', 'registered_variables', 'config'])
def _gravitational_source_term_along_axis(
        gravitational_potential: FIELD_TYPE,
        primitive_state: STATE_TYPE,
        grid_spacing: float,
        registered_variables: RegisteredVariables,
        dt: Union[float, Float[Array, ""]],
        gamma: Union[float, Float[Array, ""]],
        config: SimulationConfig,
        helper_data: HelperData,
        axis: int,
) -> STATE_TYPE:
    
    """
    Compute the source term for the self-gravity solver along a single axis.
    Currently, simply density * gravitational_acceleration for the momentum 
    and density * velocity * gravitational_acceleration for the energy.

    Args:
        gravitational_potential: The gravitational potential.
        primitive_state: The primitive state.
        grid_spacing: The grid spacing.
        registered_variables: The registered variables.
        dt: The time step.
        gamma: The adiabatic index.
        config: The simulation configuration.
        helper_data: The helper data.
        axis: The axis along which to compute the source term.

    Returns:
        The source term.
    
    """

    rho = primitive_state[registered_variables.density_index]
    v_axis = primitive_state[axis]

    # a_i = - (phi_{i+1} - phi_{i-1}) / (2 * dx)
    acceleration = -_stencil_add(gravitational_potential, 1, -1, axis - 1, factorB = -1.0) / (2 * grid_spacing)
    # it is axis - 1 because the axis is 1-indexed as usually the zeroth axis are the different
    # fields in the state vector not the spatial dimensions, but here we only have the spatial dimensions

    source_term = jnp.zeros_like(primitive_state)

    # set momentum source
    source_term = source_term.at[axis].set(primitive_state[registered_variables.density_index] * acceleration)

    # set energy source
    source_term = source_term.at[registered_variables.pressure_index].set(rho * v_axis * acceleration)

    # ===============================================

    # attempt at implementing the ATHENA source term
    # https://github.com/PrincetonUniversity/athena/blob/master/src/hydro/srcterms/self_gravity.cpp

    # num_cells = primitive_state.shape[axis]

    # primitive_state_left, primitive_state_right = _reconstruct_at_interface(primitive_state, dt, gamma, config, helper_data, registered_variables, axis)

    # # these are now approximate fluxes, starting at the flux between cell 1 and 2
    # # fluxesX = ((fluxes_left + fluxes_right) / 2)[0]
    # fluxes = _hllc_solver(primitive_state_left, primitive_state_right, gamma, config, registered_variables, axis)[0]

    # # these are the accelerations at the cell interfaces, starting at the interface between cell 1 and 2
    # acc = -(jax.lax.slice_in_dim(gravitational_potential, 2, num_cells - 1, axis = axis - 1) - jax.lax.slice_in_dim(gravitational_potential, 1, num_cells - 2, axis = axis - 1)) / (grid_spacing)

    # sources = 0.5 * (jax.lax.slice_in_dim(fluxes, 0, -1, axis = axis - 1) * jax.lax.slice_in_dim(acc, 0, -1, axis = axis - 1) + jax.lax.slice_in_dim(fluxes, 1, None, axis = axis - 1) * jax.lax.slice_in_dim(acc, 1, None, axis = axis - 1))

    # selection2 = (slice(None),) * (axis - 1) + (slice(2,-2),) + (slice(None),)*(primitive_state.ndim - axis - 2)

    # source_term = source_term.at[(registered_variables.pressure_index,) + selection2].set(sources)

    # ===============================================

    return source_term

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'registered_variables'])
def _apply_self_gravity(
    primitive_state: STATE_TYPE,
    old_primitive_state: STATE_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    helper_data: HelperData,
    gamma: Union[float, Float[Array, ""]],
    gravitational_constant: Union[float, Float[Array, ""]],
    dt: Union[float, Float[Array, ""]]
) -> STATE_TYPE:

    rho = old_primitive_state[registered_variables.density_index]

    potential = _compute_gravitational_potential(rho, config.grid_spacing, config, gravitational_constant)

    source_term = jnp.zeros_like(primitive_state)

    for i in range(config.dimensionality):
        source_term = source_term + _gravitational_source_term_along_axis(
                                        potential,
                                        old_primitive_state,
                                        config.grid_spacing,
                                        registered_variables,
                                        dt,
                                        gamma,
                                        config,
                                        helper_data,
                                        i + 1
                                    )

    conserved_state = conserved_state_from_primitive(primitive_state, gamma, config, registered_variables)

    conserved_state = conserved_state + dt * source_term

    primitive_state = primitive_state_from_conserved(conserved_state, gamma, config, registered_variables)

    primitive_state = _boundary_handler(primitive_state, config)

    return primitive_state


# -------------------------------------------------------------
# ================= ↓ not yet finished stuff ↓ ================
# -------------------------------------------------------------


# attempt at a first proof of concept implementation of
# https://arxiv.org/abs/2012.01340
# currently not working

# @jaxtyped(typechecker=typechecker)
# @partial(jax.jit, static_argnames=['axis', 'grid_spacing', 'registered_variables', 'config'])
# def _mullen_source_along_axis(
#         gravitational_potential_zero: FIELD_TYPE,
#         gravitational_potential_one: FIELD_TYPE,
#         primitive_state_zero: STATE_TYPE,
#         grid_spacing: float,
#         dt: Union[float, Float[Array, ""]],
#         gamma: Union[float, Float[Array, ""]],
#         helper_data: HelperData,
#         config: SimulationConfig,
#         registered_variables: RegisteredVariables,
#         axis: int,
# ) -> STATE_TYPE:

#     num_cells = primitive_state_zero.shape[axis]

#     selection = (slice(None),) * (axis - 1) + (slice(1,-1),) + (slice(None),)*(primitive_state_zero.ndim - axis - 2)

#     acceleration_zero_left = jnp.zeros_like(gravitational_potential_zero)
#     acceleration_zero_left = -acceleration_zero_left.at[selection].set((jax.lax.slice_in_dim(gravitational_potential_zero, 1, num_cells - 1, axis = axis - 1) - jax.lax.slice_in_dim(gravitational_potential_zero, 0, num_cells - 2, axis = axis - 1)) / (grid_spacing))

#     acceleration_one_left = jnp.zeros_like(gravitational_potential_one)
#     acceleration_one_left = -acceleration_one_left.at[selection].set((jax.lax.slice_in_dim(gravitational_potential_one, 1, num_cells - 1, axis = axis - 1) - jax.lax.slice_in_dim(gravitational_potential_one, 0, num_cells - 2, axis = axis - 1)) / (grid_spacing))

#     mean_acceleration_left = (acceleration_zero_left + acceleration_one_left) / 2

#     acceleration_zero_right = jnp.zeros_like(gravitational_potential_zero)
#     acceleration_zero_right = -acceleration_zero_right.at[selection].set((jax.lax.slice_in_dim(gravitational_potential_zero, 2, num_cells, axis = axis - 1) - jax.lax.slice_in_dim(gravitational_potential_zero, 1, num_cells - 1, axis = axis - 1)) / (grid_spacing))

#     acceleration_one_right = jnp.zeros_like(gravitational_potential_one)
#     acceleration_one_right = -acceleration_one_right.at[selection].set((jax.lax.slice_in_dim(gravitational_potential_one, 2, num_cells, axis = axis - 1) - jax.lax.slice_in_dim(gravitational_potential_one, 1, num_cells - 1, axis = axis - 1)) / (grid_spacing))

#     mean_acceleration_right = (acceleration_zero_right + acceleration_one_right) / 2

#     source_term = jnp.zeros_like(primitive_state_zero)

#     source_term = source_term.at[axis].set(primitive_state_zero[registered_variables.density_index] * (acceleration_zero_left + acceleration_zero_right) / 2)

#     # ===============================================

#     num_cells = primitive_state_zero.shape[axis]

#     if config.first_order_fallback:
#         primitive_state_left = jax.lax.slice_in_dim(primitive_state_zero, 1, num_cells - 2, axis = axis)
#         primitive_state_right = jax.lax.slice_in_dim(primitive_state_zero, 2, num_cells - 1, axis = axis)
#     else:
#         primitive_state_left, primitive_state_right = _reconstruct_at_interface(primitive_state_zero, dt, gamma, config, helper_data, registered_variables, axis)

#     if config.riemann_solver == HLL:
#         fluxes = _hll_solver(primitive_state_left, primitive_state_right, gamma, config, registered_variables, axis)
#     elif config.riemann_solver == HLLC:
#         fluxes = _hllc_solver(primitive_state_left, primitive_state_right, gamma, config, registered_variables, axis)
#     else:
#         raise ValueError("Riemann solver not supported.")

#     flux_length = fluxes.shape[axis]

#     selection = (slice(None),) * (axis - 1) + (slice(2,-2),) + (slice(None),)*(primitive_state_zero.ndim - axis - 2)

#     fluxes_right = jnp.zeros_like(mean_acceleration_right)
#     fluxes_right = fluxes_right.at[selection].set(jax.lax.slice_in_dim(fluxes, 1, flux_length, axis = axis)[0])

#     fluxes_left = jnp.zeros_like(mean_acceleration_left)
#     fluxes_left = fluxes_left.at[selection].set(jax.lax.slice_in_dim(fluxes, 0, flux_length - 1, axis = axis)[0])

#     # ===============================================

#     source_term = source_term.at[registered_variables.pressure_index].set((fluxes_left * mean_acceleration_left + fluxes_right * mean_acceleration_right) / 2)

#     return source_term

# @jaxtyped(typechecker=typechecker)
# @partial(jax.jit, static_argnames=['axis', 'grid_spacing', 'registered_variables', 'config'])
# def _mullen_source_along_axis2(
#         gravitational_potential_zero: FIELD_TYPE,
#         gravitational_potential_one: FIELD_TYPE,
#         gravitational_potential_two: FIELD_TYPE,
#         primitive_state_one: STATE_TYPE,
#         grid_spacing: float,
#         dt: Union[float, Float[Array, ""]],
#         gamma: Union[float, Float[Array, ""]],
#         helper_data: HelperData,
#         config: SimulationConfig,
#         registered_variables: RegisteredVariables,
#         axis: int,
# ) -> STATE_TYPE:

#     num_cells = primitive_state_one.shape[axis]

#     selection = (slice(None),) * (axis - 1) + (slice(1,-1),) + (slice(None),)*(primitive_state_one.ndim - axis - 2)

#     acceleration_zero_left = jnp.zeros_like(gravitational_potential_zero)
#     acceleration_zero_left = -acceleration_zero_left.at[selection].set((jax.lax.slice_in_dim(gravitational_potential_zero, 1, num_cells - 1, axis = axis - 1) - jax.lax.slice_in_dim(gravitational_potential_zero, 0, num_cells - 2, axis = axis - 1)) / (grid_spacing))

#     acceleration_one_left = jnp.zeros_like(gravitational_potential_one)
#     acceleration_one_left = -acceleration_one_left.at[selection].set((jax.lax.slice_in_dim(gravitational_potential_one, 1, num_cells - 1, axis = axis - 1) - jax.lax.slice_in_dim(gravitational_potential_one, 0, num_cells - 2, axis = axis - 1)) / (grid_spacing))

#     acceleration_two_left = jnp.zeros_like(gravitational_potential_two)
#     acceleration_two_left = -acceleration_two_left.at[selection].set((jax.lax.slice_in_dim(gravitational_potential_two, 1, num_cells - 1, axis = axis - 1) - jax.lax.slice_in_dim(gravitational_potential_two, 0, num_cells - 2, axis = axis - 1)) / (grid_spacing))

#     acceleration_zero_right = jnp.zeros_like(gravitational_potential_zero)
#     acceleration_zero_right = -acceleration_zero_right.at[selection].set((jax.lax.slice_in_dim(gravitational_potential_zero, 2, num_cells, axis = axis - 1) - jax.lax.slice_in_dim(gravitational_potential_zero, 1, num_cells - 1, axis = axis - 1)) / (grid_spacing))

#     acceleration_one_right = jnp.zeros_like(gravitational_potential_one)
#     acceleration_one_right = -acceleration_one_right.at[selection].set((jax.lax.slice_in_dim(gravitational_potential_one, 2, num_cells, axis = axis - 1) - jax.lax.slice_in_dim(gravitational_potential_one, 1, num_cells - 1, axis = axis - 1)) / (grid_spacing))

#     acceleration_two_right = jnp.zeros_like(gravitational_potential_two)
#     acceleration_two_right = -acceleration_two_right.at[selection].set((jax.lax.slice_in_dim(gravitational_potential_two, 2, num_cells, axis = axis - 1) - jax.lax.slice_in_dim(gravitational_potential_two, 1, num_cells - 1, axis = axis - 1)) / (grid_spacing))

#     source_term = jnp.zeros_like(primitive_state_one)

#     source_term = source_term.at[axis].set(primitive_state_one[registered_variables.density_index] * (acceleration_one_left + acceleration_one_right) / 2)

#     mean_acceleration_left = (acceleration_zero_left + acceleration_two_left) / 2
#     mean_acceleration_right = (acceleration_zero_right + acceleration_two_right) / 2

#     # ===============================================

#     num_cells = primitive_state_one.shape[axis]

#     if config.first_order_fallback:
#         primitive_state_left = jax.lax.slice_in_dim(primitive_state_one, 1, num_cells - 2, axis = axis)
#         primitive_state_right = jax.lax.slice_in_dim(primitive_state_one, 2, num_cells - 1, axis = axis)
#     else:
#         primitive_state_left, primitive_state_right = _reconstruct_at_interface(primitive_state_one, dt, gamma, config, helper_data, registered_variables, axis)

#     if config.riemann_solver == HLL:
#         fluxes = _hll_solver(primitive_state_left, primitive_state_right, gamma, config, registered_variables, axis)
#     elif config.riemann_solver == HLLC:
#         fluxes = _hllc_solver(primitive_state_left, primitive_state_right, gamma, config, registered_variables, axis)
#     else:
#         raise ValueError("Riemann solver not supported.")

#     flux_length = fluxes.shape[axis]

#     selection = (slice(None),) * (axis - 1) + (slice(2,-2),) + (slice(None),)*(primitive_state_one.ndim - axis - 2)

#     fluxes_right = jnp.zeros_like(mean_acceleration_right)
#     fluxes_right = fluxes_right.at[selection].set(jax.lax.slice_in_dim(fluxes, 1, flux_length, axis = axis)[0])

#     fluxes_left = jnp.zeros_like(mean_acceleration_left)
#     fluxes_left = fluxes_left.at[selection].set(jax.lax.slice_in_dim(fluxes, 0, flux_length - 1, axis = axis)[0])

#     # ===============================================

#     source_term = source_term.at[registered_variables.pressure_index].set((fluxes_left * mean_acceleration_left + fluxes_right * mean_acceleration_right) / 2)

#     return source_term


# further debugging code

    # jax.debug.print("flux shape: {fs}, min flux right: {mifs}, max_flux_right: {mafa}, min_acc_left: {mial}, max_acc_left: {maal}, min source term: {mist}, max source term: {mast}", fs = fluxes.shape, mifs = jnp.min(fluxes_right), mafa = jnp.max(fluxes_right), mial = jnp.min(acceleration_left), maal = jnp.max(acceleration_left), mist = jnp.min(source_term), mast = jnp.max(source_term))

    # print the deviation between the two source term calculations
    # source1 = source_term[registered_variables.pressure_index]
    # source1 = source1[2:-2, 2:-2, 2:-2]
    # source2 = rho * v_axis * acceleration
    # source2 = source2[2:-2, 2:-2, 2:-2]
    # deviation = jnp.sum(jnp.abs(source1 - source2))
    # max_deviation_index = jnp.argmax(jnp.abs(source1 - source2))
    # max_deviation_index = jnp.unravel_index(max_deviation_index, source1.shape)
    # jax.debug.print("summed deviation: {d}, index: {i}, source1: {s1}, source2: {s2}", d = deviation, i = max_deviation_index, s1 = source1[max_deviation_index], s2 = source2[max_deviation_index])
    # def plot_source_terms(source_term1, source_term2, pressure):
    #     import matplotlib.pyplot as plt
    #     from mpl_toolkits.axes_grid1 import make_axes_locatable
    #     from matplotlib.colors import LogNorm

    #     min_val = jnp.minimum(jnp.min(source_term1), jnp.min(source_term2))
    #     max_val = jnp.maximum(jnp.max(source_term1), jnp.max(source_term2))

    #     fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    #     im1 = axs[0].imshow(source_term1[:, :, 32], origin='lower') # , vmin=min_val, vmax=max_val)
    #     axs[0].set_title("riemann flux based source term (problem)")
    #     divider1 = make_axes_locatable(axs[0])
    #     cax1 = divider1.append_axes('right', size='5%', pad=0.05)
    #     fig.colorbar(im1, cax=cax1, orientation='vertical')

    #     im2 = axs[1].imshow(source_term2[:, :, 32], origin='lower') # , vmin=min_val, vmax=max_val)
    #     axs[1].set_title("rho * v * acceleration based source term (works)")
    #     divider2 = make_axes_locatable(axs[1])
    #     cax2 = divider2.append_axes('right', size='5%', pad=0.05)
    #     fig.colorbar(im2, cax=cax2, orientation='vertical')

    #     # plot the pressure field with log color scale
    #     im3 = axs[2].imshow(pressure, origin='lower', norm=LogNorm())
    #     axs[2].set_title("pressure field")
    #     divider3 = make_axes_locatable(axs[2])
    #     cax3 = divider3.append_axes('right', size='5%', pad=0.05)
    #     fig.colorbar(im3, cax=cax3, orientation='vertical')


    #     plt.tight_layout()
    #     plt.savefig("source_term_comparison{axis}.png".format(axis = axis))

    #     plt.close()


    # jax.debug.callback(plot_source_terms, source1, source2, primitive_state[registered_variables.pressure_index][:, :, 32])


# -------------------------------------------------------------
# ================= ↑ not yet finished stuff ↑ ================
# -------------------------------------------------------------