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
from typing import Tuple, Union

# fft, in the future use
# https://github.com/DifferentiableUniverseInitiative/JaxPM
from jax.numpy.fft import fftn, ifftn

# jf1uids data classes
from jf1uids._physics_modules._cosmic_rays.cr_fluid_equations import speed_of_sound_crs
from jf1uids._riemann_solver._riemann_solver import _riemann_solver
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
from jf1uids.fluid_equations.fluid import conserved_state_from_primitive, primitive_state_from_conserved, speed_of_sound


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

    #                            TODO: check on this / Jeans swindle
    gas_density = gas_density # - jnp.mean(gas_density)

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
    # v_axis = primitive_state[axis]

    # a_i = - (phi_{i+1} - phi_{i-1}) / (2 * dx)
    acceleration = -_stencil_add(gravitational_potential, indices = (1, -1), factors = (1.0, -1.0), axis = axis - 1) / (2 * grid_spacing)
    # it is axis - 1 because the axis is 1-indexed as usually the zeroth axis are the different
    # fields in the state vector not the spatial dimensions, but here we only have the spatial dimensions

    source_term = jnp.zeros_like(primitive_state)

    # set momentum source
    source_term = source_term.at[axis].set(rho * acceleration)

    # set energy source
    # source_term = source_term.at[registered_variables.pressure_index].set(rho * v_axis * acceleration)

    # ===============================================

    # better energy source

    num_cells = primitive_state.shape[axis]

    primitive_state_left, primitive_state_right = _reconstruct_at_interface(primitive_state, dt, gamma, config, helper_data, registered_variables, axis)
    
    fluxes = _riemann_solver(primitive_state_left, primitive_state_right, gamma, config, registered_variables, axis)
    fluxes_i_to_ip1 = jnp.maximum(fluxes, 0)
    fluxes_ip1_to_i = jnp.minimum(fluxes, 0)

    # these are the accelerations at the cell interfaces, starting at the interface between cell 1 and 2
    acc = -(jax.lax.slice_in_dim(gravitational_potential, 2, num_cells - 1, axis = axis - 1) - jax.lax.slice_in_dim(gravitational_potential, 1, num_cells - 2, axis = axis - 1)) / (grid_spacing)

    fluxes_acc = jnp.zeros_like(primitive_state)
    selection = (slice(None),) * (axis) + (slice(2,-2),) + (slice(None),)*(primitive_state.ndim - axis - 1)
    fluxes_acc = fluxes_acc.at[selection].set(jax.lax.slice_in_dim(fluxes_i_to_ip1, 1, None, axis = axis) * jax.lax.slice_in_dim(acc, 1, None, axis = axis - 1) + jax.lax.slice_in_dim(fluxes_ip1_to_i, 0, -1, axis = axis) * jax.lax.slice_in_dim(acc, 0, -1, axis = axis - 1))

    source_term = source_term.at[registered_variables.pressure_index].set(fluxes_acc[0])

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