from functools import partial
import jax.numpy as jnp
import jax

from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker
from typing import Union

from jax.numpy.fft import fftn, ifftn

from jf1uids._geometry.boundaries import _boundary_handler
from jf1uids.fluid_equations.fluid import conserved_state_from_primitive, primitive_state_from_conserved
from jf1uids.fluid_equations.registered_variables import RegisteredVariables
from jf1uids.option_classes.simulation_config import FIELD_TYPE, OPEN_BOUNDARY, PERIODIC_BOUNDARY, STATE_TYPE, SimulationConfig


# Currently a simple source term handling of self gravity.
# For future inspiration, see e.g.
# https://arxiv.org/abs/2012.01340
# and for a multigrid method see
# https://arxiv.org/abs/2306.05332

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['grid_spacing', 'config'])
def _compute_gravitational_potential(
    gas_density: FIELD_TYPE,
    grid_spacing: float,
    config: SimulationConfig,
    G: Union[float, Float[Array, ""]] = 1.0,
    background_density: float = 0.0
) -> FIELD_TYPE:
    """
    Compute the gravitational potential using FFT to solve Poisson's equation.
    """

    # TODO: remove ghost cells in this computations (?)

    dimensionality = config.dimensionality

    # we only use outflow if not all boundaries are periodic
    # SO THERES EITHER ALL PERIODIC OR NONE
    # TODO: improve

    non_periodic_boundaries = False

    if dimensionality == 1:
        if not (config.boundary_settings.left_boundary == PERIODIC_BOUNDARY and config.boundary_settings.right_boundary == PERIODIC_BOUNDARY):
            non_periodic_boundaries = True
    elif dimensionality == 2:
        if not (config.boundary_settings.x.left_boundary == PERIODIC_BOUNDARY and config.boundary_settings.x.right_boundary == PERIODIC_BOUNDARY and config.boundary_settings.y.left_boundary == PERIODIC_BOUNDARY and config.boundary_settings.y.right_boundary == PERIODIC_BOUNDARY):
            non_periodic_boundaries = True
    elif dimensionality == 3:
        if not (config.boundary_settings.x.left_boundary == PERIODIC_BOUNDARY and config.boundary_settings.x.right_boundary == PERIODIC_BOUNDARY and config.boundary_settings.y.left_boundary == PERIODIC_BOUNDARY and config.boundary_settings.y.right_boundary == PERIODIC_BOUNDARY and config.boundary_settings.z.left_boundary == PERIODIC_BOUNDARY and config.boundary_settings.z.right_boundary == PERIODIC_BOUNDARY):
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
        extended_density = jnp.zeros(extended_shape, dtype=gas_density.dtype) + background_density
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
@partial(jax.jit, static_argnames=['axis', 'grid_spacing', 'registered_variables'])
def _conservative_gravitational_source_term_along_axis(
        gravitational_potential: FIELD_TYPE,
        primitive_state: STATE_TYPE,
        grid_spacing: float,
        registered_variables: RegisteredVariables,
        axis: int,
) -> STATE_TYPE:

    num_cells = primitive_state.shape[axis]

    rho = primitive_state[registered_variables.density_index]
    v_axis = primitive_state[axis]
    
    acceleration = jnp.zeros_like(gravitational_potential)
    selection = (slice(None),) * (axis - 1) + (slice(1,-1),) + (slice(None),)*(primitive_state.ndim - axis - 2)
    acceleration = -acceleration.at[selection].set((jax.lax.slice_in_dim(gravitational_potential, 2, num_cells, axis = axis - 1) - jax.lax.slice_in_dim(gravitational_potential, 0, num_cells - 2, axis = axis - 1)) / (2 * grid_spacing))

    source_term = jnp.zeros_like(primitive_state)

    source_term = source_term.at[axis].set(primitive_state[registered_variables.density_index])

    source_term = source_term.at[registered_variables.pressure_index].set(rho * v_axis)

    source_term = source_term * acceleration

    return source_term


@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'registered_variables'])
def _apply_self_gravity(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    gamma: Union[float, Float[Array, ""]],
    gravitational_constant: Union[float, Float[Array, ""]],
    dt: Union[float, Float[Array, ""]]
) -> STATE_TYPE:

    rho = primitive_state[registered_variables.density_index]

    potential = _compute_gravitational_potential(rho, config.grid_spacing, config, gravitational_constant)

    source_term = jnp.zeros_like(primitive_state)

    for i in range(config.dimensionality):
        source_term = source_term + _conservative_gravitational_source_term_along_axis(potential, primitive_state, config.grid_spacing, registered_variables, i + 1)

    conserved_state = conserved_state_from_primitive(primitive_state, gamma, config, registered_variables)

    conserved_state = conserved_state + dt * source_term

    primitive_state = primitive_state_from_conserved(conserved_state, gamma, config, registered_variables)

    primitive_state = _boundary_handler(primitive_state, config)

    return primitive_state