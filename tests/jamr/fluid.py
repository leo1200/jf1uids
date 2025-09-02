from typing import Union

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

FloatScalar = float | Float[Array, ""]
IntScalar = int | Int[Array, ""]
BoolScalar = bool | Bool[Array, ""]


@jax.jit
def _minmod(a, b):
    return 0.5 * (jnp.sign(a) + jnp.sign(b)) * jnp.minimum(jnp.abs(a), jnp.abs(b))


@jax.jit
def total_energy_from_primitives(rho, u, p, gamma):
    """Calculate the total energy from the primitive variables.

    Args:
        rho: The density.
        u: The velocity.
        p: The pressure.
        gamma: The adiabatic index.

    Returns:
        The total energy.
    """

    return p / (gamma - 1) + 0.5 * rho * u**2


@jax.jit
def pressure_from_internal_energy(e, rho, gamma):
    """
    Calculate the pressure from the internal energy.

    Args:
        e: The internal energy.
        rho: The density.
        gamma: The adiabatic index.

    Returns:
        The pressure.
    """
    return (gamma - 1) * rho * e


@jax.jit
def internal_energy_from_energy(E, rho, u):
    """Calculate the internal energy from the total energy.

    Args:
        E: The total energy.
        rho: The density.
        u: The velocity.

    Returns:
        The internal energy.
    """
    return E / rho - 0.5 * u**2


@jax.jit
def pressure_from_energy(E, rho, u, gamma):
    """Calculate the pressure from the total energy.

    Args:
        E: The total energy.
        rho: The density.
        u: The velocity.
        gamma: The adiabatic index.

    Returns:
        The pressure.
    """

    e = internal_energy_from_energy(E, rho, u)
    return pressure_from_internal_energy(e, rho, gamma)


@jax.jit
def speed_of_sound(rho, p, gamma):
    """Calculate the speed of sound.

    Args:
        rho: The density.
        p: The pressure.
        gamma: The adiabatic index.

    Returns:
        The speed of sound.
    """

    # produces nans if p or rho are negative
    return jnp.sqrt(gamma * p / rho)


@jax.jit
def conserved_state_from_primitive(
    primitive_states: Float[Array, "num_vars num_cells"],
    gamma: Union[float, Float[Array, ""]],
) -> Float[Array, "num_vars num_cells"]:
    """Convert the primitive state to the conserved state.

    Args:
        primitive_state: The primitive state.
        gamma: The adiabatic index of the fluid.

    Returns:
        The conserved state.
    """

    rho, u, p = primitive_states
    E = total_energy_from_primitives(rho, u, p, gamma)
    return jnp.array([rho, rho * u, E])


@jax.jit
def primitive_state_from_conserved(
    conserved_state: Float[Array, "num_vars num_cells"],
    gamma: Union[float, Float[Array, ""]],
) -> Float[Array, "num_vars num_cells"]:
    """Convert the conserved state to the primitive state.

    Args:
        conserved_state: The conserved state.
        gamma: The adiabatic index of the fluid.

    Returns:
        The primitive state.
    """
    rho, m, E = conserved_state
    u = m / rho
    p = pressure_from_energy(E, rho, u, gamma)
    return jnp.stack([rho, u, p], axis=0)


@jax.jit
def _euler_flux(
    primitive_states: Float[Array, "num_vars num_cells"],
    gamma: Union[float, Float[Array, ""]],
) -> Float[Array, "num_vars num_cells"]:
    """Compute the Euler fluxes for the given primitive states.

    Args:
        primitive_states: The primitive state of the fluid on all cells.
        gamma: The adiabatic index of the fluid.

    Returns:
        The Euler fluxes for the given primitive states.

    """
    rho, u, p = primitive_states
    m = rho * u
    E = total_energy_from_primitives(rho, u, p, gamma)
    return jnp.stack([m, m * u + p, u * (E + p)], axis=0)


@jax.jit
def _calculate_limited_gradients(
    primitive_states: Float[Array, "num_vars num_cells"], r: Float[Array, "num_cells"]
) -> Float[Array, "num_vars num_cells-2"]:
    """
    Calculate the limited gradients of the primitive variables.

    Args:
        primitive_states: The primitive state array.
        dx: The cell width.
        geometry: The geometry of the domain.
        rv: The volumetric centers of the cells.

    Returns:
        The limited gradients of the primitive variables.

    """
    cell_distances_left = r[1:-1] - r[:-2]  # distances r_i - r_{i-1}
    cell_distances_right = r[2:] - r[1:-1]  # distances r_{i+1} - r_i

    # formulation 2:
    limited_gradients = _minmod(
        (primitive_states[:, 1:-1] - primitive_states[:, :-2]) / cell_distances_left,
        (primitive_states[:, 2:] - primitive_states[:, 1:-1]) / cell_distances_right,
    )

    return limited_gradients


@jax.jit
def _calculate_average_gradients(
    primitive_states: Float[Array, "num_vars num_cells"], r: Float[Array, "num_cells"]
) -> Float[Array, "num_vars num_cells-2"]:
    """
    Calculate the limited gradients of the primitive variables.

    Args:
        primitive_states: The primitive state array.
        dx: The cell width.
        geometry: The geometry of the domain.
        rv: The volumetric centers of the cells.

    Returns:
        The limited gradients of the primitive variables.

    """
    cell_distances_left = r[1:-1] - r[:-2]  # distances r_i - r_{i-1}
    cell_distances_right = r[2:] - r[1:-1]  # distances r_{i+1} - r_i

    # formulation 2:
    limited_gradients = 0.5 * (
        (primitive_states[:, 1:-1] - primitive_states[:, :-2]) / cell_distances_left
        + (primitive_states[:, 2:] - primitive_states[:, 1:-1]) / cell_distances_right
    )

    return limited_gradients


@jax.jit
def _hll_solver(
    primitives_left: Float[Array, "num_vars num_interfaces"],
    primitives_right: Float[Array, "num_vars num_interfaces"],
    gamma: Union[float, Float[Array, ""]],
) -> Float[Array, "num_vars num_interfaces"]:
    """
    Returns the conservative fluxes.

    Args:
        primitives_left: States left of the interfaces.
        primitives_right: States right of the interfaces.
        gamma: The adiabatic index.

    Returns:
        The conservative fluxes at the interfaces.

    """
    rho_L, u_L, p_L = primitives_left
    rho_R, u_R, p_R = primitives_right

    # calculate the sound speeds
    c_L = speed_of_sound(rho_L, p_L, gamma)
    c_R = speed_of_sound(rho_R, p_R, gamma)

    # get the left and right states and fluxes
    fluxes_left = _euler_flux(primitives_left, gamma)
    fluxes_right = _euler_flux(primitives_right, gamma)

    # very simple approach for the wave velocities
    wave_speeds_right_plus = jnp.maximum(jnp.maximum(u_L + c_L, u_R + c_R), 0)
    wave_speeds_left_minus = jnp.minimum(jnp.minimum(u_L - c_L, u_R - c_R), 0)

    # get the left and right conserved variables
    conservatives_left = conserved_state_from_primitive(primitives_left, gamma)
    conservatives_right = conserved_state_from_primitive(primitives_right, gamma)

    # calculate the interface HLL fluxes
    # F = (S_R * F_L - S_L * F_R + S_L * S_R * (U_R - U_L)) / (S_R - S_L)
    fluxes = (
        wave_speeds_right_plus * fluxes_left
        - wave_speeds_left_minus * fluxes_right
        + wave_speeds_left_minus
        * wave_speeds_right_plus
        * (conservatives_right - conservatives_left)
    ) / (wave_speeds_right_plus - wave_speeds_left_minus)

    return fluxes


@jax.jit
def _open_boundaries(
    primitive_states: Float[Array, "num_vars buffer_size"], num_cells: int
) -> Float[Array, "num_vars buffer_size"]:
    """
    Apply open boundary conditions.

    Args:
        primitive_states: The primitive state.

    Returns:
        The primitive state with open boundary conditions applied.

    """
    # left boundary
    primitive_states = primitive_states.at[:, 0].set(primitive_states[:, 1])

    # right boundary
    primitive_states = primitive_states.at[:, num_cells].set(
        primitive_states[:, num_cells - 1]
    )

    return primitive_states