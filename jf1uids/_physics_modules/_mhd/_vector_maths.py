import jax
import jax.numpy as jnp


@jax.jit
def cross(a, b):
    result = jnp.zeros_like(a)
    result = result.at[0, ...].set(a[1, ...] * b[2, ...] - a[2, ...] * b[1, ...])
    result = result.at[1, ...].set(a[2, ...] * b[0, ...] - a[0, ...] * b[2, ...])
    result = result.at[2, ...].set(a[0, ...] * b[1, ...] - a[1, ...] * b[0, ...])
    return result


@jax.jit
def divergence2D(field, grid_spacing: float):
    """Calculate the divergence of a 3d field on a 2d grid.

    Args:
        field: The field to calculate the divergence of.
        grid_spacing: The width of the cells.

    Returns:
        The divergence of the field.
    """
    divergence = jnp.zeros_like(field)

    divergence = divergence.at[0, 1:-1, 1:-1].add((field[0, 2:, 1:-1] - field[0, :-2, 1:-1]) / (2 * grid_spacing))
    divergence = divergence.at[1, 1:-1, 1:-1].add((field[1, 1:-1, 2:] - field[1, 1:-1, :-2]) / (2 * grid_spacing))

    divergence = jnp.sum(divergence, axis = 0)

    return divergence


@jax.jit
def curl3D(field, grid_spacing: float):
    """Calculate the curl of a 3d field on a 3d grid.

    Args:
        field: The field to calculate the curl of.
        grid_spacing: The width of the cells.

    Returns:
        The curl of the field.
    """
    curl = jnp.zeros_like(field)

    curl = curl.at[0, :, 1:-1, :].add(0.5 * (field[2, :, 2:, :] - field[2, :, :-2, :]) / grid_spacing)
    curl = curl.at[0, :, :, 1:-1].add(-0.5 * (field[1, :, :, 2:] - field[1, :, :, :-2]) / grid_spacing)

    curl = curl.at[1, :, :, 1:-1].add(0.5 * (field[0, :, :, 2:] - field[0, :, :, :-2]) / grid_spacing)
    curl = curl.at[1, 1:-1, :, :].add(-0.5 * (field[2, 2:, :, :] - field[2, :-2, :, :]) / grid_spacing)

    curl = curl.at[2, 1:-1, :, :].add(0.5 * (field[1, 2:, :, :] - field[1, :-2, :, :]) / grid_spacing)
    curl = curl.at[2, :, 1:-1, :].add(-0.5 * (field[0, :, 2:, :] - field[0, :, :-2, :]) / grid_spacing)

    return curl


@jax.jit
def curl2D(field, grid_spacing: float):
    """Calculate the curl of a 3d field on a 2d grid.
    assumes grid_spacing = dy

    Args:
        field: The field to calculate the curl of.
        grid_spacing: The width of the cells.

    Returns:
        The curl of the field.
    """
    curl = jnp.zeros_like(field)

    curl = curl.at[0, 1:-1, 1:-1].add(0.5 * (field[2, 1:-1, 2:] - field[2, 1:-1, :-2]) / grid_spacing)
    curl = curl.at[1, 1:-1, 1:-1].add(-0.5 * (field[2, 2:, 1:-1] - field[2, :-2, 1:-1]) / grid_spacing)

    curl = curl.at[2, 1:-1, 1:-1].add(0.5 * (field[1, 2:, 1:-1] - field[1, :-2, 1:-1]) / grid_spacing)
    curl = curl.at[2, 1:-1, 1:-1].add(-0.5 * (field[0, 1:-1, 2:] - field[0, 1:-1, :-2]) / grid_spacing)

    return curl