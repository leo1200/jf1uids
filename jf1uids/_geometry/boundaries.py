import jax.numpy as jnp
from functools import partial
import jax

from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

OPEN_BOUNDARY = 0
REFLECTIVE_BOUNDARY = 1

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['left_boundary', 'right_boundary'])
def _boundary_handler(primitive_states: Float[Array, "num_vars num_cells"], left_boundary: int, right_boundary: int) -> Float[Array, "num_vars num_cells"]:
    """Apply the boundary conditions to the primitive states.

    Args:
        primitive_states: The primitive state array.
        left_boundary: The left boundary condition.
        right_boundary: The right boundary condition.

    Returns:
        The primitive state array with the boundary conditions applied.
    """
    if left_boundary == OPEN_BOUNDARY:
        primitive_states = _open_left_boundary(primitive_states)
    elif left_boundary == REFLECTIVE_BOUNDARY:
        primitive_states = _reflective_left_boundary(primitive_states)
    else:
        raise ValueError("Unknown left boundary condition")
    
    if right_boundary == OPEN_BOUNDARY:
        primitive_states = _open_right_boundary(primitive_states)
    elif right_boundary == REFLECTIVE_BOUNDARY:
        primitive_states = _reflective_right_boundary(primitive_states)
    else:
        raise ValueError("Unknown right boundary condition")
    
    return primitive_states

@jaxtyped(typechecker=typechecker)
@jax.jit
def _open_right_boundary(primitive_states: Float[Array, "num_vars num_cells"]) -> Float[Array, "num_vars num_cells"]:
    """Apply the open boundary condition to the right boundary.
    
    Args:
        primitive_states: The primitive state array.

    Returns:
        The primitive state array with the open boundary condition applied.
    """

    primitive_states = primitive_states.at[:, -1].set(primitive_states[:, -3])
    primitive_states = primitive_states.at[:, -2].set(primitive_states[:, -3])
    return primitive_states

@jaxtyped(typechecker=typechecker)
@jax.jit
def _open_left_boundary(primitive_states: Float[Array, "num_vars num_cells"]) -> Float[Array, "num_vars num_cells"]:
    """Apply the open boundary condition to the left boundary.

    Args:
        primitive_states: The primitive state array.

    Returns:
        The primitive state array with
        the open boundary condition applied.

    """

    primitive_states = primitive_states.at[:, 0].set(primitive_states[:, 2])
    primitive_states = primitive_states.at[:, 1].set(primitive_states[:, 2])
    return primitive_states

@jaxtyped(typechecker=typechecker)
@jax.jit
def _reflective_left_boundary(primitive_states: Float[Array, "num_vars num_cells"]) -> Float[Array, "num_vars num_cells"]:
    """Apply the reflective boundary condition to the left boundary.

    Args:
        primitive_states: The primitive state array.

    Returns:
        The primitive state array with
        the reflective boundary condition applied.

    """

    primitive_states = primitive_states.at[:, 0].set(primitive_states[:, 2])
    primitive_states = primitive_states.at[1, 0].set(-primitive_states[1, 2])

    primitive_states = primitive_states.at[:, 1].set(primitive_states[:, 2])
    primitive_states = primitive_states.at[1, 1].set(-primitive_states[1, 2])

    return primitive_states

@jaxtyped(typechecker=typechecker)
@jax.jit
def _reflective_right_boundary(primitive_states: Float[Array, "num_vars num_cells"]) -> Float[Array, "num_vars num_cells"]:
    """Apply the reflective boundary condition to the right boundary.

    Args:
        primitive_states: The primitive state array.

    Returns:
        The primitive state array with
        the reflective boundary condition applied.

    """

    primitive_states = primitive_states.at[:, -1].set(primitive_states[:, -3])
    primitive_states = primitive_states.at[1, -1].set(-primitive_states[1, -3])

    primitive_states = primitive_states.at[:, -2].set(primitive_states[:, -3])
    primitive_states = primitive_states.at[1, -2].set(-primitive_states[1, -3])

    return primitive_states