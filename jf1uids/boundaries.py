import jax.numpy as jnp
from functools import partial
import jax

OPEN_BOUNDARY = 0
REFLECTIVE_BOUNDARY = 1

@partial(jax.jit, static_argnames=['left_boundary', 'right_boundary'])
def _boundary_handler(primitive_states, left_boundary, right_boundary):
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

@jax.jit
def _open_right_boundary(primitive_states):
    primitive_states = primitive_states.at[:, -1].set(primitive_states[:, -3])
    primitive_states = primitive_states.at[:, -2].set(primitive_states[:, -3])
    return primitive_states

@jax.jit
def _open_left_boundary(primitive_states):
    primitive_states = primitive_states.at[:, 0].set(primitive_states[:, 2])
    primitive_states = primitive_states.at[:, 1].set(primitive_states[:, 2])
    return primitive_states

@jax.jit
def _reflective_left_boundary(primitive_states):

    primitive_states = primitive_states.at[:, 0].set(primitive_states[:, 2])
    primitive_states = primitive_states.at[1, 0].set(-primitive_states[1, 2])

    primitive_states = primitive_states.at[:, 1].set(primitive_states[:, 2])
    primitive_states = primitive_states.at[1, 1].set(-primitive_states[1, 2])

    return primitive_states

@jax.jit
def _reflective_right_boundary(primitive_states):

    primitive_states = primitive_states.at[:, -1].set(primitive_states[:, -3])
    primitive_states = primitive_states.at[1, -1].set(-primitive_states[1, -3])

    primitive_states = primitive_states.at[:, -2].set(primitive_states[:, -3])
    primitive_states = primitive_states.at[1, -2].set(-primitive_states[1, -3])

    return primitive_states