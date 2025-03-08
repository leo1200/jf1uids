"""
Convenience functions for operations that combine multiple elements
of an array based on some stencil, e.g. b_i <- a_{i + 1} + a_{i - 1}.
Allows for code "closer to the math".
"""

# TODO: use stencil operations throughout the codebase

# general
import jax
import jax.numpy as jnp
from functools import partial

# typechecking
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped
from typing import Tuple, Union

    
@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['indices', 'axis', 'zero_pad'])
def _stencil_add(
        input_array: jnp.ndarray,
        indices: Tuple[int, ...],
        factors: Tuple[Union[float, Float[Array, ""]], ...],
        axis: int,
        zero_pad: bool = True
) -> jnp.ndarray:
    """
    Combines elements of an array additively
        output_i <- sum_j factors_j * input_array_{i + indices_j}

    By default, the output is zero-padded to the same shape as 
    the input array (as we handle boundaries via ghost cells in 
    the overall simulation code). This behavior can be disabled,
    then the output will have a different shape along the specified
    axis.

    Args:
        input_array: The array to operate on.
        indices: output_i <- sum_j factors_j * input_array_{i + indices_j}
        factors: output_i <- sum_j factors_j * input_array_{i + indices_j}
        axis: The axis along which to operate.
        zero_pad: Whether to zero-pad the output to have the same shape as the input.
        
    Returns:
        output_i <- sum_j factors_j * input_array_{i + indices_j}
    """

    num_cells = input_array.shape[axis]

    first_write_index = -min(0, min(indices))
    last_write_index = num_cells - max(0, max(indices))

    # for the first write index, the elements considered are
    first_handled_indices = tuple(first_write_index + index for index in indices)

    # for the last write index, the elements considered are
    last_handled_indices = tuple(last_write_index + index for index in indices)

    output = (
        sum(
            factor * jax.lax.slice_in_dim(
                input_array,
                first_handled_index,
                last_handled_index,
                axis = axis
            )
            for factor, first_handled_index, last_handled_index in zip(
                factors, first_handled_indices, last_handled_indices
            )
        )
    )

    if zero_pad:
        result = jnp.zeros_like(input_array)
        selection = (
            (slice(None),) * axis +
            (slice(first_write_index, last_write_index),) +
            (slice(None),)*(input_array.ndim - axis - 1)
        )
        result = result.at[selection].set(output)
        return result
    else:
        return output