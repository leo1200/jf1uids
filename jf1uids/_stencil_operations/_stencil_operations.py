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
from typing import Union

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['indexA', 'indexB', 'axis', 'zero_pad'])
def _stencil_add(
        input_array: jnp.ndarray,
        indexA: int,
        indexB: int,
        axis: int,
        zero_pad: bool = True,
        factorA: Union[float, Float[Array, ""]] = 1.0,
        factorB: Union[float, Float[Array, ""]] = 1.0
) -> jnp.ndarray:
    """
    Combines elements of an array additively
        output_i <- factorA * input_array_{i + indexA} + factorB * input_array_{i + indexB}

    By default, the output is zero-padded to the same shape as 
    the input array (as we handle boundaries via ghost cells in 
    the overall simulation code). This behavior can be disabled,
    then the output will have a different shape along the specified
    axis.

    Args:
        input_array: The array to operate on.
        indexA: output_i <- factorA * input_array_{i + indexA} + factorB * input_array_{i + indexB}
        indexB: output_i <- factorA * input_array_{i + indexA} + factorB * input_array_{i + indexB}
        axis: The axis along which to operate.
        zero_pad: Whether to zero-pad the output to have the same shape as the input.
        factorA: output_i <- factorA * input_array_{i + indexA} + factorB * input_array_{i + indexB}
        factorB: output_i <- factorA * input_array_{i + indexA} + factorB * input_array_{i + indexB}
        
    Returns:
        output_i <- factorA * input_array_{i + indexA} + factorB * input_array_{i + indexB}

    """

    num_cells = input_array.shape[axis]

    first_write_index = -min(0, min(indexA, indexB))
    last_write_index = num_cells - max(0, max(indexA, indexB))

    # for the first write index, the elements considered are
    first_handled_indexA = first_write_index + indexA
    first_handled_indexB = first_write_index + indexB

    # for the last write index, the elements considered are
    last_handled_indexA = last_write_index + indexA
    last_handled_indexB = last_write_index + indexB

    output = (
        factorA * jax.lax.slice_in_dim(
            input_array,
            first_handled_indexA,
            last_handled_indexA,
            axis = axis
        ) +
        factorB * jax.lax.slice_in_dim(
            input_array,
            first_handled_indexB,
            last_handled_indexB,
            axis = axis
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