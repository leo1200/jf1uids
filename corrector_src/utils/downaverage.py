import jax.numpy as jnp


def downaverage_state(state: jnp.ndarray, downsample_factor: int) -> jnp.ndarray:
    """
    Downaverages the spatial dimensions of a state array using block reshaping.

    This function is designed for a state array with the shape
    (NUM_VARS, H, W) and reduces it to (NUM_VARS, h, w) by
    averaging over non-overlapping blocks.

    Args:
        state: The input JAX array with shape (NUM_VARS, H, W).
        target_shape: A tuple (h, w) representing the desired output
                      spatial dimensions. H must be divisible by h, and W
                      must be divisible by w.

    Returns:
        The downaveraged JAX array with shape (NUM_VARS, h, w).
    """
    num_vars, h_in, w_in = state.shape
    h_out, w_out = h_in // downsample_factor, w_in // downsample_factor

    if h_in % h_out != 0 or w_in % w_out != 0:
        raise ValueError(
            f"Input shape {(h_in, w_in)} is not divisible by target shape {(h_out, w_out)}"
        )
    h_factor = h_in // h_out
    w_factor = w_in // w_out

    reshaped = state.reshape(num_vars, h_out, h_factor, w_out, w_factor)
    downaveraged = reshaped.mean(axis=(2, 4))

    return downaveraged
