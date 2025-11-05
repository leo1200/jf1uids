# import jax.numpy as jnp


# def downaverage_state(state: jnp.ndarray, downscale_factor: int) -> jnp.ndarray:
#     """
#     Downaverages the spatial dimensions of a state array using block reshaping.

#     This function is designed for a state array with the shape
#     (NUM_VARS, H, W) and reduces it to (NUM_VARS, h, w) by
#     averaging over non-overlapping blocks.

#     Args:
#         state: The input JAX array with shape (NUM_VARS, H, W).
#         target_shape: A tuple (h, w) representing the desired output
#                       spatial dimensions. H must be divisible by h, and W
#                       must be divisible by w.

#     Returns:
#         The downaveraged JAX array with shape (NUM_VARS, h, w).
#     """
#     num_vars, h_in, w_in, d_in = state.shape
#     downscale_factor = int(downscale_factor)
#     h_out, w_out, d_out = (
#         h_in // downscale_factor,
#         w_in // downscale_factor,
#         d_in // downscale_factor,
#     )

#     if h_in % h_out != 0 or w_in % w_out != 0:
#         raise ValueError(
#             f"Input shape {(h_in, w_in)} is not divisible by target shape {(h_out, w_out)}"
#         )
#     h_factor = h_in // h_out
#     w_factor = w_in // w_out
#     d_factor = d_in // d_out

#     reshaped = state.reshape(
#         num_vars, h_out, h_factor, w_out, w_factor, d_out, d_factor
#     )
#     downaveraged = reshaped.mean(axis=(2, 4, 6))

#     return downaveraged


# def downaverage_states(state: jnp.ndarray, downscale_factor: int) -> jnp.ndarray:
#     """
#     Downaverages the spatial dimensions of a state array using block reshaping.

#     This function is designed for a state array with the shape
#     (batch, NUM_VARS, H, W, Z) and reduces it to (NUM_VARS, h, w) by
#     averaging over non-overlapping blocks.

#     Args:
#         state: The input JAX array with shape (NUM_VARS, H, W).
#         target_shape: A tuple (h, w) representing the desired output
#                       spatial dimensions. H must be divisible by h, and W
#                       must be divisible by w.

#     Returns:
#         The downaveraged JAX array with shape (NUM_VARS, h, w).
#     """
#     n_states, num_vars, h_in, w_in, d_in = state.shape
#     h_out, w_out, d_out = (
#         h_in // downscale_factor,
#         w_in // downscale_factor,
#         d_in // downscale_factor,
#     )

#     if h_in % h_out != 0 or w_in % w_out != 0:
#         raise ValueError(
#             f"Input shape {(h_in, w_in)} is not divisible by target shape {(h_out, w_out)}"
#         )
#     h_factor = h_in // h_out
#     w_factor = w_in // w_out
#     d_factor = d_in // d_out

#     reshaped = state.reshape(
#         n_states, num_vars, h_out, h_factor, w_out, w_factor, d_out, d_factor
#     )
#     downaveraged = reshaped.mean(axis=(3, 5, 7))

#     return downaveraged


import jax.numpy as jnp


def downaverage(state: jnp.ndarray, downscale_factor: int) -> jnp.ndarray:
    """
    Downaverage spatial (and depth) dimensions by non-overlapping block averaging.

    This function accepts either:
      - unbatched input of shape (NUM_VARS, H, W, D)
      - batched input of shape (N, NUM_VARS, H, W, D)

    The downscale_factor is an integer factor by which each spatial/depth
    dimension (H, W, D) is reduced:
        h_out = H // downscale_factor
        w_out = W // downscale_factor
        d_out = D // downscale_factor

    Args:
        state: JAX ndarray with shape (NUM_VARS, H, W, D) or (N, NUM_VARS, H, W, D).
        downscale_factor: integer factor > 0 that divides H, W and D.

    Returns:
        downaveraged array with shape:
            - (NUM_VARS, h_out, w_out, d_out) for unbatched input
            - (N, NUM_VARS, h_out, w_out, d_out) for batched input

    Raises:
        ValueError: if input ndim is not 4 or 5, or if spatial/depth dims are not divisible
                    by downscale_factor.
    """
    downscale_factor = int(downscale_factor)
    if downscale_factor <= 0:
        raise ValueError("downscale_factor must be a positive integer")

    if state.ndim == 4:
        # (NUM_VARS, H, W, D)
        num_vars, H, W, D = state.shape
        if (
            (H % downscale_factor) != 0
            or (W % downscale_factor) != 0
            or (D % downscale_factor) != 0
        ):
            raise ValueError(
                f"Spatial/depth dims {(H, W, D)} must be divisible by downscale_factor={downscale_factor}"
            )
        h_out = H // downscale_factor
        w_out = W // downscale_factor
        d_out = D // downscale_factor

        # reshape into blocks and mean over block axes
        reshaped = state.reshape(
            num_vars,
            h_out,
            downscale_factor,
            w_out,
            downscale_factor,
            d_out,
            downscale_factor,
        )
        # mean over the block axes (2, 4, 6)
        downaveraged = reshaped.mean(axis=(2, 4, 6))
        return downaveraged

    elif state.ndim == 5:
        # (N, NUM_VARS, H, W, D)
        N, num_vars, H, W, D = state.shape
        if (
            (H % downscale_factor) != 0
            or (W % downscale_factor) != 0
            or (D % downscale_factor) != 0
        ):
            raise ValueError(
                f"Spatial/depth dims {(H, W, D)} must be divisible by downscale_factor={downscale_factor}"
            )
        h_out = H // downscale_factor
        w_out = W // downscale_factor
        d_out = D // downscale_factor

        reshaped = state.reshape(
            N,
            num_vars,
            h_out,
            downscale_factor,
            w_out,
            downscale_factor,
            d_out,
            downscale_factor,
        )
        # mean over the block axes (3, 5, 7)
        downaveraged = reshaped.mean(axis=(3, 5, 7))
        return downaveraged

    else:
        raise ValueError(
            f"Unsupported input ndim {state.ndim}. Expected 4 (NUM_VARS,H,W,D) or "
            f"5 (N,NUM_VARS,H,W,D)."
        )
