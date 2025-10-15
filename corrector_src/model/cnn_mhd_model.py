import jax
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray

# general
from functools import partial

from jax.experimental import checkify


class CorrectorCNN(eqx.Module):
    """
    A simple CNN that maps an input of shape (C, H, W) to an output of the same shape.
    """

    layers: eqx.nn.Sequential

    def __init__(self, in_channels: int, hidden_channels: int, *, key: PRNGKeyArray):
        # We need a key for each convolutional layer
        key1, key2, key3 = jax.random.split(key, 3)

        # A simple 3-layer CNN.
        # Note the use of padding=1 with kernel_size=3 to keep the
        # spatial dimensions (height and width) the same.
        self.layers = eqx.nn.Sequential(
            [
                eqx.nn.Conv3d(in_channels, hidden_channels, 3, padding=1, key=key1),
                eqx.nn.Lambda(jax.nn.relu),
                eqx.nn.Conv3d(hidden_channels, hidden_channels, 3, padding=1, key=key2),
                eqx.nn.Lambda(jax.nn.relu),
                eqx.nn.Conv3d(
                    hidden_channels, in_channels, 3, padding=1, key=key3, use_bias=False
                ),
            ]
        )

    def __call__(self, x: Float[Array, "num_vars h w"]) -> Float[Array, "num_vars h w"]:
        """
        The forward pass of the model.
        """
        # Pass the input through the network to get the correction term
        correction = self.layers(x)
        # Add the learned correction to the original input
        return correction
