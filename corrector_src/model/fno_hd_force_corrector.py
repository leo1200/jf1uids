import jax
import equinox as eqx
import jax.numpy as jnp

# equinox_spectral_conv3d.py
import math
from typing import Optional

from jaxtyping import Array, Float, PRNGKeyArray


class SpectralConv3d(eqx.Module):
    """
    3D spectral convolution in JAX/Equinox.

    - Operates at the input resolution (no spectral upsampling by default).
    - Uses rfftn/irfftn: stores only non-redundant kz >= 0 modes.
    - Splits the (kx, ky) plane into 4 regions (quadrants) and uses separate
      complex weights for each quadrant (weights1..4).
    - shifting_modes >= 0 shifts the window of modes away from DC.
    """

    in_channels: int
    out_channels: int
    modes1: int
    modes2: int
    modes3: int
    shifting_modes: int

    # weight arrays (complex64)
    weights1: jnp.ndarray
    weights2: jnp.ndarray
    weights3: jnp.ndarray
    weights4: jnp.ndarray

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        modes3: int,
        shifting_modes: int = 0,
        *,
        key: Optional[jax.random.KeyArray] = None,
    ):
        if key is None:
            key = jax.random.PRNGKey(0)
        k1, k2, k3, k4 = jax.random.split(key, 4)

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        self.modes3 = int(modes3)
        self.shifting_modes = int(max(0, shifting_modes))

        # Recommended initialization scale (as discussed)
        scale = 1.0 / math.sqrt(max(1, self.in_channels))

        def _init_complex(key, shape):
            # real + 1j * imag both normal(0,1) scaled
            k_r, k_i = jax.random.split(key)
            r = jax.random.normal(k_r, shape)
            i = jax.random.normal(k_i, shape)
            return (scale * (r + 1j * i)).astype(jnp.complex64)

        wshape = (
            self.in_channels,
            self.out_channels,
            self.modes1,
            self.modes2,
            self.modes3,
        )
        self.weights1 = _init_complex(k1, wshape)
        self.weights2 = _init_complex(k2, wshape)
        self.weights3 = _init_complex(k3, wshape)
        self.weights4 = _init_complex(k4, wshape)

    # complex multiplication shorthand
    def compl_mul3d(self, x_ft_region: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
        # x_ft_region: (batch, in_c, kx, ky, kz)
        # w: (in_c, out_c, kx, ky, kz)
        # returns (batch, out_c, kx, ky, kz)
        return jnp.einsum("bixyz,ioxyz->boxyz", x_ft_region, w)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Arguments
        ---------
        x: jnp.ndarray
            shape (batch, in_channels, nx, ny, nz) with real dtype (float32/64).
        Returns
        -------
        out: jnp.ndarray
            shape (batch, out_channels, nx, ny, nz) real-valued output.
        """
        assert x.ndim == 5, "input must be (batch, in_c, nx, ny, nz)"
        batch, in_c, nx, ny, nz = x.shape
        assert in_c == self.in_channels

        # rfftn keeps kz >= 0 (last axis reduced to nz//2 + 1)
        x_ft = jnp.fft.rfftn(
            x, axes=(-3, -2, -1)
        )  # shape: (b, in_c, nx, ny, nz//2 + 1)

        # compute how many modes we'll actually use (bounded by available spectrum)
        # note: for nx, we treat available kx indices as nx (full signed range), but usable positive half is nx//2
        m1 = int(min(self.modes1, x_ft.shape[-3] // 2))
        m2 = int(min(self.modes2, x_ft.shape[-2] // 2))
        m3 = int(min(self.modes3, x_ft.shape[-1]))

        s = int(self.shifting_modes)
        assert m1 + s <= x_ft.shape[-3] // 2 + 0, "modes1 + shift too large for input"
        assert m2 + s <= x_ft.shape[-2] // 2 + 0, "modes2 + shift too large for input"
        assert m3 + s <= x_ft.shape[-1], "modes3 + shift too large for input"

        # define slices
        kx_pos = slice(s, s + m1)
        ky_pos = slice(s, s + m2)
        kz_pos = slice(s, s + m3)

        if s > 0:
            kx_neg = slice(-m1 - s, -s)
            ky_neg = slice(-m2 - s, -s)
        else:
            kx_neg = slice(-m1, None)
            ky_neg = slice(-m2, None)

        # prepare out FT buffer at the same resolution as if we would call irfftn with the original sizes
        out_ft = jnp.zeros(
            (batch, self.out_channels, nx, ny, nz // 2 + 1), dtype=jnp.complex64
        )

        # Quadrant 1: (+kx, +ky)
        if (m1 > 0) and (m2 > 0) and (m3 > 0):
            out_ft = out_ft.at[..., kx_pos, ky_pos, kz_pos].set(
                self.compl_mul3d(
                    x_ft[..., kx_pos, ky_pos, kz_pos], self.weights1[..., :m1, :m2, :m3]
                )
            )

            # Quadrant 2: (-kx, +ky)
            out_ft = out_ft.at[..., kx_neg, ky_pos, kz_pos].set(
                self.compl_mul3d(
                    x_ft[..., kx_neg, ky_pos, kz_pos], self.weights2[..., :m1, :m2, :m3]
                )
            )

            # Quadrant 3: (+kx, -ky)
            out_ft = out_ft.at[..., kx_pos, ky_neg, kz_pos].set(
                self.compl_mul3d(
                    x_ft[..., kx_pos, ky_neg, kz_pos], self.weights3[..., :m1, :m2, :m3]
                )
            )

            # Quadrant 4: (-kx, -ky)
            out_ft = out_ft.at[..., kx_neg, ky_neg, kz_pos].set(
                self.compl_mul3d(
                    x_ft[..., kx_neg, ky_neg, kz_pos], self.weights4[..., :m1, :m2, :m3]
                )
            )

        # inverse rFFT -> real physical space
        out = jnp.fft.irfftn(out_ft, s=(nx, ny, nz), axes=(-3, -2, -1))

        # output shape: (batch, out_c, nx, ny, nz) real-valued
        return jnp.real(out)


class FNO3dLayer(eqx.Module):
    spectral_conv: SpectralConv3d
    w: jnp.ndarray  # pointwise linear weight (in_c -> out_c)
    b: jnp.ndarray  # pointwise bias

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        modes3: int,
        shifting_modes: int = 0,
        *,
        key: jax.random.KeyArray = None,
    ):
        if key is None:
            key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)

        # spectral convolution
        self.spectral_conv = SpectralConv3d(
            in_channels, out_channels, modes1, modes2, modes3, shifting_modes, key=k1
        )

        # pointwise linear: weight shape (in_channels, out_channels), bias shape (out_channels,)
        scale = 1.0 / jnp.sqrt(in_channels)
        self.w = scale * jax.random.normal(k2, (in_channels, out_channels))
        self.b = jnp.zeros((out_channels,))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: (batch, in_channels, nx, ny, nz)
        returns: (batch, out_channels, nx, ny, nz)
        """
        # spectral convolution
        x_spec = self.spectral_conv(x)

        # pointwise linear
        x_lin = (
            jnp.einsum("bixyz,io->boxyz", x, self.w) + self.b[None, :, None, None, None]
        )

        return x_spec + x_lin


class FNO(eqx.Module):
    layers: list
    activation: callable = eqx.static_field()

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        n_fourier_layers: int,
        fourier_modes: int,
        shifting_modes: int,
        key: jax.random.KeyArray,
        activation=jax.nn.gelu,
    ):
        self.activation = activation
        keys = jax.random.split(key, n_fourier_layers + 2)
        self.layers = []

        # Initial Conv3d layer
        self.layers.append(
            eqx.nn.Conv3d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1,
                key=keys[0],
            )
        )

        # Fourier layers with residual connections
        for i in range(n_fourier_layers):
            self.layers.append(
                FNO3dLayer(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    modes1=fourier_modes,
                    modes2=fourier_modes,
                    modes3=fourier_modes,
                    shifting_modes=shifting_modes,
                    key=keys[i + 1],
                )
            )

        # Final Conv3d layer
        self.layers.append(
            eqx.nn.Conv3d(
                in_channels=hidden_channels,
                out_channels=in_channels,
                kernel_size=3,
                padding=1,
                key=keys[-1],
            )
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        out = self.layers[0](x)

        for layer in self.layers[1:-1]:
            # Residual connection
            y = layer(out)
            out = self.activation(out + y)
        out = self.layers[-1](out)
        return out
